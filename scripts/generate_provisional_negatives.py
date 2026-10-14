#!/usr/bin/env python3
"""Prepare and generate provisional LLM hard-negative cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from thought_action_retrieval.inference.client import InferenceClient, InferenceError
from thought_action_retrieval.matching.provisional_negatives import (
    PILOT_TIER,
    PROVISIONAL_STATUS,
    build_generation_request,
    build_split_indexes,
    prepare_provisional_rows,
    select_validated_negatives,
    validate_generation_response,
)


ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "prompts" / "provisional_negative_generation.md"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain an object")
        rows.append(value)
    return rows


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_publish(output_dir: Path, files: dict[str, bytes]) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for relative, content in files.items():
            path = staged / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        staged.replace(output_dir)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_update_directory(output_dir: Path, files: dict[str, bytes]) -> None:
    """Publish a related file set by swapping a fully staged directory."""

    if not output_dir.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {output_dir}")
    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.finalize.", dir=output_dir.parent))
    backup = output_dir.parent / f".{output_dir.name}.backup"
    if backup.exists():
        raise FileExistsError(f"stale finalize backup exists: {backup}")
    try:
        shutil.copytree(output_dir, staged, dirs_exist_ok=True)
        for relative, content in files.items():
            path = staged / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        output_dir.replace(backup)
        try:
            staged.replace(output_dir)
        except BaseException:
            backup.replace(output_dir)
            raise
        shutil.rmtree(backup)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        if backup.exists() and not output_dir.exists():
            backup.replace(output_dir)
        raise


def _load_prepared_dataset(
    dataset_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pairs_path = dataset_dir / "provisional_pairs.jsonl"
    requests_path = dataset_dir / "generation_requests.jsonl"
    manifest_path = dataset_dir / "prepare_manifest.json"
    if not pairs_path.is_file() or not requests_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("prepared pairs, requests, or manifest are missing")
    pairs = _read_jsonl(pairs_path)
    requests = _read_jsonl(requests_path)
    manifest = _read_json(manifest_path)
    if manifest.get("candidate_count") != len(pairs):
        raise ValueError("prepared candidate count does not match manifest")
    if manifest.get("request_count") != len(requests) or len(requests) != len(pairs):
        raise ValueError("prepared request count does not match pairs")
    prompt_hash = hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()
    if (manifest.get("prompt") or {}).get("sha256") != prompt_hash:
        raise ValueError("generation prompt hash does not match prepare manifest")
    pair_ids = {str(row.get("candidate_id")) for row in pairs}
    request_ids = {str(row.get("candidate_id")) for row in requests}
    if len(pair_ids) != len(pairs) or pair_ids != request_ids:
        raise ValueError("prepared pair and request candidate IDs do not reconcile")
    return pairs, requests, manifest


def _prompt_sections() -> tuple[str, str]:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    try:
        system_and_user = text.split("## System Prompt", 1)[1]
        system, user = system_and_user.split("## User Prompt", 1)
    except (IndexError, ValueError):
        raise ValueError("negative-generation prompt sections are malformed") from None
    return system.strip(), user.strip()


def _cache_key(
    request: dict[str, Any],
    manifest: dict[str, Any],
    *,
    endpoint: str,
    request_path: str,
    model: str | None,
    json_response_format: bool,
) -> str:
    material = {
        "request_id": request["request_id"],
        "request_sha256": hashlib.sha256(_json_bytes(request)).hexdigest(),
        "prompt_sha256": manifest["prompt"]["sha256"],
        "source_manifest_sha256": manifest["input"]["build_manifest_sha256"],
        "endpoint": endpoint.rstrip("/"),
        "request_path": request_path,
        "model": model,
        "json_response_format": json_response_format,
    }
    return hashlib.sha256(_json_bytes(material)).hexdigest()


def _state_path(state_dir: Path, request: dict[str, Any]) -> Path:
    digest = hashlib.sha256(str(request["request_id"]).encode("utf-8")).hexdigest()
    return state_dir / f"{digest}.json"


def _read_matching_state(
    state_dir: Path,
    request: dict[str, Any],
    cache_key: str,
) -> dict[str, Any] | None:
    path = _state_path(state_dir, request)
    if not path.is_file():
        return None
    state = _read_json(path)
    if state.get("request_id") != request["request_id"] or state.get("cache_key") != cache_key:
        return None
    return state


def _rebuild_generation_aggregates(
    dataset_dir: Path,
    pairs: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    endpoint: str,
    request_path: str,
    model: str | None,
    json_response_format: bool,
) -> tuple[int, int]:
    state_dir = dataset_dir / ".generation_state"
    pair_by_id = {row["candidate_id"]: row for row in pairs}
    indexes = build_split_indexes(pairs)
    responses: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for request in requests:
        key = _cache_key(
            request,
            manifest,
            endpoint=endpoint,
            request_path=request_path,
            model=model,
            json_response_format=json_response_format,
        )
        state = _read_matching_state(state_dir, request, key)
        if state is None:
            continue
        if state.get("status") == "success":
            parent = pair_by_id[request["candidate_id"]]
            validated = validate_generation_response(
                parent, request, state["response"], indexes
            )
            responses.append(
                {
                    "request_id": request["request_id"],
                    "candidate_id": request["candidate_id"],
                    "cache_key": key,
                    "response": state["response"],
                    "accepted_count": len(validated.accepted),
                    "rejected": list(validated.rejected),
                    "supervision_status": PROVISIONAL_STATUS,
                    "experiment_tier": PILOT_TIER,
                    "human_reviewed": False,
                }
            )
            for negative in select_validated_negatives(
                request["candidate_id"], validated.accepted
            ):
                candidates.append({"candidate_id": request["candidate_id"], **negative})
        elif state.get("status") == "failure":
            failures.append(
                {
                    "request_id": request["request_id"],
                    "candidate_id": request["candidate_id"],
                    "cache_key": key,
                    "error": state.get("error", "unknown generation failure"),
                    "experiment_tier": PILOT_TIER,
                }
            )
    _atomic_write(dataset_dir / "negative_responses.jsonl", _jsonl_bytes(responses))
    _atomic_write(dataset_dir / "negative_failures.jsonl", _jsonl_bytes(failures))
    _atomic_write(dataset_dir / "negative_candidates.jsonl", _jsonl_bytes(candidates))
    return len(responses), len(failures)


def prepare(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    candidate_path = input_dir / "annotation_candidates.jsonl"
    source_manifest_path = input_dir / "build_manifest.json"
    if not candidate_path.is_file() or not source_manifest_path.is_file():
        raise FileNotFoundError("Phase 1 candidates or build manifest are missing")
    candidates = _read_jsonl(candidate_path)
    source_manifest = _read_json(source_manifest_path)
    expected_count = (source_manifest.get("counts") or {}).get(
        "annotation_candidates"
    )
    if expected_count != len(candidates):
        raise ValueError(
            f"source candidate count mismatch: {expected_count} != {len(candidates)}"
        )
    rows = prepare_provisional_rows(candidates, seed=args.seed)
    indexes = build_split_indexes(rows)
    requests = tuple(
        build_generation_request(row, indexes, max_pool_actions=args.max_pool_actions)
        for row in rows
    )
    split_counts = Counter(str(row["split"]) for row in rows)
    prompt_hash = hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "provisional_negative_prepare_v1",
        "candidate_count": len(rows),
        "request_count": len(requests),
        "split_counts": dict(sorted(split_counts.items())),
        "seed": args.seed,
        "max_pool_actions": args.max_pool_actions,
        "supervision_status": PROVISIONAL_STATUS,
        "experiment_tier": PILOT_TIER,
        "human_reviewed": False,
        "input": {
            "candidate_file": candidate_path.name,
            "candidate_sha256": _sha256(candidate_path),
            "build_manifest_sha256": _sha256(source_manifest_path),
        },
        "prompt": {
            "path": str(PROMPT_PATH.relative_to(ROOT)),
            "sha256": prompt_hash,
        },
    }
    _atomic_publish(
        output_dir,
        {
            "provisional_pairs.jsonl": _jsonl_bytes(rows),
            "generation_requests.jsonl": _jsonl_bytes(requests),
            "prepare_manifest.json": _json_bytes(manifest),
        },
    )
    print(f"prepared {len(rows)} provisional pairs in {output_dir}")
    return 0


def generate(args: argparse.Namespace) -> int:
    dataset_dir = args.dataset_dir.resolve()
    api_key = os.environ.get("INF_API_KEY", "").strip()
    if not api_key:
        raise ValueError("INF_API_KEY must be set in the environment")
    pairs, requests, manifest = _load_prepared_dataset(dataset_dir)
    endpoint = args.endpoint or os.environ.get("INF_API_ENDPOINT", "").strip()
    if not endpoint:
        raise ValueError("an inference endpoint is required")
    request_path = args.request_path or os.environ.get("INF_API_REQUEST_PATH", "/")
    model = args.model if args.model is not None else os.environ.get("INF_MODEL_NAME")
    model = model.strip() if isinstance(model, str) and model.strip() else None
    state_dir = dataset_dir / ".generation_state"
    pending: list[tuple[dict[str, Any], str]] = []
    for request in requests:
        key = _cache_key(
            request,
            manifest,
            endpoint=endpoint,
            request_path=request_path,
            model=model,
            json_response_format=args.json_response_format,
        )
        state = _read_matching_state(state_dir, request, key)
        if state is None or state.get("status") != "success":
            pending.append((request, key))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        pending = pending[: args.limit]

    if pending:
        client = InferenceClient(
            endpoint=endpoint,
            api_key=api_key,
            request_path=request_path,
            model=model,
            timeout=args.timeout,
            retries=args.retries,
            json_response_format=args.json_response_format,
        )
        client.preflight()
        system_prompt, user_template = _prompt_sections()
        pair_by_id = {row["candidate_id"]: row for row in pairs}
        indexes = build_split_indexes(pairs)
        consecutive_failures = 0
        for request, key in pending:
            try:
                user_prompt = user_template.replace(
                    "{request_json}",
                    json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True),
                )
                response = client.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                validated = validate_generation_response(
                    pair_by_id[request["candidate_id"]], request, response, indexes
                )
                state = {
                    "schema_version": "provisional_negative_state_v1",
                    "status": "success",
                    "request_id": request["request_id"],
                    "candidate_id": request["candidate_id"],
                    "cache_key": key,
                    "response": response,
                    "accepted_count": len(validated.accepted),
                    "rejected_count": len(validated.rejected),
                }
                consecutive_failures = 0
            except (InferenceError, ValueError) as error:
                state = {
                    "schema_version": "provisional_negative_state_v1",
                    "status": "failure",
                    "request_id": request["request_id"],
                    "candidate_id": request["candidate_id"],
                    "cache_key": key,
                    "error": str(error),
                }
                consecutive_failures += 1
            state_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write(_state_path(state_dir, request), _json_bytes(state))
            if consecutive_failures >= args.max_consecutive_failures:
                break

    response_count, failure_count = _rebuild_generation_aggregates(
        dataset_dir,
        pairs,
        requests,
        manifest,
        endpoint=endpoint,
        request_path=request_path,
        model=model,
        json_response_format=args.json_response_format,
    )
    print(
        f"generation state: {response_count}/{len(requests)} responses, "
        f"{failure_count} failures"
    )
    return 1 if failure_count else 0


def finalize(args: argparse.Namespace) -> int:
    dataset_dir = args.dataset_dir.resolve()
    pairs, requests, prepare_manifest = _load_prepared_dataset(dataset_dir)
    response_path = dataset_dir / "negative_responses.jsonl"
    failure_path = dataset_dir / "negative_failures.jsonl"
    if not response_path.is_file() or not failure_path.is_file():
        raise FileNotFoundError("generation aggregates are missing")
    responses = _read_jsonl(response_path)
    failures = _read_jsonl(failure_path)
    if failures:
        raise ValueError("generation contains failures and cannot be finalized")
    request_by_id = {request["request_id"]: request for request in requests}
    response_by_id = {row.get("request_id"): row for row in responses}
    if len(response_by_id) != len(responses):
        raise ValueError("duplicate generation response request_id")
    if set(response_by_id) != set(request_by_id):
        raise ValueError("generation response set is incomplete or inconsistent")

    pair_by_id = {row["candidate_id"]: row for row in pairs}
    indexes = build_split_indexes(pairs)
    records: list[dict[str, Any]] = []
    rejected_audit: list[dict[str, Any]] = []
    negative_type_counts: Counter[str] = Counter()
    for request in requests:
        parent = pair_by_id[request["candidate_id"]]
        response_row = response_by_id[request["request_id"]]
        validated = validate_generation_response(
            parent, request, response_row["response"], indexes
        )
        negatives = select_validated_negatives(parent["candidate_id"], validated.accepted)
        negative_type_counts.update(row["negative_type"] for row in negatives)
        rejected_audit.extend(
            {"candidate_id": parent["candidate_id"], **row}
            for row in validated.rejected
        )
        records.append(
            {
                "candidate_id": parent["candidate_id"],
                "trajectory_id": parent["trajectory_id"],
                "thought_step": parent.get("thought_step"),
                "split": parent["split"],
                "intent_text": parent["intent_text"],
                "intent_input": parent["intent_input"],
                "positive_action": parent["action"],
                "negative_actions": list(negatives),
                "source": "swe",
                "supervision_status": PROVISIONAL_STATUS,
                "experiment_tier": PILOT_TIER,
                "human_reviewed": False,
            }
        )
    records.sort(key=lambda row: row["candidate_id"])
    split_rows = {
        split: [row for row in records if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    audit = {
        "schema_version": "provisional_negative_audit_v1",
        "label": "FEASIBILITY ONLY — NOT GOLD EVALUATION",
        "candidate_count": len(records),
        "selected_negative_count": sum(len(row["negative_actions"]) for row in records),
        "negative_type_counts": dict(sorted(negative_type_counts.items())),
        "rejected_count": len(rejected_audit),
        "rejected": rejected_audit,
        "supervision_status": PROVISIONAL_STATUS,
        "experiment_tier": PILOT_TIER,
        "human_reviewed": False,
    }
    manifest = {
        "schema_version": "provisional_intent_action_pilot_v1",
        "label": "FEASIBILITY ONLY — NOT GOLD EVALUATION",
        "candidate_count": len(records),
        "response_count": len(responses),
        "negative_count": audit["selected_negative_count"],
        "split_counts": {key: len(value) for key, value in split_rows.items()},
        "negative_type_counts": dict(sorted(negative_type_counts.items())),
        "kimi_row_count": 0,
        "supervision_status": PROVISIONAL_STATUS,
        "experiment_tier": PILOT_TIER,
        "human_reviewed": False,
        "prepare_manifest_sha256": _sha256(dataset_dir / "prepare_manifest.json"),
    }
    files = {
        "provisional_training_records.jsonl": _jsonl_bytes(records),
        "splits/train.jsonl": _jsonl_bytes(split_rows["train"]),
        "splits/validation.jsonl": _jsonl_bytes(split_rows["validation"]),
        "splits/test.jsonl": _jsonl_bytes(split_rows["test"]),
        "negative_audit.json": _json_bytes(audit),
        "pilot_manifest.json": _json_bytes(manifest),
    }
    _atomic_update_directory(dataset_dir, files)
    print(f"finalized {len(records)} provisional training records in {dataset_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--input-dir", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--seed", type=int, default=20260921)
    prepare_parser.add_argument("--max-pool-actions", type=int, default=24)
    prepare_parser.set_defaults(handler=prepare)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--dataset-dir", type=Path, required=True)
    generate_parser.add_argument(
        "--endpoint",
        default=os.environ.get(
            "INF_API_ENDPOINT",
            "https://qjkpcombh9jkcecah8jdkgd5d5beqege.openapi-qb-ai.sii.edu.cn",
        ),
    )
    generate_parser.add_argument(
        "--request-path",
        default=os.environ.get("INF_API_REQUEST_PATH", "/v1/chat/completions"),
    )
    generate_parser.add_argument("--model", default=None)
    generate_parser.add_argument("--json-response-format", action="store_true")
    generate_parser.add_argument("--timeout", type=int, default=120)
    generate_parser.add_argument("--retries", type=int, default=2)
    generate_parser.add_argument("--limit", type=int)
    generate_parser.add_argument("--max-consecutive-failures", type=int, default=3)
    generate_parser.set_defaults(handler=generate)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--dataset-dir", type=Path, required=True)
    finalize_parser.set_defaults(handler=finalize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (FileExistsError, FileNotFoundError, InferenceError, ValueError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
