#!/usr/bin/env python3
"""Build and finalize the auditable Phase 1 Intent--Action dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from thought_action_retrieval.matching.phase1_dataset import (
    BuildConfig,
    export_gold_relations,
    extract_candidates,
    split_by_group,
    validate_reviewed_annotations,
)


ANNOTATION_FIELDS = (
    "candidate_id",
    "source",
    "trajectory_id",
    "thought_step",
    "intent_id",
    "intent_text",
    "source_quote",
    "action_raw",
    "action_serialized",
    "label",
    "annotator",
    "notes",
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _validate_phase0_manifest(
    manifest: dict[str, Any], records: Sequence[dict[str, Any]]
) -> None:
    if manifest.get("schema_version") != "thought_intent_manifest_v1":
        raise ValueError("invalid Phase 0 manifest schema_version")
    trajectory_count = len(records)
    if manifest.get("selected_trajectory_count") != trajectory_count:
        raise ValueError(
            "manifest trajectory count does not match trajectory files: "
            f"{manifest.get('selected_trajectory_count')} != {trajectory_count}"
        )
    source_counts = Counter(str(record["source_type"]) for record in records)
    if manifest.get("source_counts") != dict(sorted(source_counts.items())):
        raise ValueError("manifest source counts do not match trajectory files")
    step_count = sum(len(record["steps"]) for record in records)
    if manifest.get("step_count") != step_count:
        raise ValueError(
            "manifest step count does not match trajectory files: "
            f"{manifest.get('step_count')} != {step_count}"
        )


def _validate_step_schema(step: Any, path: Path, index: int) -> None:
    valid = (
        isinstance(step, dict)
        and type(step.get("thought_step")) is int
        and isinstance(step.get("thought"), str)
        and isinstance(step.get("status"), str)
        and isinstance(step.get("actual_actions"), list)
        and isinstance(step.get("intent_result"), dict)
        and isinstance(step["intent_result"].get("execution_intents"), list)
    )
    if not valid:
        raise ValueError(f"invalid Thought-to-Intent step: {path}:steps[{index}]")


def _load_records(
    input_dir: Path,
) -> tuple[list[dict[str, Any]], list[Path], Path]:
    paths = sorted(input_dir.glob("*.json"))
    records: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    seen_trajectories: set[tuple[str, str]] = set()
    for path in paths:
        if path.name == "manifest.json":
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != "thought_intent_v1"
            or not isinstance(value.get("source_type"), str)
            or not value.get("source_type")
            or not isinstance(value.get("trajectory_id"), str)
            or not value.get("trajectory_id")
            or not isinstance(value.get("steps"), list)
        ):
            raise ValueError(f"invalid Thought-to-Intent trajectory: {path}")
        for index, step in enumerate(value["steps"]):
            _validate_step_schema(step, path, index)
        identity = (value["source_type"], value["trajectory_id"])
        if identity in seen_trajectories:
            raise ValueError(f"duplicate Thought-to-Intent trajectory: {identity}")
        seen_trajectories.add(identity)
        records.append(value)
        source_paths.append(path)
    if not records:
        raise ValueError(f"no Thought-to-Intent trajectory files found in {input_dir}")
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Phase 0 manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Phase 0 manifest must contain a JSON object")
    _validate_phase0_manifest(manifest, records)
    return records, source_paths, manifest_path


def _annotation_csv(rows: Sequence[dict[str, Any]]) -> bytes:
    with tempfile.TemporaryFile(mode="w+", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        for row in rows:
            action = row.get("action") or {}
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "source": row["source"],
                    "trajectory_id": row["trajectory_id"],
                    "thought_step": row["thought_step"],
                    "intent_id": row["intent_id"],
                    "intent_text": row["intent_text"],
                    "source_quote": row["source_quote"],
                    "action_raw": action.get("raw", ""),
                    "action_serialized": action.get("serialized", ""),
                    "label": "",
                    "annotator": "",
                    "notes": "",
                }
            )
        handle.seek(0)
        return handle.read().encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    records, source_paths, phase0_manifest_path = _load_records(input_dir)
    result = extract_candidates(
        records,
        BuildConfig(
            max_action_tokens=args.max_action_tokens,
            max_payload_tokens=args.max_payload_tokens,
        ),
    )
    phase0_manifest = json.loads(phase0_manifest_path.read_text(encoding="utf-8"))
    if result.audit_summary["steps"] != phase0_manifest["step_count"]:
        raise ValueError(
            "candidate audit step count does not match Phase 0 manifest: "
            f"{result.audit_summary['steps']} != {phase0_manifest['step_count']}"
        )
    all_review_rows = result.annotation_candidates + result.unfulfilled_candidates
    manifest = {
        "schema_version": "intent_action_phase1_build_v1",
        "input_directory": str(input_dir),
        "source_files": [
            {"name": path.name, "sha256": _sha256(path)} for path in source_paths
        ],
        "phase0_manifest": {
            "name": phase0_manifest_path.name,
            "sha256": _sha256(phase0_manifest_path),
        },
        "configuration": {
            "max_action_tokens": args.max_action_tokens,
            "max_payload_tokens": args.max_payload_tokens,
        },
        "counts": {
            "source_files": len(source_paths),
            "annotation_candidates": len(result.annotation_candidates),
            "unfulfilled_candidates": len(result.unfulfilled_candidates),
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        _atomic_write(temporary_dir / "audit_summary.json", _json_bytes(result.audit_summary))
        _atomic_write(
            temporary_dir / "annotation_candidates.jsonl",
            _jsonl_bytes(result.annotation_candidates),
        )
        _atomic_write(
            temporary_dir / "unfulfilled_candidates.jsonl",
            _jsonl_bytes(result.unfulfilled_candidates),
        )
        _atomic_write(
            temporary_dir / "annotation_template.csv",
            _annotation_csv(all_review_rows),
        )
        _atomic_write(temporary_dir / "build_manifest.json", _json_bytes(manifest))
        temporary_dir.replace(output_dir)
    except BaseException:
        for path in sorted(temporary_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        temporary_dir.rmdir()
        raise

    print(
        f"built {len(result.annotation_candidates)} annotation candidates and "
        f"{len(result.unfulfilled_candidates)} unfulfilled candidates in {output_dir}"
    )
    return 0


def _ensure_absent(paths: Iterable[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"finalized output already exists: {existing[0]}")


def publish_finalized_artifacts(
    staged_dir: Path,
    dataset_dir: Path,
    *,
    replace: Callable[[Path, Path], Any] = os.replace,
) -> None:
    """Publish a staged artifact set with rollback and manifest-last commit."""

    relative_paths = (
        Path("positives.jsonl"),
        Path("evaluation_relations.jsonl"),
        Path("splits"),
        Path("split_manifest.json"),
    )
    published: list[Path] = []
    try:
        for relative in relative_paths:
            source = staged_dir / relative
            destination = dataset_dir / relative
            replace(source, destination)
            published.append(destination)
    except BaseException:
        for path in reversed(published):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staged_dir, ignore_errors=True)


def finalize_dataset(args: argparse.Namespace) -> int:
    dataset_dir = args.dataset_dir.resolve()
    reviewed_path = args.reviewed_annotations or dataset_dir / "reviewed_annotations.jsonl"
    if not reviewed_path.is_file():
        raise FileNotFoundError(f"reviewed annotations not found: {reviewed_path}")

    candidates = _read_jsonl(dataset_dir / "annotation_candidates.jsonl")
    candidates.extend(_read_jsonl(dataset_dir / "unfulfilled_candidates.jsonl"))
    reviews = _read_jsonl(reviewed_path)
    review_result = validate_reviewed_annotations(candidates, reviews)
    positives, evaluations = export_gold_relations(review_result)
    split_result = split_by_group(list(positives), seed=args.seed)

    paths = [
        dataset_dir / "positives.jsonl",
        dataset_dir / "evaluation_relations.jsonl",
        dataset_dir / "split_manifest.json",
        dataset_dir / "splits",
    ]
    _ensure_absent(paths)

    manifest = dict(split_result.manifest)
    manifest["reviewed_rows"] = len(review_result.reviewed_rows)
    manifest["unreviewed_rows"] = len(review_result.unreviewed_candidate_ids)
    manifest["positive_rows"] = len(positives)
    manifest["evaluation_rows"] = len(evaluations)

    staged_dir = Path(tempfile.mkdtemp(prefix=".finalize.", dir=dataset_dir))
    try:
        _atomic_write(staged_dir / "positives.jsonl", _jsonl_bytes(positives))
        _atomic_write(
            staged_dir / "evaluation_relations.jsonl", _jsonl_bytes(evaluations)
        )
        split_dir = staged_dir / "splits"
        split_dir.mkdir()
        _atomic_write(split_dir / "train.jsonl", _jsonl_bytes(split_result.train))
        _atomic_write(
            split_dir / "validation.jsonl", _jsonl_bytes(split_result.validation)
        )
        _atomic_write(split_dir / "test.jsonl", _jsonl_bytes(split_result.test))
        _atomic_write(staged_dir / "split_manifest.json", _json_bytes(manifest))
        publish_finalized_artifacts(staged_dir, dataset_dir)
    except BaseException:
        shutil.rmtree(staged_dir, ignore_errors=True)
        raise

    print(
        f"finalized {len(positives)} positives and {len(evaluations)} evaluation "
        f"relations in {dataset_dir}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    build = subparsers.add_parser("build", help="build annotation queues")
    build.add_argument("--input-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--max-action-tokens", type=int, default=128)
    build.add_argument("--max-payload-tokens", type=int, default=64)
    build.set_defaults(handler=build_dataset)

    finalize = subparsers.add_parser("finalize", help="export reviewed gold data")
    finalize.add_argument("--dataset-dir", type=Path, required=True)
    finalize.add_argument("--reviewed-annotations", type=Path)
    finalize.add_argument("--seed", type=int, default=20260916)
    finalize.set_defaults(handler=finalize_dataset)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
