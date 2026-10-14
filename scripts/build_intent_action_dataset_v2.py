#!/usr/bin/env python3
"""Initialize the frozen Intent--Action Matching Pilot v2 run root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from thought_action_retrieval.intent.extraction import load_prompt_bundle
from thought_action_retrieval.intent.validation import (
    export_blind_intents,
    load_intent_run_records,
)
from thought_action_retrieval.matching.action_atomicization import (
    audit_intent_run,
    rows_jsonl,
)
from thought_action_retrieval.matching.v2_io import publish_stage, read_verified_jsonl
from thought_action_retrieval.matching.v2_schema import (
    build_pilot_run_config,
    canonical_json_bytes,
    prompt_bundle_sha256,
    validate_pilot_run_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPORAL_PROMPT_PATH = REPOSITORY_ROOT / "prompts" / "thought_to_temporal_intent.md"


def approved_temporal_prompt_sha256() -> str:
    """Hash both reviewed temporal Prompt sections from the repository."""
    bundle = load_prompt_bundle(TEMPORAL_PROMPT_PATH)
    return prompt_bundle_sha256(bundle.system, bundle.user_template)


def _write_config(path: Path, config: dict[str, object]) -> None:
    with path.open("wb") as handle:
        handle.write(canonical_json_bytes(config) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def init_pilot(output_dir: Path) -> int:
    output_dir = output_dir.expanduser().resolve()
    prompt_hash = approved_temporal_prompt_sha256()
    config = build_pilot_run_config(prompt_hash)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"output path is not a directory: {output_dir}")
        config_path = output_dir / "run_config.json"
        if not config_path.is_file():
            raise FileExistsError(f"run root exists without run_config.json: {output_dir}")
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            validate_pilot_run_config(existing, prompt_hash)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"existing run root configuration does not match Pilot config: {error}") from error
        print(f"Pilot run root already initialized and matches: {output_dir}")
        return 0

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        _write_config(staging / "run_config.json", config)
        os.rename(staging, output_dir)
        staging = None  # type: ignore[assignment]
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    print(f"Initialized Pilot run root: {output_dir}")
    return 0


def audit_actions(intent_run_dir: Path, output_dir: Path) -> int:
    rows, manifest = audit_intent_run(intent_run_dir.expanduser().resolve())
    custom_manifest = canonical_json_bytes(manifest) + b"\n"
    publish_stage(
        output_dir.expanduser().resolve(),
        {
            "actions_enriched.jsonl": rows_jsonl(rows),
            "action_structure_manifest.json": custom_manifest,
        },
    )
    print(f"Published action structure audit: {output_dir}")
    return 0


def export_intents(intent_run_dir: Path, action_audit_dir: Path, output_dir: Path) -> int:
    """Publish Stage 1 Thoughts-derived Intents without raw Action payloads."""
    records, record_provenance = load_intent_run_records(intent_run_dir)
    audit_dir = action_audit_dir.expanduser().resolve()
    actions_path = audit_dir / "actions_enriched.jsonl"
    structure_path = audit_dir / "action_structure_manifest.json"
    if not actions_path.is_file() or not structure_path.is_file():
        raise ValueError("action-audit directory needs actions_enriched.jsonl and action_structure_manifest.json")
    action_rows = read_verified_jsonl(actions_path)
    try:
        action_structure = json.loads(structure_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid action structure manifest: {structure_path}") from error
    if not isinstance(action_structure, dict) or action_structure.get("schema_version") != "matching_v2_action_structure_manifest":
        raise ValueError("invalid action structure manifest schema")
    rows = export_blind_intents(records, action_rows)
    risk_distribution = {"high": 0, "medium": 0, "low": 0}
    for row in rows:
        for intent in row["execution_intents"]:
            risk_distribution[intent["risk_level"]] += 1
    audit_stage_manifest = audit_dir / "manifest.json"
    if not audit_stage_manifest.is_file():
        raise ValueError("action-audit directory needs publish_stage manifest.json")
    extractor_manifest = intent_run_dir.expanduser().resolve() / "manifest.json"
    manifest = {
        "schema_version": "matching_v2_intents_blind_manifest",
        "thought_row_count": len(rows),
        "intent_count": sum(len(row["execution_intents"]) for row in rows),
        "risk_distribution": risk_distribution,
        "upstream": {
            "intent_run": {
                "record_count": len(records),
                "files": record_provenance,
                "manifest_sha256": hashlib.sha256(extractor_manifest.read_bytes()).hexdigest() if extractor_manifest.is_file() else None,
            },
            "action_audit": {
                "action_row_count": len(action_rows),
                "raw_action_count": action_structure.get("raw_action_count"),
                "stage_manifest_sha256": hashlib.sha256(audit_stage_manifest.read_bytes()).hexdigest(),
                "action_structure_manifest_sha256": hashlib.sha256(structure_path.read_bytes()).hexdigest(),
            },
        },
    }
    publish_stage(
        output_dir.expanduser().resolve(),
        {
            "intents_blind.jsonl": rows_jsonl(rows),
            "intents_manifest.json": canonical_json_bytes(manifest) + b"\n",
        },
    )
    print(f"Published blind Intent export: {output_dir.expanduser().resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init-pilot", help="initialize Pilot-100 run configuration")
    init_parser.add_argument("--output-dir", required=True, type=Path)
    audit_parser = subparsers.add_parser("audit-actions", aliases=["atomicize"], help="audit trajectory action structure")
    audit_parser.add_argument("--intent-run-dir", required=True, type=Path)
    audit_parser.add_argument("--output-dir", required=True, type=Path)
    export_parser = subparsers.add_parser("export-intents", help="export validated action-blind temporal Intents")
    export_parser.add_argument("--intent-run-dir", required=True, type=Path)
    export_parser.add_argument("--action-audit-dir", required=True, type=Path)
    export_parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "init-pilot":
        return init_pilot(args.output_dir)
    if args.command in {"audit-actions", "atomicize"}:
        return audit_actions(args.intent_run_dir, args.output_dir)
    if args.command == "export-intents":
        return export_intents(args.intent_run_dir, args.action_audit_dir, args.output_dir)
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=os.sys.stderr)
        raise SystemExit(1)
