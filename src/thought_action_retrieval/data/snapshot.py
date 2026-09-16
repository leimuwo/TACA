"""Validate and register a selected SWE trajectory snapshot without copying it."""

from __future__ import annotations

import csv
import json
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from thought_action_retrieval.data.manifests import (
    atomic_write_manifest,
    build_dataset_manifest,
    describe_file,
    sha256_file,
)
from thought_action_retrieval.data.paths import resolve_under_data_root


CASE_NAME_RE = re.compile(r"CASE-\d{4}")
SNAPSHOT_NAME = "swe-selected-500"
SNAPSHOT_VERSION = "2026-09-16"
EXPECTED_TARGET_COUNTS = {"true": 250, "false": 250}
EXPECTED_MODEL_COUNTS = {
    "swe-agent-llama-70b": 350,
    "swe-agent-llama-8b": 100,
    "swe-agent-llama-405b": 50,
}
EXPECTED_SELECTION_GROUP_COUNTS = {"paired": 250, "diversity": 250}


def _read_json_object(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object: {path}")
    return parsed


def _unique_ids(values: Sequence[str], label: str) -> set[str]:
    counts = Counter(values)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {label}: {duplicates[0]}")
    return set(values)


def _case_counts(cases: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "target_counts": dict(Counter(str(case.get("target")).lower() for case in cases)),
        "model_counts": dict(Counter(str(case.get("model_name")) for case in cases)),
        "selection_group_counts": dict(
            Counter(str(case.get("selection_group")) for case in cases)
        ),
    }


def validate_selected_snapshot(
    source_dir: Path, expected_count: int
) -> dict[str, Any]:
    """Validate CASE, JSONL, CSV, and summary consistency for one snapshot."""
    source_dir = source_dir.resolve()
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    case_paths = sorted(source_dir.glob("CASE-*.json"))
    if len(case_paths) != expected_count:
        raise ValueError(
            f"expected {expected_count} CASE files, found {len(case_paths)}"
        )
    cases: list[dict[str, Any]] = []
    case_ids: list[str] = []
    for path in case_paths:
        if not CASE_NAME_RE.fullmatch(path.stem):
            raise ValueError(f"invalid four-digit CASE filename: {path.name}")
        case = _read_json_object(path)
        cases.append(case)
        case_ids.append(str(case.get("case_id") or ""))
    case_id_set = _unique_ids(case_ids, "CASE ID")
    for path, case_id in zip(case_paths, case_ids):
        if case_id != path.stem:
            raise ValueError(f"CASE ID does not match filename: {path.name}")

    jsonl_path = source_dir / f"selected_{expected_count}.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"missing selected JSONL: {jsonl_path}")
    jsonl_rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        jsonl_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} is not an object")
        jsonl_rows.append(row)
    jsonl_ids = _unique_ids(
        [str(row.get("case_id") or "") for row in jsonl_rows], "JSONL CASE ID"
    )
    if len(jsonl_rows) != expected_count or jsonl_ids != case_id_set:
        raise ValueError("JSONL IDs do not match CASE files")

    manifest_path = source_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest CSV: {manifest_path}")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest_ids = _unique_ids(
        [str(row.get("selection_id") or "") for row in manifest_rows],
        "manifest selection ID",
    )
    if len(manifest_rows) != expected_count or manifest_ids != case_id_set:
        raise ValueError("manifest IDs do not match CASE files")

    summary_path = source_dir / "selection_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing selection summary: {summary_path}")
    summary = _read_json_object(summary_path)
    if summary.get("selected_count") != expected_count:
        raise ValueError(
            "selection_summary selected_count does not match expected_count"
        )
    derived_counts = _case_counts(cases)
    for field, derived in derived_counts.items():
        if summary.get(field) != derived:
            raise ValueError(f"selection_summary {field} does not match CASE files")
    if expected_count == 500:
        frozen = {
            "target_counts": EXPECTED_TARGET_COUNTS,
            "model_counts": EXPECTED_MODEL_COUNTS,
            "selection_group_counts": EXPECTED_SELECTION_GROUP_COUNTS,
        }
        for field, expected in frozen.items():
            if summary.get(field) != expected:
                raise ValueError(f"selection_summary {field} violates frozen quotas")

    return {
        "selected_count": expected_count,
        "case_paths": case_paths,
        "cases": cases,
        "case_ids": case_id_set,
        "jsonl_path": jsonl_path,
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "summary": summary,
    }


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle, tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=destination.name + ".",
        suffix=".tmp",
        delete=False,
    ) as output_handle:
        shutil.copyfileobj(input_handle, output_handle)
        temporary = Path(output_handle.name)
    temporary.replace(destination)


def _describe_repository_file(repository_root: Path, path: Path) -> dict[str, object]:
    resolved_root = repository_root.resolve()
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(resolved_root).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def import_selected_snapshot(
    *,
    data_root: Path,
    source_relative: Path,
    repository_root: Path,
    sample_ids: Sequence[str],
    raw_shards_relative: Path,
    expected_count: int = 500,
) -> dict[str, Any]:
    """Import only metadata and requested samples for an external snapshot."""
    data_root = data_root.resolve()
    repository_root = repository_root.resolve()
    source_dir = resolve_under_data_root(data_root, source_relative)
    raw_shards_dir = resolve_under_data_root(data_root, raw_shards_relative)
    validated = validate_selected_snapshot(source_dir, expected_count)
    requested = list(sample_ids)
    if len(set(requested)) != len(requested):
        raise ValueError("sample IDs must be unique")
    missing = sorted(set(requested) - validated["case_ids"])
    if missing:
        raise ValueError(f"requested sample does not exist: {missing[0]}")

    raw_shards = sorted(raw_shards_dir.glob("*.jsonl"))
    if not raw_shards:
        raise ValueError(f"no raw JSONL shards found in {raw_shards_dir}")

    manifest_dir = repository_root / "data/manifests" / SNAPSHOT_NAME
    samples_dir = repository_root / "data/samples" / SNAPSHOT_NAME
    copied_paths: list[Path] = []
    for source_name in ("manifest.csv", "selection_summary.json"):
        destination = manifest_dir / source_name
        _atomic_copy(source_dir / source_name, destination)
        copied_paths.append(destination)
    case_path_by_id = {path.stem: path for path in validated["case_paths"]}
    for sample_id in requested:
        destination = samples_dir / f"{sample_id}.json"
        _atomic_copy(case_path_by_id[sample_id], destination)
        copied_paths.append(destination)

    external_files = [
        validated["jsonl_path"],
        validated["manifest_path"],
        validated["summary_path"],
        *raw_shards,
    ]
    manifest = build_dataset_manifest(
        name=SNAPSHOT_NAME,
        version=SNAPSHOT_VERSION,
        data_root=data_root,
        files=external_files,
        metadata={},
    )
    summary = validated["summary"]
    manifest["metadata"] = {
        "source_relative": Path(source_relative).as_posix(),
        "selected_count": validated["selected_count"],
        "seed": summary.get("seed"),
        "target_counts": summary.get("target_counts"),
        "model_counts": summary.get("model_counts"),
        "selection_group_counts": summary.get("selection_group_counts"),
        "unique_instance_ids": summary.get("unique_instance_ids"),
        "unique_repositories": summary.get("unique_repositories"),
        "raw_shards": [describe_file(data_root, path) for path in raw_shards],
        "tracked_artifacts": [
            _describe_repository_file(repository_root, path)
            for path in sorted(copied_paths)
        ],
    }
    dataset_manifest_path = manifest_dir / "dataset_manifest.json"
    atomic_write_manifest(dataset_manifest_path, manifest)
    return {
        "selected_count": validated["selected_count"],
        "dataset_manifest": dataset_manifest_path,
        "sample_count": len(requested),
        "raw_shard_count": len(raw_shards),
    }
