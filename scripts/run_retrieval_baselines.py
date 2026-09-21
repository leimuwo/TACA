#!/usr/bin/env python3
"""Run deterministic sparse baselines on a finalized provisional dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

from thought_action_retrieval.training.baselines import (
    FEASIBILITY_LABEL,
    evaluate_records,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing split file: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"split must contain JSON objects: {path}")
    for row in rows:
        if (
            row.get("supervision_status") != "provisional_auto_pair"
            or row.get("experiment_tier") != "feasibility_only"
            or row.get("human_reviewed") is not False
        ):
            raise ValueError("baseline input is missing provisional watermarks")
    return rows


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def run(args: argparse.Namespace) -> int:
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    reports: dict[str, dict[str, Any]] = {}
    for split in ("validation", "test"):
        rows = _read_jsonl(dataset_dir / "splits" / f"{split}.jsonl")
        for method in ("tfidf", "bm25"):
            report = evaluate_records(rows, method=method)
            report["split"] = split
            reports[f"{method}_{split}.json"] = report
    summary = {
        "schema_version": "sparse_baseline_summary_v1",
        "label": FEASIBILITY_LABEL,
        "reports": {
            name.removesuffix(".json"): {
                "recall_at_1": report["retrieval"]["recall_at_1"],
                "mrr": report["retrieval"]["mrr"],
                "pairwise_accuracy": report["hard_negative"]["pairwise_accuracy"],
            }
            for name, report in sorted(reports.items())
        },
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for name, report in reports.items():
            (staged / name).write_bytes(_json_bytes(report))
        (staged / "sparse_baseline_summary.json").write_bytes(_json_bytes(summary))
        staged.replace(output_dir)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    print(f"wrote sparse feasibility reports to {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
