#!/usr/bin/env python3
"""Judge a provisional bi-encoder training report with fixed thresholds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence


FEASIBILITY_LABEL = "FEASIBILITY ONLY — NOT GOLD EVALUATION"
METRIC_FIELDS = {
    "recall_at_1": ("retrieval", "recall_at_1"),
    "recall_at_3": ("retrieval", "recall_at_3"),
    "mrr": ("retrieval", "mrr"),
    "mean_rank": ("retrieval", "mean_rank"),
    "pairwise_accuracy": ("hard_negative", "pairwise_accuracy"),
    "mean_similarity_margin": ("hard_negative", "mean_similarity_margin"),
}
EPSILON = 1e-12


class ReportValidationError(ValueError):
    """Raised when a report lacks the required numeric schema."""


def _numeric_at(value: Any, path: tuple[str, ...]) -> float:
    current = value
    try:
        for component in path:
            current = current[component]
    except (KeyError, TypeError) as error:
        raise ReportValidationError(
            f"missing report field: {'.'.join(path)}"
        ) from error
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ReportValidationError(
            f"report field must be numeric: {'.'.join(path)}"
        )
    number = float(current)
    if not math.isfinite(number):
        raise ReportValidationError(
            f"report field must be finite: {'.'.join(path)}"
        )
    return number


def extract_metrics(report: Any) -> dict[str, dict[str, dict[str, float]]]:
    if not isinstance(report, dict):
        raise ReportValidationError("training report must be a JSON object")
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for method in ("frozen", "fine_tuned"):
        metrics[method] = {}
        for split in ("validation", "test"):
            metrics[method][split] = {
                name: _numeric_at(report, (method, split, *path))
                for name, path in METRIC_FIELDS.items()
            }
    return metrics


def judge_metrics(
    metrics: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    frozen_validation = metrics["frozen"]["validation"]
    tuned_validation = metrics["fine_tuned"]["validation"]
    frozen_test = metrics["frozen"]["test"]
    tuned_test = metrics["fine_tuned"]["test"]
    deltas = {
        name: tuned_test[name] - frozen_test[name] for name in METRIC_FIELDS
    }
    checks = {
        "validation_mrr": tuned_validation["mrr"] + EPSILON
        >= frozen_validation["mrr"],
        "test_mrr_gain": deltas["mrr"] + EPSILON >= 0.03,
        "test_recall_at_1": tuned_test["recall_at_1"] + EPSILON
        >= frozen_test["recall_at_1"],
        "test_pairwise_accuracy": tuned_test["pairwise_accuracy"] + EPSILON
        >= frozen_test["pairwise_accuracy"],
        "test_mean_rank": tuned_test["mean_rank"]
        < frozen_test["mean_rank"] - EPSILON,
    }
    passed = all(checks.values())
    strong = passed and all(
        (
            tuned_test["mrr"] + EPSILON >= 0.60,
            tuned_test["recall_at_1"] + EPSILON >= 0.45,
            tuned_test["pairwise_accuracy"] + EPSILON >= 0.80,
        )
    )
    classification = "STRONG_PASS" if strong else "PASS" if passed else "FAIL"
    return {
        "classification": classification,
        "checks": checks,
        "deltas": deltas,
    }


def format_result(
    metrics: dict[str, dict[str, dict[str, float]]], result: dict[str, Any]
) -> str:
    lines = [
        FEASIBILITY_LABEL,
        "",
        "method       split       R@1    R@3    MRR    MeanRank  PairAcc  Margin",
    ]
    for method in ("frozen", "fine_tuned"):
        for split in ("validation", "test"):
            item = metrics[method][split]
            lines.append(
                f"{method:12s} {split:10s} "
                f"{item['recall_at_1']:.3f}  {item['recall_at_3']:.3f}  "
                f"{item['mrr']:.3f}  {item['mean_rank']:8.3f}  "
                f"{item['pairwise_accuracy']:.3f}    "
                f"{item['mean_similarity_margin']:+.3f}"
            )
    lines.extend(("", "test deltas (fine_tuned - frozen):"))
    for name in METRIC_FIELDS:
        lines.append(f"  {name}: {result['deltas'][name]:+.3f}")
    lines.extend(("", "criteria:"))
    for name, passed in result["checks"].items():
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    lines.extend(("", f"FINAL: {result['classification']}"))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="path to training_report.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        metrics = extract_metrics(report)
        result = judge_metrics(metrics)
    except (OSError, json.JSONDecodeError, ReportValidationError) as error:
        print(f"error: {error}", file=__import__("sys").stderr)
        return 2
    print(format_result(metrics, result))
    return 1 if result["classification"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
