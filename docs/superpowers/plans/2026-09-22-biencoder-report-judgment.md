# Bi-encoder Report Judgment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed-threshold CLI that classifies provisional bi-encoder training reports as `STRONG_PASS`, `PASS`, or `FAIL`.

**Architecture:** Keep the feature in one dependency-free Python script with small pure functions for extraction, judgment, and formatting. Exercise the public functions and subprocess CLI from one focused unittest module using synthetic reports.

**Tech Stack:** Python 3.10+, standard-library `argparse`, `json`, `math`, `pathlib`, and `unittest`.

---

## File Structure

- Create `scripts/judge_biencoder_report.py`: validate report fields, calculate deltas and checks, format results, and map outcomes to exit statuses.
- Create `tests/test_judge_biencoder_report.py`: cover fixed thresholds, failure criteria, malformed input, output labels, and CLI exit statuses.

### Task 1: Implement fixed report judgment with TDD

**Files:**
- Create: `scripts/judge_biencoder_report.py`
- Test: `tests/test_judge_biencoder_report.py`

- [ ] **Step 1: Write failing unit tests for strong pass, pass, and every mandatory failure**

Create `tests/test_judge_biencoder_report.py` with:

```python
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "judge_biencoder_report.py"
SPEC = importlib.util.spec_from_file_location("judge_biencoder_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
JUDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JUDGE)


def _split(mrr, recall_at_1, mean_rank, pairwise_accuracy):
    return {
        "retrieval": {
            "recall_at_1": recall_at_1,
            "recall_at_3": min(1.0, recall_at_1 + 0.25),
            "mrr": mrr,
            "mean_rank": mean_rank,
        },
        "hard_negative": {
            "pairwise_accuracy": pairwise_accuracy,
            "mean_similarity_margin": 0.1,
        },
    }


def _report():
    return {
        "label": "FEASIBILITY ONLY — NOT GOLD EVALUATION",
        "frozen": {
            "validation": _split(0.55, 0.40, 1.8, 0.76),
            "test": _split(0.56, 0.40, 1.9, 0.78),
        },
        "fine_tuned": {
            "validation": _split(0.61, 0.47, 1.5, 0.82),
            "test": _split(0.62, 0.46, 1.5, 0.82),
        },
    }


class JudgmentTests(unittest.TestCase):
    def test_classifies_strong_pass(self):
        result = JUDGE.judge_metrics(JUDGE.extract_metrics(_report()))
        self.assertEqual(result["classification"], "STRONG_PASS")
        self.assertTrue(all(result["checks"].values()))

    def test_classifies_pass_below_strong_absolute_thresholds(self):
        report = _report()
        report["fine_tuned"]["test"] = _split(0.59, 0.44, 1.5, 0.79)
        result = JUDGE.judge_metrics(JUDGE.extract_metrics(report))
        self.assertEqual(result["classification"], "PASS")

    def test_each_mandatory_condition_can_fail_the_report(self):
        mutations = {
            "validation_mrr": lambda report: report["fine_tuned"]["validation"]["retrieval"].update(mrr=0.54),
            "test_mrr_gain": lambda report: report["fine_tuned"]["test"]["retrieval"].update(mrr=0.589),
            "test_recall_at_1": lambda report: report["fine_tuned"]["test"]["retrieval"].update(recall_at_1=0.39),
            "test_pairwise_accuracy": lambda report: report["fine_tuned"]["test"]["hard_negative"].update(pairwise_accuracy=0.77),
            "test_mean_rank": lambda report: report["fine_tuned"]["test"]["retrieval"].update(mean_rank=1.9),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                report = _report()
                mutate(report)
                result = JUDGE.judge_metrics(JUDGE.extract_metrics(report))
                self.assertEqual(result["classification"], "FAIL")
                self.assertFalse(result["checks"][name])

    def test_rejects_missing_and_non_numeric_metrics(self):
        missing = _report()
        del missing["frozen"]["test"]["retrieval"]["mrr"]
        with self.assertRaisesRegex(JUDGE.ReportValidationError, "frozen.test"):
            JUDGE.extract_metrics(missing)
        non_numeric = _report()
        non_numeric["fine_tuned"]["test"]["retrieval"]["mrr"] = "high"
        with self.assertRaisesRegex(JUDGE.ReportValidationError, "numeric"):
            JUDGE.extract_metrics(non_numeric)


class JudgmentCliTests(unittest.TestCase):
    def _run(self, content):
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "training_report.json"
            report_path.write_text(content, encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(SCRIPT), str(report_path)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

    def test_cli_prints_label_and_strong_pass(self):
        result = self._run(json.dumps(_report()))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FEASIBILITY ONLY — NOT GOLD EVALUATION", result.stdout)
        self.assertIn("FINAL: STRONG_PASS", result.stdout)

    def test_cli_returns_one_for_failed_valid_report(self):
        report = _report()
        report["fine_tuned"]["test"]["retrieval"]["mrr"] = 0.57
        result = self._run(json.dumps(report))
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("FINAL: FAIL", result.stdout)

    def test_cli_returns_two_for_invalid_json(self):
        result = self._run("not json\n")
        self.assertEqual(result.returncode, 2)
        self.assertIn("error:", result.stderr)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify RED**

Run: `PYTHONPATH=src python -m unittest tests.test_judge_biencoder_report -v`

Expected: import fails because `scripts/judge_biencoder_report.py` does not exist.

- [ ] **Step 3: Write the minimal evaluator**

Create `scripts/judge_biencoder_report.py` with:

```python
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
        raise ReportValidationError(f"missing report field: {'.'.join(path)}") from error
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ReportValidationError(f"report field must be numeric: {'.'.join(path)}")
    number = float(current)
    if not math.isfinite(number):
        raise ReportValidationError(f"report field must be finite: {'.'.join(path)}")
    return number


def extract_metrics(report: Any) -> dict[str, dict[str, dict[str, float]]]:
    if not isinstance(report, dict):
        raise ReportValidationError("training report must be a JSON object")
    metrics = {}
    for method in ("frozen", "fine_tuned"):
        metrics[method] = {}
        for split in ("validation", "test"):
            metrics[method][split] = {
                name: _numeric_at(report, (method, split, *path))
                for name, path in METRIC_FIELDS.items()
            }
    return metrics


def judge_metrics(metrics: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    frozen_validation = metrics["frozen"]["validation"]
    tuned_validation = metrics["fine_tuned"]["validation"]
    frozen_test = metrics["frozen"]["test"]
    tuned_test = metrics["fine_tuned"]["test"]
    deltas = {name: tuned_test[name] - frozen_test[name] for name in METRIC_FIELDS}
    checks = {
        "validation_mrr": tuned_validation["mrr"] + EPSILON >= frozen_validation["mrr"],
        "test_mrr_gain": deltas["mrr"] + EPSILON >= 0.03,
        "test_recall_at_1": tuned_test["recall_at_1"] + EPSILON >= frozen_test["recall_at_1"],
        "test_pairwise_accuracy": tuned_test["pairwise_accuracy"] + EPSILON >= frozen_test["pairwise_accuracy"],
        "test_mean_rank": tuned_test["mean_rank"] < frozen_test["mean_rank"] - EPSILON,
    }
    passed = all(checks.values())
    strong = passed and all((
        tuned_test["mrr"] + EPSILON >= 0.60,
        tuned_test["recall_at_1"] + EPSILON >= 0.45,
        tuned_test["pairwise_accuracy"] + EPSILON >= 0.80,
    ))
    classification = "STRONG_PASS" if strong else "PASS" if passed else "FAIL"
    return {"classification": classification, "checks": checks, "deltas": deltas}


def format_result(metrics, result):
    lines = [FEASIBILITY_LABEL, "", "method       split       R@1    R@3    MRR    MeanRank  PairAcc  Margin"]
    for method in ("frozen", "fine_tuned"):
        for split in ("validation", "test"):
            item = metrics[method][split]
            lines.append(
                f"{method:12s} {split:10s} {item['recall_at_1']:.3f}  "
                f"{item['recall_at_3']:.3f}  {item['mrr']:.3f}  "
                f"{item['mean_rank']:8.3f}  {item['pairwise_accuracy']:.3f}    "
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


def build_parser():
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
```

- [ ] **Step 4: Run test to verify GREEN**

Run: `PYTHONPATH=src python -m unittest tests.test_judge_biencoder_report -v`

Expected: 7 tests pass with exit status `0`.

- [ ] **Step 5: Check formatting and commit**

```bash
git diff --check -- scripts/judge_biencoder_report.py tests/test_judge_biencoder_report.py
git add scripts/judge_biencoder_report.py tests/test_judge_biencoder_report.py
git commit -m "feat: judge bi-encoder training reports"
```

If Git identity remains unset, record the blocked commit without changing Git configuration.

### Task 2: Verify integration and a real report

**Files:**
- Verify: `scripts/judge_biencoder_report.py`
- Verify: `tests/test_judge_biencoder_report.py`

- [ ] **Step 1: Run the complete test suite**

Run: `PYTHONPATH=src python -m unittest discover -s tests -v`

Expected: all tests pass with zero failures and zero errors.

- [ ] **Step 2: Evaluate the completed H200 report**

```bash
python scripts/judge_biencoder_report.py \
  /inspire/hdd/project/control-technology/public/workspace/wx/data/SWE-agent-trajectories/experiments/provisional_biencoder_h200_bge_base_20260922T064148Z/training_report.json
```

Expected: feasibility label, four metric rows, deltas, five criteria, and one final classification. Exit status `0` means `PASS`/`STRONG_PASS`; status `1` means a valid `FAIL` result.

- [ ] **Step 3: Verify invalid input behavior**

```bash
set +e
python scripts/judge_biencoder_report.py /path/that/does/not/exist
status=$?
set -e
test "$status" -eq 2
```

Expected: stderr begins with `error:` and the status assertion succeeds.

- [ ] **Step 4: Review the final diff**

```bash
git diff --check
git status --short
git diff -- scripts/judge_biencoder_report.py tests/test_judge_biencoder_report.py
```

Expected: no whitespace errors; only planned feature files plus known pre-existing untracked files are present.
