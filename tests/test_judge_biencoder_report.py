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
            "validation_mrr": lambda report: report["fine_tuned"]["validation"][
                "retrieval"
            ].update(mrr=0.54),
            "test_mrr_gain": lambda report: report["fine_tuned"]["test"][
                "retrieval"
            ].update(mrr=0.589),
            "test_recall_at_1": lambda report: report["fine_tuned"]["test"][
                "retrieval"
            ].update(recall_at_1=0.39),
            "test_pairwise_accuracy": lambda report: report["fine_tuned"]["test"][
                "hard_negative"
            ].update(pairwise_accuracy=0.77),
            "test_mean_rank": lambda report: report["fine_tuned"]["test"][
                "retrieval"
            ].update(mean_rank=1.9),
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
