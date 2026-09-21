import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from thought_action_retrieval.training.baselines import evaluate_records
from thought_action_retrieval.training.metrics import (
    pairwise_metrics,
    retrieval_metrics,
)


class RetrievalMetricTests(unittest.TestCase):
    def test_computes_hand_checked_recall_reciprocal_rank_and_mean_rank(self):
        result = retrieval_metrics(
            rankings=[["a", "b", "c"], ["x", "y", "z"]],
            positive_ids=[{"b"}, {"x"}],
            recall_ks=(1, 3),
        )

        self.assertEqual(
            result,
            {
                "query_count": 2,
                "recall_at_1": 0.5,
                "recall_at_3": 1.0,
                "mrr": 0.75,
                "mean_rank": 1.5,
            },
        )

    def test_computes_pairwise_accuracy_and_average_margin_per_comparison(self):
        result = pairwise_metrics(
            positive_scores=[0.8, 0.2],
            negative_scores=[[0.5, 0.9], [0.1]],
        )

        self.assertEqual(result["comparison_count"], 3)
        self.assertAlmostEqual(result["pairwise_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["mean_similarity_margin"], 0.1)

    def test_rejects_empty_or_misaligned_metric_inputs(self):
        with self.assertRaises(ValueError):
            retrieval_metrics([], [])
        with self.assertRaises(ValueError):
            retrieval_metrics([["a"]], [])
        with self.assertRaises(ValueError):
            pairwise_metrics([0.5], [])


class SparseBaselineTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "candidate_id": "one",
                "supervision_status": "provisional_auto_pair",
                "experiment_tier": "feasibility_only",
                "human_reviewed": False,
                "intent_input": "[INTENT] Open alpha.py.",
                "positive_action": {
                    "serialized": '[ACTION] open(path="alpha.py")'
                },
                "negative_actions": [
                    {"serialized": '[ACTION] run(path="alpha.py")'},
                    {"serialized": '[ACTION] open(path="beta.py")'},
                ],
            },
            {
                "candidate_id": "two",
                "supervision_status": "provisional_auto_pair",
                "experiment_tier": "feasibility_only",
                "human_reviewed": False,
                "intent_input": "[INTENT] Execute beta_test.py using python.",
                "positive_action": {
                    "serialized": '[ACTION] python(command="beta_test.py")'
                },
                "negative_actions": [
                    {"serialized": '[ACTION] open(path="beta_test.py")'}
                ],
            },
        ]

    def test_tfidf_and_bm25_emit_the_same_feasibility_metric_schema(self):
        tfidf = evaluate_records(self.records, method="tfidf")
        bm25 = evaluate_records(self.records, method="bm25")

        expected_keys = {
            "label",
            "method",
            "retrieval",
            "hard_negative",
            "document_count",
        }
        self.assertEqual(set(tfidf), expected_keys)
        self.assertEqual(set(bm25), expected_keys)
        self.assertEqual(
            tfidf["label"], "FEASIBILITY ONLY — NOT GOLD EVALUATION"
        )
        self.assertEqual(tfidf["retrieval"]["recall_at_1"], 1.0)
        self.assertEqual(bm25["retrieval"]["recall_at_1"], 1.0)
        self.assertEqual(tfidf["hard_negative"]["comparison_count"], 3)

    def test_baseline_cli_writes_validation_and_test_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            output = root / "reports"
            (dataset / "splits").mkdir(parents=True)
            payload = "".join(
                json.dumps(record, sort_keys=True) + "\n" for record in self.records
            )
            for split in ("validation", "test"):
                (dataset / "splits" / f"{split}.jsonl").write_text(
                    payload, encoding="utf-8"
                )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_retrieval_baselines.py",
                    "--dataset-dir",
                    str(dataset),
                    "--output-dir",
                    str(output),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "bm25_test.json",
                    "bm25_validation.json",
                    "sparse_baseline_summary.json",
                    "tfidf_test.json",
                    "tfidf_validation.json",
                },
            )
            summary = json.loads(
                (output / "sparse_baseline_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["label"], "FEASIBILITY ONLY — NOT GOLD EVALUATION"
            )


if __name__ == "__main__":
    unittest.main()
