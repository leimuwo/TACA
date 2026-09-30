"""Offline checks for the ClawGym Silver training and evaluation contracts."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_bge_clawgym import known_positive_metrics, load_split
from train_bge_clawgym import load_train


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


class BgeClawgymContractTests(unittest.TestCase):
    def test_known_positive_metrics(self):
        result = known_positive_metrics([1, 2, 10])
        self.assertEqual(result["query_count"], 3)
        self.assertAlmostEqual(result["known_positive_at_1"], 1 / 3)
        self.assertAlmostEqual(result["known_positive_at_5"], 2 / 3)
        self.assertAlmostEqual(result["known_positive_mrr"], (1 + 0.5 + 0.1) / 3)

    def test_loader_detects_cross_row_positive_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            write(path, [{"query": "[INTENT] read", "pos": ["[ACTION] a"], "neg": ["[ACTION] b"]},
                         {"query": "[INTENT] read", "pos": ["[ACTION] b"], "neg": ["[ACTION] c"]}])
            with self.assertRaisesRegex(ValueError, "collides"):
                load_train(path)

    def test_split_loader_requires_positive_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = Path(tmp) / "dev"
            write(split / "queries.jsonl", [{"query_id": "q", "query": "[INTENT] read"}])
            write(split / "actions.jsonl", [{"action_id": "a", "action": "[ACTION] a"}])
            write(split / "qrels.jsonl", [{"query_id": "q", "action_id": "missing", "relevance": 1}])
            with self.assertRaisesRegex(ValueError, "unknown query or action"):
                load_split(Path(tmp), "dev")

    def test_split_loader_rejects_unknown_query_qrel(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = Path(tmp) / "test"
            write(split / "queries.jsonl", [{"query_id": "q", "query": "[INTENT] read"}])
            write(split / "actions.jsonl", [{"action_id": "a", "action": "[ACTION] a"}])
            write(split / "qrels.jsonl", [{"query_id": "other", "action_id": "a", "relevance": 1}])
            with self.assertRaisesRegex(ValueError, "unknown query or action"):
                load_split(Path(tmp), "test")


if __name__ == "__main__":
    unittest.main()
