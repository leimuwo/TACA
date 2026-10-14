"""Offline checks for fixed-candidate reranking."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_reranker_clawgym import filter_clear_queries, load_candidates, make_training_groups, prioritize_action, summarize


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


class RerankerContractTests(unittest.TestCase):
    def test_pairwise_group_keeps_positive_first(self):
        positive = '[ACTION] {"function":"exec","args":{"command":"ls input"}}'
        negative = '[ACTION] {"function":"exec","args":{"command":"cat input/file"}}'
        groups = make_training_groups([{"query": "[INTENT] list input", "pos": [positive], "neg": [negative]}])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["positive_tool"], "exec")
        self.assertEqual(groups[0]["actions"], [prioritize_action(positive), prioritize_action(negative)])

    def test_action_target_precedes_long_content(self):
        action = '[ACTION] {"args":{"content":"long body","path":"/tmp/report.md"},"function":"write"}'
        serialized = prioritize_action(action)
        self.assertLess(serialized.index('"function"'), serialized.index('"path"'))
        self.assertLess(serialized.index('"path"'), serialized.index('"content"'))
        self.assertEqual(json.loads(serialized[len("[ACTION] "):]), json.loads(action[len("[ACTION] "):]))

    def make_data(self, root):
        write(root / "dev/queries.jsonl", [{"query_id": "q", "query": "[INTENT] read x"}])
        write(root / "dev/actions.jsonl", [
            {"action_id": "a", "action": '[ACTION] {"function":"read","args":{"path":"x"}}'},
            {"action_id": "b", "action": '[ACTION] {"function":"write","args":{"path":"x"}}'}])
        write(root / "dev/qrels.jsonl", [{"query_id": "q", "action_id": "a", "relevance": 1}])
        write(root / "rankings.jsonl", [{"query_id": "q", "top_actions": [
            {"action_id": "b", "score": 0.9}, {"action_id": "a", "score": 0.8}]}])

    def test_reordering_improves_metric_without_changing_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_data(root)
            candidates = load_candidates(root, root / "rankings.jsonl", root / "dev/qrels.jsonl", 2)
            before = summarize(candidates)
            after = summarize(candidates, [[1, 0]])
            self.assertEqual(before["known_positive_candidate_recall"], 1)
            self.assertEqual(before["known_positive_at_1"], 0)
            self.assertEqual(after["known_positive_at_1"], 1)
            self.assertEqual(after["known_positive_mrr_at_k"], 1)

    def test_rejects_duplicate_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_data(root)
            write(root / "rankings.jsonl", [{"query_id": "q", "top_actions": [
                {"action_id": "a", "score": 0.9}, {"action_id": "a", "score": 0.8}]}])
            with self.assertRaisesRegex(ValueError, "duplicate or unknown"):
                load_candidates(root, root / "rankings.jsonl", root / "dev/qrels.jsonl", 2)

    def test_clear_query_filter_uses_explicit_decisions(self):
        candidates = [{"query_id": "clear"}, {"query_id": "vague"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "screening.jsonl"
            write(path, [{"query_id": "clear", "ambiguous": False},
                         {"query_id": "vague", "ambiguous": True}])
            self.assertEqual(filter_clear_queries(candidates, path), [candidates[0]])
            write(path, [{"query_id": "clear", "ambiguous": False}])
            with self.assertRaisesRegex(ValueError, "screening missing"):
                filter_clear_queries(candidates, path)

    def test_clear_filter_accepts_excluded_only_source_ids(self):
        candidates = [{"query_id": "clear"}, {"query_id": "vague"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ambiguous_test.jsonl"
            write(path, [{"query_id": "screened-id", "source_query_id": "vague", "ambiguous": True}])
            self.assertEqual(filter_clear_queries(candidates, path), [candidates[0]])

    def test_load_test_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_data(root)
            (root / "dev").rename(root / "test")
            candidates = load_candidates(root, root / "rankings.jsonl", root / "test/qrels.jsonl", 2, split="test")
            self.assertEqual(len(candidates), 1)


if __name__ == "__main__":
    unittest.main()
