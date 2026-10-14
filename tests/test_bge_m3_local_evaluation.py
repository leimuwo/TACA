"""Contracts for clear-dev candidate construction and step scoring."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_bge_m3_local import build_pools, clear_ids, metrics, rank_from_saved_scores
from evaluate_step_consistency import score


def action(aid, tool, step):
    return {"action_id": aid, "action": '[ACTION] {"function":"' + tool + '","args":{}}',
            "trajectory_id": "t", "thought_step": step}


class LocalEvaluationTests(unittest.TestCase):
    def test_clear_ids_require_complete_decisions(self):
        queries = [{"query_id": "q"}]
        screen = [{"query_id": "s", "source_query_id": "q", "split": "dev"}]
        self.assertEqual(clear_ids(queries, screen, [{"query_id": "s", "ambiguous": False}]), {"q"})
        with self.assertRaises(ValueError):
            clear_ids(queries, screen, [])

    def test_hybrid_excludes_same_step_and_does_not_force_positive(self):
        queries = [{"query_id": "q", "query": "[INTENT] run", "trajectory_id": "t", "thought_step": 2}]
        actions = [action("near", "exec", 1), action("positive", "exec", 2),
                   action("semantic", "exec", 4)]
        ranking = [{"query_id": "q", "top_actions": [
            {"action_id": "semantic", "score": 0.9}, {"action_id": "positive", "score": 0.8}]}]
        pools = build_pools(queries, actions, {"q": {"positive"}}, ranking, {"q"}, "hybrid", 5, 2)
        self.assertEqual(pools[0]["candidate_action_ids"], ["near", "semantic", "positive"])
        self.assertEqual(metrics(pools)["candidate_coverage"], 1.0)
        ranking[0]["top_actions"] = [{"action_id": "semantic", "score": 0.9}]
        pools = build_pools(queries, actions, {"q": {"positive"}}, ranking, {"q"}, "hybrid", 5, 2)
        self.assertEqual(pools[0]["candidate_action_ids"], ["near", "semantic"])
        self.assertEqual(metrics(pools)["candidate_coverage"], 0.0)

    def test_saved_scores_only_for_present_candidates(self):
        pool = [{"query_id": "q", "candidate_action_ids": ["a", "b"],
                 "known_positive_action_ids": ["b"]}]
        ranking = [{"query_id": "q", "top_actions": [
            {"action_id": "a", "score": 0.2}, {"action_id": "b", "score": 0.8}]}]
        scored = rank_from_saved_scores(pool, ranking)
        self.assertEqual(scored[0]["top_actions"][0]["action_id"], "b")
        with self.assertRaises(ValueError):
            rank_from_saved_scores(pool, [{"query_id": "q", "top_actions": ranking[0]["top_actions"][:1]}])

    def test_step_score_requires_adjudicated_gold_and_prediction(self):
        pair = {"query_id": "q", "action_id": "a", "action_tool": "exec",
                "gold_label": "partial", "predicted_label": "faithful", "review_reason": "missing output",
                "due_now_review": True}
        result = score([pair])
        self.assertEqual(result["confusion_matrix"]["partial"]["faithful"], 1)
        with self.assertRaises(ValueError):
            score([{**pair, "gold_label": None}])


if __name__ == "__main__":
    unittest.main()
