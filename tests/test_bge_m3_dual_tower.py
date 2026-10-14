"""Offline contracts for the independent BGE-M3 towers."""

import sys
import unittest
import json
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_bge_m3_exec_supervision import counterfactuals, target_inventory  # noqa: E402
from build_bge_m3_operation_supervision import operation_counterfactual  # noqa: E402
from audit_bge_m3_full_ranks import summarize, validate_reference  # noqa: E402
from apply_bge_m3_dev_reviews import additions_from_reviews  # noqa: E402
from train_bge_m3_dual_tower import candidate_mask, epoch_order  # noqa: E402


def action(tool, **args):
    return "[ACTION] " + json.dumps({"function": tool, "args": args}, sort_keys=True)


class BgeM3DualTowerTests(unittest.TestCase):
    def test_other_known_positive_is_masked_but_own_positive_is_kept(self):
        batch = [
            {"query": "q", "pos": ["a"], "neg": ["c"]},
            {"query": "q", "pos": ["b"], "neg": ["d"]},
        ]
        candidates, mask = candidate_mask(batch, defaultdict(set, {"q": {"a", "b"}}))
        self.assertEqual(candidates, ["a", "b", "c", "d"])
        self.assertEqual(mask, [[True, False, True, True],
                                [False, True, True, True]])

    def test_exec_repeat_adds_only_exec_rows(self):
        rows = [{"query": "q1", "pos": [action("read", path="a")], "neg": []},
                {"query": "q2", "pos": [action("exec", command="python a.py")], "neg": []}]
        order = epoch_order(rows, 4, 11, 1)
        self.assertEqual(order.count(0), 1)
        self.assertEqual(order.count(1), 4)
        self.assertEqual(order, epoch_order(rows, 4, 11, 1))

    def test_counterfactual_requires_explicit_script_target(self):
        positive = action("exec", command="python3 tools/required.py")
        other = action("exec", command="python3 tools/other.py")
        inventory = target_inventory([{"pos": [positive]}, {"pos": [other]}])
        generated = counterfactuals("[INTENT] Run required.py", positive, inventory)
        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0][0], "wrong_script")
        self.assertIn("other.py", generated[0][1])
        self.assertEqual(counterfactuals("[INTENT] Run the script", positive, inventory), [])

    def test_file_replacement_keeps_parent_directory(self):
        positive = action("exec", command="cat data/a.csv")
        other = action("exec", command="cat other/b.csv")
        inventory = target_inventory([{"pos": [positive]}, {"pos": [other]}])
        generated = counterfactuals("[INTENT] Read data/a.csv", positive, inventory)
        self.assertEqual(generated[0][0], "wrong_file")
        self.assertEqual(json.loads(generated[0][1][len("[ACTION] "):])["args"]["command"],
                         "cat data/b.csv")

    def test_operation_counterfactual_keeps_target_and_changes_operation(self):
        positive = action("exec", command="python3 scripts/report.py", workdir="/workspace")
        category, negative = operation_counterfactual("[INTENT] Run report.py", positive)
        self.assertEqual(category, "list_script_instead_of_run")
        self.assertEqual(json.loads(negative[len("[ACTION] "):])["args"],
                         {"command": "ls -l scripts/report.py", "workdir": "/workspace"})
        self.assertIsNone(operation_counterfactual("[INTENT] Inspect report.py", positive))
        category, negative = operation_counterfactual(
            "[INTENT] Read the contents of data.csv", action("exec", command="cat data.csv"))
        self.assertEqual(category, "list_file_instead_of_read")
        self.assertIn("ls -l data.csv", negative)

    def test_full_rank_summary_buckets_only_exec_top20_misses(self):
        queries = [{"query_id": "q", "query": "[INTENT] Run a script"}]
        actions = [{"action_id": str(i), "action": action("exec" if i == 120 else "read", path="x")}
                   for i in range(121)]
        top = [{"action_id": str(i), "score": 1 - i / 1000} for i in range(100)]
        summary, review = summarize(queries, actions, {"q": {"120"}},
                                    [{"query_id": "q", "known_positive_rank": 121,
                                      "top_actions": top}])
        self.assertEqual(summary["original_top20_misses"], 1)
        self.assertEqual(summary["miss_rank_over_100"], 1)
        self.assertEqual(review[0]["review_label"], "unjudged")

    def test_full_rank_must_reproduce_saved_top20(self):
        top = [{"action_id": str(i)} for i in range(100)]
        full = [{"query_id": "q", "top_actions": top}]
        reference = [{"query_id": "q", "top_actions": top[:20]}]
        validate_reference(full, reference)
        reference[0]["top_actions"][0] = {"action_id": "wrong"}
        with self.assertRaises(ValueError):
            validate_reference(full, reference)

    def test_dev_qrels_require_explicit_review_and_reason(self):
        queries = [{"query_id": "q"}]
        actions = [{"action_id": "a"}, {"action_id": "b"}]
        with self.assertRaises(ValueError):
            additions_from_reviews([{"query_id": "q", "review_label": "equivalent",
                                     "equivalent_action_ids": ["b"]}], queries, actions, {"q": {"a"}})
        added = additions_from_reviews([{"query_id": "q", "review_label": "equivalent",
                                         "review_reason": "same command and target",
                                         "equivalent_action_ids": ["b"]}],
                                        queries, actions, {"q": {"a"}})
        self.assertEqual(added[0]["action_id"], "b")


if __name__ == "__main__":
    unittest.main()
