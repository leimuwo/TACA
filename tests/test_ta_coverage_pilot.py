"""Contracts for frozen TA windows, pair aggregation, and review sampling."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_ta_coverage_pilot import split_index, window  # noqa: E402
from predict_ta_coverage_bge import aggregate  # noqa: E402
from review_ta_coverage_gqq import SYSTEM, validate  # noqa: E402
from render_ta_coverage_review import choose  # noqa: E402
from evaluate_ta_coverage_pilot import evaluate  # noqa: E402
from merge_ta_model_reviews import merge  # noqa: E402
from ta_coverage_aggregation import aggregate_ta_label  # noqa: E402


class TACoveragePilotTests(unittest.TestCase):
    def test_split_index_rejects_trajectory_and_template_leakage(self):
        with self.assertRaises(ValueError):
            split_index([{"trajectory_id": "t", "split": "train"},
                         {"trajectory_id": "t", "split": "test"}])
        with self.assertRaises(ValueError):
            split_index([{"trajectory_id": "a", "split": "train", "template_cluster_id": "x"},
                         {"trajectory_id": "b", "split": "dev", "template_cluster_id": "x"}])

    def test_window_keeps_all_calls_and_corresponding_results(self):
        row = {"trajectory_id": "t", "thought_step": 1, "thought": "Read A and B",
               "execution_intents": [{"intent_id": "I1", "intent_text": "Read A", "temporal_scope": "current"},
                                     {"intent_id": "I2", "intent_text": "Read B", "temporal_scope": "current"}]}
        raw = {"messages": [{"role": "assistant", "assistant_thought": {"content": "Read A and B"},
                             "tool_calls": [{"id": "a", "function": "read", "args": {"path": "A"}},
                                            {"id": "b", "function": "read", "args": {"path": "B"}}]},
                            {"role": "tool", "tool_call_id": "a", "content": "contents A"},
                            {"role": "tool", "tool_call_id": "b", "content": "contents B"}]}
        result = window(row, raw, "train")
        self.assertEqual(len(result["pairs"]), 4)
        self.assertEqual(result["actions"][1]["execution_result"], ["contents B"])

    def test_gqq_aggregation_is_derived_from_pairs(self):
        w = {"intents": [{"intent_id": "i", "temporal_scope": "current"}],
             "pairs": [{"pair_id": "p", "intent_id": "i"}]}
        verdict = {"intents": [{"intent_id": "i", "due_now": True, "assessable": True, "reason": "clear"}],
                   "pairs": [{"pair_id": "p", "covered": False, "reason": "wrong action"}],
                   "suggested_ta_label": "faithful"}
        result = validate(w, verdict)
        self.assertEqual(result["suggested_ta_label"], "not_executed")
        self.assertTrue(result["aggregation_disagrees"])
        verdict.pop("suggested_ta_label")
        verdict.pop("model_raw_ta_label")
        verdict.pop("aggregation_disagrees")
        result = validate(w, verdict)
        self.assertEqual(result["suggested_ta_label"], "not_executed")
        self.assertNotIn("model_raw_ta_label", result)

    def test_review_prompt_covers_known_atomicity_failures(self):
        self.assertIn("private calculation already performed inside Thought", SYSTEM)
        self.assertIn("the skill directories", SYSTEM)
        self.assertIn("A single write Action may demonstrate classification", SYSTEM)
        self.assertIn("ONE Action alone fully covers the WHOLE Intent", SYSTEM)

    def test_similarity_baseline_aggregation(self):
        w = {"intents": [{"intent_id": "i1", "temporal_scope": "current"},
                         {"intent_id": "i2", "temporal_scope": "current"}],
             "pairs": [{"pair_id": "p1", "intent_id": "i1"}, {"pair_id": "p2", "intent_id": "i2"}]}
        self.assertEqual(aggregate(w, {"p1": .7, "p2": .1}, .5), "partial")

    def test_window_aggregation_uses_intent_coverage(self):
        w = {"intents": [{"intent_id": "i1"}, {"intent_id": "i2"}],
             "pairs": [{"pair_id": "p1", "intent_id": "i1"},
                       {"pair_id": "p2", "intent_id": "i2"}]}
        decisions = {key: {"due_now": True, "assessable": True} for key in ("i1", "i2")}
        self.assertEqual(aggregate_ta_label(w, decisions, {"p1": True, "p2": True}), "faithful")
        self.assertEqual(aggregate_ta_label(w, decisions, {"p1": True, "p2": False}), "partial")
        self.assertEqual(aggregate_ta_label(w, decisions, {"p1": False, "p2": False}), "not_executed")
        decisions["i2"]["assessable"] = False
        self.assertEqual(aggregate_ta_label(w, decisions, {"p1": True, "p2": False}), "unassessable")

    def test_selection_uses_distinct_trajectories(self):
        windows = [{"window_id": str(i), "trajectory_id": str(i // 2),
                    "intents": [{"intent_id": "i"}], "actions": [{"action_id": "a"}]}
                   for i in range(8)]
        reviews = [{"window_id": str(i), "verdict": {"suggested_ta_label": "faithful"}}
                   for i in range(8)]
        chosen = choose(windows, reviews, 4)
        self.assertEqual(len({x["window"]["trajectory_id"] for x in chosen}), 4)

    def test_evaluation_requires_pair_consistent_human_gold(self):
        windows = [{"window_id": "w", "intents": [{"intent_id": "i", "temporal_scope": "current"}],
                    "pairs": [{"pair_id": "p", "intent_id": "i"}],
                    "actions": [{"raw_call": {"function": "exec"}}]}]
        human = [{"window_id": "w", "gold_label": "faithful", "review_reason": "runs target",
                  "intents": {"i": {"due_now": "yes", "assessable": "yes"}}, "pairs": {"p": "yes"}}]
        pred = [{"window_id": "w", "predicted_label": "faithful"}]
        self.assertEqual(evaluate(windows, human, pred)["accuracy"], 1)
        human[0]["pairs"]["p"] = "no"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            evaluate(windows, human, pred)
        human[0]["gold_label"] = ""
        self.assertEqual(evaluate(windows, human, pred)["accuracy"], 0)

    def test_overall_human_gold_does_not_require_pair_review(self):
        windows = [{"window_id": wid, "intents": [], "pairs": [],
                    "actions": [{"raw_call": {"function": "exec"}}]} for wid in ("a", "b")]
        human = [{"window_id": "a", "gold_label": "partial", "review_reason": "",
                  "intents": {}, "pairs": {}},
                 {"window_id": "b", "gold_label": ""}]
        pred = [{"window_id": "a", "predicted_label": "partial"},
                {"window_id": "b", "predicted_label": "faithful"}]
        report = evaluate(windows, human, pred, gold_source="human-overall")
        self.assertEqual(report["reviewed_pairs"], 1)
        self.assertEqual(report["accuracy"], 1)
        self.assertEqual(report["skipped"], {"unlabeled_human_gold": 1})

    def test_review_merge_requires_valid_result_for_every_window(self):
        from tempfile import TemporaryDirectory
        import json
        with TemporaryDirectory() as directory:
            source = Path(directory) / "reviews.jsonl"
            source.write_text(json.dumps({"window_id": "w", "review_failed": True,
                                          "verdict": None}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing valid review"):
                merge([{"window_id": "w"}], [source])
            source.write_text(json.dumps({"window_id": "w", "review_failed": False,
                                          "verdict": {"suggested_ta_label": "faithful"}}) + "\n",
                              encoding="utf-8")
            self.assertEqual(merge([{"window_id": "w"}], [source])[0]["review_source"], str(source))


if __name__ == "__main__":
    unittest.main()
