"""Held-out GLM evaluation sampling and weak-label filtering."""
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_ta_glm_eval_sample import features, select, train_profile  # noqa: E402
from finalize_ta_glm_eval import finalize, finalize_pairs  # noqa: E402
from match_ta_glm_eval_pairs import choose, validate_splits  # noqa: E402


class GLMEvalTests(unittest.TestCase):
    def test_sample_uses_input_features_only(self):
        train = [{"window_id": f"t{i}", "tool": "exec" if i % 2 else "read"}
                 for i in range(20)]
        profile = train_profile(train)
        pool = [{"window_id": f"w{i}", "actions": [{"raw_call": {
            "function": "exec" if i % 2 else "read"}}], "pairs": [{"pair_id": f"p{i}"}]}
            for i in range(20)]
        chosen, distance = select(pool, profile, 10, 4)
        self.assertEqual(len(chosen), 10)
        self.assertEqual(distance, 0)
        self.assertEqual({features(w)[0] for w in chosen}, {"read", "exec"})

    def test_finalizer_requires_clear_agreeing_review(self):
        window = {"window_id": "w", "split": "dev", "trajectory_id": "heldout",
                  "intents": [{"intent_id": "i", "text": "Read input/a.txt", "temporal_scope": "current"}],
                  "actions": [{"action_id": "a", "action_text": "read input/a.txt",
                               "raw_call": {"function": "read"}}],
                  "pairs": [{"pair_id": "p", "intent_id": "i", "action_id": "a"}]}
        verdict = {"intents": [{"intent_id": "i", "due_now": True,
                               "assessable": True, "reason": "explicit"}],
                   "pairs": [{"pair_id": "p", "covered": True, "reason": "exact read"}],
                   "uncertainties": [], "suggested_ta_label": "faithful"}
        review = {"window_id": "w", "review_failed": False, "verdict": verdict,
                  "model": "glm-5.2", "prompt_version": "test"}
        accepted, pairs, queue, _, _ = finalize([window], [review], [])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(pairs[0]["label"], 1)
        self.assertFalse(queue)
        verdict["uncertainties"] = ["Target is ambiguous."]
        accepted, _, queue, _, _ = finalize([window], [review], [])
        self.assertFalse(accepted)
        self.assertEqual(queue[0]["reason"], "uncertain_review")
        verdict["uncertainties"] = []
        train = [{"trajectory_id": "other", "intent": "Read input/a.txt",
                  "action": "read input/a.txt"}]
        accepted, _, queue, _, _ = finalize([window], [review], train)
        self.assertFalse(accepted)
        self.assertEqual(queue[0]["reason"], "duplicate_text_pair_across_splits")

    def test_pair_finalizer_keeps_agreeing_negative_from_uncertain_window(self):
        window = {"window_id": "w", "split": "test", "trajectory_id": "heldout",
                  "intents": [{"intent_id": "i", "text": "Read input/a.txt", "temporal_scope": "current"}],
                  "actions": [{"action_id": "a", "action_text": "write output/b.txt",
                               "raw_call": {"function": "write"}}],
                  "pairs": [{"pair_id": "p", "intent_id": "i", "action_id": "a"}]}
        review = {"window_id": "w", "review_failed": False, "model": "glm-5.2",
                  "prompt_version": "test", "verdict": {
                      "intents": [{"intent_id": "i", "due_now": True,
                                   "assessable": True, "reason": "clear"}],
                      "pairs": [{"pair_id": "p", "covered": False, "reason": "wrong operation"}],
                      "uncertainties": ["The target file might be ambiguous."],
                      "suggested_ta_label": "not_executed"}}
        second = {"window_id": "w", "review_failed": False, "verdict": {"pairs": [
            {"pair_id": "p", "decision": "not_covered", "evidence_sufficient": True,
             "remaining_doubt": "", "reason": "write cannot read"}]}}
        accepted, pairs, queue, _, _ = finalize_pairs([window], [review], [second], [])
        self.assertEqual(accepted[0]["label"], "not_executed")
        self.assertEqual(accepted[0]["label_source"], "program_aggregation_of_glm-5.2_weak_pairs")
        self.assertEqual(pairs[0]["label"], 0)
        self.assertFalse(queue)
        second["verdict"]["pairs"][0]["decision"] = "covered"
        accepted, pairs, queue, _, _ = finalize_pairs([window], [review], [second], [])
        self.assertFalse(accepted)
        self.assertFalse(pairs)
        self.assertEqual(queue[0]["reason"], "pair_label_conflict")

    def test_stratified_pairs_preserve_split_isolation(self):
        def pair(pid, split, trajectory, label, tool="exec"):
            return {"pair_id": pid, "split": split, "trajectory_id": trajectory,
                    "intent": f"intent {pid}", "action": f"action {pid}",
                    "label": label, "tool": tool}
        train = [pair("t0", "train", "t", 0), pair("t1", "train", "t", 1)]
        dev = [pair("d0", "dev", "d", 0), pair("d1", "dev", "d", 1)]
        test = [pair("e0", "test", "e", 0)]
        validate_splits(train, dev, test)
        self.assertEqual(choose(dev, Counter({("exec", 0): 1, ("exec", 1): 1}), 2, 1), dev)
        test[0]["trajectory_id"] = "t"
        with self.assertRaisesRegex(ValueError, "train/test"):
            validate_splits(train, dev, test)


if __name__ == "__main__":
    unittest.main()
