"""Versioned TA repair and weak-label training contracts."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_ta_coverage_train_v2 import EMAIL_WINDOW, SCOPE_CORRECTIONS, repair  # noqa: E402
from prepare_ta_pair_train import prepare  # noqa: E402
from resolve_ta_uncertainties import resolve  # noqa: E402
from train_ta_pair_reranker import DEFAULT_MODEL, validate_data  # noqa: E402
from train_ta_pair_cross_encoder import main as cross_encoder_main  # noqa: E402


class TAPairTrainingTests(unittest.TestCase):
    def test_resolve_field_pollution_and_unaffected_pair(self):
        window = {"window_id": "w", "intents": [
            {"intent_id": "i1", "text": "Read input/a.txt"},
            {"intent_id": "i2", "text": "Write output/b.txt"}],
            "pairs": [{"pair_id": "p1", "intent_id": "i1"},
                      {"pair_id": "p2", "intent_id": "i2"}]}
        verdict = {"pairs": [{"pair_id": "p1", "covered": True},
                             {"pair_id": "p2", "covered": False}],
                   "uncertainties": ["No uncertainty: paths are explicit.",
                                     "The intent 'Write output/b.txt' may refer to another file."]}
        result = resolve(window, verdict)
        self.assertEqual(result["cleared_pair_ids"], ["p1"])
        self.assertEqual(result["blocked_pair_ids"], ["p2"])

    def test_resolve_preparatory_negative_only(self):
        window = {"window_id": "w", "intents": [{"intent_id": "i", "text": "Create output.txt"}],
                  "pairs": [{"pair_id": "p", "intent_id": "i"}]}
        note = "The action is a preparatory step, but it does not itself create the file."
        negative = {"pairs": [{"pair_id": "p", "covered": False}], "uncertainties": [note]}
        positive = {"pairs": [{"pair_id": "p", "covered": True}], "uncertainties": [note]}
        self.assertEqual(resolve(window, negative)["cleared_pair_ids"], ["p"])
        self.assertEqual(resolve(window, positive)["blocked_pair_ids"], ["p"])

    def test_resolve_keeps_target_or_result_ambiguity_blocked(self):
        window = {"window_id": "w", "intents": [{"intent_id": "i", "text": "Read input files"}],
                  "pairs": [{"pair_id": "p", "intent_id": "i"}]}
        verdict = {"pairs": [{"pair_id": "p", "covered": True}],
                   "uncertainties": ["The intent does not specify which input files are intended."]}
        self.assertEqual(resolve(window, verdict)["blocked_pair_ids"], ["p"])
        verdict["uncertainties"] = ["The result is not shown; it may have failed."]
        self.assertEqual(resolve(window, verdict)["blocked_pair_ids"], ["p"])

    def test_prepare_keeps_cleared_pair_from_uncertain_window(self):
        windows = [{"window_id": "w", "trajectory_id": "t", "split": "train",
                    "intents": [{"intent_id": "i1", "text": "Read input/a.txt", "temporal_scope": "current"},
                                {"intent_id": "i2", "text": "Write output/b.txt", "temporal_scope": "current"}],
                    "actions": [{"action_id": "a", "action_text": "read a", "raw_call": {"function": "read"}}],
                    "pairs": [{"pair_id": "p1", "intent_id": "i1", "action_id": "a"},
                              {"pair_id": "p2", "intent_id": "i2", "action_id": "a"}]}]
        reviews = [{"window_id": "w", "verdict": {
            "intents": [{"intent_id": "i1", "due_now": True, "assessable": True},
                        {"intent_id": "i2", "due_now": True, "assessable": True}],
            "pairs": [{"pair_id": "p1", "covered": True, "reason": "read"},
                      {"pair_id": "p2", "covered": False, "reason": "not written"}],
            "uncertainties": ["The intent 'Write output/b.txt' is ambiguous."]}}]
        resolutions = {"w": resolve(windows[0], reviews[0]["verdict"])}
        # Add another negative trajectory so the training split has both classes.
        other = {**windows[0], "window_id": "w2", "trajectory_id": "t2",
                 "intents": [{"intent_id": "i3", "text": "Write c", "temporal_scope": "current"}],
                 "pairs": [{"pair_id": "p3", "intent_id": "i3", "action_id": "a"}]}
        reviews.append({"window_id": "w2", "verdict": {
            "intents": [{"intent_id": "i3", "due_now": True, "assessable": True}],
            "pairs": [{"pair_id": "p3", "covered": False, "reason": "not written"}],
            "uncertainties": []}})
        # Split placement varies with trajectory shuffle; include enough examples.
        for n in range(4, 15):
            copied = {**other, "window_id": f"w{n}", "trajectory_id": f"t{n}",
                      "intents": [{"intent_id": f"i{n}", "text": f"Write c{n}", "temporal_scope": "current"}],
                      "pairs": [{"pair_id": f"p{n}", "intent_id": f"i{n}", "action_id": "a"}]}
            windows.append(copied)
            reviews.append({"window_id": f"w{n}", "verdict": {
                "intents": [{"intent_id": f"i{n}", "due_now": True, "assessable": True}],
                "pairs": [{"pair_id": f"p{n}", "covered": n % 2 == 0, "reason": "checked"}],
                "uncertainties": []}})
        windows.append(other)
        train, validation, excluded = prepare(windows, reviews, [], resolutions=resolutions)
        self.assertIn("p1", {x["pair_id"] for x in train + validation})
        self.assertNotIn("p2", {x["pair_id"] for x in train + validation})
        self.assertEqual(excluded["uncertain_pair"], 1)

    def test_pair_classifier_defaults_to_bge_m3(self):
        self.assertEqual(DEFAULT_MODEL.name, "bge-m3")

    def test_cross_encoder_entrypoint_is_available(self):
        self.assertTrue(callable(cross_encoder_main))

    def test_scope_repair_adds_all_action_pairs(self):
        wid, corrected = next(iter(SCOPE_CORRECTIONS.items()))
        intents = [{"intent_id": iid, "temporal_scope": "future"} for iid in sorted(corrected)]
        record = {"window_id": wid, "intents": intents,
                  "actions": [{"action_id": "a1"}, {"action_id": "a2"}], "pairs": []}
        fixed, changed = repair(record)
        self.assertTrue(changed)
        self.assertEqual(len(fixed["pairs"]), len(corrected) * 2)
        self.assertTrue(all(x["temporal_scope"] == "current" for x in fixed["intents"]))
        self.assertTrue(all(x["temporal_scope"] == "future" for x in record["intents"]))

    def test_plural_email_intent_is_split(self):
        record = {"window_id": EMAIL_WINDOW,
                  "intents": [{"intent_id": "original", "source_intent_id": "T1-I1",
                               "text": "Verify the email versions", "temporal_scope": "current"}],
                  "actions": [{"action_id": "a1"}, {"action_id": "a2"}], "pairs": []}
        fixed, changed = repair(record)
        self.assertTrue(changed)
        self.assertEqual(len(fixed["intents"]), 2)
        self.assertEqual(len(fixed["pairs"]), 4)
        self.assertEqual({x["parent_intent_id"] for x in fixed["intents"]}, {"original"})

    def test_prepare_filters_quarantine_and_separates_trajectories(self):
        windows = []
        reviews = []
        for n in range(20):
            wid, iid, aid, pid = f"w{n}", f"i{n}", f"a{n}", f"p{n}"
            windows.append({"window_id": wid, "trajectory_id": f"t{n}", "split": "train",
                            "intents": [{"intent_id": iid, "text": f"Read x{n}", "temporal_scope": "current"}],
                            "actions": [{"action_id": aid, "action_text": "[ACTION] {}",
                                         "raw_call": {"function": "read"}}],
                            "pairs": [{"pair_id": pid, "intent_id": iid, "action_id": aid}]})
            reviews.append({"window_id": wid, "verdict": {
                "intents": [{"intent_id": iid, "due_now": True, "assessable": True}],
                "pairs": [{"pair_id": pid, "covered": n % 2 == 0, "reason": "checked"}],
                "suggested_ta_label": "faithful" if n % 2 == 0 else "not_executed",
                "uncertainties": []}})
        train, validation, excluded = prepare(windows, reviews, [
            {"scope": "window", "window_id": "w0"}, {"scope": "pair", "pair_id": "p1"}])
        self.assertFalse({x["trajectory_id"] for x in train} &
                         {x["trajectory_id"] for x in validation})
        self.assertEqual(len(train) + len(validation), 18)
        self.assertEqual(excluded["quarantined_window"], 1)
        self.assertEqual(excluded["quarantined_pair"], 1)
        validate_data(train, validation)

    def test_prepare_removes_conflicting_duplicate_text_labels(self):
        windows = []
        reviews = []
        for n in range(20):
            wid, iid, aid, pid = f"w{n}", f"i{n}", f"a{n}", f"p{n}"
            text = "same text" if n < 2 else f"unique {n}"
            windows.append({"window_id": wid, "trajectory_id": f"t{n}", "split": "train",
                            "intents": [{"intent_id": iid, "text": text, "temporal_scope": "current"}],
                            "actions": [{"action_id": aid, "action_text": "[ACTION] {}",
                                         "raw_call": {"function": "read"}}],
                            "pairs": [{"pair_id": pid, "intent_id": iid, "action_id": aid}]})
            reviews.append({"window_id": wid, "verdict": {
                "intents": [{"intent_id": iid, "due_now": True, "assessable": True}],
                "pairs": [{"pair_id": pid, "covered": n % 2 == 0, "reason": "checked"}],
                "suggested_ta_label": "faithful" if n % 2 == 0 else "not_executed",
                "uncertainties": []}})
        train, validation, excluded = prepare(windows, reviews, [])
        self.assertEqual(excluded["conflicting_duplicate_text_pair"], 2)
        self.assertEqual(len(train) + len(validation), 18)


if __name__ == "__main__":
    unittest.main()
