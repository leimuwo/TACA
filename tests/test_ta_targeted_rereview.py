"""Targeted GQQ review must release only independently supported pair labels."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from merge_ta_targeted_reviews import merge  # noqa: E402
from rereview_ta_uncertain_pairs_gqq import parse_response, validate  # noqa: E402


class TargetedReviewTests(unittest.TestCase):
    def setUp(self):
        self.window = {"window_id": "w", "intents": [
            {"intent_id": "i1", "temporal_scope": "current"},
            {"intent_id": "i2", "temporal_scope": "future"}],
            "pairs": [{"pair_id": "p1", "intent_id": "i1"},
                      {"pair_id": "p2", "intent_id": "i1"},
                      {"pair_id": "p3", "intent_id": "i2"}]}
        self.resolution = {"window_id": "w", "cleared_pair_ids": [],
                           "blocked_pair_ids": ["p1", "p2", "p3"], "notes": []}
        self.original = {"verdict": {
            "intents": [{"intent_id": "i1", "due_now": True, "assessable": True},
                        {"intent_id": "i2", "due_now": False, "assessable": True}],
            "pairs": [{"pair_id": "p1", "covered": True},
                      {"pair_id": "p2", "covered": False},
                      {"pair_id": "p3", "covered": False}]}}

    def test_parse_rejects_extra_json(self):
        with self.assertRaisesRegex(ValueError, "extra content"):
            parse_response('{"pairs":[]} {"pairs":[]}')

    def test_validate_requires_structured_evidence(self):
        valid = {"pairs": [{"pair_id": "p1", "decision": "covered",
                            "evidence_sufficient": True, "remaining_doubt": "", "reason": "direct"}]}
        self.assertEqual(validate(valid, {"p1"}), valid)
        valid["pairs"][0]["evidence_sufficient"] = False
        with self.assertRaisesRegex(ValueError, "evidence conflict"):
            validate(valid, {"p1"})

    def test_merge_releases_only_agreeing_current_pairs(self):
        targeted = {"review_failed": False, "verdict": {"pairs": [
            {"pair_id": "p1", "decision": "covered", "evidence_sufficient": True,
             "remaining_doubt": "", "reason": "direct"},
            {"pair_id": "p2", "decision": "covered", "evidence_sufficient": True,
             "remaining_doubt": "", "reason": "disagrees"},
            {"pair_id": "p3", "decision": "not_covered", "evidence_sufficient": True,
             "remaining_doubt": "", "reason": "future"}]}}
        result = merge(self.window, self.resolution, self.original, targeted)
        self.assertEqual(result["cleared_pair_ids"], ["p1"])
        self.assertEqual(result["blocked_pair_ids"], ["p2", "p3"])
        self.assertEqual([x["status"] for x in result["targeted_audit"]],
                         ["released", "label_conflict", "intent_not_eligible"])

    def test_failed_review_keeps_all_blocked(self):
        result = merge(self.window, self.resolution, self.original,
                       {"review_failed": True, "verdict": None})
        self.assertEqual(result["blocked_pair_ids"], ["p1", "p2", "p3"])

    def test_ambiguous_positive_stays_blocked(self):
        self.resolution["notes"] = [{"disposition": "unresolved_window",
                                     "text": "The intent does not specify which files are meant."}]
        targeted = {"review_failed": False, "verdict": {"pairs": [
            {"pair_id": "p1", "decision": "covered", "evidence_sufficient": True,
             "remaining_doubt": "", "reason": "The action reads a plausible file."},
            {"pair_id": "p2", "decision": "not_covered", "evidence_sufficient": True,
             "remaining_doubt": "", "reason": "Clearly unrelated."},
            {"pair_id": "p3", "decision": "not_covered", "evidence_sufficient": True,
             "remaining_doubt": "", "reason": "Future intent."}]}}
        result = merge(self.window, self.resolution, self.original, targeted)
        self.assertEqual(result["cleared_pair_ids"], ["p2"])
        self.assertEqual(result["targeted_audit"][0]["status"], "ambiguous_positive")


if __name__ == "__main__":
    unittest.main()
