"""Candidate-pool construction keeps SWE issues and ClawGym splits separate."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_ta_10k_candidates import choose_claw, swe_split  # noqa: E402


class TA10KCandidateTests(unittest.TestCase):
    def test_claw_selection_preserves_repaired_window_and_exact_pair_budget(self):
        fixed = {"window_id": "fixed", "split": "train", "pairs": [{"pair_id": "p1"}]}
        candidates = [fixed, {"window_id": "a", "split": "train", "pairs": [{"pair_id": "p2"}]},
                      {"window_id": "b", "split": "train", "pairs": [{"pair_id": "p3"}]}]
        selected = choose_claw([fixed], candidates, 3, 1)
        self.assertEqual(len(selected), 3)
        self.assertIn(fixed, selected)
        self.assertEqual(sum(len(x["pairs"]) for x in selected), 3)

    def test_swe_instance_split_is_stable(self):
        self.assertEqual(swe_split("repo__name-123"), swe_split("repo__name-123"))
        self.assertIn(swe_split("repo__name-123"), {"train", "dev", "test"})


if __name__ == "__main__":
    unittest.main()
