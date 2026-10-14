"""Contracts for conservative Action-pool negative mining."""

import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_nightowl_progressive import (  # noqa: E402
    build_safe_pool, is_confirmed_target_mismatch, select_pool_negatives,
)


def action(tool, **args):
    return "[ACTION] " + json.dumps({"function": tool, "args": args}, sort_keys=True)


class ProgressiveNegativeTests(unittest.TestCase):
    def test_same_target_is_not_a_negative(self):
        query = "[INTENT] Run scripts/build.py to generate the report."
        positive = action("exec", command="python3 scripts/build.py")
        equivalent = action("exec", command="python scripts/build.py --verbose")
        wrong = action("exec", command="python3 scripts/clean.py")
        self.assertFalse(is_confirmed_target_mismatch(query, positive, equivalent))
        self.assertTrue(is_confirmed_target_mismatch(query, positive, wrong))

    def test_named_script_without_directory(self):
        query = "[INTENT] Run validate.py to check the generated report."
        positive = action("exec", command="python3 tests/validate.py")
        self.assertTrue(is_confirmed_target_mismatch(
            query, positive, action("exec", command="python3 tools/cleanup.py")))
        self.assertFalse(is_confirmed_target_mismatch(
            query, positive, action("exec", command="python tests/validate.py --verbose")))

    def test_unknown_or_cross_tool_candidate_is_not_mined(self):
        query = "[INTENT] Read the file."
        positive = action("read", path="input/data.csv")
        self.assertFalse(is_confirmed_target_mismatch(query, positive,
                                                       action("read", path="input/other.csv")))
        self.assertFalse(is_confirmed_target_mismatch(query, positive,
                                                       action("write", path="input/other.csv")))

    def test_other_target_named_by_multi_target_intent_is_not_negative(self):
        query = "[INTENT] Read input/a.csv and input/b.csv."
        positive = action("read", path="input/a.csv")
        self.assertFalse(is_confirmed_target_mismatch(
            query, positive, action("read", path="input/b.csv")))

    def test_url_conflict_and_known_positives(self):
        query = "[INTENT] Download https://example.org/a.json with curl."
        positive = action("exec", command="curl https://example.org/a.json")
        equivalent = action("exec", command="curl -L https://example.org/a.json")
        conflict = action("exec", command="curl https://example.org/b.json")
        groups = [(query, positive, [])]
        safe = build_safe_pool(groups, [positive, equivalent, conflict],
                               defaultdict(set, {query: {positive, conflict}}))
        self.assertEqual(safe, [[]])
        safe = build_safe_pool(groups, [positive, equivalent, conflict],
                               defaultdict(set, {query: {positive}}))
        self.assertEqual(safe, [[2]])

    def test_hard_and_random_samples_do_not_overlap(self):
        import random

        self.assertEqual(select_pool_negatives([1, 2, 3], [0.0, 0.1, 0.3, 0.2],
                                               hard_count=1, random_count=2,
                                               rng=random.Random(7))[0], 2)


if __name__ == "__main__":
    unittest.main()
