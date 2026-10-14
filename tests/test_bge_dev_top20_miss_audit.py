"""Checks for the fixed stratified dev audit."""

import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_bge_dev_top20_misses import DECISIONS, sample_misses


class MissAuditTests(unittest.TestCase):
    def test_sample_is_reproducible_and_excludes_reachable(self):
        rankings20 = []
        rankings50 = {}
        qrels = defaultdict(set)
        for index in range(33):
            query_id = str(index)
            qrels[query_id].add("positive")
            rankings20.append({"query_id": query_id, "top_actions": [{"action_id": "other"}]})
            rankings50[query_id] = {"known_positive_rank": 25 if index < 11 else 75 if index < 22 else 150}
        rankings20[0]["top_actions"][0]["action_id"] = "positive"
        first = sample_misses(rankings20, rankings50, qrels, per_band=3)
        self.assertEqual(first, sample_misses(rankings20, rankings50, qrels, per_band=3))
        self.assertEqual({band: len(group) for band, group in first.items()},
                         {"21-50": 3, "51-100": 3, "101+": 3})
        self.assertNotIn("0", first["21-50"])

    def test_review_decisions_are_nonautomatic(self):
        self.assertEqual(len(DECISIONS), 30)
        self.assertEqual({label for label, _ in DECISIONS.values()},
                         {"retrieval_miss_likely", "possible_missing_qrel", "positive_label_problem", "needs_context"})


if __name__ == "__main__":
    unittest.main()
