"""Safety checks for reviewed exec additions."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_reviewed_exec_train import action, build_additions  # noqa: E402


def edge(edge_id, command, intent="Run target.py"):
    return {"edge_id": edge_id, "split": "train", "eligibility": "candidate",
            "trajectory_id": "train-trace", "intent_text": intent,
            "positive_action": {"function": "exec", "args": {"command": command}}}


class ReviewedExecTrainTests(unittest.TestCase):
    def setUp(self):
        self.source = edge("source", "python target.py")
        self.other = edge("other", "python other.py", "Run other.py")
        self.queue = [{"source_edge_id": "source", "candidate_edge_id": "other",
                       "positive_action": action(self.source), "candidate_action": action(self.other)}]
        self.review = [{"source_edge_id": "source", "intent_review_reason": "Target is explicit",
                        "negatives": [{"candidate_edge_id": "other", "review_reason": "Wrong target"}]}]

    def test_new_pair_has_source_and_review_provenance(self):
        added = build_additions([], [self.source, self.other], self.queue, self.review)
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]["trajectory_id"], "train-trace")
        self.assertEqual(json.loads(added[0]["neg"][0][9:])["args"]["command"], "python other.py")

    def test_duplicate_or_held_out_positive_is_rejected(self):
        base = [{"query": "[INTENT] Run target.py", "pos": [action(self.source)],
                 "neg": [action(self.other)]}]
        with self.assertRaises(ValueError):
            build_additions(base, [self.source, self.other], self.queue, self.review)
        with self.assertRaises(ValueError):
            build_additions([], [self.source, self.other], self.queue, self.review, {"train-trace"})

    def test_known_positive_cannot_be_used_as_negative(self):
        self.other["intent_text"] = self.source["intent_text"]
        with self.assertRaises(ValueError):
            build_additions([], [self.source, self.other], self.queue, self.review)


if __name__ == "__main__":
    unittest.main()
