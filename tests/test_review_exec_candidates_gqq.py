"""Offline checks for GQQ candidate selection."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_reviewed_exec_train import action  # noqa: E402
from review_exec_candidates_gqq import audit, select  # noqa: E402


def edge(edge_id, intent, command, split="train"):
    return {"edge_id": edge_id, "split": split, "eligibility": "candidate",
            "trajectory_id": "trace", "intent_text": intent,
            "positive_action": {"function": "exec", "args": {"command": command}}}


class GqqSelectionTests(unittest.TestCase):
    def test_excludes_existing_and_known_positive_negatives(self):
        source = edge("source", "Run target.py", "python target.py")
        equivalent = edge("equivalent", "Run target.py", "python3 target.py")
        wrong = edge("wrong", "Run other.py", "python other.py")
        queue = [{"source_edge_id": "source", "candidate_edge_id": item["edge_id"],
                  "positive_action": action(source), "candidate_action": action(item)}
                 for item in (equivalent, wrong)]
        selected = select([source, equivalent, wrong], queue, [], set(), 10)
        self.assertEqual([item["candidate_edge_id"] for item in selected[0]["negatives"]], ["wrong"])
        base = [{"query": "[INTENT] Run target.py", "pos": [action(source)], "neg": [action(wrong)]}]
        self.assertEqual(select([source, equivalent, wrong], queue, base, set(), 10), [])
        self.assertEqual(select([source, equivalent, wrong], queue, [], {"trace"}, 10), [])

    def test_ambiguous_intent_cannot_approve_negative(self):
        item = {"intent": "Check if the input files exist.",
                "negatives": [{"candidate_edge_id": "n", "action": "[ACTION] {}"}]}
        verdict = {"ambiguous": True, "target_evidence": None, "positive_match": True,
                   "positive_reason": "plausible", "ambiguity_reason": "unspecified files",
                   "negatives": [{"candidate_edge_id": "n", "safe_negative": True,
                                  "category": "different_operation", "reason": "wrong"}]}
        response = {"choices": [{"message": {"content": json.dumps(verdict)}}]}
        class Stub:
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return False
            def read(self):
                return json.dumps(response).encode()
        with patch("urllib.request.OpenerDirector.open", return_value=Stub()):
            with self.assertRaisesRegex(ValueError, "ambiguous Intent"):
                audit(item, "https://example.invalid", "secret", "model", 1)


if __name__ == "__main__":
    unittest.main()
