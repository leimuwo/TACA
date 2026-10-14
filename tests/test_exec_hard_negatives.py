"""Ensure exec augmentation uses validated labels and leaves unjudged candidates out."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from assemble_exec_hard_negatives import action_text, augment, promote_clear_script_targets


class ExecHardNegativeTests(unittest.TestCase):
    def test_validated_candidate_is_added_only_once(self):
        positive = {"function": "exec", "args": {"command": "python3 report.py"}}
        negative = {"function": "exec", "args": {"command": "cat report.py"}}
        source = {"query": "[INTENT] Run report.py", "pos": [action_text(positive)],
                  "neg": [action_text({"function": "exec", "args": {"command": "cat other.py"}})],
                  "candidate_ids": ["old"], "trajectory_id": "train-1"}
        edge = {"edge_id": "edge", "intent_text": "Run report.py", "positive_action": positive}
        candidate = {"edge_id": "edge", "candidate_id": "new", "candidate_action": negative,
                     "category": "P2", "validation": {"accepted": True}}
        train, provenance, additions, audit, _, _ = augment([source], {"edge": edge}, iter([candidate, candidate]))
        self.assertEqual(audit["added"], 1)
        self.assertEqual(len(train), 2)
        self.assertEqual(train[-1]["neg"], [action_text(negative)])
        self.assertEqual(provenance[-1]["candidate_ids"], ["new"])

    def test_rule_requires_explicit_script_name(self):
        positive = action_text({"function": "exec", "args": {"command": "python3 report.py"}})
        negative = action_text({"function": "exec", "args": {"command": "python3 other.py"}})
        queue = [{"query": "[INTENT] Run report.py", "positive_action": positive,
                  "candidate_action": negative, "source_edge_id": "s", "candidate_edge_id": "n"},
                 {"query": "[INTENT] Run the script", "positive_action": positive,
                  "candidate_action": negative, "source_edge_id": "s2", "candidate_edge_id": "n2"}]
        train, provenance = [], []
        promoted = promote_clear_script_targets(queue, train, provenance)
        self.assertEqual(len(promoted), 1)
        self.assertEqual(len(train), 1)
        self.assertEqual(json.loads(train[0]["neg"][0][len("[ACTION] "):])["args"]["command"],
                         "python3 other.py")


if __name__ == "__main__":
    unittest.main()
