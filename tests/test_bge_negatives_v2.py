"""Check false-negative guards for v2 negative candidates."""

import unittest

from scripts.build_bge_negatives_v2 import candidate, check_candidate, compact, action_view


def edge(intent, action):
    return {
        "edge_id": "edge-1",
        "action_id": "action-1",
        "intent_id": "intent-1",
        "trajectory_id": "trajectory-1",
        "thought_step": 1,
        "intent_text": intent,
        "positive_action": action,
    }


def validate(source, proposed, category, other_positives=()):
    row = candidate(source, proposed, category, "test")
    positives = {compact(action_view(source["positive_action"]))}
    positives.update(compact(action_view(action)) for action in other_positives)
    return check_candidate(source, row, {source["edge_id"]: positives})


class NegativeGuardTests(unittest.TestCase):
    def test_generic_read_does_not_make_other_input_file_negative(self):
        source = edge("Read the input files", {"function": "read", "args": {"path": "input/a.csv"}})
        proposed = {"function": "read", "args": {"path": "input/b.csv"}}
        self.assertEqual(validate(source, proposed, "P1"), "p1_target_not_explicit")

    def test_explicit_target_can_be_replaced_once(self):
        source = edge("Read input/a.csv", {"function": "read", "args": {"path": "input/a.csv"}})
        proposed = {"function": "read", "args": {"path": "input/b.csv"}}
        self.assertIsNone(validate(source, proposed, "P1"))

    def test_known_alternative_positive_is_excluded(self):
        source = edge("Read input/a.csv", {"function": "read", "args": {"path": "input/a.csv"}})
        proposed = {"function": "read", "args": {"path": "input/b.csv"}}
        self.assertEqual(validate(source, proposed, "P1", [proposed]), "positive_collision")

    def test_wrong_operation_requires_real_tool_and_same_target(self):
        source = edge("Write report.md", {"function": "write", "args": {"path": "report.md", "content": "report"}})
        self.assertIsNone(validate(source, {"function": "read", "args": {"path": "report.md"}}, "P2"))
        self.assertEqual(validate(source, {"function": "append", "args": {"path": "report.md"}}, "P2"), "p2_unsupported_operation")
        self.assertEqual(validate(source, {"function": "read", "args": {"path": "other.md"}}, "P2"), "p2_changed_target")

    def test_content_change_requires_explicit_intent_constraint(self):
        positive = {"function": "write", "args": {"path": "notes.md", "content": "Meeting notes: status PASS and next steps."}}
        proposed = {"function": "write", "args": {"path": "notes.md", "content": "Meeting notes: status FAIL and next steps."}}
        self.assertEqual(validate(edge("Create the meeting notes file", positive), proposed, "P1"), "p1_content_not_constrained")
        self.assertIsNone(validate(edge("Write notes.md with status PASS", positive), proposed, "P1"))


if __name__ == "__main__":
    unittest.main()
