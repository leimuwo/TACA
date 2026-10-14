"""Clear-Intent TA split selection contracts."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_ta_clear_split import build_clear_windows, clear_keys  # noqa: E402
from ta_evaluation_scope import exclusion_reason


class TAClearEvaluationTests(unittest.TestCase):
    def test_bulk_file_content_objectives_are_excluded(self):
        for text in ('Read all input files from input/', 'Read the HTML files in input/pages/',
                     'Review the skill package files under input/seo-analyzer-v2/', '阅读所有文件'):
            self.assertEqual(exclusion_reason(text), 'non_atomic_bulk_file_read')
        for text in ('Read input/config.yaml', 'Inspect the input directory to see what files are there.',
                     'Write a script that reads all input files', 'Check what input files exist'):
            self.assertIsNone(exclusion_reason(text))

    def test_bulk_intent_excludes_whole_window(self):
        source = {'trajectory_id': 't', 'thought_step': 1,
                  'execution_intents': [{'intent_id': 'i1', 'temporal_scope': 'current',
                                        'intent_text': 'Read all files'},
                                       {'intent_id': 'i2', 'temporal_scope': 'current',
                                        'intent_text': 'Read config.yaml'}]}
        selected, excluded = build_clear_windows('dev', [source], {'t': 'dev'}, {},
                                                 {('t', 1, 'i1'), ('t', 1, 'i2')})
        self.assertEqual(selected, [])
        self.assertEqual(excluded['non_atomic_bulk_file_read'], 1)
    def test_test_screen_uses_source_query_id_and_checks_identity(self):
        queries = [{"query_id": "q1", "trajectory_id": "t", "thought_step": 1, "intent_id": "i1"},
                   {"query_id": "q2", "trajectory_id": "t", "thought_step": 1, "intent_id": "i2"}]
        excluded = [{"source_query_id": "q2", "split": "test", "ambiguous": True,
                     "trajectory_id": "t", "thought_step": 1, "intent_id": "i2"}]
        self.assertEqual(clear_keys("test", [], queries, excluded), {("t", 1, "i1")})
        excluded[0]["intent_id"] = "i1"
        with self.assertRaisesRegex(ValueError, "disagrees"):
            clear_keys("test", [], queries, excluded)

    def test_mixed_window_is_excluded_in_full(self):
        rows = [{"trajectory_id": "t", "thought_step": 1,
                 "execution_intents": [{"intent_id": "i1", "temporal_scope": "current"},
                                       {"intent_id": "i2", "temporal_scope": "current"}]}]
        selected, excluded = build_clear_windows("dev", rows, {"t": "dev"}, {}, {("t", 1, "i1")})
        self.assertEqual(selected, [])
        self.assertEqual(excluded["ambiguous_or_unscreened_current_intent"], 1)


if __name__ == "__main__":
    unittest.main()
