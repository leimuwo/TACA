import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_ta_pair_cross_encoder import prepare


class PairEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.w = {'window_id': 'w', 'trajectory_id': 't', 'split': 'dev',
                  'intents': [{'intent_id': 'i', 'text': 'Read a.txt', 'temporal_scope': 'current'}],
                  'actions': [{'action_id': 'a', 'action_text': '[ACTION] read a.txt'}],
                  'pairs': [{'pair_id': 'p', 'intent_id': 'i', 'action_id': 'a'}]}
        self.l = {'window_id': 'w', 'trajectory_id': 't', 'split': 'dev', 'label': 'faithful'}
        self.p = {'window_id': 'w', 'pair_id': 'p', 'intent_id': 'i', 'action_id': 'a',
                  'trajectory_id': 't', 'split': 'dev', 'label': 1,
                  'intent': 'Read a.txt', 'action': '[ACTION] read a.txt'}

    def test_accepts_exact_held_out_pair_labels(self):
        self.assertEqual(len(prepare([self.w], [self.l], [self.p], 'dev', [])[0]), 1)

    def test_rejects_train_trajectory_and_text_leakage(self):
        for tid in ('t', 'other'):
            train = [{'trajectory_id': tid, 'intent': self.p['intent'], 'action': self.p['action']}]
            with self.assertRaises(ValueError):
                prepare([self.w], [self.l], [self.p], 'dev', train)

    def test_rejects_missing_and_mismatched_pairs(self):
        with self.assertRaises(ValueError):
            prepare([self.w], [self.l], [], 'dev', [])
        self.p['action'] = 'wrong text'
        with self.assertRaises(ValueError):
            prepare([self.w], [self.l], [self.p], 'dev', [])


if __name__ == '__main__':
    unittest.main()
