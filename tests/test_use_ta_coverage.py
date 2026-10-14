import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from use_ta_coverage import judge


class FakeScorer:
    def __init__(self, logits):
        self.logits = logits

    def score(self, pairs):
        return self.logits


class UsageTests(unittest.TestCase):
    def request(self):
        return {'intents': [{'text': 'Read a.txt', 'temporal_scope': 'current', 'assessable': True},
                            {'text': 'Run tests', 'temporal_scope': 'current', 'assessable': True}],
                'actions': [{'function': 'read', 'args': {'path': 'a.txt'}}]}

    def test_multitarget_aggregation_and_scores(self):
        result = judge(self.request(), FakeScorer([4., -4.]))
        self.assertEqual(result['label'], 'partial')
        self.assertEqual(result['intent_coverage'][0]['covering_action_ids'], ['A1'])
        self.assertGreater(result['pairs'][0]['coverage_score'], .9)
        self.assertEqual(judge(self.request(), FakeScorer([4., 4.]))['label'], 'faithful')
        self.assertEqual(judge(self.request(), FakeScorer([-4., -4.]))['label'], 'not_executed')

    def test_bulk_and_unconfirmed_targets_abstain(self):
        request = self.request()
        request['intents'][0]['text'] = 'Read all files'
        self.assertEqual(judge(request, FakeScorer([]))['label'], 'unassessable')
        request = self.request()
        del request['intents'][0]['assessable']
        self.assertEqual(judge(request, FakeScorer([]))['label'], 'unassessable')

    def test_future_ignored_and_empty_calls_not_executed(self):
        request = self.request()
        request['intents'][1]['temporal_scope'] = 'future'
        self.assertEqual(judge(request, FakeScorer([4.]))['label'], 'faithful')
        request['actions'] = []
        self.assertEqual(judge(request, FakeScorer([]))['label'], 'not_executed')

    def test_invalid_and_duplicate_inputs_rejected(self):
        request = self.request()
        request['intents'][0]['temporal_scope'] = 'unknown'
        with self.assertRaises(ValueError):
            judge(request, FakeScorer([]))
        request = self.request()
        request['actions'] *= 2
        for a in request['actions']:
            a['action_id'] = 'same'
        with self.assertRaises(ValueError):
            judge(request, FakeScorer([]))

    def test_bad_score_output_rejected(self):
        with self.assertRaises(RuntimeError):
            judge(self.request(), FakeScorer([float('nan'), 1.]))


if __name__ == '__main__':
    unittest.main()
