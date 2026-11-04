import copy
import unittest
from pathlib import Path
from evaluate_smoke import load, evaluate


class SmokeEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.refs = load(Path(__file__).resolve().parent / 'testset/references.jsonl')
        self.predictions = {k: {**copy.deepcopy(v), 'status': 'completed'} for k, v in self.refs.items()}

    def test_exact(self):
        r = evaluate(self.refs, self.predictions)
        self.assertEqual(r['step_agreement'], 1)
        self.assertEqual(r['reference_pair_agreement_full_denominator'], 1)
        self.assertEqual(r['reference_pairs'], 21)

    def test_missing_counts_as_error(self):
        del self.predictions['S001']
        r = evaluate(self.refs, self.predictions)
        self.assertEqual(r['step_correct_windows'], 19)
        self.assertEqual(r['missing_or_unaligned_reference_pairs'], 1)
        self.assertEqual(r['reference_pairs'], 21)

    def test_wrong_quote_cannot_align_by_id(self):
        self.predictions['S001']['intents'][0]['intent_quote'] = 'Different quote'
        r = evaluate(self.refs, self.predictions)
        self.assertEqual(r['missing_or_unaligned_reference_pairs'], 1)
        self.assertEqual(r['extra_predicted_pairs'], 1)

    def test_technical_failure_is_not_correct_uncertain(self):
        self.predictions['S015']['status'] = 'technical_failure'
        r = evaluate(self.refs, self.predictions)
        self.assertEqual(r['step_correct_windows'], 19)


if __name__ == '__main__':
    unittest.main()
