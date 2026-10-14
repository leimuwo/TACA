import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_ta_independent_evaluation import select_distinct, template
from finalize_ta_independent_gold import validate_reviews


class IndependentGoldTests(unittest.TestCase):
    def setUp(self):
        self.w = {'window_id': 'w', 'trajectory_id': 't', 'split': 'test',
                  'intents': [{'intent_id': 'i', 'temporal_scope': 'future'}],
                  'pairs': [{'pair_id': 'p', 'intent_id': 'i', 'action_id': 'a'}]}
        self.r = template(self.w, True)
        self.r.update(gold_label='faithful', annotator='human-1',
                      review_reason='target is covered', review_status='human_reviewed')
        self.r['intents']['i'] = {'due_now': 'yes', 'assessable': 'yes'}
        self.r['pairs']['p'] = 'yes'

    def test_selection_is_order_independent_and_distinct(self):
        items = [{'window_id': str(n), 'trajectory_id': str(n // 2)} for n in range(20)]
        chosen = select_distinct(items, 8, 13)
        self.assertEqual(chosen, select_distinct(list(reversed(items)), 8, 13))
        self.assertEqual(len({w['trajectory_id'] for w in chosen}), 8)

    def test_unfinished_and_train_reviews_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            validate_reviews([self.w], [template(self.w, True)], True)
        w = copy.deepcopy(self.w)
        w['split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'train'):
            validate_reviews([w], [self.r], True)

    def test_pair_conflict_and_uncertainty_are_rejected(self):
        self.r['pairs']['p'] = 'no'
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            validate_reviews([self.w], [self.r], True)
        self.r['pairs']['p'] = 'uncertain'
        with self.assertRaisesRegex(ValueError, 'uncertain'):
            validate_reviews([self.w], [self.r], True)

    def test_corrected_due_now_future_intent_can_be_evaluated(self):
        self.assertEqual(validate_reviews([self.w], [self.r], True)[0]['gold_label'], 'faithful')

    def test_missing_and_duplicate_windows_are_rejected(self):
        for reviews in ([], [self.r, self.r]):
            with self.assertRaisesRegex(ValueError, 'IDs'):
                validate_reviews([self.w], reviews)

    def test_complex_exclusion_requires_explanation(self):
        self.r['gold_label'] = 'unassessable'
        with self.assertRaisesRegex(ValueError, 'explain'):
            validate_reviews([self.w], [self.r], True)
        self.r['disputes'] = 'joint coverage cannot be atomized'
        validate_reviews([self.w], [self.r], True)


if __name__ == '__main__':
    unittest.main()
