#!/usr/bin/env python3
"""Evaluate authored development smoke references; never claim independent accuracy."""
import argparse
import collections
import json
from pathlib import Path


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if len({r['unit_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate unit IDs')
    return {r['unit_id']: r for r in rows}


def evaluate(refs, predictions):
    if set(predictions) - set(refs):
        raise ValueError('Unexpected prediction IDs')
    extraction_exact = step_correct = pair_correct = pair_total = missing_pair = 0
    gold_counts = collections.Counter()
    tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter()
    cases = []; statuses = collections.Counter(); extra_pair_count = 0
    for uid, ref in refs.items():
        pred = predictions.get(uid)
        statuses[pred.get('status', 'unknown') if pred else 'missing_window'] += 1
        actual_intents = pred.get('intents', []) if pred else []
        # Exact reference text+scope, not locally assigned model IDs.
        expected_intents = [(i['intent_quote'], i['temporal_scope']) for i in ref['intents']]
        observed_intents = [(i['intent_quote'], i['temporal_scope']) for i in actual_intents]
        valid = pred is not None and pred.get('status') != 'technical_failure'
        exact = valid and observed_intents == expected_intents
        extraction_exact += exact
        step_ok = valid and pred.get('step_label') == ref['step_label']
        step_correct += step_ok
        gold_map = {i['intent_id']: (i['intent_quote'], i['temporal_scope']) for i in ref['intents']}
        pred_map = {i['intent_id']: (i['intent_quote'], i['temporal_scope']) for i in actual_intents}
        actual = {}
        for pair in pred.get('pairs', []) if pred else []:
            if pair['intent_id'] not in pred_map:
                raise ValueError('Prediction pair references unknown Intent')
            key = (pred_map[pair['intent_id']], pair['action_id'])
            if key in actual:
                raise ValueError('Duplicate predicted pair')
            actual[key] = pair['operation_match'] if valid else 'missing'
        expected_keys = set(); errors = []
        for pair in ref['pairs']:
            key = (gold_map[pair['intent_id']], pair['action_id'])
            expected_keys.add(key)
            label = pair['operation_match']; observed = actual.get(key, 'missing')
            pair_total += 1; gold_counts[label] += 1; pair_correct += observed == label
            missing_pair += observed == 'missing'
            if observed == label:
                tp[label] += 1
            else:
                fn[label] += 1
                fp[observed] += 1
                errors.append({'intent_quote': key[0][0], 'action_id': key[1], 'expected': label, 'observed': observed})
        extra = set(actual) - expected_keys
        extra_pair_count += len(extra)
        for key in extra:
            fp[actual[key]] += 1
        cases.append({'unit_id': uid, 'case_type': ref['case_type'], 'extraction_exact': exact,
                      'step_correct': step_ok, 'expected_step': ref['step_label'],
                      'observed_step': pred.get('step_label') if pred else None,
                      'pair_errors': errors, 'extra_predicted_pairs': len(extra)})
    metrics = {}
    for label in ['yes', 'no', 'uncertain']:
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.
        recall = tp[label] / gold_counts[label] if gold_counts[label] else None
        metrics[label] = {'support': gold_counts[label], 'precision': precision, 'recall': recall,
                          'f1': 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else 0.}
    return {'limit': 'Authored exposed protocol smoke; NOT independent human Gold or natural trajectory accuracy.',
            'windows': len(refs), 'predicted_windows': len(predictions), 'statuses': dict(statuses),
            'extraction_exact_windows': extraction_exact, 'extraction_exact_rate': extraction_exact / len(refs),
            'step_correct_windows': step_correct, 'step_agreement': step_correct / len(refs),
            'reference_pairs': pair_total, 'correct_reference_pairs': pair_correct,
            'reference_pair_agreement_full_denominator': pair_correct / pair_total if pair_total else None,
            'missing_or_unaligned_reference_pairs': missing_pair, 'extra_predicted_pairs': extra_pair_count,
            'per_class_end_to_end': metrics, 'cases': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--references', type=Path, default=Path(__file__).resolve().parent / 'testset/references.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(load(args.references), load(args.predictions))
    if args.output.exists():
        raise FileExistsError('Use a new report path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
