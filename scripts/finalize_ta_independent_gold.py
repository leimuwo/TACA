#!/usr/bin/env python3
"""Validate completed human overall/detailed reviews and freeze held-out Gold."""
import argparse
import json
from collections import Counter
from pathlib import Path

from evaluate_bge_clawgym import rows, sha256
from evaluate_step_consistency import CLASSES
from ta_coverage_aggregation import aggregate_ta_label
from build_ta_coverage_pilot import write


def validate_reviews(windows, reviews, detailed=False):
    expected = {w['window_id']: w for w in windows}
    if len(expected) != len(windows):
        raise ValueError('duplicate source window')
    indexed = {r['window_id']: r for r in reviews}
    if len(indexed) != len(reviews) or set(indexed) != set(expected):
        raise ValueError('review IDs must match the frozen queue exactly')
    result = []
    for wid, w in expected.items():
        r = indexed[wid]
        if w['split'] not in {'dev', 'test'}:
            raise ValueError('train cannot enter independent Gold')
        if (r.get('trajectory_id') != w['trajectory_id'] or r.get('split') != w['split']
                or r.get('review_status') != 'human_reviewed'
                or not str(r.get('annotator') or '').strip()
                or not str(r.get('review_reason') or '').strip()
                or r.get('gold_label') not in (*CLASSES, 'unassessable')):
            raise ValueError(f'incomplete human annotation or source mismatch: {wid}')
        if detailed:
            intents, pairs = r.get('intents', {}), r.get('pairs', {})
            if (set(intents) != {i['intent_id'] for i in w['intents']}
                    or set(pairs) != {p['pair_id'] for p in w['pairs']}
                    or any(d.get('due_now') not in {'yes', 'no'} or d.get('assessable') not in {'yes', 'no'}
                           for d in intents.values())
                    or any(v not in {'yes', 'no', 'uncertain'} for v in pairs.values())):
                raise ValueError(f'incomplete Intent/pair annotation: {wid}')
            due = {i for i, d in intents.items() if d['due_now'] == 'yes'}
            if r['gold_label'] != 'unassessable' and any(
                    pairs[p['pair_id']] == 'uncertain' for p in w['pairs'] if p['intent_id'] in due):
                raise ValueError(f'uncertain pair cannot enter strict Gold: {wid}')
            derived = aggregate_ta_label(w,
                {i: {k: v == 'yes' for k, v in d.items()} for i, d in intents.items()},
                {p: v == 'yes' for p, v in pairs.items()})
            if r['gold_label'] != 'unassessable' and r['gold_label'] != derived:
                raise ValueError(f'overall label conflicts with pair coverage: {wid}')
            if r['gold_label'] == 'unassessable' and derived != 'unassessable' and not r.get('disputes'):
                raise ValueError(f'explain complex/uncertain exclusion in disputes: {wid}')
        result.append(r)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--dev-review', type=Path, required=True)
    parser.add_argument('--test-review', type=Path, required=True)
    parser.add_argument('--pair-review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.dataset / 'manifest.json').read_text())
    for name, digest in manifest['output_sha256'].items():
        if sha256(args.dataset / name) != digest:
            raise ValueError(f'frozen file changed: {name}')
    assignments = {r['trajectory_id']: r['split'] for r in rows(args.dataset / 'split_assignments.jsonl')}
    groups = {}
    for split, path in [('dev', args.dev_review), ('test', args.test_review), ('pair30', args.pair_review)]:
        windows = list(rows(args.dataset / split / 'windows.jsonl'))
        for w in windows:
            if assignments.get(w['trajectory_id']) != w['split'] or (split != 'pair30' and w['split'] != split):
                raise ValueError('source split mismatch')
        groups[split] = validate_reviews(windows, list(rows(path)), detailed=split == 'pair30')
    overall = {r['window_id']: r for s in ('dev', 'test') for r in groups[s]}
    for r in groups['pair30']:
        if overall[r['window_id']]['gold_label'] != r['gold_label']:
            raise ValueError(f'overall/detailed disagreement needs adjudication: {r["window_id"]}')
    args.output.mkdir(parents=True)
    for s, records in groups.items():
        write(args.output / (s + '_gold.jsonl'), records)
    report = {'status': 'human_gold_frozen', 'train_overlap': 0,
              'labels': {s: dict(Counter(r['gold_label'] for r in records)) for s, records in groups.items()},
              'dataset_manifest_sha256': sha256(args.dataset / 'manifest.json'),
              'annotation_sha256': {s: sha256(p) for s, p in [('dev', args.dev_review), ('test', args.test_review), ('pair30', args.pair_review)]},
              'test_policy': 'no threshold calibration, rule development, or training on test',
              'limitation': 'Gold tracks human provenance; agreement needs independent second annotation'}
    (args.output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
