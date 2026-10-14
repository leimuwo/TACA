#!/usr/bin/env python3
"""Evaluate a checkpoint on finalized, held-out GLM weak labels."""
import argparse
import json
from pathlib import Path

from evaluate_bge_clawgym import rows, sha256
from evaluate_ta_clear_split import predict, write_jsonl
from evaluate_step_consistency import score, CLASSES
from train_ta_pair_reranker import classification_metrics
from finalize_ta_glm_eval import text_key
from ta_evaluation_scope import window_exclusions, POLICY_VERSION


def prepare(windows, labels, pairs, split, train):
    by_window = {w['window_id']: w for w in windows}
    by_label = {r['window_id']: r for r in labels}
    by_pair = {p['pair_id']: p for p in pairs}
    if len(by_window) != len(windows) or len(by_label) != len(labels) or len(by_pair) != len(pairs):
        raise ValueError('duplicate window/label/pair ID')
    if not labels or not set(by_label) <= set(by_window):
        raise ValueError('empty or unknown labeled windows')
    train_ids = {r['trajectory_id'] for r in train}
    train_keys = {text_key(r['intent'], r['action']) for r in train}
    expected_pairs = {p['pair_id'] for wid in by_label for p in by_window[wid]['pairs']}
    if set(by_pair) != expected_pairs:
        raise ValueError('finalized pair labels must exactly cover labeled windows')
    selected, excluded = [], []
    for wid, label in by_label.items():
        w = by_window[wid]
        if (w['split'] != split or label['split'] != split
                or w['trajectory_id'] != label['trajectory_id']
                or w['trajectory_id'] in train_ids):
            raise ValueError('held-out split/source violation')
        if label['label'] not in CLASSES:
            raise ValueError('invalid window label')
        intents = {i['intent_id']: i['text'] for i in w['intents']}
        actions = {a['action_id']: a['action_text'] for a in w['actions']}
        for p in w['pairs']:
            r = by_pair[p['pair_id']]
            if (r['window_id'] != wid or r['split'] != split or r['label'] not in (0, 1)
                    or r['trajectory_id'] != w['trajectory_id']
                    or r['intent_id'] != p['intent_id'] or r['action_id'] != p['action_id']
                    or r['intent'] != intents[p['intent_id']] or r['action'] != actions[p['action_id']]):
                raise ValueError('pair source/text/label mismatch')
            if text_key(r['intent'], r['action']) in train_keys:
                raise ValueError('training text pair in evaluation')
        exclusions = window_exclusions(w)
        if exclusions:
            excluded.append({'window_id': wid, 'intents': exclusions})
        else:
            selected.append(w)
    if not selected:
        raise ValueError('no eligible windows')
    return selected, by_label, by_pair, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--windows', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--train-data', type=Path, required=True)
    parser.add_argument('--split', choices=['dev', 'test'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--threshold', type=float, default=0.0, help='Logit threshold; calibrate on dev only')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train = list(rows(args.train_data / 'train.jsonl')) + list(rows(args.train_data / 'internal_validation.jsonl'))
    selected, labels, pairs, excluded = prepare(list(rows(args.windows)), list(rows(args.labels)),
                                               list(rows(args.pairs)), args.split, train)
    report = {'split': args.split, 'windows': len(selected),
              'pairs': sum(len(w['pairs']) for w in selected), 'bulk_read_excluded': len(excluded),
              'scope_policy': POLICY_VERSION, 'label_status': 'GLM weak supervision; NOT human Gold',
              'threshold': args.threshold, 'checkpoint': str(args.checkpoint),
              'input_sha256': {name: sha256(path) for name, path in
                               [('windows', args.windows), ('labels', args.labels), ('pairs', args.pairs),
                                ('train', args.train_data / 'train.jsonl'),
                                ('internal_validation', args.train_data / 'internal_validation.jsonl')]}}
    if args.dry_run:
        print(json.dumps({**report, 'mode': 'dry_run'}, ensure_ascii=False))
        return
    predictions = predict(selected, args.checkpoint, args.device, args.batch_size, args.max_length, args.threshold)
    by_pred = {p['window_id']: p for p in predictions}
    comparisons, logits, targets = [], [], []
    for w in selected:
        pred = by_pred[w['window_id']]
        comparisons.append({'query_id': w['window_id'], 'action_id': w['window_id'],
                            'gold_label': labels[w['window_id']]['label'],
                            'predicted_label': pred['predicted_label'],
                            'action_tool': '+'.join(sorted({a['raw_call']['function'] for a in w['actions']}))})
        for p in w['pairs']:
            logits.append(pred['pair_logits'][p['pair_id']] - args.threshold)
            targets.append(pairs[p['pair_id']]['label'])
    report['pair_metrics'] = classification_metrics(logits, targets)
    report['window_metrics'] = score(comparisons, require_review=False)
    per_class = report['window_metrics']['per_class']
    report['window_metrics']['macro_precision'] = sum(per_class[c]['precision'] for c in CLASSES) / len(CLASSES)
    report['window_metrics']['macro_recall'] = sum(per_class[c]['recall'] for c in CLASSES) / len(CLASSES)
    report['metrics_ready'] = True
    args.output.mkdir(parents=True)
    write_jsonl(args.output / 'windows.jsonl', selected)
    write_jsonl(args.output / 'predictions.jsonl', predictions)
    write_jsonl(args.output / 'excluded_windows.jsonl', excluded)
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    pair = report['pair_metrics']
    step = report['window_metrics']
    print(f"[{args.split}] Pair (positive=covered): "
          f"ACC={pair['accuracy']:.4f} PRE={pair['precision']:.4f} "
          f"REC={pair['recall']:.4f} F1={pair['f1']:.4f}")
    print(f"[{args.split}] Step (macro over 3 classes): "
          f"ACC={step['accuracy']:.4f} PRE={step['macro_precision']:.4f} "
          f"REC={step['macro_recall']:.4f} F1={step['macro_f1']:.4f}")
    print(f"Report: {args.output / 'report.json'}")


if __name__ == '__main__':
    main()
