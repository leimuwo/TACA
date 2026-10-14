#!/usr/bin/env python3
"""Evaluate a TA Cross-Encoder on a frozen labeled pair subset."""
import argparse
import json
from pathlib import Path

from evaluate_bge_clawgym import rows, sha256
from evaluate_ta_clear_split import write_jsonl
from finalize_ta_glm_eval import text_key
from train_reranker_clawgym import prioritize_action
from train_ta_pair_reranker import classification_metrics
from ta_evaluation_scope import exclusion_reason


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--train-data', type=Path, required=True)
    p.add_argument('--split', choices=['dev', 'test'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--threshold', type=float, default=0.0)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.output.exists() or a.batch_size < 1:
        raise ValueError('output must not exist and batch size must be positive')
    train = list(rows(a.train_data / 'train.jsonl')) + list(rows(a.train_data / 'internal_validation.jsonl'))
    tids = {r['trajectory_id'] for r in train}
    keys = {text_key(r['intent'], r['action']) for r in train}
    source = list(rows(a.pairs))
    if not source or len({r['pair_id'] for r in source}) != len(source):
        raise ValueError('empty or duplicate evaluation pairs')
    selected, excluded = [], []
    for r in source:
        if r['split'] != a.split or r['label'] not in (0, 1):
            raise ValueError('invalid split or binary label')
        if r['trajectory_id'] in tids or text_key(r['intent'], r['action']) in keys:
            raise ValueError('training trajectory or text pair overlaps evaluation')
        reason = exclusion_reason(r['intent'])
        if reason:
            excluded.append({'pair_id': r['pair_id'], 'reason': reason})
        else:
            selected.append(r)
    if not selected:
        raise ValueError('no eligible evaluation pairs')
    setup = {'split': a.split, 'pairs': len(selected), 'excluded': len(excluded),
             'label_status': 'GLM weak supervision; NOT human Gold',
             'threshold': a.threshold, 'checkpoint': str(a.checkpoint),
             'pairs_sha256': sha256(a.pairs),
             'train_sha256': sha256(a.train_data / 'train.jsonl'),
             'internal_validation_sha256': sha256(a.train_data / 'internal_validation.jsonl')}
    if a.dry_run:
        print(json.dumps(setup))
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    tokenizer = AutoTokenizer.from_pretrained(a.checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(a.checkpoint, local_files_only=True).to(a.device).eval()
    if model.config.num_labels != 1:
        raise ValueError('checkpoint must output one logit')
    logits = []
    with torch.inference_mode():
        for start in range(0, len(selected), a.batch_size):
            batch = selected[start:start+a.batch_size]
            tokens = tokenizer([r['intent'] for r in batch],
                               [prioritize_action(r['action']) for r in batch],
                               padding=True, truncation=True, max_length=a.max_length,
                               return_tensors='pt').to(a.device)
            logits.extend(model(**tokens).logits[:, 0].float().cpu().tolist())
    metrics = classification_metrics([v-a.threshold for v in logits], [r['label'] for r in selected])
    a.output.mkdir(parents=True)
    write_jsonl(a.output / 'predictions.jsonl', [{**r, 'logit': v, 'predicted_label': int(v >= a.threshold)}
                                               for r, v in zip(selected, logits)])
    write_jsonl(a.output / 'excluded_pairs.jsonl', excluded)
    (a.output / 'report.json').write_text(json.dumps({**setup, 'metrics': metrics}, indent=2)+'\n')
    print(f"[{a.split}] ACC={metrics['accuracy']:.4f} PRE={metrics['precision']:.4f} REC={metrics['recall']:.4f} F1={metrics['f1']:.4f}")


if __name__ == '__main__':
    main()
