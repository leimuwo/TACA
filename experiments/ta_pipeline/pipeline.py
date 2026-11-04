#!/usr/bin/env python3
"""Local Thought -> extracted Intent -> legacy pair scorer -> operation aggregation."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contracts import parse_prediction, prioritize_action

# Portable defaults; download_models.py writes these local directories.
INTENT = Path(__file__).resolve().parent / 'models/intent'
RERANKER = Path(__file__).resolve().parent / 'models/reranker'
PROMPT = Path(__file__).resolve().parent / 'student_intent.en.txt'


def normalize(row):
    if not isinstance(row, dict) or not isinstance(row.get('thought'), str):
        raise ValueError('Each row requires thought string')
    uid = row.get('unit_id')
    if not isinstance(uid, str) or not uid.strip():
        raise ValueError('Each row requires a nonempty unit_id string')
    if not isinstance(row.get('actions'), list) or not isinstance(row.get('window_complete'), bool):
        raise ValueError('actions list and explicit window_complete boolean required')
    actions, seen = [], set()
    for index, a in enumerate(row['actions'], 1):
        if not isinstance(a, dict):
            raise ValueError('Action must be object')
        call = a.get('raw_call', a)
        tool = call.get('tool', call.get('function'))
        arguments = call.get('arguments', call.get('args'))
        if isinstance(tool, dict):
            arguments = tool.get('arguments')
            tool = tool.get('name')
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(tool, str) or not tool.strip() or not isinstance(arguments, dict):
            raise ValueError('Action requires tool/function name and arguments/args object')
        aid = str(a.get('action_id', a.get('id', call.get('id', f'A{index}'))))
        if not aid or aid in seen:
            raise ValueError('Empty or duplicate Action ID')
        seen.add(aid)
        text = prioritize_action('[ACTION] ' + json.dumps({'function': tool, 'args': arguments}, ensure_ascii=False))
        actions.append({'action_id': aid, 'tool': tool, 'arguments': arguments, 'model_text': text})
    return {**row, 'actions': actions}


def aggregate(intents, actions, pairs, complete):
    decisions = []
    for i in intents:
        if i['temporal_scope'] != 'current':
            continue
        related = [p for p in pairs if p['intent_id'] == i['intent_id']]
        if any(p['operation_match'] == 'yes' for p in related):
            state = 'yes'
        elif complete and actions and len(related) == len(actions) and all(p['operation_match'] == 'no' for p in related):
            state = 'no'
        else:
            state = 'uncertain'
        decisions.append({'intent_id': i['intent_id'], 'operation_match': state,
                          'matched_action_ids': [p['action_id'] for p in related if p['operation_match'] == 'yes']})
    states = {i['operation_match'] for i in decisions}
    if not states:
        step = 'not_applicable'
    elif states == {'yes'}:
        step = 'fully_consistent'
    elif states == {'no'}:
        step = 'fully_inconsistent'
    elif {'yes', 'no'} <= states:
        step = 'partially_consistent'
    else:
        step = 'uncertain'
    return {'intent_results': decisions, 'step_label': step}


class Models:
    def __init__(self, args):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        self.torch, self.args = torch, args
        dtype = torch.bfloat16 if args.device == 'cuda' and torch.cuda.is_bf16_supported() else (torch.float16 if args.device == 'cuda' else torch.float32)
        self.it = AutoTokenizer.from_pretrained(args.intent_model, local_files_only=True)
        self.im = AutoModelForCausalLM.from_pretrained(args.intent_model, local_files_only=True, torch_dtype=dtype).to(args.device).eval()
        self.rt = AutoTokenizer.from_pretrained(args.reranker, local_files_only=True)
        self.rm = AutoModelForSequenceClassification.from_pretrained(args.reranker, local_files_only=True, torch_dtype=dtype).to(args.device).eval()
        if self.rm.config.num_labels != 1:
            raise ValueError('Legacy reranker must have one logit')
        self.prompt = Path(args.prompt).read_text()
        self.stop_ids = self.im.generation_config.eos_token_id
        self.stop_ids = set(self.stop_ids if isinstance(self.stop_ids, list) else [self.stop_ids]) | {self.it.eos_token_id}

    def extract(self, thought):
        messages = [{'role': 'system', 'content': self.prompt}, {'role': 'user', 'content': thought}]
        text = self.it.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = self.it(text, add_special_tokens=False, truncation=False, return_tensors='pt').to(self.args.device)
        width = tokens['input_ids'].shape[1]
        if width + self.args.max_new_tokens > self.im.config.max_position_embeddings:
            raise ValueError('Intent context overflow; refusing truncation')
        with self.torch.inference_mode():
            output = self.im.generate(**tokens, max_new_tokens=self.args.max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None,
                                      pad_token_id=self.it.pad_token_id or self.it.eos_token_id)
        ids = output[0, width:].tolist()
        raw = self.it.decode(ids, skip_special_tokens=True)
        if not any(token in self.stop_ids for token in ids):
            return None, {'raw': raw, 'status': 'generation_truncated', 'input_tokens': width}
        values, status = parse_prediction(raw, thought)
        if values is None:
            return None, {'raw': raw, 'status': status, 'input_tokens': width}
        return [{'intent_id': f'I{k}', **v} for k, v in enumerate(values, 1)], {'raw': raw, 'status': status, 'input_tokens': width}

    def match(self, intents, actions):
        pairs, jobs = [], []
        for i in intents:
            if i['temporal_scope'] != 'current':
                continue
            for a in actions:
                length = len(self.rt(i['intent_quote'], a['model_text'], truncation=False)['input_ids'])
                pair = {'intent_id': i['intent_id'], 'action_id': a['action_id'], 'input_tokens': length,
                        'operation_match': 'uncertain', 'logit': None, 'sigmoid_score': None,
                        'beyond_legacy_default_512': length > 512}
                pairs.append(pair)
                if length > min(self.args.max_pair_tokens, self.rm.config.max_position_embeddings - 2):
                    pair['reason'] = 'context_overflow'
                else:
                    jobs.append((pair, i['intent_quote'], a['model_text']))
        # Batch close lengths together; long pairs run alone, padding remains dynamic.
        jobs.sort(key=lambda x: x[0]['input_tokens'])
        pos = 0
        while pos < len(jobs):
            size = 1 if jobs[pos][0]['input_tokens'] > 4096 else self.args.batch_size
            batch = jobs[pos:pos + size]
            if any(x[0]['input_tokens'] > 4096 for x in batch):
                batch = [jobs[pos]]
            pos += len(batch)
            try:
                tokens = self.rt([x[1] for x in batch], [x[2] for x in batch], padding=True, truncation=False, return_tensors='pt').to(self.args.device)
                with self.torch.inference_mode():
                    scores = self.rm(**tokens).logits[:, 0].float().cpu().tolist()
                if len(scores) != len(batch) or not all(math.isfinite(s) for s in scores):
                    raise ValueError('Invalid model scores')
                for (pair, _, _), score in zip(batch, scores):
                    pair.update(logit=score, sigmoid_score=1 / (1 + math.exp(-max(-700, min(700, score)))),
                                operation_match='yes' if score >= self.args.threshold + self.args.gray_margin else 'no' if score < self.args.threshold - self.args.gray_margin else 'uncertain')
            except self.torch.cuda.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                for pair, _, _ in batch:
                    pair['reason'] = 'out_of_memory'
        return pairs


def run_unit(row, models):
    started = time.monotonic()
    result = {'unit_id': row['unit_id'], 'thought': row['thought'], 'actions': row['actions'],
              'window_complete': row['window_complete'], 'protocol': 'operation-pair-v2',
              'label_status': 'legacy_reranker_experimental_not_gold',
              'semantic_acceptance_passed': False, 'intents': [], 'pairs': [], 'intent_results': []}
    try:
        intents, extraction = models.extract(row['thought'])
        result['extraction'] = extraction
        if intents is None:
            raise ValueError('Intent extraction invalid: ' + extraction['status'])
        result['intents'] = intents
        result['pairs'] = models.match(intents, row['actions'])
        result.update(aggregate(intents, row['actions'], result['pairs'], row['window_complete']))
        result['status'] = 'completed_with_uncertain' if any(p['operation_match'] == 'uncertain' for p in result['pairs']) or result['step_label'] == 'uncertain' else 'completed'
    except Exception as exc:
        result.update(status='technical_failure', step_label='uncertain', error=f'{type(exc).__name__}: {exc}')
    result['seconds'] = round(time.monotonic() - started, 3)
    return result


def fingerprint(path):
    p = Path(path).resolve()
    if not p.is_dir():
        raise ValueError(f'Model directory missing: {p}')
    return {f.name: {'bytes': f.stat().st_size, 'mtime_ns': f.stat().st_mtime_ns} for f in p.iterdir() if f.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory, or same settings for resume')
    parser.add_argument('--intent-model', default=str(INTENT))
    parser.add_argument('--reranker', default=str(RERANKER))
    parser.add_argument('--prompt', type=Path, default=PROMPT)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--threshold', type=float, default=0.0, help='Legacy logit threshold; not calibrated for new rules')
    parser.add_argument('--gray-margin', type=float, default=0.0)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-pair-tokens', type=int, default=8192)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or not math.isfinite(args.gray_margin) or args.gray_margin < 0 or min(args.batch_size, args.max_pair_tokens, args.max_new_tokens) < 1 or args.max_pair_tokens > 8192:
        parser.error('Invalid thresholds, batch size or token limits')
    rows = [normalize(json.loads(l)) for l in args.input.read_text().splitlines() if l.strip()]
    if not rows or len({r['unit_id'] for r in rows}) != len(rows):
        raise ValueError('Empty input or duplicate unit_id')
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ('check_only', 'output')}
    settings.update(input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    prompt_sha256=hashlib.sha256(args.prompt.read_bytes()).hexdigest(),
                    intent_files=fingerprint(args.intent_model), reranker_files=fingerprint(args.reranker),
                    protocol='operation-pair-v2', legacy_reranker_semantics=True,
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.check_only:
        print(json.dumps({'windows': len(rows), 'actions': sum(len(r['actions']) for r in rows), 'model_files_present': True, 'runtime_inference_tested': False}, ensure_ascii=False))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != settings:
            raise ValueError('Settings changed; use a new output directory')
    else:
        if any(args.output.iterdir()):
            raise ValueError('Nonempty output without manifest')
        manifest.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + '\n')
    predictions = args.output / 'predictions.jsonl'
    done = [json.loads(l) for l in predictions.read_text().splitlines() if l.strip()] if predictions.exists() else []
    ids = {r['unit_id'] for r in done}
    if len(ids) != len(done) or not ids <= {r['unit_id'] for r in rows}:
        raise ValueError('Invalid cached IDs')
    models = Models(args) if len(done) < len(rows) else None
    with predictions.open('a') as stream:
        for row in rows:
            if row['unit_id'] in ids:
                continue
            result = run_unit(row, models)
            stream.write(json.dumps(result, ensure_ascii=False) + '\n'); stream.flush()
            done.append(result)
            print(json.dumps({'done': len(done), 'total': len(rows), 'unit_id': row['unit_id'], 'status': result['status']}, ensure_ascii=False), flush=True)
    from collections import Counter
    (args.output / 'summary.json').write_text(json.dumps({'windows': len(done), 'statuses': dict(Counter(r['status'] for r in done)), 'step_labels': dict(Counter(r['step_label'] for r in done)), 'limit': 'Experimental legacy reranker; scores are not calibrated probabilities; no new-rule semantic acceptance.'}, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
