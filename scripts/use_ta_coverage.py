#!/usr/bin/env python3
"""Apply the trained pair classifier to actual tool calls (JSONL or HTTP)."""
import argparse
import json
import math
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

from train_reranker_clawgym import prioritize_action
from ta_evaluation_scope import exclusion_reason

DEFAULT = Path('outputs/ta_pair_cross_encoder_10k_v3/best')


class CoverageModel:
    def __init__(self, checkpoint, device='cuda', batch_size=16, max_length=512):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        self.torch, self.device = torch, device
        self.batch_size, self.max_length = batch_size, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
        if self.model.config.num_labels != 1:
            raise ValueError('checkpoint must output one logit')

    def score(self, pairs):
        result = []
        with self.torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start:start+self.batch_size]
                tokens = self.tokenizer([p[0] for p in batch], [p[1] for p in batch],
                                        padding=True, truncation=True, max_length=self.max_length,
                                        return_tensors='pt').to(self.device)
                result.extend(self.model(**tokens).logits[:, 0].float().cpu().tolist())
        return result


def judge(request, scorer, threshold=0.0):
    if not isinstance(request, dict):
        raise ValueError('input must be an object')
    intents, actions = request.get('intents'), request.get('actions')
    if not isinstance(intents, list) or not isinstance(actions, list):
        raise ValueError('intents and actions must be lists')
    current, excluded, seen = [], [], set()
    for index, i in enumerate(intents):
        if not isinstance(i, dict) or not isinstance(i.get('text'), str) or not i['text'].strip():
            raise ValueError('each Intent needs nonempty text')
        iid = str(i.get('intent_id', f'I{index+1}'))
        if iid in seen:
            raise ValueError('duplicate Intent ID')
        seen.add(iid)
        temporal = i.get('temporal_scope')
        if temporal not in ('current', 'future', 'conditional'):
            raise ValueError('each Intent needs current/future/conditional temporal_scope')
        if temporal != 'current':
            continue
        reason = exclusion_reason(i['text'])
        if i.get('assessable') is not True:
            reason = 'assessability_not_confirmed'
        if reason:
            excluded.append({'intent_id': iid, 'reason': reason})
        current.append({'intent_id': iid, 'text': i['text']})
    normalized, action_ids = [], set()
    for index, a in enumerate(actions):
        if not isinstance(a, dict):
            raise ValueError('Action must be an object')
        call = a.get('raw_call', a)
        if not isinstance(call, dict) or not isinstance(call.get('function'), str) or not isinstance(call.get('args'), dict):
            raise ValueError('Action requires function string and args object')
        aid = str(a.get('action_id', call.get('id', f'A{index+1}')))
        if aid in action_ids:
            raise ValueError('duplicate Action ID')
        action_ids.add(aid)
        text = '[ACTION] ' + json.dumps(call, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        normalized.append((aid, prioritize_action(text)))
    result = {'request_id': request.get('request_id'), 'threshold': threshold,
              'judgment_scope': 'tool_call_semantic_coverage', 'excluded_intents': excluded,
              'pairs': [], 'intent_coverage': []}
    if not current or excluded:
        result['label'] = 'unassessable'
        result['reason'] = 'no_current_intent' if not current else 'current_intent_not_assessable'
        return result
    pairs = [(i, aid, text) for i in current for aid, text in normalized]
    logits = scorer.score([(i['text'], text) for i, aid, text in pairs]) if pairs else []
    if len(logits) != len(pairs) or any(not math.isfinite(v) for v in logits):
        raise RuntimeError('invalid model scores')
    for (i, aid, _), logit in zip(pairs, logits):
        probability = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
        result['pairs'].append({'intent_id': i['intent_id'], 'action_id': aid,
                                'logit': logit, 'coverage_score': probability, 'covered': logit >= threshold})
    for i in current:
        covered_actions = [p['action_id'] for p in result['pairs'] if p['intent_id'] == i['intent_id'] and p['covered']]
        result['intent_coverage'].append({'intent_id': i['intent_id'], 'covered': bool(covered_actions),
                                          'covering_action_ids': covered_actions})
    count = sum(i['covered'] for i in result['intent_coverage'])
    result['label'] = 'faithful' if count == len(current) else 'partial' if count else 'not_executed'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or args.batch_size < 1 or args.max_length < 32:
        parser.error('invalid threshold/batch-size/max-length')
    if args.serve and (args.input or args.output):
        parser.error('serve cannot be combined with input/output')
    if not args.serve and (not args.input or not args.output):
        parser.error('batch mode needs input and output')
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    model = CoverageModel(args.checkpoint, args.device, args.batch_size, args.max_length)
    if not args.serve:
        results = []
        for line in args.input.read_text().splitlines():
            if line.strip():
                results.append(judge(json.loads(line), model, args.threshold))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            for result in results:
                stream.write(json.dumps(result, ensure_ascii=False)+'\n')
        print(json.dumps({'windows': len(results), 'output': str(args.output)}))
        return
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.respond(200 if self.path == '/health' else 404,
                         {'ready': True, 'checkpoint': str(args.checkpoint)} if self.path == '/health' else {'error': 'not_found'})

        def do_POST(self):
            if self.path != '/judge':
                self.respond(404, {'error': 'not_found'})
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 4 * 1024 * 1024:
                    raise ValueError('request body must be between 1 byte and 4 MiB')
                body = json.loads(self.rfile.read(length))
                self.respond(200, judge(body, model, args.threshold))
            except (ValueError, TypeError, KeyError) as exc:
                self.respond(400, {'error': str(exc)})
            except Exception:
                self.respond(500, {'error': 'model_inference_failed'})
    print(f'TA coverage ready at http://{args.host}:{args.port}', flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
