#!/usr/bin/env python3
"""Prepare blind dev/test overall review and a 30-window detailed audit."""
import argparse
import base64
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_ta_coverage_pilot import ROOT, INTENTS, SPLITS, raw_index, split_index, window, write, stable_id
from evaluate_bge_clawgym import rows, sha256
from render_ta_coverage_review import HTML


def select_distinct(windows, count, seed):
    by_trajectory = {}
    for item in windows:
        by_trajectory.setdefault(item['trajectory_id'], []).append(item)
    if count < 1 or len(by_trajectory) < count:
        raise ValueError('insufficient distinct held-out trajectories')
    tids = sorted(by_trajectory, key=lambda tid: stable_id(seed, tid))[:count]
    return [min(by_trajectory[tid], key=lambda w: stable_id(seed, w['window_id'])) for tid in tids]


def template(w, detailed):
    return {'window_id': w['window_id'], 'trajectory_id': w['trajectory_id'],
            'split': w['split'], 'gold_label': '', 'review_reason': '', 'disputes': '',
            'annotator': '', 'review_status': 'unreviewed',
            'intents': {i['intent_id']: {'due_now': '', 'assessable': ''} for i in w['intents']} if detailed else {},
            'pairs': {p['pair_id']: '' for p in w['pairs']} if detailed else {}}


def page(windows, output, detailed, reviews=None):
    cases = [{'category': 'blind/' + w['split'], 'window': w,
              'model_review': {'verdict': None}} for w in windows]
    data = base64.b64encode(json.dumps(cases, ensure_ascii=False).encode()).decode()
    html = HTML.replace('__DATA__', data).replace(
        "pairs:Object.fromEntries(c.window.pairs.map(p=>[p.pair_id,''])}",
        "pairs:Object.fromEntries(c.window.pairs.map(p=>[p.pair_id,'']))}")
    html = html.replace('· train ·', '· ${esc(w.split)} ·')
    html = html.replace('<h2>Thought</h2>', '<details><summary>当前步骤之前的上下文</summary><pre>${esc(w.preceding_context)}</pre></details><h2>Thought</h2>')
    key = hashlib.sha256(data.encode()).hexdigest()[:16]
    if reviews is not None:
        indexed = {r['window_id']: r for r in reviews}
        if len(indexed) != len(reviews) or set(indexed) != {w['window_id'] for w in windows}:
            raise ValueError('model reviews must match all page windows exactly')
        from review_ta_coverage_gqq import validate
        for c in cases:
            review = indexed[c['window']['window_id']]
            if review.get('review_failed') or review.get('verdict') is None:
                raise ValueError('model review must contain a valid verdict')
            validate(c['window'], review['verdict'])
            c['model_review'] = review
        model_data = base64.b64encode(json.dumps(cases, ensure_ascii=False).encode()).decode()
        html = html.replace(data, model_data)
        panel = r'''function modelPanel(c){const w=c.window,r=c.model_review,v=r.verdict;const intentName=id=>'I'+(w.intents.findIndex(i=>i.intent_id===id)+1);const actionName=id=>'A'+(w.actions.findIndex(a=>a.action_id===id)+1);return `<section><details><summary><strong>GQQ 模型建议 · 展开对照</strong></summary><p class="meta">模型建议不是人工 Gold。请先完成独立判断；仅评调用语义，不评执行成功。</p><div class="meta">${esc(r.model)} · ${esc(r.prompt_version)}</div><h3>整步聚合：${esc(v.suggested_ta_label)}</h3><p>${esc(v.reason)}</p><h3>逐 Intent 判断</h3>${v.intents.map(i=>`<div class="pair"><strong>${intentName(i.intent_id)}</strong> · due-now：${i.due_now?'是':'否'} · 可评测：${i.assessable?'是':'否'}<p>${esc(i.reason)}</p></div>`).join('')}<h3>逐 pair 覆盖判断</h3>${v.pairs.map(p=>{const pair=w.pairs.find(x=>x.pair_id===p.pair_id);return `<div class="pair"><strong>${intentName(pair.intent_id)} → ${actionName(pair.action_id)}</strong> · ${p.covered?'完整覆盖':'未完整覆盖'}<p>${esc(p.reason)}</p></div>`}).join('')}<h3>不确定性</h3><pre>${esc((v.uncertainties||[]).join('\n')||'无')}</pre><details><summary>原始模型判断 JSON</summary><pre>${esc(JSON.stringify(v,null,2))}</pre></details></details></section>`}'''
        html = html.replace('function render(){', panel + 'function render(){')
        html = html.replace(";document.getElementById('review').innerHTML", "+modelPanel(c);document.getElementById('review').innerHTML")
    html = html.replace("ta-coverage-review-v1", 'ta-independent-' + key + ('-pair' if detailed else '-overall'))
    html = html.replace('<details><summary>GQQ 建议（完成独立判断后查看）</summary><pre>${esc(JSON.stringify(v,null,2))}</pre></details>', '')
    html = html.replace("gold_label:'',review_reason:'',disputes:''", "gold_label:'',review_reason:'',disputes:'',annotator:'',review_status:'unreviewed',split:c.window.split")
    html = html.replace('<h2>人工标签</h2>', '<h2>人工标签</h2><label>标注者 <input value="${esc(s.annotator)}" onchange="setField(\'annotator\',this.value)"></label>')
    html = html.replace("function exportReviews(){const rows=", "function exportReviews(){const rows=")
    html = html.replace("const blob=new Blob", "rows.forEach(r=>{r.review_status=r.gold_label&&r.review_reason&&r.annotator?'human_reviewed':'unreviewed'});const blob=new Blob")
    if not detailed:
        start = html.index('${w.intents.map((i,n)=>`<div class="pair">')
        end = html.index('<div class="row"><label>整步 Gold', start)
        html = html[:start] + html[end:]
        html = html.replace('先核对 due-now、可评测性和每个 pair，再给整步标签。',
                            '独立核对全部当前目标与实际调用；任一当前目标不可评测或需要联合覆盖裁定时标为不可评测。仅判断调用语义，不以执行失败判负。')
    (output / 'review.html').write_text(html, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT)
    parser.add_argument('--intents', type=Path, default=INTENTS)
    parser.add_argument('--splits', type=Path, default=SPLITS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count-per-split', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261013)
    args = parser.parse_args()
    if args.output.exists() or args.count_per_split < 15:
        raise ValueError('output must not exist; at least 15 windows per split required')
    assignments = split_index(rows(args.splits))
    raw = raw_index(args.raw)
    eligible = {'dev': [], 'test': []}
    for record in rows(args.intents):
        tid = record['trajectory_id']
        split = assignments.get(tid)
        if split not in eligible or tid not in raw:
            continue
        w = window(record, raw[tid], split)
        # Include ALL extracted Intents, even future ones: reviewers may correct due-now.
        w['pairs'] = [{'pair_id': stable_id(i['intent_id'], a['action_id']),
                       'window_id': w['window_id'], 'intent_id': i['intent_id'],
                       'action_id': a['action_id'], 'covered': None,
                       'review_status': 'unreviewed', 'review_reason': None}
                      for i in w['intents'] for a in w['actions']]
        w['preceding_context'] = record.get('preceding_context', '')
        if w['intents'] and w['actions']:
            eligible[split].append(w)
    selected = {s: select_distinct(v, args.count_per_split, args.seed) for s, v in eligible.items()}
    detailed = [w for s, v in selected.items() for w in select_distinct(v, 15, args.seed + 1)]
    train_ids = {t for t, s in assignments.items() if s == 'train'}
    assert not train_ids.intersection(w['trajectory_id'] for v in selected.values() for w in v)
    args.output.mkdir(parents=True)
    write(args.output / 'split_assignments.jsonl', [{'trajectory_id': t, 'split': s} for t, s in sorted(assignments.items())])
    for name, chosen in [*selected.items(), ('pair30', detailed)]:
        out = args.output / name
        out.mkdir()
        write(out / 'windows.jsonl', chosen)
        write(out / 'annotation_template.jsonl', [template(w, name == 'pair30') for w in chosen])
        page(chosen, out, name == 'pair30')
    raw_hashes = {}
    chosen_ids = {w['trajectory_id'] for v in selected.values() for w in v}
    for path in sorted(args.raw.glob('*.json')):
        record = json.loads(path.read_text())
        if record.get('id') in chosen_ids:
            raw_hashes[path.name] = sha256(path)
    manifest = {'schema_version': 'ta_independent_evaluation_v1', 'seed': args.seed,
                'counts': {s: len(v) for s, v in selected.items()}, 'pair_audit_windows': len(detailed),
                'pair_audit_pairs': sum(len(w['pairs']) for w in detailed),
                'pair_audit_splits': dict(Counter(w['split'] for w in detailed)),
                'train_overlap': 0, 'label_status': 'human_annotation_pending',
                'selection': 'hash-random; one window per trajectory per split; no model labels',
                'dev_role': 'threshold calibration and rule development',
                'test_role': 'sealed final evaluation after protocol freeze; never train',
                'input_sha256': {'intents': sha256(args.intents), 'splits': sha256(args.splits)},
                'raw_sha256': raw_hashes,
                'output_sha256': {str(p.relative_to(args.output)): sha256(p) for p in args.output.rglob('*.jsonl')}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith('sha256')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
