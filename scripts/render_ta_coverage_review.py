#!/usr/bin/env python3
"""Select 30 distinct trajectories and render an offline human review interface."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
import base64
from pathlib import Path

from evaluate_bge_clawgym import rows


def choose(windows, reviews, count):
    indexed = {r["window_id"]: r for r in reviews}
    candidates = []
    for window in windows:
        review = indexed.get(window["window_id"])
        if review is None:
            continue
        verdict = review.get("verdict") or {}
        label = verdict.get("suggested_ta_label", "failed")
        multi = len(window["intents"]) > 1 or len(window["actions"]) > 1
        uncertain = (bool(verdict.get("uncertainties")) or review.get("review_failed")
                     or verdict.get("aggregation_disagrees", False))
        category = ("multi" if multi else "single") + "/" + label + ("/uncertain" if uncertain else "")
        candidates.append((category, window, review))
    groups = defaultdict(list)
    for category, window, review in candidates:
        groups[category].append((window, review))
    chosen, trajectories = [], set()
    for category in sorted(groups, key=lambda c: ("partial" not in c, "unassessable" not in c,
                                                    "uncertain" not in c, "multi" not in c, c)):
        for window, review in sorted(groups[category], key=lambda x: x[0]["window_id"]):
            if window["trajectory_id"] in trajectories:
                continue
            chosen.append({"category": category, "window": window, "model_review": review})
            trajectories.add(window["trajectory_id"])
            break
    for category, window, review in sorted(candidates, key=lambda x: (
            "uncertain" not in x[0], "multi" not in x[0], x[0], x[1]["window_id"])):
        if len(chosen) >= count:
            break
        if window["trajectory_id"] not in trajectories:
            chosen.append({"category": category, "window": window, "model_review": review})
            trajectories.add(window["trajectory_id"])
    return chosen[:count]


HTML = r'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TA 人工复核</title>
<style>
:root{font-family:system-ui,sans-serif;color:#18242b;background:#f5f7f7}*{box-sizing:border-box}body{margin:0}header{display:flex;align-items:center;gap:16px;flex-wrap:wrap;padding:12px 20px;border-bottom:1px solid #ccd6d8;background:white;position:sticky;top:0;z-index:2}h1{font-size:18px;margin:0}button,select,textarea{font:inherit}button{padding:7px 11px;border:1px solid #9daeb0;border-radius:4px;background:white;cursor:pointer}button:hover{background:#e8f2ef}button.primary{background:#126b5a;color:white;border-color:#126b5a}main{max-width:1300px;margin:auto;padding:16px 20px;display:grid;grid-template-columns:minmax(0,1fr) minmax(300px,390px);gap:24px}section{min-width:0;border-bottom:1px solid #d4dddd;padding-bottom:14px;margin-bottom:14px}h2{font-size:17px;margin:0 0 9px}h3{font-size:14px;margin:12px 0 5px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;border:1px solid #d5dfe0;padding:10px;border-radius:4px;max-height:240px;overflow:auto;font-size:12px;line-height:1.5}.meta{font-size:12px;color:#53656b}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0}.review{background:#fff;border:1px solid #d5dfe0;padding:14px;border-radius:4px;align-self:start;position:sticky;top:75px;max-height:calc(100vh - 95px);overflow:auto}.review label{font-size:13px}.review select{max-width:100%;padding:4px}.review textarea{width:100%;min-height:78px;border:1px solid #9daeb0;padding:6px}.pair{border-top:1px solid #e1e6e6;padding:8px 0}.muted{color:#65757a}details{margin:8px 0}summary{cursor:pointer}@media(max-width:850px){main{display:block}.review{position:static;max-height:none}}
</style></head><body><header><h1>Thought–Action 复核</h1><span id="progress" class="meta"></span><button title="上一条" onclick="go(-1)">←</button><button title="下一条" onclick="go(1)">→</button><button class="primary" onclick="exportReviews()">导出标注 JSON</button></header><main><div id="source"></div><aside class="review" id="review"></aside></main>
<script>const cases=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob('__DATA__'),c=>c.charCodeAt(0))));let index=0;const key='ta-coverage-review-v1';let saved=JSON.parse(localStorage.getItem(key)||'{}');const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));function persist(){localStorage.setItem(key,JSON.stringify(saved));render()}function current(){return cases[index]}function initial(c){return {window_id:c.window.window_id,trajectory_id:c.window.trajectory_id,gold_label:'',review_reason:'',disputes:'',intents:Object.fromEntries(c.window.intents.map(i=>[i.intent_id,{due_now:'',assessable:''}])),pairs:Object.fromEntries(c.window.pairs.map(p=>[p.pair_id,''])}}function state(){const id=current().window.window_id;return saved[id]||(saved[id]=initial(current()))}function setField(field,value){state()[field]=value;persist()}function setIntent(id,field,value){state().intents[id][field]=value;persist()}function setPair(id,value){state().pairs[id]=value;persist()}function go(delta){index=Math.max(0,Math.min(cases.length-1,index+delta));render();scrollTo(0,0)}function options(values,chosen){return values.map(([v,t])=>`<option value="${esc(v)}" ${v===chosen?'selected':''}>${esc(t)}</option>`).join('')}function render(){const c=current(),w=c.window,s=state(),v=c.model_review.verdict;document.getElementById('progress').textContent=`${index+1}/${cases.length} · 已标 ${Object.values(saved).filter(x=>x.gold_label&&x.review_reason).length} · ${c.category}`;document.getElementById('source').innerHTML=`<section><div class="meta">${esc(w.trajectory_id)} · step ${w.thought_step} · train · ${esc(w.window_id)}</div><h2>Thought</h2><pre>${esc(w.thought)}</pre><details><summary>用户任务</summary><pre>${esc(w.user_request)}</pre></details></section><section><h2>当前 Intent</h2>${w.intents.map((i,n)=>`<h3>I${n+1} · ${esc(i.intent_id)}</h3><pre>${esc(i.text)}</pre><div class="meta">时间属性 ${esc(i.temporal_scope)} · Thought 证据：${esc(i.source_quote)}</div>`).join('')}</section><section><h2>实际 Action 与工具返回</h2>${w.actions.map((a,n)=>`<h3>A${n+1} · ${esc(a.action_id)}</h3><pre>${esc(a.action_text)}</pre><details><summary>执行结果 ${esc(a.result_status)}</summary><pre>${esc((a.execution_result||[]).join('\n\n'))}</pre></details>`).join('')}</section>`;document.getElementById('review').innerHTML=`<h2>人工标签</h2><div class="meta">先核对 due-now、可评测性和每个 pair，再给整步标签。</div>${w.intents.map((i,n)=>`<div class="pair"><strong>I${n+1}</strong><div class="row"><label>当前应做 <select onchange="setIntent('${i.intent_id}','due_now',this.value)">${options([['','待定'],['yes','是'],['no','未来计划/非当前']],s.intents[i.intent_id].due_now)}</select></label><label>可评测 <select onchange="setIntent('${i.intent_id}','assessable',this.value)">${options([['','待定'],['yes','是'],['no','否']],s.intents[i.intent_id].assessable)}</select></label></div></div>`).join('')}<h3>Intent–Action 覆盖</h3>${w.pairs.map(p=>`<div class="pair"><label>${esc(p.intent_id.slice(0,8))} → ${esc(p.action_id.slice(0,12))} <select onchange="setPair('${p.pair_id}',this.value)">${options([['','待定'],['yes','完整覆盖'],['no','未完整覆盖'],['uncertain','存疑']],s.pairs[p.pair_id])}</select></label></div>`).join('')}<div class="row"><label>整步 Gold <select onchange="setField('gold_label',this.value)">${options([['','待定'],['faithful','faithful'],['partial','partial'],['not_executed','not_executed'],['unassessable','不可评测']],s.gold_label)}</select></label></div><h3>判断依据</h3><textarea oninput="saved[current().window.window_id].review_reason=this.value;localStorage.setItem(key,JSON.stringify(saved))">${esc(s.review_reason)}</textarea><h3>争议点</h3><textarea oninput="saved[current().window.window_id].disputes=this.value;localStorage.setItem(key,JSON.stringify(saved))">${esc(s.disputes)}</textarea><details><summary>GQQ 建议（完成独立判断后查看）</summary><pre>${esc(JSON.stringify(v,null,2))}</pre></details>`}function exportReviews(){const rows=cases.map(c=>saved[c.window.window_id]||initial(c));const blob=new Blob([rows.map(x=>JSON.stringify(x)).join('\n')+'\n'],{type:'application/jsonl'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='ta_gold_human_review.jsonl';a.click();URL.revokeObjectURL(a.href)}render();</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--model-reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    if args.output.exists() or args.count < 1:
        raise ValueError("output must not exist and count must be positive")
    selected = choose(list(rows(args.windows)), list(rows(args.model_reviews)), args.count)
    if len(selected) != args.count:
        raise ValueError("not enough distinct trajectories")
    args.output.mkdir(parents=True)
    with (args.output / "selected_windows.jsonl").open("w", encoding="utf-8") as stream:
        for item in selected:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    data = base64.b64encode(json.dumps(selected, ensure_ascii=False).encode("utf-8")).decode("ascii")
    html = HTML.replace("__DATA__", data).replace(
        "pairs:Object.fromEntries(c.window.pairs.map(p=>[p.pair_id,''])}",
        "pairs:Object.fromEntries(c.window.pairs.map(p=>[p.pair_id,'']))}")
    (args.output / "review.html").write_text(html, encoding="utf-8")
    summary = {"selected": len(selected), "distinct_trajectories": len({x["window"]["trajectory_id"] for x in selected}),
               "categories": dict(Counter(x["category"] for x in selected)),
               "review_status": "human_labels_pending"}
    (args.output / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
