#!/usr/bin/env python3
"""Build a standalone, readable browser view of BGE-M3 dev rankings."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evaluate_bge_clawgym import rows

DATASET = Path("data/clawgym/bge_dataset_split_20260930_v2/dev")
QRELS = Path("data/clawgym/bge_dev_qrels_exact_text_20261009/qrels_review_closed.jsonl")
RANKINGS = Path("outputs/bge_m3_dual_tower_v1/best_dev_rankings.jsonl")


def build_data(queries, actions, qrels, rankings):
    query_by_id = {row["query_id"]: row for row in queries}
    action_by_id = {row["action_id"]: row for row in actions}
    if len(query_by_id) != len(queries) or len(action_by_id) != len(actions):
        raise ValueError("duplicate query or Action ID")
    positives = defaultdict(list)
    for row in qrels:
        qid, aid = row["query_id"], row["action_id"]
        if qid not in query_by_id or aid not in action_by_id or row.get("relevance") != 1:
            raise ValueError("invalid qrel")
        if aid not in positives[qid]:
            positives[qid].append(aid)
    output = []
    seen = set()
    used_actions = set()
    for row in rankings:
        qid = row["query_id"]
        if qid not in query_by_id or qid in seen or not positives[qid]:
            raise ValueError(f"invalid or duplicate ranking query: {qid}")
        seen.add(qid)
        top = row["top_actions"]
        if len(top) != 20 or len({item["action_id"] for item in top}) != 20:
            raise ValueError(f"expected 20 distinct ranked Actions: {qid}")
        if any(item["action_id"] not in action_by_id for item in top):
            raise ValueError(f"ranking references unknown Action: {qid}")
        if any(top[i]["score"] < top[i + 1]["score"] for i in range(len(top) - 1)):
            raise ValueError(f"ranking is not sorted by score: {qid}")
        positive = sorted(positives[qid])
        for item in top:
            if item["is_known_positive"] != (item["action_id"] in positive):
                raise ValueError(f"ranking label disagrees with qrels: {qid}")
        rank = next((i for i, item in enumerate(top, 1) if item["action_id"] in positive), None)
        used_actions.update(positive)
        used_actions.update(item["action_id"] for item in top)
        output.append({"id": qid, "intent": query_by_id[qid]["query"],
                       "trajectory": query_by_id[qid].get("trajectory_id"),
                       "rank": rank, "positive": positive,
                       "top": [[item["action_id"], item["score"]] for item in top],
                       "exec": any(json.loads(action_by_id[aid]["action"][len("[ACTION] "):])["function"] == "exec"
                                   for aid in positive)})
    if seen != set(query_by_id):
        raise ValueError("rankings must include every dev query")
    action_text = {aid: {"text": action_by_id[aid]["action"],
                         "trajectory": action_by_id[aid].get("trajectory_id")}
                   for aid in sorted(used_actions)}
    return {"queries": output, "actions": action_text,
            "source": {"rankings": str(RANKINGS), "qrels": str(QRELS)}}


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BGE-M3 Intent / Action 排名审阅</title>
<style>
:root { color-scheme: light; font-family: system-ui, sans-serif; color: #1c2824; background: #f7f9f8; }
* { box-sizing: border-box; }
body { margin: 0; }
header { padding: 14px 20px; border-bottom: 1px solid #d7e1dc; background: #fff; }
h1 { font-size: 18px; margin: 0 0 4px; font-weight: 680; }
header p { margin: 0; color: #53655e; font-size: 13px; }
.shell { display: grid; grid-template-columns: minmax(250px, 350px) minmax(0, 1fr); height: calc(100vh - 72px); }
.sidebar { border-right: 1px solid #d7e1dc; display: flex; flex-direction: column; min-width: 0; background: #fff; }
.controls { padding: 12px; display: grid; gap: 9px; border-bottom: 1px solid #d7e1dc; }
input, select { width: 100%; padding: 9px 10px; border: 1px solid #b9c9c1; border-radius: 4px; background: #fff; color: #1c2824; font: inherit; font-size: 13px; }
.filters { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
#count { font-size: 12px; color: #53655e; }
#items { overflow: auto; flex: 1; }
.item { width: 100%; text-align: left; display: block; background: #fff; border: 0; border-bottom: 1px solid #e7eeea; padding: 10px 12px; cursor: pointer; color: inherit; }
.item:hover, .item.active { background: #eaf3ed; }
.item strong { display: block; font-size: 13px; line-height: 1.4; font-weight: 600; overflow-wrap: anywhere; }
.item small { display: block; margin-top: 4px; color: #62746b; font-size: 11px; }
main { overflow: auto; padding: 20px 26px 60px; min-width: 0; }
.inner { max-width: 1100px; margin: 0 auto; }
h2 { font-size: 20px; line-height: 1.4; margin: 0 0 12px; overflow-wrap: anywhere; }
h3 { font-size: 15px; margin: 25px 0 10px; }
.meta { color: #53655e; font-size: 12px; line-height: 1.6; overflow-wrap: anywhere; }
.label { display: inline-block; padding: 3px 6px; border-radius: 3px; background: #eaf3ed; color: #146540; font-size: 11px; font-weight: 650; }
.miss { background: #fff0e0; color: #915014; }
.actions { display: grid; gap: 10px; }
.action { border: 1px solid #d7e1dc; border-radius: 4px; background: #fff; padding: 11px 13px; min-width: 0; }
.action.known { border-left: 4px solid #188358; }
.action-head { display: flex; flex-wrap: wrap; gap: 5px 12px; align-items: baseline; font-size: 12px; }
.action-head strong { font-size: 14px; }
.action-head code { overflow-wrap: anywhere; }
.action-head .score { color: #385f4d; font-variant-numeric: tabular-nums; }
.action-head .source { color: #62746b; }
pre { font: 12px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace; white-space: pre-wrap; overflow-wrap: anywhere; margin: 10px 0 0; color: #27332d; }
.note { margin-top: 20px; font-size: 12px; line-height: 1.5; color: #5b6e63; }
@media (max-width: 760px) { .shell { display: block; height: auto; } .sidebar { border-right: 0; border-bottom: 1px solid #d7e1dc; } #items { max-height: 220px; } main { padding: 16px 12px 40px; } }
</style>
</head>
<body>
<header><h1>BGE-M3 Intent / Action 排名审阅</h1><p>Dev 集 · 每条 Intent 的已知正例与 Top-20 Action 全文 · 分数按降序排列</p></header>
<div class="shell">
<aside class="sidebar"><div class="controls">
<input id="search" type="search" placeholder="搜索 Intent 文本或 query_id" aria-label="搜索 Intent">
<div class="filters"><select id="status" aria-label="排名状态"><option value="all">全部排名</option><option value="miss">正例未入 Top-20</option><option value="wrong">正例未排首位</option><option value="hit">正例排首位</option></select><select id="tool" aria-label="正例类型"><option value="all">全部正例</option><option value="exec">exec 正例</option></select></div>
<span id="count"></span></div><div id="items"></div></aside>
<main><div class="inner" id="detail"></div></main>
</div>
<script id="dataset" type="application/json">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById('dataset').textContent);
const byId = new Map(data.queries.map(q => [q.id, q]));
const items = document.getElementById('items');
const detail = document.getElementById('detail');
const search = document.getElementById('search');
const status = document.getElementById('status');
const tool = document.getElementById('tool');
let selected = null;
function el(tag, cls, text) { const node = document.createElement(tag); if (cls) node.className = cls; if (text != null) node.textContent = text; return node; }
function actionNode(id, score, rank, known, queryTrajectory) {
  const entry = data.actions[id];
  const box = el('section', 'action' + (known ? ' known' : ''));
  const head = el('div', 'action-head');
  head.append(el('strong', '', rank ? '#' + rank : '已知正例'));
  if (known) head.append(el('span', 'label', '已标注正例'));
  if (score != null) head.append(el('span', 'score', 'score ' + score.toFixed(6)));
  head.append(el('code', '', id));
  if (entry.trajectory) head.append(el('span', 'source', (entry.trajectory === queryTrajectory ? '同轨迹 ' : '其他轨迹 ') + entry.trajectory));
  box.append(head);
  const text = entry.text.startsWith('[ACTION] ') ? entry.text.slice(9) : entry.text;
  let formatted = text;
  try { formatted = JSON.stringify(JSON.parse(text), null, 2); } catch (_) {}
  box.append(el('pre', '', formatted));
  return box;
}
function show(id) {
  const q = byId.get(id); if (!q) return;
  selected = id; location.hash = encodeURIComponent(id);
  detail.replaceChildren();
  detail.append(el('h2', '', q.intent));
  const rank = q.rank == null ? '已知正例未进入 Top-20' : '首个已知正例排名 #' + q.rank;
  detail.append(el('div', 'meta', 'query_id ' + q.id + ' · ' + rank + ' · ' + (q.exec ? 'exec 正例 · ' : '') + '轨迹 ' + (q.trajectory || '未知')));
  detail.append(el('h3', '', '已知正例 Action（' + q.positive.length + '）'));
  const positives = el('div', 'actions');
  for (const aid of q.positive) {
    const found = q.top.findIndex(x => x[0] === aid);
    positives.append(actionNode(aid, found >= 0 ? q.top[found][1] : null, found >= 0 ? found + 1 : null, true, q.trajectory));
  }
  detail.append(positives);
  detail.append(el('h3', '', '模型 Top-20 Action'));
  const ranked = el('div', 'actions');
  q.top.forEach(([aid, score], index) => ranked.append(actionNode(aid, score, index + 1, q.positive.includes(aid), q.trajectory)));
  detail.append(ranked);
  detail.append(el('p', 'note', '“已标注正例”来自 dev qrels；未标注的 Action 尚未经判错，可能是等价答案。Top-20 之外的正例无分数，因为原排名文件只保存了前 20 名。'));
  for (const node of items.querySelectorAll('.item')) node.classList.toggle('active', node.dataset.id === id);
  document.querySelector('main').scrollTop = 0;
}
function refresh() {
  const needle = search.value.trim().toLowerCase();
  const filtered = data.queries.filter(q => {
    if (needle && !q.intent.toLowerCase().includes(needle) && !q.id.includes(needle)) return false;
    if (tool.value === 'exec' && !q.exec) return false;
    if (status.value === 'miss' && q.rank != null) return false;
    if (status.value === 'wrong' && q.rank === 1) return false;
    if (status.value === 'hit' && q.rank !== 1) return false;
    return true;
  });
  document.getElementById('count').textContent = filtered.length + ' / ' + data.queries.length + ' 条查询';
  items.replaceChildren();
  for (const q of filtered) {
    const button = el('button', 'item' + (q.id === selected ? ' active' : ''));
    button.type = 'button'; button.dataset.id = q.id;
    button.append(el('strong', '', q.intent));
    button.append(el('small', '', (q.rank == null ? 'Top-20 漏召回' : '正例 #' + q.rank) + (q.exec ? ' · exec' : '') + ' · ' + q.id));
    button.addEventListener('click', () => show(q.id));
    items.append(button);
  }
  if (filtered.length && !filtered.some(q => q.id === selected)) show(filtered[0].id);
  if (!filtered.length) detail.replaceChildren(el('p', '', '没有符合条件的查询。'));
}
search.addEventListener('input', refresh);
status.addEventListener('change', refresh);
tool.addEventListener('change', refresh);
const initial = decodeURIComponent(location.hash.slice(1));
if (byId.has(initial)) selected = initial;
refresh();
if (selected && byId.has(selected)) show(selected);
</script>
</body></html>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--qrels", type=Path, default=QRELS)
    parser.add_argument("--rankings", type=Path, default=RANKINGS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = build_data(list(rows(args.dataset / "queries.jsonl")),
                      list(rows(args.dataset / "actions.jsonl")),
                      list(rows(args.qrels)), list(rows(args.rankings)))
    data["source"] = {"rankings": str(args.rankings), "qrels": str(args.qrels)}
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML.replace("__DATA__", encoded), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "queries": len(data["queries"]),
                      "actions": len(data["actions"]), "bytes": args.output.stat().st_size},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
