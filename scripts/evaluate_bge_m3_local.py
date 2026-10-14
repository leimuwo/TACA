#!/usr/bin/env python3
"""Build and optionally score clear-dev, trajectory-local BGE-M3 candidates."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_bge_clawgym import embed, load_split, rows, sha256

DATASET = Path("data/clawgym/bge_dataset_split_20260930_v2")
QRELS = Path("data/clawgym/bge_dev_qrels_exact_text_20261009/qrels_review_closed.jsonl")
SCREEN_INPUT = Path("../data/clawgym/ambiguous/screen_input.jsonl")
DECISIONS = Path("../data/clawgym/ambiguous/decisions.jsonl")
RANKINGS = Path("outputs/bge_m3_dual_tower_v1/best_dev_rankings.jsonl")
CHECKPOINT = Path("outputs/bge_m3_dual_tower_v1/best")


def clear_ids(queries, screen_input, decisions):
    source = {row["query_id"]: row for row in screen_input if row["split"] == "dev"}
    target = {row["query_id"] for row in queries}
    mapped = {}
    for screen_id, row in source.items():
        qid = row.get("source_query_id") or screen_id
        if qid in mapped:
            raise ValueError(f"duplicate screen mapping: {qid}")
        mapped[qid] = screen_id
    if set(mapped) != target:
        raise ValueError("ambiguity screen must cover exactly the evaluated dev queries")
    by_id = {}
    for row in decisions:
        sid = row["query_id"]
        if sid in by_id or not isinstance(row.get("ambiguous"), bool):
            raise ValueError(f"duplicate or invalid ambiguity decision: {sid}")
        by_id[sid] = row["ambiguous"]
    if not set(mapped.values()) <= set(by_id):
        raise ValueError("missing ambiguity decisions")
    return {qid for qid, sid in mapped.items() if not by_id[sid]}


def action_tool(row):
    return json.loads(row["action"].removeprefix("[ACTION] "))["function"]


def build_pools(queries, actions, qrels, rankings, clear, method, k, step_window):
    if k not in (5, 10, 20) or step_window < 0:
        raise ValueError("K must be 5, 10 or 20 and step window nonnegative")
    action_by_id = {row["action_id"]: row for row in actions}
    by_trajectory = defaultdict(list)
    for row in actions:
        by_trajectory[row["trajectory_id"]].append(row)
    ranked = {row["query_id"]: row["top_actions"] for row in rankings}
    if set(ranked) != {row["query_id"] for row in queries}:
        raise ValueError("saved retrieval rankings do not cover the dev queries")
    pools = []
    for query in queries:
        qid = query["query_id"]
        if qid not in clear:
            continue
        positives = qrels[qid]
        tools = {action_tool(action_by_id[aid]) for aid in positives}
        trajectory = by_trajectory[query["trajectory_id"]]
        step = query.get("thought_step")
        nearest = sorted(trajectory, key=lambda row: (
            abs(row["thought_step"] - step) if isinstance(row.get("thought_step"), int) and isinstance(step, int) else 10**9,
            row["action_id"]))
        same_tool = [row for row in nearest if action_tool(row) in tools]
        nearby = [row for row in nearest if isinstance(row.get("thought_step"), int)
                  and isinstance(step, int) and 0 < abs(row["thought_step"] - step) <= step_window]
        semantic = [action_by_id[item["action_id"]] for item in ranked[qid][:k]]
        if method == "same_tool":
            sources = (same_tool,)
        elif method == "nearby":
            sources = (nearby,)
        elif method == "semantic":
            sources = (semantic,)
        elif method == "hybrid":
            sources = (nearby, semantic)
        else:
            raise ValueError(f"unknown candidate method: {method}")
        selected = []
        seen = set()
        for group in sources:
            for row in group:
                aid = row["action_id"]
                if aid not in seen and len(selected) < k:
                    selected.append(aid)
                    seen.add(aid)
        pools.append({"query_id": qid, "query": query["query"], "trajectory_id": query["trajectory_id"],
                      "known_positive_action_ids": sorted(positives), "candidate_action_ids": selected,
                      "known_positive_in_candidates": bool(seen & positives),
                      "positive_tools": sorted(tools)})
    return pools


def metrics(pools, rankings=None):
    report = {"queries": len(pools), "known_positive_in_candidates": sum(
        row["known_positive_in_candidates"] for row in pools),
        "mean_candidate_count": sum(len(row["candidate_action_ids"]) for row in pools) / len(pools)}
    report["candidate_coverage"] = report["known_positive_in_candidates"] / len(pools)
    if rankings is not None:
        by_id = {row["query_id"]: row for row in rankings}
        if len(by_id) != len(rankings) or set(by_id) != {row["query_id"] for row in pools}:
            raise ValueError("scored rankings must cover each local query once")
        ranks = []
        for pool in pools:
            order = [item["action_id"] for item in by_id[pool["query_id"]]["top_actions"]]
            rank = next((i for i, aid in enumerate(order, 1)
                         if aid in pool["known_positive_action_ids"]), None)
            ranks.append(rank)
        report.update({"known_positive_at_1": sum(rank == 1 for rank in ranks) / len(ranks),
                       "known_positive_mrr": sum(1 / rank for rank in ranks if rank) / len(ranks)})
    return report


def rank_from_saved_scores(pools, full_rankings):
    by_id = {row["query_id"]: row for row in full_rankings}
    output = []
    for pool in pools:
        score_by_id = {item["action_id"]: item["score"] for item in by_id[pool["query_id"]]["top_actions"]}
        candidates = pool["candidate_action_ids"]
        if not set(candidates) <= set(score_by_id):
            raise ValueError("saved rankings lack local candidates; use GPU scoring")
        ordered = sorted(candidates, key=lambda aid: (-score_by_id[aid], aid))
        output.append({"query_id": pool["query_id"], "top_actions": [
            {"action_id": aid, "score": score_by_id[aid],
             "is_known_positive": aid in pool["known_positive_action_ids"]} for aid in ordered]})
    return output


def score_pools(pools, actions, checkpoint, device, batch_size):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install torch and transformers on the GPU machine") from error
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "query_model", local_files_only=True)
    query_model = AutoModel.from_pretrained(checkpoint / "query_model", local_files_only=True).to(device).eval()
    action_model = AutoModel.from_pretrained(checkpoint / "action_model", local_files_only=True).to(device).eval()
    projection = torch.nn.Linear(action_model.config.hidden_size, query_model.config.hidden_size,
                                 bias=False).to(device)
    projection.load_state_dict(torch.load(checkpoint / "projection.pt", map_location=device,
                                          weights_only=True))
    projection.eval()
    action_by_id = {row["action_id"]: row for row in actions}
    needed = sorted({aid for pool in pools for aid in pool["candidate_action_ids"]})
    av = embed([action_by_id[aid]["action"] for aid in needed], torch=torch, tokenizer=tokenizer,
               model=action_model, device=device, batch_size=batch_size, max_length=512)
    qv = embed([row["query"] for row in pools], torch=torch, tokenizer=tokenizer,
               model=query_model, device=device, batch_size=batch_size, max_length=256)
    with torch.inference_mode():
        av = torch.nn.functional.normalize(projection(av.to(device)), dim=-1)
        scores = (qv.to(device) @ av.T).cpu()
    position = {aid: i for i, aid in enumerate(needed)}
    output = []
    for pool, values in zip(pools, scores):
        candidates = pool["candidate_action_ids"]
        order = sorted(candidates, key=lambda aid: (-values[position[aid]].item(), aid))
        output.append({"query_id": pool["query_id"], "top_actions": [
            {"action_id": aid, "score": values[position[aid]].item(),
             "is_known_positive": aid in pool["known_positive_action_ids"]} for aid in order]})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--qrels", type=Path, default=QRELS)
    parser.add_argument("--screen-input", type=Path, default=SCREEN_INPUT)
    parser.add_argument("--decisions", type=Path, default=DECISIONS)
    parser.add_argument("--rankings", type=Path, default=RANKINGS)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--method", choices=("same_tool", "nearby", "semantic", "hybrid"), default="hybrid")
    parser.add_argument("--k", type=int, choices=(5, 10, 20), default=20)
    parser.add_argument("--step-window", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--score-from-rankings", action="store_true",
                        help="Reuse saved scores when every local candidate is in the rankings")
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1:
        raise ValueError("output must not exist and batch size must be positive")
    queries, actions, qrels = load_split(args.dataset, "dev", qrels_path=args.qrels)
    clear = clear_ids(queries, rows(args.screen_input), rows(args.decisions))
    saved_rankings = list(rows(args.rankings))
    pools = build_pools(queries, actions, qrels, saved_rankings, clear,
                        args.method, args.k, args.step_window)
    if not pools or any(not row["candidate_action_ids"] for row in pools):
        raise ValueError("one or more clear queries have an empty candidate pool")
    if args.coverage_only and args.score_from_rankings:
        raise ValueError("choose coverage-only or score-from-rankings")
    ranked = (None if args.coverage_only else
              rank_from_saved_scores(pools, saved_rankings) if args.score_from_rankings else
              score_pools(pools, actions, args.checkpoint, args.device, args.batch_size))
    report = {"schema_version": "bge_m3_local_dev_v1", "method": args.method,
              "k": args.k, "step_window": args.step_window, "clear_queries": len(pools),
              "excluded_ambiguous_queries": len(queries) - len(pools),
              "overall": metrics(pools, ranked),
              "by_positive_tool": {tool: metrics([p for p in pools if tool in p["positive_tools"]],
                                               [r for r in ranked if r["query_id"] in
                                                {p["query_id"] for p in pools if tool in p["positive_tools"]}]
                                               if ranked else None)
                                   for tool in sorted({t for p in pools for t in p["positive_tools"]})},
              "ranked": ranked is not None, "label_limit": "Silver known positives only",
              "candidate_note": "same_tool uses qrels positive tool for diagnosis; nearby/hybrid exclude the query's own step",
              "input_sha256": {"qrels": sha256(args.qrels), "rankings": sha256(args.rankings),
                               "screen_input": sha256(args.screen_input), "decisions": sha256(args.decisions)}}
    args.output.mkdir(parents=True)
    with (args.output / "candidates.jsonl").open("w", encoding="utf-8") as stream:
        for row in pools:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if ranked:
        with (args.output / "rankings.jsonl").open("w", encoding="utf-8") as stream:
            for row in ranked:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output / "local_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
