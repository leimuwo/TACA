#!/usr/bin/env python3
"""Sample BGE-M3 exec-positive retrieval errors for manual review."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_bge_clawgym import rows, sha256

DATASET = Path("data/clawgym/bge_dataset_split_20260930_v2")
QRELS = Path("data/clawgym/bge_dev_qrels_exact_text_20261009/qrels_review_closed.jsonl")
RANKINGS = Path("outputs/bge_m3_dual_tower_v1/best_dev_rankings.jsonl")


def tool(text):
    return json.loads(text.removeprefix("[ACTION] "))["function"]


def collect(queries, actions, qrels, rankings):
    query_by_id = {row["query_id"]: row for row in queries}
    action_by_id = {row["action_id"]: row for row in actions}
    positive_ids = defaultdict(set)
    for row in qrels:
        positive_ids[row["query_id"]].add(row["action_id"])
    cases = []
    for ranking in rankings:
        query_id = ranking["query_id"]
        positives = positive_ids[query_id]
        if not any(tool(action_by_id[action_id]["action"]) == "exec" for action_id in positives):
            continue
        top = ranking["top_actions"]
        rank = next((index for index, item in enumerate(top, 1)
                     if item["action_id"] in positives), None)
        if rank == 1:
            continue
        first = action_by_id[top[0]["action_id"]]
        positive = [action_by_id[action_id] for action_id in sorted(positives)]
        first_known = next((item for item in top if item["action_id"] in positives), None)
        cases.append({"query_id": query_id, "query": query_by_id[query_id]["query"],
                      "query_trajectory_id": query_by_id[query_id]["trajectory_id"],
                      "group": "top20_miss" if rank is None else "candidate_misrank",
                      "known_positive_rank_at_20": rank,
                      "top1": {"action_id": first["action_id"], "action": first["action"],
                               "score": top[0]["score"],
                               "trajectory_id": first["trajectory_id"]},
                      "known_positives": [{"action_id": row["action_id"], "action": row["action"],
                                            "trajectory_id": row["trajectory_id"]} for row in positive],
                      "first_known_positive_score": first_known["score"] if first_known else None,
                      "top3": [{"action_id": item["action_id"],
                                "action": action_by_id[item["action_id"]]["action"],
                                "rank": index} for index, item in enumerate(top[:3], 1)]})
    return cases


def stratum(case):
    top_tool = tool(case["top1"]["action"])
    if case["group"] == "top20_miss":
        return "exec_top1" if top_tool == "exec" else "other_top1"
    rank = case["known_positive_rank_at_20"]
    return "rank_2_5" if rank <= 5 else "rank_6_10" if rank <= 10 else "rank_11_20"


def sample_cases(cases, count_per_stratum, seed):
    groups = defaultdict(list)
    for case in cases:
        groups[(case["group"], stratum(case))].append(case)
    rng = random.Random(seed)
    selected = []
    for key in sorted(groups):
        selected.extend(rng.sample(sorted(groups[key], key=lambda row: row["query_id"]),
                                   min(count_per_stratum, len(groups[key]))))
    return sorted(selected, key=lambda row: (row["group"], stratum(row), row["query_id"])), groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--qrels", type=Path, default=QRELS)
    parser.add_argument("--rankings", type=Path, default=RANKINGS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count-per-stratum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    if args.count_per_stratum < 1 or args.output.exists():
        raise ValueError("count must be positive and output must not exist")
    folder = args.dataset / "dev"
    cases = collect(rows(folder / "queries.jsonl"), rows(folder / "actions.jsonl"),
                    rows(args.qrels), rows(args.rankings))
    selected, groups = sample_cases(cases, args.count_per_stratum, args.seed)
    args.output.mkdir(parents=True)
    with (args.output / "review_queue.jsonl").open("w", encoding="utf-8") as stream:
        for case in selected:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    summary = {"schema_version": "bge_m3_exec_error_audit_v1",
               "population": dict(Counter(case["group"] for case in cases)),
               "population_strata": {f"{group}/{name}": len(items)
                                     for (group, name), items in sorted(groups.items())},
               "sample": dict(Counter(case["group"] for case in selected)),
               "sample_strata": dict(Counter(f"{case['group']}/{stratum(case)}" for case in selected)),
               "seed": args.seed, "input_sha256": {"rankings": sha256(args.rankings),
                                                     "qrels": sha256(args.qrels)}}
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
