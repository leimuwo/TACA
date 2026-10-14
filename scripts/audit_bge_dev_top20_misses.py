#!/usr/bin/env python3
"""Record a fixed, stratified audit of dev queries missing known positives in Top-20."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_bge_clawgym import load_split, rows, sha256


DATASET = Path("data/clawgym/bge_dataset_split_20260930_v2")
RANK20 = Path("outputs/bge_clawgym_silver_v2_dev_top20_20261009/rankings.jsonl")
RANK50 = Path("outputs/bge_clawgym_silver_v2_dev_top50/rankings.jsonl")
QRELS = Path("data/clawgym/bge_dev_qrels_exact_text_20261009/qrels_review_closed.jsonl")

# Decisions use the visible Intent and Action arguments only. None authorizes a qrel edit.
DECISIONS = {
    "5afa42782b47ae2624273e0b": ("needs_context", "Generic input-files intent; cross-task reads cannot be judged without the source task."),
    "db3deb059ea934afa72ae056": ("needs_context", "Required files are unspecified; other input reads may be relevant in their own tasks."),
    "32664bc4e70500c8d9d0732c": ("needs_context", "Official pages are unspecified; same-trajectory pages need the source goal to judge."),
    "a1000d145fc32d21fbb3a084": ("retrieval_miss_likely", "Top candidates fetch or download UNESCO material; known action controls the running process."),
    "1e011009cb40e230b6e23c2b": ("needs_context", "Broad input-files intent omits which files belong to this task."),
    "5881146270b1236829527b52": ("retrieval_miss_likely", "Top candidate runs run_analysis.sh; the intent asks to create that script."),
    "fa46ad4c5d43558ba949b047": ("retrieval_miss_likely", "Intent specifies curl and a different search engine; high candidates use Serper or inspect its script."),
    "74e4c886d1b17c8082455860": ("needs_context", "A same-trajectory search may be broader; query scope requires review."),
    "0ffbc007c59d9740db65a1af": ("possible_missing_qrel", "Other direct Serper script calls may satisfy the broad test intent despite different query strings."),
    "e894a75506471b9e53214675": ("needs_context", "Multiple validation commands capture output; task context identifies which validator is intended."),
    "d8d4d9438617d99e592f6e3e": ("needs_context", "Requirements and input-file reads are ambiguous across tasks."),
    "2f0c401a86e91360528d4613": ("needs_context", "Input-files intent does not identify the task's directory contents."),
    "7ab094ec0ade6995de8aca10": ("retrieval_miss_likely", "Known action creates web/, logs/, reports/; high candidates write files or create different directories."),
    "20293638ae0e7296f18a380e": ("retrieval_miss_likely", "Intent verifies existing files; high candidates mostly create directories or read unrelated outputs."),
    "5664a1c932c9a936bec31891": ("retrieval_miss_likely", "Known action lists output/; high actions operate on outputs/ or perform other work."),
    "0cbd50fd045e9ccb82482669": ("needs_context", "The script is unnamed; a different run script could be relevant in another task."),
    "fe37d7fa04122a692d301587": ("needs_context", "Broad input-files intent does not specify the relevant project."),
    "fe60089f6bcdb5febe523353": ("positive_label_problem", "Top-20 edit sets csv.DictWriter lineterminator to \\n on the same file; compare it with the known edit before treating it as negative."),
    "d64da3665da4acb08f587639": ("possible_missing_qrel", "Top-1 and Top-2 run the Serper script with the exact quoted query; known action instead posts to the Serper API directly."),
    "9cd5da33de94ebeff2ec5687": ("retrieval_miss_likely", "Intent creates an ad-copy JSONL file; high candidates edit or validate other files."),
    "57803ded5b5bf98f13779755": ("retrieval_miss_likely", "Known grep targets essay.txt markdown symbols; high grep commands inspect unrelated HTML or headings."),
    "d03866233e0a3fed44ab9c8c": ("needs_context", "Unspecified input files permit many cross-task reads without project context."),
    "42bbdf6b49ceddd593edc67d": ("retrieval_miss_likely", "Intent asks chmod; high commands run scripts or inspect the environment."),
    "95660376b4bca40d885ffd32": ("retrieval_miss_likely", "High writes target violation and other files, not the named INT specification path."),
    "270421210d03f470930e00e1": ("needs_context", "Generic verify-outputs intent; candidate file checks might count depending on required outputs."),
    "52ee2da88fb3d76d80cc2a6e": ("retrieval_miss_likely", "Known chmod targets two named scripts; high actions mostly inspect, create, or run scripts."),
    "ef010e283d8a024c67b236dc": ("retrieval_miss_likely", "High writes do not create output/alerts.json with the requested sort order."),
    "6332ccc8cecd7a19db76dc5b": ("needs_context", "Intent omits the three document names; same-trajectory URL checks need source context."),
    "2ee72b23c900d95a87b161a6": ("possible_missing_qrel", "A Top-20 ps command prints a more detailed process list; output completeness needs review."),
    "6e7bd1b1e609b2ff671ccb0a": ("needs_context", "Output directory structure is unspecified; writing under output/posts may imply it exists but does not explicitly create it."),
}


def sample_misses(rank20, rank50, qrels, per_band=10, seed=20261009):
    groups = defaultdict(list)
    for row in rank20:
        query_id = row["query_id"]
        if any(item["action_id"] in qrels[query_id] for item in row["top_actions"][:20]):
            continue
        rank = rank50[query_id]["known_positive_rank"]
        band = "21-50" if rank <= 50 else "51-100" if rank <= 100 else "101+"
        groups[band].append(query_id)
    # Match the frozen sample used for the manual review above.
    return {band: random.Random(seed).sample(groups[band], min(per_band, len(groups[band])))
            for band in ("21-50", "51-100", "101+")}


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    queries, actions, qrels = load_split(args.dataset, "dev", qrels_path=args.qrels)
    queries = {row["query_id"]: row for row in queries}
    actions = {row["action_id"]: row for row in actions}
    rank20 = list(rows(args.rank20))
    rank50 = {row["query_id"]: row for row in rows(args.rank50)}
    if len(rank20) != len(queries) or set(rank50) != set(queries):
        raise ValueError("rankings do not cover the same dev queries")
    bands = sample_misses(rank20, rank50, qrels)
    selected = {query_id for group in bands.values() for query_id in group}
    if selected != set(DECISIONS):
        raise ValueError("frozen sample changed; review decisions must be refreshed")
    ranking_by_id = {row["query_id"]: row for row in rank20}
    output_rows = []
    for band, query_ids in bands.items():
        for query_id in query_ids:
            query = queries[query_id]
            label, reason = DECISIONS[query_id]
            candidates = []
            for place, item in enumerate(ranking_by_id[query_id]["top_actions"][:20], 1):
                action = actions[item["action_id"]]
                candidates.append({"rank": place, "action_id": item["action_id"],
                                   "score": item["score"], "action": action["action"],
                                   "same_trajectory": action.get("trajectory_id") == query.get("trajectory_id")})
            output_rows.append({"query_id": query_id, "band": band, "known_positive_rank": rank50[query_id]["known_positive_rank"],
                                "intent": query["query"], "trajectory_id": query["trajectory_id"],
                                "known_positives": [{"action_id": aid, "action": actions[aid]["action"]}
                                                    for aid in sorted(qrels[query_id])],
                                "top20": candidates, "preliminary_label": label, "reason": reason,
                                "qrel_changed": False})
    args.output.mkdir(parents=True)
    with (args.output / "sample_review.jsonl").open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    miss_count = len([row for row in rank20 if not any(item["action_id"] in qrels[row["query_id"]]
                                                          for item in row["top_actions"][:20])])
    summary = {"population_top20_misses": miss_count, "sample_size": len(output_rows),
               "sample_by_rank_band": {band: len(ids) for band, ids in bands.items()},
               "preliminary_labels": dict(Counter(row["preliminary_label"] for row in output_rows)),
               "qrels_changed": False, "sampling_seed": 20261009,
               "input_sha256": {"rank20": sha256(args.rank20), "rank50": sha256(args.rank50),
                                "qrels": sha256(args.qrels)}}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--rank20", type=Path, default=RANK20)
    parser.add_argument("--rank50", type=Path, default=RANK50)
    parser.add_argument("--qrels", type=Path, default=QRELS)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
