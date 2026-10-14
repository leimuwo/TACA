#!/usr/bin/env python3
"""Apply text-reviewed train near-neighbor negatives with source provenance."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from build_bge_m3_exec_supervision import normalized_action
from evaluate_bge_clawgym import rows, sha256
from train_bge_clawgym import load_train

DATASET = Path("data/clawgym/bge_dataset_split_20260930_v2_exec_operation_20261012")
QUEUE = Path("data/clawgym/bge_dataset_split_20260930_v2_exec_augmented_v2/exec_review_queue.jsonl")
REVIEWS = Path("data/clawgym/bge_m3_train_constraint_reviews_20261012.json")
POOL = Path("data/clawgym/bge_negatives_v2_20260929/positive_pool.jsonl")


def approved_rows(train, queue, decisions, pool):
    positive = defaultdict(set)
    existing = set()
    source_pairs = set()
    for row in train:
        pair = (row["query"], normalized_action(row["pos"][0]))
        source_pairs.add(pair)
        positive[row["query"]].add(pair[1])
        existing.update((*pair, normalized_action(negative)) for negative in row["neg"])
    indexed = {item["source_edge_id"] + "/" + item["candidate_edge_id"]: item for item in queue}
    if len(indexed) != len(queue):
        raise ValueError("duplicate queue source/candidate IDs")
    additions = []
    for key, reason in decisions.items():
        if not isinstance(reason, str) or not reason.strip() or key not in indexed:
            raise ValueError(f"invalid review decision: {key}")
        item = indexed[key]
        source = pool.get(item["source_edge_id"])
        candidate_edge = pool.get(item["candidate_edge_id"])
        if (source is None or source.get("split") != "train"
                or source.get("eligibility") != "candidate"
                or source.get("trajectory_id") != item.get("trajectory_id")
                or item["query"] != "[INTENT] " + source.get("intent_text", "")
                or candidate_edge is None or candidate_edge.get("split") != "train"
                or candidate_edge.get("trajectory_id") != item.get("trajectory_id")):
            raise ValueError(f"review source is not an eligible train edge: {key}")
        source_action = "[ACTION] " + json.dumps(
            {"function": source["positive_action"]["function"],
             "args": source["positive_action"].get("args", {})},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if normalized_action(source_action) != normalized_action(item["positive_action"]):
            raise ValueError(f"review source Action changed: {key}")
        pair = item["query"], normalized_action(item["positive_action"])
        negative = normalized_action(item["candidate_action"])
        if negative in positive[item["query"]] or (*pair, negative) in existing:
            raise ValueError(f"review collides with existing training label: {key}")
        existing.add((*pair, negative))
        additions.append({"query": item["query"], "pos": [item["positive_action"]],
                          "neg": [item["candidate_action"]], "source_edge_id": item["source_edge_id"],
                          "candidate_edge_id": item["candidate_edge_id"], "review_reason": reason,
                          "new_positive_pair": pair not in source_pairs,
                          "review_status": "manual_text_review_not_execution_verified"})
        source_pairs.add(pair)
        positive[item["query"]].add(pair[1])
    return additions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--queue", type=Path, default=QUEUE)
    parser.add_argument("--reviews", type=Path, default=REVIEWS)
    parser.add_argument("--positive-pool", type=Path, default=POOL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train, _ = load_train(args.dataset / "train" / "train.jsonl")
    decisions = json.loads(args.reviews.read_text(encoding="utf-8"))
    pool = {row["edge_id"]: row for row in rows(args.positive_pool)}
    additions = approved_rows(train, list(rows(args.queue)), decisions, pool)
    folder = args.output / "train"
    folder.mkdir(parents=True)
    with (folder / "train.jsonl").open("w", encoding="utf-8") as stream:
        for row in train:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in additions:
            stream.write(json.dumps({key: row[key] for key in ("query", "pos", "neg")},
                                    ensure_ascii=False) + "\n")
    load_train(folder / "train.jsonl")
    with (args.output / "reviewed_constraint_negatives.jsonl").open("w", encoding="utf-8") as stream:
        for row in additions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    for split in ("dev", "test"):
        shutil.copytree(args.dataset / split, args.output / split)
    manifest = {"schema_version": "bge_m3_train_reviews_v1", "source_train_rows": len(train),
                "added_reviewed_rows": len(additions), "dev_test_unchanged": True,
                "new_positive_pairs": sum(row["new_positive_pair"] for row in additions),
                "review_status": "manual text and command review; execution not verified",
                "input_sha256": {"train": sha256(args.dataset / "train" / "train.jsonl"),
                                 "queue": sha256(args.queue), "reviews": sha256(args.reviews),
                                 "positive_pool": sha256(args.positive_pool)},
                "output_sha256": sha256(folder / "train.jsonl")}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
