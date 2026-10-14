#!/usr/bin/env python3
"""Append reviewed train-trajectory exec pairs to a clear training split."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from build_bge_m3_exec_supervision import normalized_action
from evaluate_bge_clawgym import rows, sha256
from train_bge_clawgym import load_train

BASE = Path("data/clawgym/bge_dataset_split_20260930_v2_clear_exec_constraints_20261013")
POOL = Path("data/clawgym/bge_negatives_v2_20260929/positive_pool.jsonl")
QUEUE = Path("data/clawgym/bge_dataset_split_20260930_v2_exec_augmented_v2/exec_review_queue.jsonl")
REVIEWS = Path("data/clawgym/bge_m3_exec_train_reviews_20261013.json")


def action(edge):
    item = edge["positive_action"]
    return "[ACTION] " + json.dumps({"function": item["function"], "args": item.get("args", {})},
                                     ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_additions(train, pool_rows, queue_rows, reviews, held_out_trajectories=()):
    pool = {edge["edge_id"]: edge for edge in pool_rows}
    if len(pool) != len(pool_rows):
        raise ValueError("duplicate pool edge ID")
    queue = {(item["source_edge_id"], item["candidate_edge_id"]): item for item in queue_rows}
    if len(queue) != len(queue_rows):
        raise ValueError("duplicate review queue pair")
    known_positive = defaultdict(set)
    for edge in pool_rows:
        if edge["split"] == "train" and edge["eligibility"] == "candidate":
            known_positive["[INTENT] " + edge["intent_text"]].add(normalized_action(action(edge)))
    existing_pairs = {(item["query"], normalized_action(item["pos"][0])) for item in train}
    existing_triples = {(item["query"], normalized_action(item["pos"][0]), normalized_action(neg))
                        for item in train for neg in item["neg"]}
    additions = []
    for review in reviews:
        source_id = review["source_edge_id"]
        source = pool[source_id]
        query = "[INTENT] " + source["intent_text"]
        positive = action(source)
        pair = (query, normalized_action(positive))
        if (source["split"] != "train" or source["eligibility"] != "candidate"
                or source["positive_action"]["function"] != "exec"
                or source["trajectory_id"] in held_out_trajectories
                or pair in existing_pairs or not review.get("intent_review_reason", "").strip()):
            raise ValueError(f"ineligible, duplicate, or unreviewed positive: {source_id}")
        negatives = []
        for item in review["negatives"]:
            candidate_id = item["candidate_edge_id"]
            candidate = pool[candidate_id]
            queued = queue.get((source_id, candidate_id))
            negative = action(candidate)
            triple = (*pair, normalized_action(negative))
            if (queued is None or candidate["split"] != "train"
                    or candidate["trajectory_id"] != source["trajectory_id"]
                    or normalized_action(queued["positive_action"]) != pair[1]
                    or normalized_action(queued["candidate_action"]) != triple[2]
                    or triple[2] in known_positive[query] or triple in existing_triples
                    or not item.get("review_reason", "").strip()):
                raise ValueError(f"invalid or possibly positive negative: {source_id}/{candidate_id}")
            existing_triples.add(triple)
            negatives.append(negative)
        if not 1 <= len(negatives) <= 3 or len(set(negatives)) != len(negatives):
            raise ValueError(f"invalid negative count: {source_id}")
        existing_pairs.add(pair)
        additions.append({"query": query, "pos": [positive], "neg": negatives,
                          "source_edge_id": source_id, "trajectory_id": source["trajectory_id"],
                          "intent_review_reason": review["intent_review_reason"],
                          "negative_reviews": review["negatives"],
                          "review_status": "manual_text_review_not_execution_verified"})
    return additions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=BASE)
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--queue", type=Path, default=QUEUE)
    parser.add_argument("--reviews", type=Path, default=REVIEWS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train_path = args.dataset / "train/train.jsonl"
    train, _ = load_train(train_path)
    held_out = {row["trajectory_id"] for split in ("dev", "test")
                for row in rows(args.dataset / split / "queries.jsonl")}
    pool = list(rows(args.pool))
    queue = list(rows(args.queue))
    reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
    additions = build_additions(train, pool, queue, reviews, held_out)
    folder = args.output / "train"
    folder.mkdir(parents=True)
    with (folder / "train.jsonl").open("w", encoding="utf-8") as stream:
        for item in train + additions:
            stream.write(json.dumps({key: item[key] for key in ("query", "pos", "neg")}, ensure_ascii=False) + "\n")
    load_train(folder / "train.jsonl")
    with (args.output / "reviewed_additions.jsonl").open("w", encoding="utf-8") as stream:
        for item in additions:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    for split in ("dev", "test"):
        shutil.copytree(args.dataset / split, args.output / split)
    report = {"schema_version": "reviewed_exec_train_v1", "source_rows": len(train),
              "added_unique_exec_positive_pairs": len(additions),
              "added_reviewed_negative_pairs": sum(len(item["neg"]) for item in additions),
              "dev_test_unchanged": True, "review_status": "manual text review; execution not verified",
              "input_sha256": {"train": sha256(train_path), "pool": sha256(args.pool),
                               "queue": sha256(args.queue), "reviews": sha256(args.reviews)},
              "output_sha256": sha256(folder / "train.jsonl")}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
