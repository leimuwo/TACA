#!/usr/bin/env python3
"""Package omitted validated exec negatives and unjudged nearby exec candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import posixpath
import shlex
from collections import Counter, defaultdict
from pathlib import Path

from train_bge_clawgym import load_train

DATASET = Path("data/clawgym/bge_dataset_split_20260930_v2")
POOL = Path("data/clawgym/bge_negatives_v2_20260929/positive_pool.jsonl")
VALIDATED = Path("data/clawgym/bge_negatives_v2_20260929/validated_candidates.jsonl")


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_rows(path, records):
    with path.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def action_text(action):
    return "[ACTION] " + json.dumps({"function": action["function"], "args": action.get("args", {})},
                                     ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def augment(grouped, pool, validated):
    positives = defaultdict(set)
    existing = set()
    sources = {}
    used_ids = set()
    for row in grouped:
        key = (row["query"], row["pos"][0])
        sources.setdefault(key, row)
        positives[row["query"]].add(row["pos"][0])
        existing.update((row["query"], row["pos"][0], neg) for neg in row["neg"])
        used_ids.update(row["candidate_ids"])
    additions = defaultdict(list)
    audit = Counter()
    for item in validated:
        edge = pool.get(item["edge_id"])
        if not edge or edge["positive_action"].get("function") != "exec":
            continue
        if item.get("validation", {}).get("accepted") is not True:
            continue
        audit["validated_exec"] += 1
        if item["candidate_id"] in used_ids:
            audit["already_used"] += 1
            continue
        query = "[INTENT] " + edge["intent_text"]
        positive = action_text(edge["positive_action"])
        negative = action_text(item["candidate_action"])
        key = (query, positive)
        triple = (query, positive, negative)
        if key not in sources:
            audit["source_not_in_train"] += 1
        elif negative in positives[query] or triple in existing:
            audit["collision_or_duplicate"] += 1
        else:
            additions[key].append({"neg": negative, "candidate_id": item["candidate_id"],
                                   "category": item["category"], "source_edge_id": item["edge_id"]})
            existing.add(triple)
            audit["added"] += 1
    train = [{key: row[key] for key in ("query", "pos", "neg")} for row in grouped]
    provenance = list(grouped)
    for (query, positive), items in additions.items():
        source = sources[(query, positive)]
        for start in range(0, len(items), 3):
            batch = items[start:start + 3]
            row = {"query": query, "pos": [positive], "neg": [item["neg"] for item in batch]}
            train.append(row)
            provenance.append({**row, "candidate_ids": [item["candidate_id"] for item in batch],
                               "negative_categories": [item["category"] for item in batch],
                               "source_edge_ids": [item["source_edge_id"] for item in batch],
                               "trajectory_id": source["trajectory_id"], "split": "train",
                               "supervision_tier": "silver_model_validated"})
    return train, provenance, additions, audit, positives, existing


def review_queue(pool, positives, existing, max_per_edge):
    by_trajectory = defaultdict(list)
    eligible = [edge for edge in pool.values() if edge.get("eligibility") == "candidate"
                and edge["positive_action"].get("function") == "exec"]
    for edge in eligible:
        by_trajectory[edge["trajectory_id"]].append(edge)
    review = []
    seen = set()
    for edge in eligible:
        query = "[INTENT] " + edge["intent_text"]
        positive = action_text(edge["positive_action"])
        nearby = sorted(by_trajectory[edge["trajectory_id"]],
                        key=lambda other: (abs(other["thought_step"] - edge["thought_step"]), other["edge_id"]))
        count = 0
        for other in nearby:
            distance = abs(other["thought_step"] - edge["thought_step"])
            if other["edge_id"] == edge["edge_id"] or distance > 3:
                continue
            candidate = action_text(other["positive_action"])
            triple = (query, positive, candidate)
            if candidate == positive or candidate in positives[query] or triple in existing or triple in seen:
                continue
            seen.add(triple)
            review.append({"query": query, "positive_action": positive, "candidate_action": candidate,
                           "source_edge_id": edge["edge_id"], "candidate_edge_id": other["edge_id"],
                           "trajectory_id": edge["trajectory_id"], "step_distance": distance,
                           "label": "unjudged", "review_decision": None})
            count += 1
            if count >= max_per_edge:
                break
    return review


def promote_clear_script_targets(queue, train, provenance):
    """Promote only explicit script-target mismatches with the same interpreter."""
    promoted = []
    existing = {(row["query"], row["pos"][0], neg) for row in train for neg in row["neg"]}
    for item in queue:
        try:
            positive = json.loads(item["positive_action"][len("[ACTION] "):])["args"]["command"]
            negative = json.loads(item["candidate_action"][len("[ACTION] "):])["args"]["command"]
            left, right = shlex.split(positive), shlex.split(negative)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if len(left) != len(right) or len(left) < 2 or left[0] != right[0] or left[0] not in {"python", "python3"}:
            continue
        changed = [index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]]
        target = posixpath.basename(left[1]).lower()
        if changed != [1] or not target.endswith(".py") or len(target) < 8 or target not in item["query"].lower():
            continue
        triple = (item["query"], item["positive_action"], item["candidate_action"])
        if triple in existing:
            continue
        row = {"query": item["query"], "pos": [item["positive_action"]], "neg": [item["candidate_action"]]}
        train.append(row)
        provenance.append({**row, "candidate_ids": [item["candidate_edge_id"]],
                           "negative_categories": ["P1_exec_target_reviewed"],
                           "source_edge_ids": [item["source_edge_id"]], "split": "train",
                           "supervision_tier": "silver_rule_reviewed"})
        existing.add(triple)
        promoted.append(item)
    return promoted


def run(args):
    if args.max_review_per_edge < 1:
        raise ValueError("max review candidates per edge must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    pool = {row["edge_id"]: row for row in rows(args.positive_pool)}
    grouped = list(rows(args.dataset / "train" / "train_grouped.jsonl"))
    train, provenance, additions, audit, positives, existing = augment(grouped, pool, rows(args.validated))
    review = review_queue(pool, positives, existing, args.max_review_per_edge)
    promoted = promote_clear_script_targets(review, train, provenance)
    review = [item for item in review if item not in promoted]
    (args.output / "train").mkdir(parents=True)
    write_rows(args.output / "train" / "train.jsonl", train)
    write_rows(args.output / "train" / "train_grouped.jsonl", provenance)
    load_train(args.output / "train" / "train.jsonl")
    for split in ("dev", "test"):
        shutil.copytree(args.dataset / split, args.output / split)
    write_rows(args.output / "new_validated_exec_negatives.jsonl", [
        {"query": query, "positive_action": positive, **item}
        for (query, positive), items in additions.items() for item in items])
    write_rows(args.output / "exec_review_queue.jsonl", review)
    manifest = {"schema_version": "exec_hard_negative_augmentation_v1",
                "train_rows_before": len(grouped), "train_rows_after": len(train),
                "new_validated_exec_negatives": audit["added"], "promoted_rule_reviewed_exec_negatives": len(promoted),
                "review_candidates_unjudged": len(review),
                "validation_audit": dict(audit), "dev_test_unchanged": True,
                "input_sha256": {"train": hashlib.sha256((args.dataset / "train" / "train.jsonl").read_bytes()).hexdigest(),
                                 "positive_pool": hashlib.sha256(args.positive_pool.read_bytes()).hexdigest(),
                                 "validated": hashlib.sha256(args.validated.read_bytes()).hexdigest()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--positive-pool", type=Path, default=POOL)
    parser.add_argument("--validated", type=Path, default=VALIDATED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-review-per-edge", type=int, default=3)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
