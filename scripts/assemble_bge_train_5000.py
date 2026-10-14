#!/usr/bin/env python3
"""Assemble GQQ-checked silver negatives into BGE train and evaluation files."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action_text(action):
    return "[ACTION] " + canonical({"function": action["function"], "args": action.get("args", {})})


def intent_for(row, intents):
    intent_id = row["selected_intent_ids"][0]
    revised = {item["intent_id"]: item for item in row.get("revised_intents", [])}
    original = {item["intent_id"]: item for item in intents[(row["trajectory_id"], row["thought_step"])].get("execution_intents", [])}
    return revised.get(intent_id) or original[intent_id]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def run(args):
    negatives = list(read_jsonl(args.negatives))
    labels = Counter(row["target_label"] for row in negatives)
    if len(negatives) != 5000 or labels != {"object_mismatch": 3000, "type_mismatch": 2000}:
        raise ValueError(f"negative count or mix is incomplete: {len(negatives)}, {labels}")
    if len({row["candidate_id"] for row in negatives}) != len(negatives):
        raise ValueError("duplicate candidate_id")
    actions = {row["action_id"]: row for row in read_jsonl(args.actions)}
    intents = {(row["trajectory_id"], row["thought_step"]): row for row in read_jsonl(args.intents)}
    matches = list(read_jsonl(args.matches))
    matches_by_id = {row["action_id"]: row for row in matches}
    split_trajectories = defaultdict(set)
    for row in matches:
        split_trajectories[row["split"]].add(row["trajectory_id"])
    if any(split_trajectories[a] & split_trajectories[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("trajectory crosses splits")
    grouped = defaultdict(list)
    for row in negatives:
        source = matches_by_id[row["source_action_id"]]
        if source["split"] != row["split"] or row["split"] != "train":
            raise ValueError("source split mismatch")
        if row["donor_action_id"] and matches_by_id[row["donor_action_id"]]["split"] != "train":
            raise ValueError("donor split leakage")
        if source["label"] != "exact_match" or len(source["selected_intent_ids"]) != 1:
            raise ValueError("source is not a single-intent exact match")
        if row["source_action_sha256"] != source["raw_action_sha256"] or row["source_action_sha256"] != actions[row["source_action_id"]]["raw_action_sha256"]:
            raise ValueError("source hash mismatch")
        if canonical(row["positive_action"]) != canonical(actions[row["source_action_id"]]["action"]):
            raise ValueError("source action mismatch")
        if not row["gqq_validation"]["accepted"] or row["gqq_validation"]["label"] != row["target_label"]:
            raise ValueError("missing model validation")
        if hashlib.sha256(canonical(row["negative_action"]).encode()).hexdigest() != row["negative_action_sha256"]:
            raise ValueError("negative action hash mismatch")
        before, after = row["positive_action"], row["negative_action"]
        if row["target_label"] == "object_mismatch":
            field = "url" if before["function"] == "web_fetch" else "path"
            name = before["args"][field].split("?", 1)[0].rsplit("/", 1)[-1].lower()
            stem = name.rsplit(".", 1)[0]
            if not ((len(name) >= 5 and name in row["intent_text"].lower()) or (len(stem) >= 5 and stem in row["intent_text"].lower())):
                raise ValueError("object target is not explicit in intent")
            if before["function"] != after["function"] or set(before["args"]) != set(after["args"]) or any(before["args"][key] != after["args"][key] for key in before["args"] if key != field) or before["args"][field] == after["args"][field]:
                raise ValueError("object negative changed more than target")
        elif row["target_label"] == "type_mismatch":
            if before["function"] == after["function"] or before["args"]["path"] != after["args"]["path"] or after["function"] not in {"read", "write"}:
                raise ValueError("invalid type negative")
            expected_args = {"path"} if after["function"] == "read" else {"path", "content"}
            if set(after["args"]) != expected_args:
                raise ValueError("invalid changed operation arguments")
        else:
            raise ValueError("unknown negative label")
        grouped[row["source_action_id"]].append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    train = []
    for action_id, group in sorted(grouped.items()):
        source = matches_by_id[action_id]
        intent = intent_for(source, intents)
        if any(item["intent_text"] != intent["intent_text"] for item in group):
            raise ValueError("negative intent differs from source")
        train.append({
            "query": "[INTENT] " + intent["intent_text"],
            "pos": [action_text(actions[action_id]["action"])],
            "neg": [action_text(item["negative_action"]) for item in group],
            "source_action_id": action_id,
            "candidate_ids": [item["candidate_id"] for item in group],
            "negative_labels": [item["target_label"] for item in group],
            "supervision_tier": "silver_model_generated",
        })
    write_jsonl(args.output / "train_grouped.jsonl", train)
    flat_train = []
    for row in train:
        for negative, candidate_id, label in zip(row["neg"], row["candidate_ids"], row["negative_labels"]):
            flat_train.append({"query": row["query"], "pos": row["pos"], "neg": [negative],
                               "source_action_id": row["source_action_id"], "candidate_id": candidate_id,
                               "negative_label": label, "supervision_tier": row["supervision_tier"]})
    if len(flat_train) != 5000:
        raise ValueError("flat training rows do not match negative target")
    if len({(row["query"], row["pos"][0], row["neg"][0]) for row in flat_train}) != 5000:
        raise ValueError("duplicate training text triples")
    write_jsonl(args.output / "train.jsonl", flat_train)

    eval_counts = {}
    for split in ("dev", "test"):
        pairs = []
        for row in matches:
            if row["split"] != split or row["label"] != "exact_match" or len(row.get("selected_intent_ids", [])) != 1:
                continue
            intent = intent_for(row, intents)
            pairs.append({"query": "[INTENT] " + intent["intent_text"], "positive": action_text(actions[row["action_id"]]["action"]), "action_id": row["action_id"], "trajectory_id": row["trajectory_id"], "split": split, "source_label": "exact_match"})
        write_jsonl(args.output / f"{split}_positive_pairs.jsonl", pairs)
        eval_counts[split] = len(pairs)

    manifest = {
        "schema_version": "bge_silver_train_5000_v1",
        "date": "2026-09-29",
        "status": "silver_model_generated_not_gold",
        "train_rows": len(flat_train),
        "train_grouped_rows": len(train),
        "train_negative_count": len(negatives),
        "train_negative_labels": dict(labels),
        "evaluation_positive_pairs": eval_counts,
        "format": "FlagEmbedding style query/pos/neg; metadata fields can be ignored by trainer",
        "note": "Dev/test contain original positive pairs only; generated negatives are train-local. These pairs are not human Gold labels.",
        "input_sha256": {name: sha_file(path) for name, path in {"negatives": args.negatives, "matches": args.matches, "actions": args.actions, "intents": args.intents}.items()},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical(manifest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negatives", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--intents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
