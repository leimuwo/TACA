#!/usr/bin/env python3
"""Freeze ClawGym window schema/splits and prepare 200 train review windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_bge_clawgym import rows, sha256

ROOT = Path("../data/clawgym/selected_1000_json")
INTENTS = Path("data/clawgym/stage1_intents_blind_complete_925_v1/intents_blind.jsonl")
SPLITS = Path("data/clawgym/stage3_5_parent_split_complete_925_v1/action_intent_matches_with_split.jsonl")


def stable_id(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:24]


def write(path, items):
    with path.open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_index(records):
    result = {}
    clusters = defaultdict(set)
    for row in records:
        tid, split = row["trajectory_id"], row["split"]
        if split not in {"train", "dev", "test"} or (tid in result and result[tid] != split):
            raise ValueError(f"invalid or conflicting split: {tid}")
        result[tid] = split
        if row.get("template_cluster_id"):
            clusters[row["template_cluster_id"]].add(split)
    if any(len(value) > 1 for value in clusters.values()):
        raise ValueError("template cluster crosses split")
    return result


def raw_index(root):
    result = {}
    for path in root.glob("*.json"):
        if path.name == "selection_summary.json":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if "id" not in record or "messages" not in record:
            continue
        if record["id"] in result:
            raise ValueError("duplicate raw trajectory")
        result[record["id"]] = record
    return result


def window(row, raw, split):
    tid, step = row["trajectory_id"], row["thought_step"]
    assistants = [msg for msg in raw["messages"] if msg.get("role") == "assistant"
                  and msg.get("assistant_thought")]
    if step < 1 or step > len(assistants):
        raise ValueError(f"missing assistant step: {tid}/{step}")
    current = assistants[step - 1]
    calls = current.get("tool_calls") or []
    responses = defaultdict(list)
    for msg in raw["messages"]:
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            responses[msg["tool_call_id"]].append(msg.get("content", ""))
    wid = stable_id(tid, step)
    intents = []
    for intent in row.get("execution_intents", []):
        iid = stable_id(wid, intent["intent_id"])
        intents.append({"intent_id": iid, "source_intent_id": intent["intent_id"],
                        "window_id": wid, "text": intent["intent_text"],
                        "source_quote": intent.get("source_quote"),
                        "temporal_scope": intent.get("temporal_scope"),
                        "assessability": "unreviewed", "assessability_reason": None})
    actions = []
    for index, call in enumerate(calls):
        aid = call.get("id") or stable_id(wid, "action", index)
        actions.append({"action_id": aid, "window_id": wid, "call_index": index,
                        "raw_call": call, "action_text": "[ACTION] " + json.dumps(call, ensure_ascii=False,
                                                                               sort_keys=True, separators=(",", ":")),
                        "execution_result": responses.get(aid, []),
                        "result_status": "observed" if responses.get(aid) else "missing"})
    pairs = [{"pair_id": stable_id(intent["intent_id"], action["action_id"]),
              "window_id": wid, "intent_id": intent["intent_id"], "action_id": action["action_id"],
              "covered": None, "review_status": "unreviewed", "review_reason": None}
             for intent in intents if intent["temporal_scope"] == "current" for action in actions]
    record = {"window_id": wid, "source": "clawgym", "split": split,
              "trajectory_id": tid, "thought_step": step, "thought": row["thought"],
              "user_request": row.get("user_request"), "intents": intents, "actions": actions,
              "pairs": pairs, "ta_gold": None, "gold_review_status": "unreviewed",
              "review_notes": None}
    return record


def sample(windows, count, seed):
    groups = defaultdict(list)
    for item in windows:
        ni, na = len(item["intents"]), len(item["actions"])
        group = "multi_intent_multi_action" if ni > 1 and na > 1 else "multi_intent" if ni > 1 else "multi_action" if na > 1 else "single"
        groups[group].append(item)
    rng = random.Random(seed)
    chosen = []
    for group in sorted(groups):
        group_rows = sorted(groups[group], key=lambda x: x["window_id"])
        chosen.extend(rng.sample(group_rows, min(len(group_rows), count // len(groups))))
    remaining = sorted((x for x in windows if x["window_id"] not in {y["window_id"] for y in chosen}),
                       key=lambda x: x["window_id"])
    chosen.extend(rng.sample(remaining, min(count - len(chosen), len(remaining))))
    return sorted(chosen, key=lambda x: x["window_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT)
    parser.add_argument("--intents", type=Path, default=INTENTS)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    if args.output.exists() or args.count < 1:
        raise ValueError("output must not exist and count must be positive")
    splits = split_index(rows(args.splits))
    raw = raw_index(args.raw)
    all_windows = []
    for row in rows(args.intents):
        tid = row["trajectory_id"]
        if tid not in splits or tid not in raw:
            continue
        item = window(row, raw[tid], splits[tid])
        if item["intents"] and item["actions"]:
            all_windows.append(item)
    train = [item for item in all_windows if item["split"] == "train"]
    chosen = sample(train, args.count, args.seed)
    if len(chosen) != args.count:
        raise ValueError("insufficient train windows")
    args.output.mkdir(parents=True)
    write(args.output / "windows.jsonl", chosen)
    write(args.output / "intents.jsonl", [x for w in chosen for x in w["intents"]])
    write(args.output / "actions.jsonl", [x for w in chosen for x in w["actions"]])
    write(args.output / "pairs.jsonl", [x for w in chosen for x in w["pairs"]])
    write(args.output / "ta_gold.jsonl", [{"window_id": w["window_id"], "gold_label": None,
                                           "review_status": "unreviewed", "review_notes": None} for w in chosen])
    write(args.output / "split_assignments.jsonl", [{"trajectory_id": tid, "split": split}
                                                      for tid, split in sorted(splits.items())])
    manifest = {"schema_version": "ta_coverage_pilot_v1", "seed": args.seed,
                "source_split_is_frozen": True, "counts_by_split": dict(Counter(splits.values())),
                "eligible_windows_by_split": dict(Counter(w["split"] for w in all_windows)),
                "raw_trajectories_loaded": len(raw),
                "pilot_windows": len(chosen), "pilot_pairs": sum(len(w["pairs"]) for w in chosen),
                "pilot_trajectories": len({w["trajectory_id"] for w in chosen}),
                "input_sha256": {"intents": sha256(args.intents), "splits": sha256(args.splits)},
                "label_policy": "all labels unreviewed; retrieval qrels never imported"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
