#!/usr/bin/env python3
"""Build a versioned 10k ClawGym/SWE Intent-Action candidate pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from build_ta_coverage_pilot import INTENTS, ROOT, SPLITS, raw_index, split_index, stable_id, window, write
from evaluate_bge_clawgym import rows, sha256

REPAIRED = Path("data/clawgym/ta_coverage_train_v2_20261013/windows.jsonl")
SWE_INTENTS = Path("../data/SWE-agent-trajectories/test_data/thought_intent")


def swe_split(instance_id):
    bucket = int(hashlib.sha256(instance_id.encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else "dev" if bucket == 8 else "test"


def swe_windows(directory):
    result = []
    seen_instances = set()
    split_counts = Counter()
    for path in sorted(directory.glob("swe__*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "completed" or record.get("source_type") != "swe":
            continue
        instance_id = record["metadata"]["instance_id"]
        if not instance_id:
            raise ValueError(f"missing SWE instance_id: {path}")
        # The selected corpus should contain one trajectory per issue; repeated issues
        # need explicit grouping before they can be used in a future expansion.
        if instance_id in seen_instances:
            raise ValueError(f"duplicate SWE instance_id: {instance_id}")
        seen_instances.add(instance_id)
        split = swe_split(instance_id)
        split_counts[split] += 1
        if split != "train":
            continue
        tid = "swe:" + record["trajectory_id"]
        for step in record["steps"]:
            extracted = step.get("intent_result") or {}
            source_intents = extracted.get("execution_intents") or []
            source_actions = step.get("actual_actions") or []
            if not source_intents or not source_actions:
                continue
            number = step["thought_step"]
            wid = stable_id("swe", instance_id, tid, number)
            intents = [{"intent_id": stable_id(wid, x["intent_id"]),
                        "source_intent_id": x["intent_id"], "window_id": wid,
                        "text": x["intent_text"], "source_quote": x.get("source_quote"),
                        "temporal_scope": "current", "assessability": "unreviewed",
                        "assessability_reason": None} for x in source_intents]
            actions = []
            for index, command in enumerate(source_actions):
                if not isinstance(command, str) or not command.strip():
                    raise ValueError(f"invalid SWE action: {path}/{number}")
                aid = stable_id(wid, "action", index)
                tool = command.strip().split(maxsplit=1)[0]
                raw_call = {"function": tool, "args": {"command": command}}
                actions.append({"action_id": aid, "window_id": wid, "call_index": index,
                                "raw_call": raw_call,
                                "action_text": "[ACTION] " + json.dumps(raw_call, ensure_ascii=False,
                                                               sort_keys=True, separators=(",", ":")),
                                "execution_result": [], "result_status": "not_imported"})
            pairs = [{"pair_id": stable_id(intent["intent_id"], action["action_id"]),
                      "window_id": wid, "intent_id": intent["intent_id"],
                      "action_id": action["action_id"], "covered": None,
                      "review_status": "unreviewed", "review_reason": None}
                     for intent in intents for action in actions]
            result.append({"window_id": wid, "source": "swe_agent_trajectories",
                           "split": "train", "trajectory_id": tid, "instance_id": instance_id,
                           "thought_step": number, "thought": step["thought"],
                           "user_request": step.get("user_request") or record.get("task"),
                           "intents": intents, "actions": actions, "pairs": pairs,
                           "ta_gold": None, "gold_review_status": "unreviewed",
                           "review_notes": None})
    return result, split_counts


def choose_claw(repaired, available, target_pairs, seed):
    fixed = {x["window_id"]: x for x in repaired}
    if len(fixed) != len(repaired) or any(x["split"] != "train" for x in repaired):
        raise ValueError("repaired source must contain unique train windows")
    selected = list(repaired)
    used = sum(len(x["pairs"]) for x in selected)
    if used > target_pairs:
        raise ValueError("repaired windows exceed requested ClawGym budget")
    extra = [x for x in available if x["window_id"] not in fixed]
    random.Random(seed).shuffle(extra)
    for record in extra:
        if used == target_pairs:
            break
        if used + len(record["pairs"]) <= target_pairs:
            selected.append(record)
            used += len(record["pairs"])
    if used != target_pairs:
        raise ValueError(f"only {used}/{target_pairs} ClawGym pairs available")
    return sorted(selected, key=lambda x: x["window_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-pairs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--repaired", type=Path, default=REPAIRED)
    parser.add_argument("--swe-intents", type=Path, default=SWE_INTENTS)
    parser.add_argument("--raw", type=Path, default=ROOT)
    parser.add_argument("--intents", type=Path, default=INTENTS)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.target_pairs < 1:
        raise ValueError("output must not exist and target-pairs must be positive")
    swe, swe_counts = swe_windows(args.swe_intents)
    swe_pairs = sum(len(x["pairs"]) for x in swe)
    if not swe or swe_pairs >= args.target_pairs:
        raise ValueError("SWE candidate supply must be nonzero and below target")
    splits = split_index(rows(args.splits))
    raw = raw_index(args.raw)
    available = []
    for item in rows(args.intents):
        tid = item["trajectory_id"]
        if splits.get(tid) == "train" and tid in raw:
            built = window(item, raw[tid], "train")
            if built["intents"] and built["actions"]:
                available.append(built)
    repaired = list(rows(args.repaired))
    repaired_ids = {x["window_id"] for x in repaired}
    claw = choose_claw(repaired, available, args.target_pairs - swe_pairs, args.seed)
    combined = sorted([*claw, *swe], key=lambda x: x["window_id"])
    pairs = [x for w in combined for x in w["pairs"]]
    if len(pairs) != args.target_pairs or len({x["pair_id"] for x in pairs}) != len(pairs):
        raise ValueError("candidate count or pair ID uniqueness failed")
    args.output.mkdir(parents=True)
    write(args.output / "windows.jsonl", combined)
    write(args.output / "pairs.jsonl", pairs)
    write(args.output / "review_needed.jsonl", [x for x in combined if x["source"] == "swe_agent_trajectories"
                                                 or x["window_id"] not in repaired_ids])
    report = {"schema_version": "ta_candidate_pool_10k_v1", "label_status": "unreviewed_candidates",
              "candidate_pairs": len(pairs), "windows": len(combined),
              "trajectories": len({x["trajectory_id"] for x in combined}),
              "source_pairs": dict(Counter(w["source"] for w in combined for _ in w["pairs"])),
              "source_windows": dict(Counter(w["source"] for w in combined)),
              "swe_instance_split_counts": dict(swe_counts),
              "input_sha256": {"repaired": sha256(args.repaired), "claw_intents": sha256(args.intents),
                               "claw_splits": sha256(args.splits)},
              "policy": "Candidate generation only. No inferred positive or negative labels."}
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
