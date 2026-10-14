#!/usr/bin/env python3
"""Versioned TA train windows: repair confirmed pilot cases and add train-only windows."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from build_ta_coverage_pilot import (INTENTS, ROOT, SPLITS, raw_index, sample,
                                     split_index, stable_id, window, write)
from evaluate_bge_clawgym import rows, sha256

PILOT = Path("data/clawgym/ta_coverage_pilot_20261013/windows.jsonl")
SCOPE_CORRECTIONS = {
    "3592dafd7d2e0761c5f51598": {"ff781e8338b9f76b0f4ed452", "d71f12abe594c4fac2d8090a"},
    "efd425645387ae3fe5420334": {"4f4269ff7f9c5b8221df9835"},
}
EMAIL_WINDOW = "887dc40869ef763450e4a071"


def rebuild_pairs(record):
    record["pairs"] = [{"pair_id": stable_id(intent["intent_id"], action["action_id"]),
                        "window_id": record["window_id"], "intent_id": intent["intent_id"],
                        "action_id": action["action_id"], "covered": None,
                        "review_status": "unreviewed", "review_reason": None}
                       for intent in record["intents"] if intent["temporal_scope"] == "current"
                       for action in record["actions"]]


def repair(record):
    record = json.loads(json.dumps(record))
    wid = record["window_id"]
    changed = False
    if wid in SCOPE_CORRECTIONS:
        found = set()
        for intent in record["intents"]:
            if intent["intent_id"] in SCOPE_CORRECTIONS[wid]:
                if intent["temporal_scope"] != "future":
                    raise ValueError(f"unexpected source scope: {wid}/{intent['intent_id']}")
                intent["temporal_scope"] = "current"
                intent["scope_review"] = "manual_audit_20261013"
                found.add(intent["intent_id"])
        if found != SCOPE_CORRECTIONS[wid]:
            raise ValueError(f"missing scope correction: {wid}")
        changed = True
    if wid == EMAIL_WINDOW:
        if len(record["intents"]) != 1 or len(record["actions"]) != 2:
            raise ValueError("unexpected email window structure")
        parent = record["intents"][0]
        new = []
        for name in ("pitch_friendly.txt", "pitch_formal.txt"):
            new.append({**parent,
                        "intent_id": stable_id(wid, "atomic_email_verification", name),
                        "text": f"Read output/emails/{name} to verify this email version.",
                        "source_intent_id": parent["source_intent_id"] + ":" + name,
                        "parent_intent_id": parent["intent_id"],
                        "atomicity_review": "manual_audit_20261013"})
        record["intents"] = new
        changed = True
    if changed:
        rebuild_pairs(record)
        record["ta_gold"] = None
        record["gold_review_status"] = "unreviewed_after_repair"
    return record, changed


def build(pilot, all_train, extra_count, seed):
    if any(w["split"] != "train" for w in pilot):
        raise ValueError("pilot includes non-train window")
    pilot_ids = {w["window_id"] for w in pilot}
    if len(pilot_ids) != len(pilot):
        raise ValueError("duplicate pilot window")
    candidates = [w for w in all_train if w["window_id"] not in pilot_ids]
    extras = sample(candidates, extra_count, seed)
    if len(extras) != extra_count:
        raise ValueError("insufficient extra train windows")
    repaired = [repair(w) for w in pilot]
    combined = sorted([w for w, _ in repaired] + extras, key=lambda x: x["window_id"])
    if len({w["window_id"] for w in combined}) != len(combined):
        raise ValueError("duplicate output window")
    review_needed = sorted([w for w, changed in repaired if changed] + extras,
                           key=lambda x: x["window_id"])
    return combined, review_needed, sum(changed for _, changed in repaired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=PILOT)
    parser.add_argument("--raw", type=Path, default=ROOT)
    parser.add_argument("--intents", type=Path, default=INTENTS)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--extra-windows", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.extra_windows < 1:
        raise ValueError("output must not exist and extra-windows must be positive")
    splits = split_index(rows(args.splits))
    raw = raw_index(args.raw)
    all_train = []
    for item in rows(args.intents):
        tid = item["trajectory_id"]
        if splits.get(tid) != "train" or tid not in raw:
            continue
        built = window(item, raw[tid], "train")
        if built["intents"] and built["actions"]:
            all_train.append(built)
    combined, review_needed, changed = build(list(rows(args.pilot)), all_train,
                                             args.extra_windows, args.seed)
    args.output.mkdir(parents=True)
    for name, values in (("windows", combined), ("review_needed", review_needed),
                         ("intents", [x for w in combined for x in w["intents"]]),
                         ("actions", [x for w in combined for x in w["actions"]]),
                         ("pairs", [x for w in combined for x in w["pairs"]])):
        write(args.output / f"{name}.jsonl", values)
    write(args.output / "split_assignments.jsonl",
          [{"trajectory_id": tid, "split": split} for tid, split in sorted(splits.items())])
    report = {"schema_version": "ta_coverage_train_v2", "windows": len(combined),
              "pairs": sum(len(w["pairs"]) for w in combined),
              "review_needed": len(review_needed), "repaired_pilot_windows": changed,
              "trajectories": len({w["trajectory_id"] for w in combined}),
              "split": dict(Counter(w["split"] for w in combined)),
              "input_sha256": {"pilot": sha256(args.pilot), "intents": sha256(args.intents),
                               "splits": sha256(args.splits)}}
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
