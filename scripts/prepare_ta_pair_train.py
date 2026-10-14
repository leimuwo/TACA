#!/usr/bin/env python3
"""Build exploratory TA pair labels from valid GQQ reviews with explicit exclusions."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_bge_clawgym import rows, sha256

ROOT = Path("data/clawgym/ta_coverage_train_v2_20261013")
OLD = Path("data/clawgym/ta_coverage_pilot_20261013/model_reviews_current.jsonl")
MIDDLE = ROOT / "model_reviews_new.jsonl"
QUARANTINE = Path("data/clawgym/ta_coverage_pilot_20261013/ta_pair_audit_quarantine_20261013.jsonl")


def prepare(windows, reviews, quarantine, seed=20261013, resolutions=None):
    by_window = {w["window_id"]: w for w in windows}
    if len(by_window) != len(windows):
        raise ValueError("duplicate window")
    by_review = {}
    for review in reviews:
        wid = review["window_id"]
        if wid not in by_window:
            raise ValueError(f"review outside windows: {wid}")
        if not review.get("review_failed") and review.get("verdict") is not None:
            by_review[wid] = review
    blocked_windows = {x["window_id"] for x in quarantine if x["scope"] == "window"}
    blocked_pairs = {x["pair_id"] for x in quarantine if x["scope"] == "pair"}
    resolutions = resolutions or {}
    excluded = Counter()
    all_rows = []
    for wid, w in by_window.items():
        review = by_review.get(wid)
        if review is None:
            excluded["missing_valid_review_window"] += 1
            continue
        if wid in blocked_windows:
            excluded["quarantined_window"] += 1
            continue
        v = review["verdict"]
        resolution = resolutions.get(wid)
        if v.get("uncertainties") and resolution is None:
            excluded["uncertain_window"] += 1
            continue
        if resolution is not None:
            cleared = set(resolution["cleared_pair_ids"])
            blocked = set(resolution["blocked_pair_ids"])
            if (cleared & blocked or cleared | blocked != {p["pair_id"] for p in w["pairs"]}
                    or len(cleared) != len(resolution["cleared_pair_ids"])
                    or len(blocked) != len(resolution["blocked_pair_ids"])):
                raise ValueError(f"resolution pair IDs do not match window: {wid}")
            excluded["resolved_window"] += 1
        targeted_audit = {x["pair_id"]: x for x in (resolution or {}).get("targeted_audit", [])}
        intents = {x["intent_id"]: x for x in v["intents"]}
        pairs = {x["pair_id"]: x for x in v["pairs"]}
        if (len(intents) != len(w["intents"]) or len(pairs) != len(w["pairs"])
                or set(intents) != {x["intent_id"] for x in w["intents"]}
                or set(pairs) != {x["pair_id"] for x in w["pairs"]}):
            excluded["review_schema_mismatch_window"] += 1
            continue
        temporal_conflicts = {x["intent_id"] for x in v["intents"]
                              if x["due_now"] and x["assessable"] and
                              next(i for i in w["intents"] if i["intent_id"] == x["intent_id"])["temporal_scope"] != "current"}
        if temporal_conflicts and resolution is None:
            excluded["temporal_scope_conflict_window"] += 1
            continue
        if v.get("suggested_ta_label") == "unassessable" and resolution is None:
            excluded["unassessable_window"] += 1
            continue
        action_by_id = {x["action_id"]: x for x in w["actions"]}
        intent_by_id = {x["intent_id"]: x for x in w["intents"]}
        for pair in w["pairs"]:
            pid = pair["pair_id"]
            if pid in blocked_pairs:
                excluded["quarantined_pair"] += 1
                continue
            if resolution is not None and pid in blocked:
                excluded["uncertain_pair"] += 1
                continue
            if pair["intent_id"] in temporal_conflicts:
                excluded["temporal_scope_conflict_pair"] += 1
                continue
            i = intents[pair["intent_id"]]
            if not i["due_now"] or not i["assessable"]:
                excluded["not_assessable_pair"] += 1
                continue
            decision = pairs[pid]
            if not isinstance(decision.get("covered"), bool):
                excluded["invalid_pair"] += 1
                continue
            all_rows.append({"pair_id": pid, "window_id": wid,
                             "trajectory_id": w["trajectory_id"], "source": w.get("source", "clawgym"),
                             "split": w["split"],
                             "intent_id": pair["intent_id"], "action_id": pair["action_id"],
                             "intent": intent_by_id[pair["intent_id"]]["text"],
                             "action": action_by_id[pair["action_id"]]["action_text"],
                             "tool": action_by_id[pair["action_id"]]["raw_call"]["function"],
                             "label": int(decision["covered"]),
                             "label_source": ("gqq_targeted_agreement" if
                                              targeted_audit.get(pid, {}).get("status") == "released"
                                              else "gqq_weak_supervision"),
                             "review_reason": decision["reason"],
                             "targeted_review_reason": targeted_audit.get(pid, {}).get("reason")})
    if len({x["pair_id"] for x in all_rows}) != len(all_rows):
        raise ValueError("duplicate pair ID")
    if any(x["split"] != "train" for x in all_rows):
        raise ValueError("non-train row in training data")
    by_text = defaultdict(list)
    for row in all_rows:
        key = (" ".join(row["intent"].casefold().split()),
               " ".join(row["action"].casefold().split()))
        by_text[key].append(row)
    unique_rows = []
    for group in by_text.values():
        if len({x["label"] for x in group}) > 1:
            excluded["conflicting_duplicate_text_pair"] += len(group)
            continue
        unique_rows.append(group[0])
        excluded["duplicate_text_pair"] += len(group) - 1
    all_rows = unique_rows
    trajectories = sorted({x["trajectory_id"] for x in all_rows})
    random.Random(seed).shuffle(trajectories)
    holdout = set(trajectories[:max(1, round(len(trajectories) * .1))])
    train = [x for x in all_rows if x["trajectory_id"] not in holdout]
    validation = [x for x in all_rows if x["trajectory_id"] in holdout]
    if not {x["label"] for x in train} == {0, 1}:
        raise ValueError("training data lacks a class")
    return train, validation, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=ROOT / "windows.jsonl")
    parser.add_argument("--old-reviews", type=Path, default=OLD)
    parser.add_argument("--new-reviews", type=Path, default=ROOT / "model_reviews_new.jsonl")
    parser.add_argument("--review", type=Path, action="append",
                        help="Review files in increasing precedence order; overrides old/new defaults")
    parser.add_argument("--quarantine", type=Path, default=QUARANTINE)
    parser.add_argument("--resolutions", type=Path,
                        help="Auditable pair-level decisions from resolve_ta_uncertainties.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    review_paths = args.review or [args.old_reviews, args.new_reviews]
    train, validation, excluded = prepare(list(rows(args.windows)),
                                          [item for path in review_paths for item in rows(path)],
                                          list(rows(args.quarantine)), args.seed,
                                          {x["window_id"]: x for x in rows(args.resolutions)} if args.resolutions else None)
    args.output.mkdir(parents=True)
    for name, values in (("train.jsonl", train), ("internal_validation.jsonl", validation)):
        with (args.output / name).open("w", encoding="utf-8") as stream:
            for item in values:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    report = {"schema_version": "ta_pair_weak_train_v1", "label_status": "GQQ weak supervision",
              "train_pairs": len(train), "internal_validation_pairs": len(validation),
              "uncertainty_resolutions_sha256": sha256(args.resolutions) if args.resolutions else None,
              "train_labels": dict(Counter(x["label"] for x in train)),
              "internal_validation_labels": dict(Counter(x["label"] for x in validation)),
              "train_source_pairs": dict(Counter(x["source"] for x in train)),
              "internal_validation_source_pairs": dict(Counter(x["source"] for x in validation)),
              "train_label_sources": dict(Counter(x["label_source"] for x in train)),
              "internal_validation_label_sources": dict(Counter(x["label_source"] for x in validation)),
              "train_trajectories": len({x["trajectory_id"] for x in train}),
              "internal_validation_trajectories": len({x["trajectory_id"] for x in validation}),
              "excluded": dict(excluded),
              "input_sha256": {"windows": sha256(args.windows),
                               "reviews": {str(path): sha256(path) for path in review_paths},
                               "quarantine": sha256(args.quarantine)}}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
