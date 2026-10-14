#!/usr/bin/env python3
"""Select a deterministic, diverse 100 case review sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


QUOTAS = {
    "partial_match": 41,
    "no_intent_match": 25,
    "exact_match": 15,
    "ambiguous": 8,
    "object_mismatch": 9,
    "type_mismatch": 2,
}
SEED = "action-intent-review-100-v1"


def confidence_bucket(row):
    value = row["confidence"]
    return "low" if value < 0.5 else "medium" if value < 0.8 else "high"


def tie_break(row):
    return hashlib.sha256(f"{SEED}:{row['action_id']}".encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    chosen = []
    used_trajectories = set()
    used_actions = set()
    seen_features = Counter()
    for label, quota in sorted(QUOTAS.items(), key=lambda item: item[1]):
        pool = [row for row in rows if row["label"] == label]
        if len(pool) < quota:
            raise ValueError(f"{label}: need {quota}, found {len(pool)}")
        for _ in range(quota):
            candidates = [row for row in pool if row["action_id"] not in used_actions]
            if not candidates:
                raise ValueError(f"not enough distinct actions for {label}")
            def rank(row):
                features = (
                    (label, "atomicity", row["atomicity"]),
                    (label, "confidence", confidence_bucket(row)),
                )
                return (row["trajectory_id"] not in used_trajectories, sum(1 / (1 + seen_features[feature]) for feature in features), tie_break(row))
            pick = max(candidates, key=rank)
            chosen.append(pick)
            used_trajectories.add(pick["trajectory_id"])
            used_actions.add(pick["action_id"])
            seen_features[(label, "atomicity", pick["atomicity"])] += 1
            seen_features[(label, "confidence", confidence_bucket(pick))] += 1
    chosen.sort(key=lambda row: (row["label"], row["trajectory_id"], row["thought_step"], row["action_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for row in chosen:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "selection_method": "fixed label quotas, maximize trajectory diversity, greedy atomicity and confidence coverage",
        "seed": SEED,
        "count": len(chosen),
        "source_count": len(rows),
        "labels": dict(Counter(row["label"] for row in chosen)),
        "atomicity": dict(Counter(row["atomicity"] for row in chosen)),
        "confidence_buckets": dict(Counter(confidence_bucket(row) for row in chosen)),
        "distinct_trajectories": len(used_trajectories),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
