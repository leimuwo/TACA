#!/usr/bin/env python3
"""Select clear-Intent held-out TA windows resembling the training input mix."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_bge_clawgym import rows, sha256

TRAIN = Path("data/clawgym/ta_pair_weak_train_10k_pool_targeted_v3_20261013/train.jsonl")
DEV = Path("data/clawgym/ta_clear_dev_eval_20261013/windows.jsonl")
TEST = Path("data/clawgym/ta_clear_test_eval_20261013/windows.jsonl")


def tool_group(tool):
    return tool if tool in {"exec", "read", "write", "web_fetch", "edit"} else "other"


def pair_bin(count):
    return "1" if count == 1 else "2" if count == 2 else "3+"


def features(window):
    tools = Counter(tool_group(a["raw_call"]["function"]) for a in window["actions"])
    return (tools.most_common(1)[0][0], pair_bin(len(window["pairs"])))


def train_profile(train):
    by_window = defaultdict(list)
    for row in train:
        by_window[row["window_id"]].append(row)
    return Counter((Counter(tool_group(r["tool"]) for r in group).most_common(1)[0][0],
                    pair_bin(len(group))) for group in by_window.values())


def select(pool, profile, count, seed):
    if count > len(pool):
        raise ValueError("requested sample exceeds clear-window pool")
    if len({w["window_id"] for w in pool}) != len(pool):
        raise ValueError("duplicate window ID")
    wanted = {key: value / sum(profile.values()) for key, value in profile.items()}
    candidates = sorted(pool, key=lambda w: w["window_id"])
    rng = random.Random(seed)
    best = None
    best_score = float("inf")
    # Fixed-seed search uses input features only, never held-out labels.
    for _ in range(400):
        chosen = rng.sample(candidates, count)
        distribution = Counter(features(w) for w in chosen)
        score = sum(abs(distribution[key] / count - wanted.get(key, 0))
                    for key in distribution.keys() | wanted.keys())
        if score < best_score:
            best, best_score = chosen, score
    return sorted(best, key=lambda w: w["window_id"]), best_score


def write(path, records):
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--dev", type=Path, default=DEV)
    parser.add_argument("--test", type=Path, default=TEST)
    parser.add_argument("--count-per-split", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.count_per_split < 1:
        raise ValueError("output must not exist and count must be positive")
    train = list(rows(args.train))
    profile = train_profile(train)
    train_tids = {r["trajectory_id"] for r in train}
    selected = {}
    distances = {}
    for split, path in (("dev", args.dev), ("test", args.test)):
        pool = list(rows(path))
        if any(w["split"] != split for w in pool):
            raise ValueError(f"wrong split in {path}")
        selected[split], distances[split] = select(
            pool, profile, args.count_per_split, args.seed + (0 if split == "dev" else 1))
    tids = {split: {w["trajectory_id"] for w in records} for split, records in selected.items()}
    if train_tids & (tids["dev"] | tids["test"]) or tids["dev"] & tids["test"]:
        raise ValueError("trajectory crosses train/dev/test")
    if {w["window_id"] for w in selected["dev"]} & {w["window_id"] for w in selected["test"]}:
        raise ValueError("window crosses dev/test")
    args.output.mkdir(parents=True)
    for split, records in selected.items():
        (args.output / split).mkdir()
        write(args.output / split / "windows.jsonl", records)
    report = {"schema_version": "ta_glm_eval_sample_v1",
              "label_status": "unreviewed", "seed": args.seed,
              "selection": "clear Intent, frozen split, input-feature stratification only",
              "train_window_profile": {"|".join(k): v for k, v in sorted(profile.items())},
              "splits": {split: {"windows": len(records), "pairs": sum(len(w["pairs"]) for w in records),
                                  "trajectories": len(tids[split]),
                                  "feature_distance_l1": distances[split],
                                  "window_profile": {"|".join(k): v for k, v in sorted(Counter(features(w) for w in records).items())}}
                         for split, records in selected.items()},
              "source_sha256": {"train": sha256(args.train), "dev": sha256(args.dev),
                                "test": sha256(args.test)}}
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
