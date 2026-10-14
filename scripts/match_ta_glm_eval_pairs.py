#!/usr/bin/env python3
"""Create tool- and label-stratified pair views of held-out GLM weak labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from evaluate_bge_clawgym import rows, sha256


def group(tool):
    return tool if tool in {"exec", "read", "write", "web_fetch", "edit"} else "other"


def text_key(row):
    return (" ".join(row["intent"].casefold().split()),
            " ".join(row["action"].casefold().split()))


def validate_splits(train, dev, test):
    splits = {"train": train, "dev": dev, "test": test}
    for name, records in splits.items():
        ids = [row["pair_id"] for row in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate pair ID in {name}")
        if name != "train" and any(row["split"] != name for row in records):
            raise ValueError(f"wrong split in {name}")
    for first, second in (("train", "dev"), ("train", "test"), ("dev", "test")):
        a, b = splits[first], splits[second]
        if ({row["trajectory_id"] for row in a} & {row["trajectory_id"] for row in b}
                or {text_key(row) for row in a} & {text_key(row) for row in b}):
            raise ValueError(f"trajectory or text pair overlaps {first}/{second}")


def choose(records, profile, count, seed):
    if count > len(records):
        raise ValueError("requested count exceeds available pairs")
    buckets = {}
    for row in records:
        buckets.setdefault((group(row["tool"]), row["label"]), []).append(row)
    for key in buckets:
        buckets[key].sort(key=lambda row: hashlib.sha256(
            f"{seed}:{row['pair_id']}".encode()).hexdigest())
    tool_profile = Counter(tool for tool, _ in profile.elements())
    label_profile = Counter(label for _, label in profile.elements())
    total = sum(profile.values())
    target_tools = {tool: count * n / total for tool, n in tool_profile.items()}
    target_labels = {label: count * n / total for label, n in label_profile.items()}
    chosen_tools, chosen_labels = Counter(), Counter()
    selected = []
    while len(selected) < count:
        options = [key for key in buckets if buckets[key]]
        if not options:
            raise ValueError("insufficient candidates")
        def gain(key):
            tool, label = key
            tool_target = target_tools.get(tool, 0)
            label_target = target_labels.get(label, 0)
            return ((2 * (tool_target - chosen_tools[tool]) - 1) / max(tool_target, 1)
                    + 2 * (2 * (label_target - chosen_labels[label]) - 1) / max(label_target, 1))
        tool, label = max(options, key=lambda key: (gain(key), key))
        selected.append(buckets[(tool, label)].pop())
        chosen_tools[tool] += 1
        chosen_labels[label] += 1
    return sorted(selected, key=lambda row: row["pair_id"])


def write(path, records):
    with path.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--dev-count", type=int, default=200)
    parser.add_argument("--test-count", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train = list(rows(args.train))
    internal_path = args.train.with_name("internal_validation.jsonl")
    internal = list(rows(internal_path)) if internal_path.exists() else []
    profile = Counter((group(row["tool"]), row["label"]) for row in train)
    sources = {split: list(rows(args.eval_root / split / "pairs.jsonl"))
               for split in ("dev", "test")}
    validate_splits(train + internal, sources["dev"], sources["test"])
    args.output.mkdir(parents=True)
    report = {"schema_version": "ta_glm52_stratified_pairs_v2",
              "selection": "tool group and GLM weak label; deterministic stratified view",
              "label_status": "glm-5.2_weak_supervision_not_human_gold",
              "train_tool_profile": dict(Counter(tool for tool, _ in profile.elements())),
              "train_tool_label_profile": {f"{tool}|{label}": n
                                           for (tool, label), n in sorted(profile.items())},
              "train_pair_positive_rate":
              sum(row["label"] for row in train) / len(train), "splits": {},
              "input_sha256": {"train": sha256(args.train),
                               "internal_validation": sha256(internal_path) if internal else None}}
    for split, count in (("dev", args.dev_count), ("test", args.test_count)):
        path = args.eval_root / split / "pairs.jsonl"
        source = sources[split]
        selected = choose(source, profile, count, args.seed + (0 if split == "dev" else 1))
        out = args.output / split
        out.mkdir()
        write(out / "pairs.jsonl", selected)
        selected_profile = Counter((group(r["tool"]), r["label"]) for r in selected)
        tool_profile = Counter(tool for tool, _ in selected_profile.elements())
        train_tools = Counter(tool for tool, _ in profile.elements())
        report["splits"][split] = {"pairs": len(selected),
                                    "trajectories": len({r["trajectory_id"] for r in selected}),
                                    "positive_rate": sum(r["label"] for r in selected) / len(selected),
                                    "tool_l1_distance": sum(abs(tool_profile[t] / len(selected)
                                        - train_tools[t] / len(train))
                                        for t in train_tools.keys() | tool_profile.keys()),
                                    "tool_label_l1_distance": sum(abs(selected_profile[k] / len(selected)
                                        - profile[k] / len(train))
                                        for k in profile.keys() | selected_profile.keys()),
                                    "labels": dict(Counter(r["label"] for r in selected)),
                                    "tools": dict(Counter(group(r["tool"]) for r in selected)),
                                    "tool_label": {f"{tool}|{label}": n for (tool, label), n
                                                   in sorted(selected_profile.items())}}
        report["input_sha256"][split] = sha256(path)
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
