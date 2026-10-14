#!/usr/bin/env python3
"""Filter ambiguous training Intent texts without rewriting them."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_bge_clawgym import rows, sha256
from train_bge_clawgym import load_train

DATASET = Path("data/clawgym/bge_dataset_split_20260930_v2_exec_constraints_20261012")
SCREEN_INPUT = Path("../data/clawgym/ambiguous/screen_input.jsonl")
DECISIONS = Path("../data/clawgym/ambiguous/decisions.jsonl")


def classify_intents(screen_input, decisions):
    by_id = {}
    for row in decisions:
        if row["query_id"] in by_id or not isinstance(row.get("ambiguous"), bool):
            raise ValueError("duplicate or invalid ambiguity decision")
        by_id[row["query_id"]] = row["ambiguous"]
    by_text = defaultdict(set)
    for row in screen_input:
        if row["split"] != "train":
            continue
        if row["query_id"] not in by_id:
            raise ValueError(f"missing screen decision: {row['query_id']}")
        by_text["[INTENT] " + row["intent"]].add(by_id[row["query_id"]])
    return by_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--screen-input", type=Path, default=SCREEN_INPUT)
    parser.add_argument("--decisions", type=Path, default=DECISIONS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train_path = args.dataset / "train" / "train.jsonl"
    train, _ = load_train(train_path)
    decisions = classify_intents(rows(args.screen_input), rows(args.decisions))
    absent = sorted({row["query"] for row in train} - set(decisions))
    kept = [row for row in train if decisions.get(row["query"]) == {False}]
    removed = [row for row in train if decisions.get(row["query"]) != {False}]
    folder = args.output / "train"
    folder.mkdir(parents=True)
    with (folder / "train.jsonl").open("w", encoding="utf-8") as stream:
        for row in kept:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    load_train(folder / "train.jsonl")
    for split in ("dev", "test"):
        shutil.copytree(args.dataset / split, args.output / split)
    exec_rows = sum(json.loads(row["pos"][0].removeprefix("[ACTION] "))["function"] == "exec"
                    for row in kept)
    report = {"schema_version": "bge_m3_clear_train_v1", "source_rows": len(train),
              "kept_rows": len(kept), "removed_rows": len(removed),
              "removed_by_reason": dict(Counter(
                  "unscreened" if row["query"] not in decisions else
                  "ambiguous" if True in decisions[row["query"]] else "inconsistent_decisions"
                  for row in removed)),
              "unscreened_unique_intents": len(absent),
              "exec_positive_rows": exec_rows,
              "unique_exec_positive_pairs": len({(row["query"], row["pos"][0]) for row in kept
                                                 if json.loads(row["pos"][0].removeprefix("[ACTION] "))["function"] == "exec"}),
              "policy": "ambiguous or conflicting Intent texts excluded; no text completion",
              "dev_test_unchanged": True,
              "input_sha256": {"train": sha256(train_path), "screen_input": sha256(args.screen_input),
                               "decisions": sha256(args.decisions)},
              "output_sha256": sha256(folder / "train.jsonl")}
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
