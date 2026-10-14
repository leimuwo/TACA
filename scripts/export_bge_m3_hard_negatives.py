#!/usr/bin/env python3
"""Group existing explicit hard negatives and expose supervision gaps."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_bge_clawgym import rows, sha256
from train_bge_clawgym import load_train

DATASET = Path("data/clawgym/bge_dataset_split_20260930_v2_clear_exec_constraints_20261013")


def classify(positive, negative):
    left = json.loads(positive.removeprefix("[ACTION] "))
    right = json.loads(negative.removeprefix("[ACTION] "))
    if left["function"] != right["function"]:
        return "different_operation_or_tool"
    largs, rargs = left.get("args", {}), right.get("args", {})
    if any(largs.get(field) != rargs.get(field) for field in
           ("path", "file", "file_path", "url", "targetUrl", "command") if field in largs or field in rargs):
        return "same_tool_different_target_or_command"
    return "same_tool_different_parameters_or_content"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train_path = args.dataset / "train" / "train.jsonl"
    train, _ = load_train(train_path)
    grouped = defaultdict(set)
    for row in train:
        grouped[(row["query"], row["pos"][0])].update(row["neg"])
    args.output.mkdir(parents=True)
    counts = Counter()
    with (args.output / "hard_negatives.jsonl").open("w", encoding="utf-8") as stream:
        for (query, positive), negatives in sorted(grouped.items()):
            labels = [{"action": negative, "category": classify(positive, negative),
                       "source": "existing_explicit_train_negative"}
                      for negative in sorted(negatives)]
            counts["groups"] += 1
            counts["negative_pairs"] += len(labels)
            counts["groups_below_3"] += len(labels) < 3
            for item in labels:
                counts[item["category"]] += 1
            stream.write(json.dumps({"query": query, "positive_action": positive,
                                     "negative_actions": labels,
                                     "review_status": "inherited_silver_or_rule_checked"},
                                    ensure_ascii=False) + "\n")
    report = {"counts": dict(counts), "train_sha256": sha256(train_path),
              "policy": "exports only existing explicit train negatives; no unjudged candidate promoted",
              "limit": "categories describe serialized differences, not adjudicated error causes"}
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
