#!/usr/bin/env python3
"""Add train-only counterfactuals for explicit execute/read operation mismatches."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from build_bge_m3_exec_supervision import action_with_command, normalized_action
from evaluate_bge_clawgym import sha256
from train_bge_clawgym import load_train
from train_nightowl_progressive import action_fields

DATASET = Path("data/clawgym/bge_dataset_split_20260930_v2_exec_supervision_20261012")
PYTHON_SCRIPT = re.compile(r"^python(?:3|\d+\.\d+)?\s+(\S+\.py)$")
CAT_FILE = re.compile(r"^cat\s+(\S+)$")
EXECUTE_INTENT = re.compile(r"\b(?:run|rerun|re-run|execute|re-execute|invoke|test)\b", re.I)
CONTENT_INTENT = re.compile(r"\b(?:contents?|read|inspect)\b", re.I)


def operation_counterfactual(query, positive):
    tool, args = action_fields(positive)
    if tool != "exec" or not isinstance(args.get("command"), str):
        return None
    command = args["command"]
    script = PYTHON_SCRIPT.fullmatch(command)
    if (script and EXECUTE_INTENT.search(query)
            and script.group(1).rsplit("/", 1)[-1].lower() in query.lower()):
        return "list_script_instead_of_run", action_with_command(positive, "ls -l " + script.group(1))
    file = CAT_FILE.fullmatch(command)
    if (file and CONTENT_INTENT.search(query) and not any(char in file.group(1) for char in "*?[]")
            and not file.group(1).startswith("-")):
        return "list_file_instead_of_read", action_with_command(positive, "ls -l " + file.group(1))
    return None


def augment(rows):
    known = defaultdict(set)
    existing = set()
    for row in rows:
        known[row["query"]].add(normalized_action(row["pos"][0]))
        existing.update((row["query"], normalized_action(row["pos"][0]),
                         normalized_action(negative)) for negative in row["neg"])
    additions = []
    seen = set()
    for row in rows:
        query, positive = row["query"], row["pos"][0]
        pair = query, normalized_action(positive)
        if pair in seen:
            continue
        seen.add(pair)
        candidate = operation_counterfactual(query, positive)
        if not candidate:
            continue
        category, negative = candidate
        normalized_negative = normalized_action(negative)
        triple = (*pair, normalized_negative)
        if normalized_negative in known[query] or triple in existing:
            continue
        existing.add(triple)
        additions.append({"query": query, "pos": [positive], "neg": [negative],
                          "category": category, "source": "train_explicit_operation_counterfactual",
                          "review_status": "rule_checked_not_execution_verified"})
    return additions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, _ = load_train(args.dataset / "train" / "train.jsonl")
    additions = augment(rows)
    train = args.output / "train"
    train.mkdir(parents=True)
    with (train / "train.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in additions:
            stream.write(json.dumps({key: row[key] for key in ("query", "pos", "neg")},
                                    ensure_ascii=False) + "\n")
    load_train(train / "train.jsonl")
    with (args.output / "new_operation_counterfactuals.jsonl").open("w", encoding="utf-8") as stream:
        for row in additions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    for split in ("dev", "test"):
        shutil.copytree(args.dataset / split, args.output / split)
    manifest = {"schema_version": "bge_m3_operation_supervision_v1",
                "source_train_rows": len(rows), "new_rows": len(additions),
                "new_by_category": dict(Counter(row["category"] for row in additions)),
                "dev_test_unchanged": True,
                "review_limit": "rule-checked counterfactuals; no execution or human Gold review",
                "input_sha256": sha256(args.dataset / "train" / "train.jsonl"),
                "output_sha256": sha256(train / "train.jsonl")}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
