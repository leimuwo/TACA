#!/usr/bin/env python3
"""Add train-only, explicit-target counterfactuals to validated exec negatives."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_bge_clawgym import sha256
from train_bge_clawgym import load_train
from train_nightowl_progressive import FILE, RUN_SCRIPT, URL, action_fields, explicit_targets

DATASET = Path("data/clawgym/bge_dataset_split_20260930_v2_exec_augmented_v2")
NUMERIC_FLAG = re.compile(r"(--[\w-]+)[ =](\d+)")


def action_with_command(text, command):
    payload = json.loads(text.removeprefix("[ACTION] "))
    payload["args"]["command"] = command
    return "[ACTION] " + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalized_action(text):
    payload = json.loads(text.removeprefix("[ACTION] "))
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def target_inventory(rows):
    inventory = defaultdict(set)
    for row in rows:
        tool, args = action_fields(row["pos"][0])
        if tool != "exec" or not isinstance(args.get("command"), str):
            continue
        command = args["command"]
        match = RUN_SCRIPT.search(command)
        if match:
            inventory["script"].add(match.group(1))
        for value in URL.findall(command):
            inventory["url"].add(value.strip(".,;"))
        for value in FILE.findall(command):
            inventory["file:" + value.rsplit(".", 1)[-1].lower()].add(value.rsplit("/", 1)[-1])
        for flag, value in NUMERIC_FLAG.findall(command):
            inventory["flag:" + flag].add(value)
    return {key: sorted(values) for key, values in inventory.items()}


def choose_decoy(values, query, original, key):
    eligible = [value for value in values if value != original and value.lower() not in query.lower()]
    if not eligible:
        return None
    digest = hashlib.sha256((query + "\0" + key).encode("utf-8")).digest()
    return eligible[int.from_bytes(digest[:8], "big") % len(eligible)]


def counterfactuals(query, positive, inventory):
    """Change one explicitly requested command target; never use an unjudged Action as a label."""
    tool, args = action_fields(positive)
    if tool != "exec" or not isinstance(args.get("command"), str):
        return []
    command = args["command"]
    proposals = []
    targets = explicit_targets(query, positive)
    script_names = {target for kind, target in targets if kind == "script"}
    if len(script_names) == 1:
        original = next(iter(script_names))
        decoy = choose_decoy(inventory.get("script", []), query, original, "script")
        match = RUN_SCRIPT.search(command)
        if decoy and match and match.group(1).lower() == original:
            start, end = match.span(1)
            proposals.append(("wrong_script", command[:start] + decoy + command[end:], original, decoy))
    urls = [target for kind, target in targets if kind == "command_url"]
    if len(set(urls)) == 1 and len(URL.findall(query)) == 1:
        original = urls[0]
        decoy = choose_decoy(inventory.get("url", []), query, original, "url")
        if decoy and command.count(original) == 1:
            proposals.append(("wrong_url", command.replace(original, decoy, 1), original, decoy))
    files = [target for kind, target in targets if kind == "command_file"]
    if len(set(files)) == 1 and len(set(FILE.findall(query))) == 1:
        original = files[0]
        if not (original.endswith(".py") and RUN_SCRIPT.search(command)):
            extension = original.rsplit(".", 1)[-1].lower()
            decoy = choose_decoy(inventory.get("file:" + extension, []), query,
                                 original.rsplit("/", 1)[-1], "file")
            if decoy and command.count(original) == 1:
                parent = original.rsplit("/", 1)[0]
                replacement = parent + "/" + decoy
                proposals.append(("wrong_file", command.replace(original, replacement, 1),
                                  original, replacement))
    requested = [(flag, value) for flag, value in NUMERIC_FLAG.findall(query)
                 if re.search(re.escape(flag) + r"[ =]" + re.escape(value) + r"\b", command)]
    if len(requested) == 1:
        flag, original = requested[0]
        decoy = choose_decoy(inventory.get("flag:" + flag, []), query, original, flag)
        pattern = re.compile(re.escape(flag) + r"([ =])" + re.escape(original) + r"\b")
        if decoy and len(pattern.findall(command)) == 1:
            altered = pattern.sub(lambda match: flag + match.group(1) + decoy, command, count=1)
            proposals.append(("wrong_parameter", altered, flag + " " + original, flag + " " + decoy))
    return [(kind, action_with_command(positive, altered), original, decoy)
            for kind, altered, original, decoy in proposals if altered != command]


def augment(rows, max_per_pair):
    inventory = target_inventory(rows)
    positives = defaultdict(set)
    existing = set()
    for row in rows:
        positives[row["query"]].add(normalized_action(row["pos"][0]))
        existing.update((row["query"], normalized_action(row["pos"][0]),
                         normalized_action(negative)) for negative in row["neg"])
    additions = []
    seen_pairs = set()
    for row in rows:
        query, positive = row["query"], row["pos"][0]
        pair = (query, normalized_action(positive))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        count = 0
        for kind, negative, original, decoy in counterfactuals(query, positive, inventory):
            normalized_negative = normalized_action(negative)
            triple = (*pair, normalized_negative)
            if normalized_negative in positives[query] or triple in existing:
                continue
            existing.add(triple)
            additions.append({"query": query, "pos": [positive], "neg": [negative],
                              "category": kind, "original_target": original,
                              "replacement_target": decoy,
                              "source": "train_only_explicit_target_counterfactual"})
            count += 1
            if count >= max_per_pair:
                break
    return additions


def run(args):
    if args.max_per_pair < 1 or args.output.exists():
        raise ValueError("max_per_pair must be positive and output must not exist")
    train_path = args.dataset / "train" / "train.jsonl"
    rows, _ = load_train(train_path)
    additions = augment(rows, args.max_per_pair)
    output_train = args.output / "train"
    output_train.mkdir(parents=True)
    with (output_train / "train.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        for item in additions:
            stream.write(json.dumps({key: item[key] for key in ("query", "pos", "neg")},
                                    ensure_ascii=False) + "\n")
    load_train(output_train / "train.jsonl")
    with (args.output / "new_exec_counterfactuals.jsonl").open("w", encoding="utf-8") as stream:
        for item in additions:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    for split in ("dev", "test"):
        shutil.copytree(args.dataset / split, args.output / split)
    manifest = {"schema_version": "bge_m3_exec_supervision_v1",
                "source_train_rows": len(rows), "new_exec_rows": len(additions),
                "new_by_category": dict(Counter(item["category"] for item in additions)),
                "affected_queries": len({item["query"] for item in additions}),
                "dev_test_unchanged": True, "max_per_pair": args.max_per_pair,
                "input_sha256": {"train": sha256(train_path)},
                "output_sha256": {"train": sha256(output_train / "train.jsonl")}}
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-pair", type=int, default=3)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
