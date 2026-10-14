#!/usr/bin/env python3
"""Join GQQ ambiguity decisions to BGE v2 Intent provenance."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def load(path):
    return [json.loads(line) for line in path.open(encoding="utf8") if line.strip()]


def write(path, rows):
    with path.open("w", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory
    source = {row["query_id"]: row for row in load(directory / "screen_input.jsonl")}
    decisions = load(directory / "decisions.jsonl")
    if len(decisions) != len(source) or {row["query_id"] for row in decisions} != set(source):
        raise ValueError("incomplete or duplicated GQQ decisions")
    if any(not isinstance(row.get("ambiguous"), bool) for row in decisions):
        raise ValueError("unresolved GQQ decisions")
    ambiguous = [{**source[row["query_id"]], **row} for row in decisions if row["ambiguous"]]
    ambiguous.sort(key=lambda row: (row["split"], row["trajectory_id"] or "", str(row["thought_step"]), row["query_id"]))
    write(directory / "ambiguous_all.jsonl", ambiguous)
    for split in ("train", "dev", "test"):
        write(directory / f"ambiguous_{split}.jsonl", [row for row in ambiguous if row["split"] == split])
    manifest = {"schema_version": "gqq_ambiguous_intent_screen_v1", "date": "2026-10-08",
                "model": "deepseek-v4-flash-gqq", "input_count": len(source), "decision_count": len(decisions),
                "ambiguous_count": len(ambiguous), "clear_count": len(source)-len(ambiguous),
                "counts_by_split": dict(collections.Counter(row["split"] for row in ambiguous)),
                "categories": dict(collections.Counter(row.get("category") for row in ambiguous)),
                "source_sha256": hashlib.sha256((directory / "screen_input.jsonl").read_bytes()).hexdigest(),
                "decision_sha256": hashlib.sha256((directory / "decisions.jsonl").read_bytes()).hexdigest()}
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
