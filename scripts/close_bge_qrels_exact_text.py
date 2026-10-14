#!/usr/bin/env python3
"""Add positive qrels for candidate Actions with identical serialized text."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    queries = read_jsonl(args.split_dir / "queries.jsonl")
    actions = read_jsonl(args.split_dir / "actions.jsonl")
    originals = read_jsonl(args.split_dir / "qrels.jsonl")
    query_ids = {row["query_id"] for row in queries}
    action_by_id = {row["action_id"]: row["action"] for row in actions}
    if len(query_ids) != len(queries) or len(action_by_id) != len(actions):
        raise ValueError("duplicate query or action ID")
    by_text = defaultdict(set)
    for action_id, text in action_by_id.items():
        by_text[text].add(action_id)
    known = defaultdict(set)
    for row in originals:
        qid, aid = row["query_id"], row["action_id"]
        if qid not in query_ids or aid not in action_by_id or row.get("relevance") != 1:
            raise ValueError("invalid source qrel")
        known[qid].add(aid)
    additions = sorted({(qid, equivalent)
                        for qid, positives in known.items()
                        for positive in positives
                        for equivalent in by_text[action_by_id[positive]]
                        if equivalent not in positives})
    augmented = list(originals) + [
        {"query_id": qid, "action_id": aid, "relevance": 1,
         "source_label": "exact_action_text_equivalent",
         "supervision_tier": "deterministic_duplicate_closure"}
        for qid, aid in additions
    ]
    augmented.sort(key=lambda row: (row["query_id"], row["action_id"]))
    args.output_dir.mkdir(parents=True)
    target = args.output_dir / "qrels_exact_text_closed.jsonl"
    with target.open("w", encoding="utf-8") as stream:
        for row in augmented:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "bge_exact_action_text_qrels_v1",
        "source_split": str(args.split_dir),
        "queries": len(queries), "candidate_actions": len(actions),
        "original_positive_edges": len(originals),
        "added_equivalent_edges": len(additions),
        "affected_queries": len({qid for qid, _ in additions}),
        "augmented_positive_edges": len(augmented),
        "source_sha256": {name: sha256(args.split_dir / f"{name}.jsonl")
                          for name in ("queries", "actions", "qrels")},
        "output_sha256": sha256(target),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
