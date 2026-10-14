#!/usr/bin/env python3
"""Close Silver qrels under byte-identical Action text and audit same-target Top-1s."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


REVIEW = {
    "16aea9af6c543569bf24aa92": ("partial_or_intermediate", "Top-1 adds a --dir argument, while the known Action removes environment-variable based path setup; neither edit alone establishes the complete transition."),
    "1782ef9ef4b4ac8d4320f4ed": ("different_content_needs_validation", "The earlier same-path list differs, including name/slug and other categories; validity and the requested 40 Functional names need a full content check."),
    "3845d7c325481b2f190da302": ("intent_missing_context", "Same path, but one summary concerns injection scans and the other a ship archive audit. Intent gives no subject."),
    "4bdd26fa5afe0f1e8f971fcd": ("wrong_read_range", "Intent explicitly requests line 245; Top-1 reads the file start without offset."),
    "8f7551867dced4163f2fd5b5": ("possible_equivalent_needs_source_check", "Same FDIC rates and path; Top-1 appends '| FDIC.gov' to source_title. Intent does not specify the exact title, but source requirements must be checked."),
    "b4e6e7c7db2750d2e12f493c": ("wrong_read_range", "Top-1 reads the file start without offset, whereas the known Action reads the main-function section starting at line 195."),
    "ed69bc0e6128905495c9d75c": ("intent_missing_context", "Same agenda path but different schedules; Intent gives no event constraints to select one version."),
    "ee4719f67ef250c6c5beeceb": ("different_content_needs_validation", "Same organization CSV path, but Top-1 changes event data and adds an organization; factual grounding requires source review."),
    "f3773630abcfdb9bc6b5f858": ("intent_missing_context", "Same run-log path, but two unrelated log contents; Intent does not identify which run."),
}


def read(path):
    with path.open(encoding="utf8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, rows):
    with path.open("w", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def target(action):
    args = json.loads(action.removeprefix("[ACTION] "))["args"]
    value = args.get("path") or args.get("file")
    return value.replace("~/.openclaw/workspace/", "/root/.openclaw/workspace/").removeprefix("/root/.openclaw/workspace/") if isinstance(value, str) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actions = {row["action_id"]: row for row in read(args.test / "actions.jsonl")}
    queries = {row["query_id"]: row for row in read(args.test / "queries.jsonl")}
    originals = read(args.test / "qrels.jsonl")
    ranked = read(args.rankings)
    by_text = collections.defaultdict(set)
    by_query = collections.defaultdict(set)
    for aid, row in actions.items():
        by_text[row["action"]].add(aid)
    for row in originals:
        by_query[row["query_id"]].add(row["action_id"])
    additions = {(qid, equivalent)
                 for qid, ids in by_query.items()
                 for aid in ids
                 for equivalent in by_text[actions[aid]["action"]]
                 if equivalent not in ids}
    repaired = list(originals) + [
        {"query_id": qid, "action_id": aid, "relevance": 1,
         "source_label": "exact_action_text_equivalent", "supervision_tier": "deterministic_duplicate_closure"}
        for qid, aid in sorted(additions)]
    repaired.sort(key=lambda row: (row["query_id"], row["action_id"]))
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "qrels_exact_text_closed.jsonl", repaired)
    fixed = []
    same_target = []
    for row in ranked:
        if row["known_positive_rank"] == 1:
            continue
        qid = row["query_id"]
        top = row["top_actions"][0]["action_id"]
        positives = sorted(by_query[qid])
        top_action = actions[top]["action"]
        if any(top_action == actions[aid]["action"] for aid in positives):
            fixed.append({"query_id": qid, "intent": queries[qid]["query"],
                          "top_action_id": top, "known_positive_action_ids": positives,
                          "original_known_positive_rank": row["known_positive_rank"],
                          "action_sha256": hashlib.sha256(top_action.encode()).hexdigest(),
                          "judgment": "same_serialized_action_as_known_positive"})
            continue
        top_fn = json.loads(top_action.removeprefix("[ACTION] "))["function"]
        if top_fn not in ("read", "write", "edit") or target(top_action) is None:
            continue
        matching = [aid for aid in positives
                    if json.loads(actions[aid]["action"].removeprefix("[ACTION] "))["function"] == top_fn
                    and target(actions[aid]["action"]) == target(top_action)]
        if not matching:
            continue
        if qid not in REVIEW:
            raise ValueError(f"new same-target case needs review: {qid}")
        judgment, reason = REVIEW[qid]
        same_target.append({"query_id": qid, "intent": queries[qid]["query"],
                            "trajectory_id": queries[qid]["trajectory_id"],
                            "thought_step": queries[qid]["thought_step"],
                            "original_known_positive_rank": row["known_positive_rank"],
                            "top_action_id": top, "top_action": top_action,
                            "matching_known_positive_action_ids": matching,
                            "matching_known_positive_actions": [actions[aid]["action"] for aid in matching],
                            "judgment": judgment, "reason": reason,
                            "add_to_qrels": False})
    if len(fixed) != 14 or len(same_target) != len(REVIEW):
        raise ValueError(f"unexpected case counts: {len(fixed)} exact, {len(same_target)} same-target")
    write(args.output / "fixed_top1_exact_duplicates.jsonl", sorted(fixed, key=lambda row: row["query_id"]))
    write(args.output / "reviewed_same_target_top1.jsonl", sorted(same_target, key=lambda row: row["query_id"]))
    manifest = {"schema_version": "bge_test_exact_action_duplicate_closure_v1",
                "original_qrels": len(originals), "added_equivalent_edges": len(additions),
                "augmented_qrels": len(repaired), "affected_queries": len({qid for qid, _ in additions}),
                "clear_top1_fixed": len(fixed), "same_target_reviewed": len(same_target),
                "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for name, path in {"qrels": args.test / "qrels.jsonl",
                                                     "actions": args.test / "actions.jsonl",
                                                     "rankings": args.rankings}.items()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
