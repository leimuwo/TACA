#!/usr/bin/env python3
"""Merge targeted pair reviews into a new auditable uncertainty resolution."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from evaluate_bge_clawgym import rows, sha256

POSITIVE_SCOPE_RISK = re.compile(
    r"\b(?:ambiguous|vague|unspecified|unclear|plural|multi-file|"
    r"not specify|does not specify|not named|inferred from (?:the )?action|"
    r"which files?|which pages?|all files)\b", re.I)


def merge(window, resolution, original, targeted):
    result = {**resolution, "cleared_pair_ids": list(resolution["cleared_pair_ids"]),
              "blocked_pair_ids": list(resolution["blocked_pair_ids"])}
    audit = []
    if targeted is None or targeted.get("review_failed"):
        return {**result, "targeted_audit": audit}
    old_pairs = {p["pair_id"]: p for p in original["verdict"]["pairs"]}
    new_pairs = {p["pair_id"]: p for p in targeted["verdict"]["pairs"]}
    blocked = set(result["blocked_pair_ids"])
    if set(new_pairs) != blocked or len(targeted["verdict"]["pairs"]) != len(blocked):
        raise ValueError(f"targeted pair IDs mismatch: {window['window_id']}")
    by_intent = {i["intent_id"]: i for i in original["verdict"]["intents"]}
    source_intents = {i["intent_id"]: i for i in window["intents"]}
    pair_intents = {p["pair_id"]: p["intent_id"] for p in window["pairs"]}
    for pid in sorted(blocked):
        new = new_pairs[pid]
        old = old_pairs[pid]
        iid = pair_intents[pid]
        source = source_intents[iid]
        intent = by_intent[iid]
        expected = "covered" if old["covered"] else "not_covered"
        scope_risk = old["covered"] and (
            POSITIVE_SCOPE_RISK.search(new["reason"]) or any(
                note["disposition"] in {"intent_isolated", "unresolved_window"}
                and (note["disposition"] == "unresolved_window"
                     or note.get("intent_id") == iid)
                and POSITIVE_SCOPE_RISK.search(note["text"])
                for note in resolution.get("notes", [])))
        if not (intent["due_now"] and intent["assessable"] and
                source["temporal_scope"] == "current"):
            status = "intent_not_eligible"
        elif new["decision"] == "uncertain":
            status = "still_uncertain"
        elif not new["evidence_sufficient"] or new["remaining_doubt"].strip():
            status = "insufficient_evidence"
        elif new["decision"] != expected:
            status = "label_conflict"
        elif scope_risk:
            status = "ambiguous_positive"
        else:
            status = "released"
            blocked.remove(pid)
        audit.append({"pair_id": pid, "status": status,
                      "original_covered": old["covered"],
                      "targeted_decision": new["decision"],
                      "reason": new["reason"],
                      "remaining_doubt": new["remaining_doubt"]})
    result["blocked_pair_ids"] = sorted(blocked)
    result["cleared_pair_ids"] = sorted(set(result["cleared_pair_ids"]) | (set(new_pairs) - blocked))
    result["targeted_audit"] = audit
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--resolution", type=Path, required=True)
    parser.add_argument("--review", type=Path, action="append", required=True)
    parser.add_argument("--targeted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    windows = {w["window_id"]: w for w in rows(args.windows)}
    originals = {}
    for path in args.review:
        for item in rows(path):
            if not item.get("review_failed") and item.get("verdict") is not None:
                originals[item["window_id"]] = item
    targeted = {r["window_id"]: r for r in rows(args.targeted)}
    resolutions = list(rows(args.resolution))
    if set(targeted) - {r["window_id"] for r in resolutions}:
        raise ValueError("targeted review outside resolutions")
    result = [merge(windows[r["window_id"]], r, originals[r["window_id"]],
                    targeted.get(r["window_id"])) for r in resolutions]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for item in result:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    stats = Counter(a["status"] for item in result for a in item["targeted_audit"])
    report = {"reviewed_windows": len(targeted), "pair_decisions": dict(stats),
              "total_cleared_pairs": sum(len(x["cleared_pair_ids"]) for x in result),
              "total_blocked_pairs": sum(len(x["blocked_pair_ids"]) for x in result),
              "input_sha256": {"windows": sha256(args.windows),
                               "resolution": sha256(args.resolution),
                               "targeted": sha256(args.targeted),
                               "reviews": {str(p): sha256(p) for p in args.review}}}
    (args.output.parent / (args.output.stem + "_manifest.json")).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
