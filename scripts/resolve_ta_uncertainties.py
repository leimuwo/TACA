#!/usr/bin/env python3
"""Conservatively resolve free-text GQQ uncertainty notes for TA pair training."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from evaluate_bge_clawgym import rows, sha256

NO_UNCERTAINTY = re.compile(r"^no uncertaint(?:y|ies)\b", re.I)
NO_EFFECT = re.compile(
    r"\b(?:does not affect (?:semantic )?coverage|"
    r"does not affect (?:the )?(?:coverage|assessment)|"
    r"no significant uncertainty)\b", re.I)
RESIDUAL_RISK = re.compile(
    r"\b(?:unclear|may|might|likely|minor uncertainty|malformed|truncated|placeholder|differ|"
    r"discrepancy|inferred|does not specify|not present|not reveal)\b", re.I)
PREPARATORY = re.compile(r"\b(?:preliminary|preparatory|prerequisite|first step)\b", re.I)
DEFINITE_NON_COVERAGE = re.compile(
    r"\b(?:does not (?:itself )?(?:cover|fulfill|perform|create|write|implement)|"
    r"clearly (?:not covered|uncovered)|not (?:done|performed) by this action)\b", re.I)
QUOTED = re.compile(r"['\"]([^'\"]{10,})['\"]")
ORDINAL_INTENT = re.compile(r"\b(first|second|third|fourth|fifth) intent\b", re.I)


def mentioned_intent(window, note):
    intents = window["intents"]
    low = note.casefold()
    exact = [i["intent_id"] for i in intents
             if len(i["text"]) >= 12 and i["text"].casefold() in low]
    if len(exact) == 1:
        return exact[0]
    quoted = [m.group(1).casefold() for m in QUOTED.finditer(note)]
    matches = [i["intent_id"] for i in intents
               if any(phrase in i["text"].casefold() for phrase in quoted)]
    if len(matches) == 1:
        return matches[0]
    ordinal = ORDINAL_INTENT.search(note)
    if ordinal:
        index = ("first", "second", "third", "fourth", "fifth").index(ordinal.group(1).lower())
        if index < len(intents):
            return intents[index]["intent_id"]
    return None


def resolve(window, verdict):
    pair_ids = {p["pair_id"] for p in window["pairs"]}
    decisions = {p["pair_id"]: p for p in verdict["pairs"]}
    if pair_ids != set(decisions):
        raise ValueError(f"review pairs do not match window: {window['window_id']}")
    blocked = set()
    notes = []
    for raw_note in verdict.get("uncertainties") or []:
        note = " ".join(raw_note.split())
        if NO_UNCERTAINTY.match(note):
            notes.append({"text": raw_note, "disposition": "field_pollution"})
            continue
        if NO_EFFECT.search(note) and not RESIDUAL_RISK.search(note):
            notes.append({"text": raw_note, "disposition": "conclusion_unaffected"})
            continue
        if (PREPARATORY.search(note) and DEFINITE_NON_COVERAGE.search(note)
                and all(not p["covered"] for p in decisions.values())):
            notes.append({"text": raw_note, "disposition": "non_coverage_unchanged"})
            continue
        intent_id = mentioned_intent(window, note)
        if intent_id is not None:
            affected = {p["pair_id"] for p in window["pairs"]
                        if p["intent_id"] == intent_id}
            blocked.update(affected)
            notes.append({"text": raw_note, "disposition": "intent_isolated",
                          "intent_id": intent_id,
                          "blocked_pair_ids": sorted(affected)})
        else:
            blocked.update(pair_ids)
            notes.append({"text": raw_note, "disposition": "unresolved_window"})
    return {"window_id": window["window_id"],
            "cleared_pair_ids": sorted(pair_ids - blocked),
            "blocked_pair_ids": sorted(blocked), "notes": notes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--review", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    windows = {w["window_id"]: w for w in rows(args.windows)}
    reviews = {}
    for path in args.review:
        for item in rows(path):
            if item["window_id"] not in windows:
                raise ValueError(f"review outside windows: {item['window_id']}")
            if not item.get("review_failed") and item.get("verdict") is not None:
                reviews[item["window_id"]] = item
    resolved = [resolve(windows[wid], item["verdict"])
                for wid, item in sorted(reviews.items()) if item["verdict"].get("uncertainties")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for item in resolved:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    counts = Counter(note["disposition"] for item in resolved for note in item["notes"])
    report = {"windows": len(resolved), "notes": dict(counts),
              "cleared_pairs": sum(len(x["cleared_pair_ids"]) for x in resolved),
              "blocked_pairs": sum(len(x["blocked_pair_ids"]) for x in resolved),
              "input_sha256": {"windows": sha256(args.windows),
                               "reviews": {str(p): sha256(p) for p in args.review}}}
    (args.output.parent / (args.output.stem + "_manifest.json")).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
