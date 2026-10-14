#!/usr/bin/env python3
"""Score completed TA human Gold against independent predictions."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluate_bge_clawgym import rows
from evaluate_step_consistency import CLASSES, score
from ta_coverage_aggregation import aggregate_ta_label


def evaluate(windows, human, predictions, *, gold_source="pair-derived"):
    if gold_source not in {"pair-derived", "human-overall"}:
        raise ValueError(f"unknown Gold source: {gold_source}")
    by_window = {w["window_id"]: w for w in windows}
    by_pred = {p["window_id"]: p for p in predictions}
    if len(by_pred) != len(predictions):
        raise ValueError("duplicate prediction window")
    eligible = []
    skipped = Counter()
    seen_human = set()
    for review in human:
        wid = review["window_id"]
        if wid in seen_human:
            raise ValueError(f"duplicate human review: {wid}")
        seen_human.add(wid)
        if wid not in by_window or wid not in by_pred:
            raise ValueError(f"missing window or prediction: {wid}")
        if gold_source == "human-overall":
            label = review.get("gold_label")
            if not label:
                skipped["unlabeled_human_gold"] += 1
                continue
            if label == "unassessable":
                skipped["unassessable"] += 1
                continue
            if label not in CLASSES:
                raise ValueError(f"invalid human Gold: {wid}")
            pred = by_pred[wid]["predicted_label"]
            if pred not in CLASSES:
                skipped["unassessable_prediction"] += 1
                continue
            tools = sorted({a["raw_call"]["function"] for a in by_window[wid]["actions"]})
            eligible.append({"query_id": wid, "action_id": wid, "action_tool": "+".join(tools),
                             "gold_label": label, "predicted_label": pred})
            continue
        if not review.get("review_reason"):
            raise ValueError(f"missing human review reason: {wid}")
        window = by_window[wid]
        intent_reviews = review.get("intents", {})
        pair_reviews = review.get("pairs", {})
        if ({x["intent_id"] for x in window["intents"]} != set(intent_reviews)
                or {x["pair_id"] for x in window["pairs"]} != set(pair_reviews)
                or any(x.get("due_now") not in {"yes", "no"} or x.get("assessable") not in {"yes", "no"}
                       for x in intent_reviews.values())
                or any(value not in {"yes", "no", "uncertain"} for value in pair_reviews.values())):
            raise ValueError(f"unfinished Intent or pair review: {wid}")
        due = [x for x in window["intents"] if intent_reviews[x["intent_id"]]["due_now"] == "yes"]
        derived = aggregate_ta_label(
            window,
            {intent_id: {key: value == "yes" for key, value in decision.items()}
             for intent_id, decision in intent_reviews.items()},
            {pair_id: value == "yes" for pair_id, value in pair_reviews.items()})
        if review.get("gold_label") and review["gold_label"] != derived:
            raise ValueError(f"Gold label conflicts with pair coverage: {wid}")
        if derived == "unassessable":
            skipped["unassessable"] += 1
            continue
        if any(pair_reviews[p["pair_id"]] == "uncertain" for p in window["pairs"]
               if p["intent_id"] in {x["intent_id"] for x in due}):
            raise ValueError(f"uncertain pair in strict Gold: {wid}")
        pred = by_pred[wid]["predicted_label"]
        if pred not in CLASSES:
            skipped["unassessable_prediction"] += 1
            continue
        tools = sorted({a["raw_call"]["function"] for a in by_window[wid]["actions"]})
        eligible.append({"query_id": wid, "action_id": wid, "action_tool": "+".join(tools),
                         "gold_label": derived, "predicted_label": pred,
                         "review_reason": review["review_reason"], "due_now_review": True})
    report = score(eligible, require_review=gold_source == "pair-derived") if eligible else {
        "reviewed_pairs": 0, "metrics_ready": False}
    report["skipped"] = dict(skipped)
    report["unit"] = "reviewed_window"
    report["gold_source"] = gold_source
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--human", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold-source", choices=("pair-derived", "human-overall"),
                        default="pair-derived")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = evaluate(list(rows(args.windows)), list(rows(args.human)),
                      list(rows(args.predictions)), gold_source=args.gold_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
