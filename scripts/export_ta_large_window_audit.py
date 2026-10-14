#!/usr/bin/env python3
"""Export all large TA windows for Intent atomicity review without pruning pairs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluate_bge_clawgym import rows, sha256


def select_large(windows, reviews, threshold):
    by_id = {review["window_id"]: review for review in reviews}
    if len(by_id) != len(reviews):
        raise ValueError("duplicate model review IDs; use the latest review for each window")
    selected = []
    for window in windows:
        if len(window["pairs"]) <= threshold:
            continue
        selected.append({"window_id": window["window_id"], "trajectory_id": window["trajectory_id"],
                         "thought_step": window["thought_step"], "pair_count": len(window["pairs"]),
                         "thought": window["thought"], "user_request": window["user_request"],
                         "intents": window["intents"], "actions": window["actions"],
                         "pairs": window["pairs"], "model_review": by_id.get(window["window_id"]),
                         "atomicity_review": None, "assessability_review": None,
                         "temporal_review": None, "review_notes": None})
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--model-reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or args.threshold < 1:
        raise ValueError("output must not exist and threshold must be positive")
    windows = list(rows(args.windows))
    reviews = {r["window_id"]: r for r in rows(args.model_reviews)}
    selected = select_large(windows, list(reviews.values()), args.threshold)
    args.output.mkdir(parents=True)
    with (args.output / "review_queue.jsonl").open("w", encoding="utf-8") as stream:
        for item in selected:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    lines = ["# Large-window Intent atomicity review", "", "This is a review queue; no pair labels are changed.", ""]
    for item in selected:
        lines += [f"## {item['window_id']}", "",
                  f"Trajectory: `{item['trajectory_id']}`; step: {item['thought_step']}; pairs: {item['pair_count']}",
                  "", "### Thought", "", "```text", item["thought"], "```", "", "### Intents", ""]
        for intent in item["intents"]:
            lines.append(f"- `{intent['source_intent_id']}` [{intent['temporal_scope']}]: {intent['text']}")
        lines += ["", "### Actions and results", ""]
        for action in item["actions"]:
            lines += [f"**{action['action_id']}**", "", "```text", action["action_text"], "```",
                      "", "Result:", "", "```text", "\n".join(map(str, action["execution_result"])), "```", ""]
        review = item["model_review"] or {}
        verdict = review.get("verdict") or {}
        lines += [f"GQQ: `{verdict.get('suggested_ta_label', 'failed')}`; "
                  f"aggregation conflict: `{verdict.get('aggregation_disagrees', False)}`", ""]
    (args.output / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {"schema_version": "ta_large_window_audit_v1", "threshold": args.threshold,
              "windows": len(selected), "pairs_preserved": sum(x["pair_count"] for x in selected),
              "model_labels": dict(Counter((x["model_review"] or {}).get("verdict", {}).get("suggested_ta_label", "failed")
                                           if (x["model_review"] or {}).get("verdict") else "failed"
                                           for x in selected)),
              "source_sha256": {"windows": sha256(args.windows), "model_reviews": sha256(args.model_reviews)}}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
