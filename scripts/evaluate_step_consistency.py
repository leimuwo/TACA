#!/usr/bin/env python3
"""Create a step-level review queue or score adjudicated three-class labels."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluate_bge_clawgym import rows

DATASET = Path("data/clawgym/bge_dataset_split_20260930_v2/dev")
CLEAR = Path("data/clawgym/bge_m3_clear_dev_20261013/intent_clear.jsonl")
RANKINGS = Path("outputs/bge_m3_dual_tower_v1/best_dev_rankings.jsonl")
THOUGHTS = Path("data/clawgym/stage1_intents_blind_complete_925_v1/intents_blind.jsonl")
CLASSES = ("faithful", "partial", "not_executed")


def make_queue(queries, actions, rankings, thoughts=None):
    by_query = {row["query_id"]: row for row in queries}
    by_action = {row["action_id"]: row for row in actions}
    by_step = {(row["trajectory_id"], row["thought_step"]): row for row in (thoughts or [])}
    output = []
    for ranking in rankings:
        qid = ranking["query_id"]
        if qid not in by_query:
            continue
        item = ranking["top_actions"][0]
        action = by_action[item["action_id"]]
        query = by_query[qid]
        context = by_step.get((query["trajectory_id"], query["thought_step"]), {})
        output.append({"query_id": qid, "intent": by_query[qid]["query"],
                       "action_id": item["action_id"], "action": action["action"],
                       "action_tool": json.loads(action["action"].removeprefix("[ACTION] "))["function"],
                       "trajectory_id": by_query[qid]["trajectory_id"],
                       "thought_step": query["thought_step"],
                       "thought": context.get("thought"),
                       "user_request": context.get("user_request"),
                       "extracted_step_intents": [
                           {"intent_id": value.get("intent_id"), "intent_text": value.get("intent_text"),
                            "temporal_scope": value.get("temporal_scope"),
                            "source_quote": value.get("source_quote")}
                           for value in context.get("execution_intents", [])],
                       "due_now_review": None,
                       "candidate_trajectory_id": action["trajectory_id"],
                       "score": item["score"], "is_known_positive": item["is_known_positive"],
                       "gold_label": None, "predicted_label": None, "review_reason": None})
    if len(output) != len(by_query):
        raise ValueError("rankings must cover all clear queries")
    return output


def score(reviewed, *, require_review=True):
    if not reviewed:
        raise ValueError("empty review file")
    matrix = {gold: {pred: 0 for pred in CLASSES} for gold in CLASSES}
    by_tool = Counter()
    by_tool_correct = Counter()
    seen = set()
    for row in reviewed:
        pair = row["query_id"], row["action_id"]
        if pair in seen:
            raise ValueError(f"duplicate reviewed pair {pair}")
        seen.add(pair)
        gold, pred = row.get("gold_label"), row.get("predicted_label")
        if gold not in CLASSES or pred not in CLASSES:
            raise ValueError(f"both three-class labels are required: {pair}")
        if require_review and (not row.get("review_reason") or row.get("due_now_review") is not True):
            raise ValueError(f"due-now confirmation and a review reason are required: {pair}")
        matrix[gold][pred] += 1
        by_tool[row["action_tool"]] += 1
        by_tool_correct[row["action_tool"]] += gold == pred
    n = len(reviewed)
    per_class = {}
    for label in CLASSES:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in CLASSES if other != label)
        fn = sum(matrix[label][other] for other in CLASSES if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {"precision": precision, "recall": recall,
                            "f1": 2 * precision * recall / (precision + recall)
                            if precision + recall else 0.0,
                            "support": sum(matrix[label].values())}
    return {"reviewed_pairs": n, "accuracy": sum(matrix[x][x] for x in CLASSES) / n,
            "macro_f1": sum(value["f1"] for value in per_class.values()) / len(CLASSES),
            "per_class": per_class, "confusion_matrix": matrix,
            "by_action_tool": {tool: {"pairs": count, "accuracy": by_tool_correct[tool] / count}
                               for tool, count in sorted(by_tool.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=CLEAR)
    parser.add_argument("--actions", type=Path, default=DATASET / "actions.jsonl")
    parser.add_argument("--rankings", type=Path, default=RANKINGS)
    parser.add_argument("--thoughts", type=Path, default=THOUGHTS)
    parser.add_argument("--reviews", type=Path, help="Completed three-class reviews to score")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.reviews:
        report = score(list(rows(args.reviews)))
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
    else:
        queue = make_queue(list(rows(args.queries)), list(rows(args.actions)),
                           list(rows(args.rankings)), list(rows(args.thoughts)))
        with args.output.open("w", encoding="utf-8") as stream:
            for row in queue:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"review_queue": str(args.output), "pairs": len(queue),
                          "metrics_ready": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
