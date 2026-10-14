#!/usr/bin/env python3
"""Create a new dev qrels version from explicitly reviewed equivalent Actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_bge_clawgym import load_split, rows, sha256

DATASET = Path("data/clawgym/bge_dataset_split_20260930_v2_exec_supervision_20261012")
QRELS = Path("data/clawgym/bge_dev_qrels_exact_text_20261009/qrels_review_closed.jsonl")


def additions_from_reviews(reviews, queries, actions, known):
    query_ids = {row["query_id"] for row in queries}
    action_ids = {row["action_id"] for row in actions}
    additions = []
    seen = set()
    for review in reviews:
        qid = review["query_id"]
        if qid not in query_ids:
            raise ValueError(f"unknown query {qid}")
        label = review.get("review_label")
        proposed = review.get("equivalent_action_ids", [])
        if label == "equivalent":
            if not review.get("review_reason") or not proposed:
                raise ValueError(f"equivalent review needs a reason and Action IDs: {qid}")
            for aid in proposed:
                if aid not in action_ids or (qid, aid) in seen or aid in known[qid]:
                    raise ValueError(f"invalid or duplicate equivalent Action {qid}/{aid}")
                seen.add((qid, aid))
                additions.append({"query_id": qid, "action_id": aid, "relevance": 1,
                                  "source_label": "manual_audit_equivalent",
                                  "supervision_tier": "dev_reviewed_equivalent",
                                  "review_reason": review["review_reason"]})
        elif proposed:
            raise ValueError(f"unreviewed Action IDs cannot be added: {qid}")
    return sorted(additions, key=lambda row: (row["query_id"], row["action_id"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--qrels", type=Path, default=QRELS)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    queries, actions, known = load_split(args.dataset, "dev", qrels_path=args.qrels)
    original = list(rows(args.qrels))
    added = additions_from_reviews(rows(args.reviews), queries, actions, known)
    if not added:
        raise ValueError("no manually reviewed equivalents to add")
    args.output.mkdir(parents=True)
    qrels_path = args.output / "qrels_reviewed.jsonl"
    with qrels_path.open("w", encoding="utf-8") as stream:
        for row in sorted(original + added, key=lambda item: (item["query_id"], item["action_id"])):
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {"source_qrels": str(args.qrels), "source_sha256": sha256(args.qrels),
              "reviews": str(args.reviews), "reviews_sha256": sha256(args.reviews),
              "added_edges": len(added), "affected_queries": len({row["query_id"] for row in added}),
              "output_sha256": sha256(qrels_path), "dev_only": True}
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
