#!/usr/bin/env python3
"""Rank the complete dev Action pool with an independent BGE-M3 checkpoint."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluate_bge_clawgym import embed, load_split, sha256

DATASET = Path("data/clawgym/bge_dataset_split_20260930_v2_exec_supervision_20261012")
QRELS = Path("data/clawgym/bge_dev_qrels_exact_text_20261009/qrels_review_closed.jsonl")
CHECKPOINT = Path("outputs/bge_m3_dual_tower_v1/best")
REFERENCE = Path("outputs/bge_m3_dual_tower_v1/best_dev_rankings.jsonl")


def validate_reference(rankings, reference):
    by_id = {row["query_id"]: row for row in reference}
    if len(by_id) != len(rankings):
        raise ValueError("reference query count does not match")
    for row in rankings:
        qid = row["query_id"]
        if qid not in by_id:
            raise ValueError(f"query missing from reference: {qid}")
        actual = [item["action_id"] for item in row["top_actions"][:20]]
        expected = [item["action_id"] for item in by_id[qid]["top_actions"]]
        if actual != expected:
            raise ValueError(f"recomputed Top-20 differs from saved checkpoint ranking: {qid}")


def summarize(queries, actions, qrels, rankings):
    action_by_id = {row["action_id"]: row for row in actions}
    query_by_id = {row["query_id"]: row for row in queries}
    if len(rankings) != len(queries) or {row["query_id"] for row in rankings} != set(query_by_id):
        raise ValueError("rankings must contain each dev query exactly once")
    counts = Counter()
    review = []
    for row in rankings:
        qid = row["query_id"]
        top = row["top_actions"]
        if len(top) < 100 or len({item["action_id"] for item in top}) != len(top):
            raise ValueError(f"query {qid} needs 100 distinct ranked Actions")
        positive = qrels[qid]
        rank = row["known_positive_rank"]
        if rank < 1 or (rank <= len(top) and top[rank - 1]["action_id"] not in positive):
            raise ValueError(f"invalid known-positive rank for {qid}")
        is_exec = any(json.loads(action_by_id[aid]["action"][len("[ACTION] "):])["function"] == "exec"
                      for aid in positive)
        if not is_exec:
            continue
        counts["exec_queries"] += 1
        for k in (1, 20, 50, 100):
            counts[f"exec_at_{k}"] += rank <= k
        if rank <= 20:
            continue
        counts["original_top20_misses"] += 1
        bucket = "21_50" if rank <= 50 else "51_100" if rank <= 100 else "over_100"
        counts[f"miss_rank_{bucket}"] += 1
        first = action_by_id[top[0]["action_id"]]
        first_tool = json.loads(first["action"][len("[ACTION] "):])["function"]
        counts[f"miss_top1_tool_{first_tool}"] += 1
        review.append({"query_id": qid, "query": query_by_id[qid]["query"],
                       "query_trajectory_id": query_by_id[qid].get("trajectory_id"),
                       "known_positive_rank": rank, "rank_bucket": bucket,
                       "known_positives": [action_by_id[aid] for aid in sorted(positive)],
                       "top_candidates": [{**item, "action": action_by_id[item["action_id"]]["action"],
                                           "trajectory_id": action_by_id[item["action_id"]].get("trajectory_id")}
                                          for item in top[:10]],
                       "review_label": "unjudged", "equivalent_action_ids": [],
                       "review_reason": None})
    return dict(counts), sorted(review, key=lambda row: row["query_id"])


def full_rank(queries, actions, qrels, checkpoint, device, batch_size, query_length, action_length):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install torch and transformers in the GPU environment") from error
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "query_model", local_files_only=True)
    query_model = AutoModel.from_pretrained(checkpoint / "query_model", local_files_only=True).to(device).eval()
    action_model = AutoModel.from_pretrained(checkpoint / "action_model", local_files_only=True).to(device).eval()
    projection = torch.nn.Linear(action_model.config.hidden_size, query_model.config.hidden_size,
                                 bias=False).to(device)
    projection.load_state_dict(torch.load(checkpoint / "projection.pt", map_location=device,
                                          weights_only=True))
    projection.eval()
    action_vectors = embed([row["action"] for row in actions], torch=torch, tokenizer=tokenizer,
                           model=action_model, device=device, batch_size=batch_size,
                           max_length=action_length)
    query_vectors = embed([row["query"] for row in queries], torch=torch, tokenizer=tokenizer,
                          model=query_model, device=device, batch_size=batch_size,
                          max_length=query_length)
    with torch.inference_mode():
        action_vectors = torch.nn.functional.normalize(projection(action_vectors.to(device)), dim=-1)
        scores = (query_vectors.to(device) @ action_vectors.T).cpu()
    ids = [row["action_id"] for row in actions]
    rankings = []
    for query, values in zip(queries, scores):
        values = values.tolist()
        order = sorted(range(len(ids)), key=lambda index: (-values[index], ids[index]))
        positives = qrels[query["query_id"]]
        rank = next(position for position, index in enumerate(order, 1) if ids[index] in positives)
        rankings.append({"query_id": query["query_id"], "known_positive_rank": rank,
                         "top_actions": [{"action_id": ids[index], "score": values[index],
                                          "is_known_positive": ids[index] in positives}
                                         for index in order[:100]]})
    return rankings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--qrels", type=Path, default=QRELS)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--rankings", type=Path, help="Analyze an existing full-pool Top-100 ranking without loading a model")
    parser.add_argument("--reference-top20", type=Path, default=REFERENCE,
                        help="Saved checkpoint Top-20 to verify the recomputed encoding")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-length", type=int, default=256)
    parser.add_argument("--action-length", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1:
        raise ValueError("output must not exist and batch-size must be positive")
    queries, actions, qrels = load_split(args.dataset, "dev", qrels_path=args.qrels)
    rankings = ([json.loads(line) for line in args.rankings.read_text(encoding="utf-8").splitlines() if line]
                if args.rankings else full_rank(queries, actions, qrels, args.checkpoint, args.device,
                                                args.batch_size, args.query_length, args.action_length))
    if args.reference_top20 is not None:
        validate_reference(rankings, list(rows(args.reference_top20)))
    summary, review = summarize(queries, actions, qrels, rankings)
    args.output.mkdir(parents=True)
    with (args.output / "rankings_top100.jsonl").open("w", encoding="utf-8") as stream:
        for row in rankings:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output / "exec_top20_miss_review.jsonl").open("w", encoding="utf-8") as stream:
        for row in review:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary.update({"checkpoint": str(args.checkpoint), "qrels_sha256": sha256(args.qrels),
                    "rankings_source": str(args.rankings) if args.rankings else "computed_from_checkpoint",
                    "label_limit": "unlisted Action IDs are unjudged, not negatives"})
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
