#!/usr/bin/env python3
"""Fine-tune BGE reranker on explicit Silver pairs and score fixed retrieval candidates."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
import os
from pathlib import Path

from evaluate_bge_clawgym import load_split, rows, sha256
from train_bge_clawgym import load_train


DEFAULT_MODEL = Path(os.environ.get("TA_MODEL_ROOT", "../model_hub")) / Path("bge-reranker-v2-m3")
DEFAULT_DATASET = Path("data/clawgym/bge_dataset_split_20260930_v2")
DEFAULT_RANKINGS = Path("outputs/bge_clawgym_silver_v2_dev_top20_20261009/rankings.jsonl")
DEFAULT_QRELS = Path("data/clawgym/bge_dev_qrels_exact_text_20261009/qrels_review_closed.jsonl")
TARGET_KEYS = ("path", "file", "url", "uri", "target", "command", "cmd", "method")


def prioritize_action(text):
    """Move tool and target fields ahead of long payloads before pair truncation."""
    prefix = "[ACTION] "
    if not text.startswith(prefix):
        raise ValueError("invalid action serialization")
    try:
        action = json.loads(text[len(prefix):])
    except json.JSONDecodeError as exc:
        raise ValueError("invalid action JSON") from exc
    if not isinstance(action, dict):
        raise ValueError("action must be a JSON object")
    args = action.get("args")
    if isinstance(args, dict):
        ordered_args = {key: args[key] for key in TARGET_KEYS if key in args}
        ordered_args.update({key: value for key, value in args.items() if key not in ordered_args})
        action = {"function": action.get("function"), "args": ordered_args,
                  **{key: value for key, value in action.items() if key not in ("function", "args")}}
    return prefix + json.dumps(action, ensure_ascii=False, separators=(",", ":"))


def load_candidates(dataset, rankings_path, qrels_path, top_k, split="dev"):
    queries, actions, qrels = load_split(dataset, split, qrels_path=qrels_path)
    query_map = {row["query_id"]: row["query"] for row in queries}
    action_map = {row["action_id"]: row["action"] for row in actions}
    rankings = list(rows(rankings_path))
    if len(rankings) != len(query_map) or {row.get("query_id") for row in rankings} != set(query_map):
        raise ValueError(f"retrieval rankings must contain exactly one row per {split} query")
    if len(rankings) != len({row["query_id"] for row in rankings}):
        raise ValueError("duplicate query in retrieval rankings")
    candidates = []
    for row in rankings:
        top = row.get("top_actions")
        if not isinstance(top, list) or len(top) < top_k:
            raise ValueError(f"query {row['query_id']} has fewer than {top_k} candidates")
        selected = top[:top_k]
        ids = [item.get("action_id") for item in selected]
        if len(set(ids)) != top_k or any(action_id not in action_map for action_id in ids):
            raise ValueError(f"query {row['query_id']} has duplicate or unknown action IDs")
        scores = [item.get("score") for item in selected]
        if any(not isinstance(score, (float, int)) or not math.isfinite(score) for score in scores):
            raise ValueError(f"query {row['query_id']} has invalid retrieval scores")
        candidates.append({"query_id": row["query_id"], "query": query_map[row["query_id"]],
                           "action_ids": ids, "actions": [prioritize_action(action_map[action_id]) for action_id in ids],
                           "retrieval_scores": scores, "positives": qrels[row["query_id"]]})
    return candidates


def summarize(candidates, reranked=None):
    total = len(candidates)
    reachable = 0
    first = {1: 0, 5: 0, 10: 0}
    reciprocal = 0.0
    for index, row in enumerate(candidates):
        order = reranked[index] if reranked is not None else list(range(len(row["action_ids"])))
        hit = next((rank for rank, action_index in enumerate(order, 1)
                    if row["action_ids"][action_index] in row["positives"]), None)
        if hit is not None:
            reachable += 1
            reciprocal += 1 / hit
            for cutoff in first:
                first[cutoff] += hit <= cutoff
    return {"queries": total, "known_positive_in_candidates": reachable,
            "known_positive_candidate_recall": reachable / total,
            "known_positive_at_1": first[1] / total,
            "known_positive_at_5": first[5] / total,
            "known_positive_at_10": first[10] / total,
            "known_positive_mrr_at_k": reciprocal / total,
            "reachable_known_positive_mrr_at_k": reciprocal / reachable if reachable else None}


def score_candidates(candidates, *, model, tokenizer, torch, device, batch_size, max_length):
    model.eval()
    pairs = [(row["query"], action) for row in candidates for action in row["actions"]]
    scores = []
    started = time.monotonic()
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        tokens = tokenizer([query for query, _ in batch], [action for _, action in batch],
                           padding=True, truncation="only_second", max_length=max_length,
                           return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**tokens).logits
        if logits.ndim != 2 or logits.shape[1] != 1:
            raise ValueError("reranker must have one regression logit per pair")
        scores.extend(logits[:, 0].float().cpu().tolist())
        completed = min(start + batch_size, len(pairs))
        if completed == len(pairs) or (start // batch_size + 1) % 100 == 0:
            elapsed = time.monotonic() - started
            print(json.dumps({"stage": "dev_evaluation", "pairs_done": completed,
                              "pairs_total": len(pairs), "elapsed_seconds": round(elapsed, 1),
                              "eta_seconds": round(elapsed * (len(pairs) - completed) / completed, 1)}), flush=True)
    ranked = []
    offset = 0
    for row in candidates:
        size = len(row["action_ids"])
        local_scores = scores[offset:offset + size]
        offset += size
        ranked.append(sorted(range(size), key=lambda index: (-local_scores[index], index)))
    return ranked, scores


def evaluate(candidates, *, model, tokenizer, torch, device, batch_size, max_length, top_k):
    ranked, scores = score_candidates(candidates, model=model, tokenizer=tokenizer, torch=torch,
                                      device=device, batch_size=batch_size, max_length=max_length)
    report = {"candidate_count": top_k, "retrieval": summarize(candidates),
              "reranked": summarize(candidates, ranked)}
    output_rows = []
    offset = 0
    for row, order in zip(candidates, ranked):
        local = scores[offset:offset + len(order)]
        offset += len(order)
        output_rows.append({"query_id": row["query_id"], "top_actions": [
            {"action_id": row["action_ids"][index], "reranker_score": local[index],
             "retrieval_score": row["retrieval_scores"][index],
             "is_known_positive": row["action_ids"][index] in row["positives"]} for index in order]})
    return report, output_rows


def save_jsonl(path, records):
    with path.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_clear_queries(candidates, screening_path):
    decisions = list(rows(screening_path))
    if not decisions:
        raise ValueError("empty screening decisions")
    by_id = {}
    for row in decisions:
        query_id = row.get("source_query_id") or row.get("query_id")
        if not isinstance(query_id, str) or query_id in by_id or not isinstance(row.get("ambiguous"), bool):
            raise ValueError("screening contains duplicate query IDs or invalid ambiguous labels")
        by_id[query_id] = row["ambiguous"]
    missing = {row["query_id"] for row in candidates} - set(by_id)
    # The supplied ambiguous_test.jsonl lists only excluded queries. When all
    # evaluated IDs use source_query_id, absence means the query screened clear.
    excluded_only = all(row.get("ambiguous") is True for row in decisions)
    if excluded_only and not set(by_id) <= {row["query_id"] for row in candidates}:
        raise ValueError("screening references unknown evaluated queries")
    if missing and not excluded_only:
        raise ValueError(f"screening missing {len(missing)} evaluated queries")
    return [row for row in candidates if not by_id.get(row["query_id"], False)]


def make_training_groups(records):
    groups = []
    for row in records:
        positive = prioritize_action(row["pos"][0])
        negatives = [prioritize_action(text) for text in row["neg"]]
        tool = json.loads(row["pos"][0][len("[ACTION] "):]).get("function")
        groups.append({"query": row["query"], "actions": [positive, *negatives],
                       "positive_tool": tool})
    return groups


def pairwise_group_loss(logits, group_sizes, weights, *, torch):
    losses = []
    offset = 0
    for size in group_sizes:
        positive = logits[offset]
        negative = logits[offset + 1:offset + size]
        losses.append(torch.nn.functional.softplus(negative.float() - positive.float()).mean())
        offset += size
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=logits.device)
    return (torch.stack(losses) * weight_tensor).sum() / weight_tensor.sum()


def run(args):
    if args.eval_only and args.dry_run:
        raise ValueError("--eval-only and --dry-run cannot be combined")
    if not args.eval_only and (args.split != "dev" or args.screening is not None):
        raise ValueError("training selects checkpoints on the full dev split only")
    if (args.top_k < 1 or args.batch_size < 1 or args.eval_batch_size < 1 or args.accumulation < 1
            or args.epochs < 1 or args.max_length < 8 or args.learning_rate <= 0
            or args.weight_decay < 0 or not 0 <= args.warmup_ratio <= 1 or args.max_grad_norm <= 0
            or args.log_every < 1 or args.exec_weight <= 0):
        raise ValueError("invalid training or evaluation parameters")
    train_path = args.dataset / "train" / "train.jsonl"
    candidates = load_candidates(args.dataset, args.rankings, args.qrels, args.top_k, split=args.split)
    if args.screening is not None:
        candidates = filter_clear_queries(candidates, args.screening)
    if not candidates:
        raise ValueError("no queries remain after screening")
    if args.eval_only:
        if args.output.exists():
            raise FileExistsError(args.output)
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install torch and transformers in the evaluation environment") from exc
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device == "auto":
            device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, local_files_only=not args.allow_download).to(device)
        if model.config.num_labels != 1:
            raise ValueError("reranker must have exactly one output logit")
        report, ranking_rows = evaluate(candidates, model=model, tokenizer=tokenizer, torch=torch,
                                        device=device, batch_size=args.eval_batch_size,
                                        max_length=args.max_length, top_k=args.top_k)
        report.update({"schema_version": "bge_clawgym_reranker_evaluation_v1", "model": str(args.model),
                       "split": args.split, "screening_source": str(args.screening) if args.screening else None,
                       "rankings_source": str(args.rankings), "qrels_source": str(args.qrels),
                       "input_sha256": {"rankings": sha256(args.rankings), "qrels": sha256(args.qrels),
                                        **({"screening": sha256(args.screening)} if args.screening else {})},
                       "label_limit": "Silver known positives only; unlisted candidates are unjudged."})
        args.output.mkdir(parents=True)
        (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        save_jsonl(args.output / "rankings.jsonl", ranking_rows)
        print(json.dumps(report["reranked"], ensure_ascii=False), flush=True)
        return
    records, _ = load_train(train_path)
    pairs = [(row["query"], prioritize_action(row["pos"][0]), 1.0) for row in records]
    pairs += [(row["query"], prioritize_action(negative), 0.0) for row in records for negative in row["neg"]]
    groups = make_training_groups(records) if args.loss == "pairwise" else None
    training_items = groups if groups is not None else pairs
    setup = {"training_pairs": len(pairs), "positive_pairs": len(records),
             "explicit_negative_pairs": len(pairs) - len(records), "dev": summarize(candidates),
             "model": str(args.model), "loss": args.loss,
             "exec_groups": sum(group["positive_tool"] == "exec" for group in groups) if groups else None,
             "mode": "validated_no_model_loaded" if args.dry_run else "train"}
    if args.dry_run:
        print(json.dumps(setup, ensure_ascii=False))
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise RuntimeError("install torch and transformers in the training environment") from exc
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=not args.allow_download).to(device)
    if model.config.num_labels != 1:
        raise ValueError("base reranker must have exactly one output logit")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    micro_batches = math.ceil(len(training_items) / args.batch_size)
    steps = math.ceil(micro_batches / args.accumulation) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * args.warmup_ratio), steps)
    args.output.mkdir(parents=True)
    print(json.dumps({"stage": "training_start", "device": device, "epochs": args.epochs,
                      "training_items": len(training_items), "loss": args.loss,
                      "micro_batches_per_epoch": micro_batches,
                      "dev_pairs": sum(len(row["actions"]) for row in candidates)}), flush=True)
    history = []
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        order = list(range(len(training_items)))
        random.Random(args.seed + epoch).shuffle(order)
        loss_sum = 0.0
        started = time.monotonic()
        for micro in range(micro_batches):
            batch = [training_items[index] for index in order[micro * args.batch_size:(micro + 1) * args.batch_size]]
            if args.loss == "pairwise":
                batch_pairs = [(row["query"], action) for row in batch for action in row["actions"]]
                group_sizes = [len(row["actions"]) for row in batch]
                weights = [args.exec_weight if row["positive_tool"] == "exec" else 1.0 for row in batch]
            else:
                batch_pairs = [(query, action) for query, action, _ in batch]
            tokens = tokenizer([query for query, _ in batch_pairs], [action for _, action in batch_pairs],
                               padding=True, truncation="only_second", max_length=args.max_length,
                               return_tensors="pt").to(device)
            logits = model(**tokens).logits[:, 0]
            if args.loss == "pairwise":
                loss = pairwise_group_loss(logits, group_sizes, weights, torch=torch)
            else:
                labels = torch.tensor([label for _, _, label in batch], dtype=torch.float32, device=device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite training loss")
            loss_sum += float(loss.detach().cpu())
            group_start = (micro // args.accumulation) * args.accumulation
            group_size = min(args.accumulation, micro_batches - group_start)
            (loss / group_size).backward()
            if (micro + 1) % args.accumulation == 0 or micro + 1 == micro_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (micro + 1) % args.log_every == 0 or micro + 1 == micro_batches:
                completed = micro + 1
                elapsed = time.monotonic() - started
                print(json.dumps({"stage": "train", "epoch": epoch, "epochs": args.epochs,
                                  "micro_batch": completed, "micro_batches": micro_batches,
                                  "mean_loss_so_far": loss_sum / completed,
                                  "elapsed_seconds": round(elapsed, 1),
                                  "eta_seconds": round(elapsed * (micro_batches - completed) / completed, 1)}), flush=True)
        print(json.dumps({"stage": "dev_evaluation_start", "epoch": epoch}), flush=True)
        report, ranking_rows = evaluate(candidates, model=model, tokenizer=tokenizer, torch=torch,
                                        device=device, batch_size=args.eval_batch_size,
                                        max_length=args.max_length, top_k=args.top_k)
        score = report["reranked"]["known_positive_mrr_at_k"]
        history.append({"epoch": epoch, "mean_loss": loss_sum / micro_batches, "dev": report})
        print(json.dumps({"epoch": epoch, "mean_loss": loss_sum / micro_batches,
                          "dev_known_positive_mrr_at_k": score}), flush=True)
        if score > best:
            best = score
            best_epoch = epoch
            best_dir = args.output / "best"
            best_dir.mkdir(exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            save_jsonl(args.output / "best_dev_rankings.jsonl", ranking_rows)
    metadata = {"schema_version": "bge_clawgym_reranker_training_v1", "base_model": str(args.model),
                "retriever_rankings": str(args.rankings), "qrels_source": str(args.qrels),
                "top_k": args.top_k, "best_epoch": best_epoch, "best_dev_known_positive_mrr_at_k": best,
                "test_evaluated": False, "label_limit": "Silver known positives only; unlisted candidates are unjudged.",
                "input_sha256": {"train": sha256(train_path), "rankings": sha256(args.rankings),
                                 "qrels": sha256(args.qrels)}, "config": {key: str(value) if isinstance(value, Path) else value
                                  for key, value in vars(args).items()}, "history": history}
    (args.output / "training_report.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_checkpoint": str(args.output / "best"), "best_epoch": best_epoch,
                      "dev_known_positive_mrr_at_k": best}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--screening", type=Path, help="JSONL ambiguous decisions; evaluate only ambiguous=false")
    parser.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--loss", choices=("bce", "pairwise"), default="bce")
    parser.add_argument("--exec-weight", type=float, default=1.0,
                        help="Weight for groups whose positive Action uses exec (pairwise only)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25, help="Print training progress every N micro-batches")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-download", action="store_true", help="Allow model files outside the local cache")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="Score fixed candidates with an existing checkpoint")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
