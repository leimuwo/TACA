#!/usr/bin/env python3
"""Train a BGE Intent tower against frozen NightOwl Action embeddings."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
import os
from pathlib import Path

from evaluate_bge_clawgym import load_split, rows, sha256
from train_bge_clawgym import load_train


DATASET = Path("data/clawgym/bge_dataset_split_20260930_v2")
BGE = Path("outputs/bge_clawgym_silver_v2/best")
NIGHTOWL = Path(os.environ.get("TA_MODEL_ROOT", "../model_hub")) / Path("NightOwl-CodeEmbedding")
NIGHTOWL_REVISION = "cfc3c6d172a93c79826db380ce82b6fee377ae1c"


def action_model_load_kwargs(model_path, revision):
    path = Path(model_path)
    if path.is_dir():
        required = ("config.json", "model.safetensors", "tokenizer.json")
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete local NightOwl model: {', '.join(missing)}")
        return {"local_files_only": True}
    return {"revision": revision}


def action_tool(text):
    return json.loads(text.removeprefix("[ACTION] ")).get("function")


def action_input(text):
    """Keep the tool and target fields visible before long content is truncated."""
    action = json.loads(text.removeprefix("[ACTION] "))
    args = action.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("Action args must be an object")
    priority = ("command", "path", "file", "file_path", "url", "targetUrl", "action", "method")
    ordered = {key: args[key] for key in priority if key in args}
    ordered.update({key: value for key, value in args.items() if key not in ordered})
    return json.dumps({"function": action.get("function"), "args": ordered}, ensure_ascii=False,
                      separators=(",", ":"))


def prepare_groups(records):
    known = defaultdict(set)
    for row in records:
        known[row["query"]].add(row["pos"][0])
    groups = []
    for row in records:
        negatives = [value for value in row["neg"] if value not in known[row["query"]]]
        if negatives:
            groups.append((row["query"], row["pos"][0], negatives))
    return groups


def rank_metrics(ranks, cutoff):
    total = len(ranks)
    found = [rank for rank in ranks if rank is not None]
    return {"queries": total, "known_positive_in_candidates": len(found),
            "known_positive_at_1": sum(rank == 1 for rank in found) / total,
            "known_positive_at_5": sum(rank <= 5 for rank in found) / total,
            "known_positive_at_10": sum(rank <= 10 for rank in found) / total,
            f"known_positive_mrr_at_{cutoff}": sum(1 / rank for rank in found) / total}


def baseline_metrics(rankings_path, actions, qrels, top_k):
    tools = {row["action_id"]: action_tool(row["action"]) for row in actions}
    overall = []; exec_ranks = []
    seen = set()
    for row in rows(rankings_path):
        if row["query_id"] in seen or row["query_id"] not in qrels:
            raise ValueError("baseline rankings contain duplicate or unknown query IDs")
        seen.add(row["query_id"])
        positives = qrels[row["query_id"]]
        order = [item["action_id"] for item in row["top_actions"][:top_k]]
        if len(order) < top_k or any(action_id not in tools for action_id in order):
            raise ValueError("baseline rankings have too few or unknown Action IDs")
        rank = next((place for place, action_id in enumerate(order, 1) if action_id in positives), None)
        overall.append(rank)
        if any(tools[action_id] == "exec" for action_id in positives):
            exec_ranks.append(rank)
    if seen != set(qrels):
        raise ValueError("baseline rankings do not cover the dev queries")
    return {"overall": rank_metrics(overall, top_k), "exec_positive": rank_metrics(exec_ranks, top_k)}


def evaluate(queries, actions, qrels, action_vectors, *, query_model, query_tokenizer, torch,
             device, batch_size, query_length, top_k):
    query_model.eval()
    vectors = []
    for start in range(0, len(queries), batch_size):
        batch = [row["query"] for row in queries[start:start + batch_size]]
        tokens = query_tokenizer(batch, padding=True, truncation=True, max_length=query_length,
                                 return_tensors="pt").to(device)
        with torch.inference_mode():
            hidden = query_model(**tokens).last_hidden_state[:, 0].float()
            vectors.append(torch.nn.functional.normalize(hidden, dim=-1).cpu())
    query_vectors = torch.cat(vectors)
    scores = query_vectors @ action_vectors.T
    ids = [row["action_id"] for row in actions]
    action_tools = {row["action_id"]: action_tool(row["action"]) for row in actions}
    ranks = []; exec_ranks = []; ranking_rows = []
    for index, query in enumerate(queries):
        values = scores[index].tolist()
        order = sorted(range(len(ids)), key=lambda position: (-values[position], ids[position]))[:top_k]
        positives = qrels[query["query_id"]]
        rank = next((place for place, position in enumerate(order, 1) if ids[position] in positives), None)
        ranks.append(rank)
        if any(action_tools[action_id] == "exec" for action_id in positives):
            exec_ranks.append(rank)
        ranking_rows.append({"query_id": query["query_id"], "top_actions": [
            {"action_id": ids[position], "score": values[position],
             "is_known_positive": ids[position] in positives} for position in order]})
    return {"overall": rank_metrics(ranks, top_k), "exec_positive": rank_metrics(exec_ranks, top_k)}, ranking_rows


def encode_actions(texts, *, model, tokenizer, torch, device, batch_size, max_length):
    model.eval()
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = [action_input(text) for text in texts[start:start + batch_size]]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                           return_tensors="pt").to(device)
        with torch.inference_mode():
            pooled = model(**tokens).last_hidden_state[:, 0].float()
            vectors.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
    return torch.cat(vectors)


def run(args):
    if (args.epochs < 1 or args.batch_size < 1 or args.encode_batch_size < 1 or args.accumulation < 1
            or args.learning_rate <= 0 or args.temperature <= 0 or args.top_k < 1
            or args.query_length < 8 or args.action_length < 8 or args.log_every < 1):
        raise ValueError("invalid training parameters")
    train_path = args.dataset / "train" / "train.jsonl"
    action_load_kwargs = action_model_load_kwargs(args.action_model, args.action_revision)
    records, _ = load_train(train_path)
    groups = prepare_groups(records)
    queries, actions, qrels = load_split(args.dataset, "dev", qrels_path=args.qrels)
    baseline = baseline_metrics(args.baseline_rankings, actions, qrels, args.top_k)
    if args.dry_run:
        print(json.dumps({"groups": len(groups), "explicit_negatives": sum(len(row[2]) for row in groups),
                          "train_exec_groups": sum(action_tool(row[1]) == "exec" for row in groups),
                          "dev_queries": len(queries), "dev_exec_queries": sum(
                              any(action_tool(next(action["action"] for action in actions if action["action_id"] == aid)) == "exec"
                                  for aid in qrels[row["query_id"]]) for row in queries),
                          "nightowl": str(args.action_model), "bge_baseline": baseline,
                          "mode": "validated_no_model_loaded"}), flush=True)
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as error:
        raise RuntimeError("install torch and transformers>=4.56.2 on the GPU machine") from error
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto": device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed); torch.manual_seed(args.seed)
    if device == "cuda": torch.cuda.manual_seed_all(args.seed)
    query_tokenizer = AutoTokenizer.from_pretrained(args.query_model)
    query_model = AutoModel.from_pretrained(args.query_model).to(device)
    action_tokenizer = AutoTokenizer.from_pretrained(args.action_model, **action_load_kwargs)
    action_model = AutoModel.from_pretrained(args.action_model, **action_load_kwargs).to(device).eval()
    if query_model.config.hidden_size != action_model.config.hidden_size:
        raise ValueError("tower dimensions differ; add a learned projection before training")
    all_actions = list(dict.fromkeys([text for _, positive, negatives in groups for text in [positive, *negatives]]
                                     + [row["action"] for row in actions]))
    print(json.dumps({"stage": "encode_actions", "unique_actions": len(all_actions)}), flush=True)
    action_vectors = encode_actions(all_actions, model=action_model, tokenizer=action_tokenizer,
                                    torch=torch, device=device, batch_size=args.encode_batch_size,
                                    max_length=args.action_length)
    del action_model
    if device == "cuda": torch.cuda.empty_cache()
    action_index = {text: index for index, text in enumerate(all_actions)}
    dev_vectors = action_vectors[[action_index[row["action"]] for row in actions]]
    optimizer = torch.optim.AdamW(query_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(math.ceil(len(groups) / args.batch_size) / args.accumulation)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps_per_epoch * args.epochs * args.warmup_ratio),
                                                 steps_per_epoch * args.epochs)
    args.output.mkdir(parents=True)
    history = []; best = -1.0; best_epoch = None
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(groups)))
        random.Random(args.seed + epoch).shuffle(order)
        query_model.train(); optimizer.zero_grad(set_to_none=True)
        micro_batches = math.ceil(len(order) / args.batch_size)
        loss_sum = 0.0
        for micro in range(micro_batches):
            batch = [groups[index] for index in order[micro * args.batch_size:(micro + 1) * args.batch_size]]
            tokens = query_tokenizer([row[0] for row in batch], padding=True, truncation=True,
                                     max_length=args.query_length, return_tensors="pt").to(device)
            query_vectors = torch.nn.functional.normalize(query_model(**tokens).last_hidden_state[:, 0].float(), dim=-1)
            row_losses = []
            for index, (_, positive, negatives) in enumerate(batch):
                indices = [action_index[text] for text in [positive, *negatives]]
                candidates = action_vectors[indices].to(device)
                logits = (candidates @ query_vectors[index]) / args.temperature
                row_losses.append(torch.nn.functional.cross_entropy(logits.unsqueeze(0),
                                                                     torch.zeros(1, dtype=torch.long, device=device)))
            loss = torch.stack(row_losses).mean()
            loss_sum += float(loss.detach().cpu())
            size = min(args.accumulation, micro_batches - (micro // args.accumulation) * args.accumulation)
            (loss / size).backward()
            if (micro + 1) % args.accumulation == 0 or micro + 1 == micro_batches:
                torch.nn.utils.clip_grad_norm_(query_model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if (micro + 1) % args.log_every == 0 or micro + 1 == micro_batches:
                print(json.dumps({"epoch": epoch, "micro_batch": micro + 1, "micro_batches": micro_batches,
                                  "mean_loss_so_far": loss_sum / (micro + 1)}), flush=True)
        report, ranking_rows = evaluate(queries, actions, qrels, dev_vectors, query_model=query_model,
                                        query_tokenizer=query_tokenizer, torch=torch, device=device,
                                        batch_size=args.encode_batch_size, query_length=args.query_length,
                                        top_k=args.top_k)
        score = report[args.select_metric][f"known_positive_mrr_at_{args.top_k}"]
        history.append({"epoch": epoch, "mean_loss": loss_sum / micro_batches, "dev": report})
        print(json.dumps({"epoch": epoch, "overall": report["overall"], "exec_positive": report["exec_positive"]}), flush=True)
        if score > best:
            best = score; best_epoch = epoch
            checkpoint = args.output / "best"; checkpoint.mkdir(exist_ok=True)
            query_model.save_pretrained(checkpoint / "query_model")
            query_tokenizer.save_pretrained(checkpoint / "query_model")
            with (args.output / "best_dev_rankings.jsonl").open("w", encoding="utf-8") as stream:
                for row in ranking_rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"schema_version": "nightowl_action_tower_v1", "action_model": str(args.action_model),
               "action_revision": args.action_revision,
               "query_model": str(args.query_model), "action_tower_frozen": True,
               "pooling": "normalized_cls", "selection_metric": args.select_metric,
               "best_epoch": best_epoch, "best_dev_mrr_at_k": best,
               "test_evaluated": False, "label_limit": "Silver known positives only",
               "bge_baseline": baseline,
               "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               "input_sha256": {"train": sha256(train_path), "qrels": sha256(args.qrels),
                                "baseline_rankings": sha256(args.baseline_rankings)}, "history": history}
    (args.output / "training_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--qrels", type=Path, default=Path("data/clawgym/bge_dev_qrels_exact_text_20261009/qrels_review_closed.jsonl"))
    parser.add_argument("--baseline-rankings", type=Path,
                        default=Path("outputs/bge_clawgym_silver_v2_dev_top20_20261009/rankings.jsonl"))
    parser.add_argument("--query-model", type=Path, default=BGE)
    parser.add_argument("--action-model", default=NIGHTOWL)
    parser.add_argument("--action-revision", default=NIGHTOWL_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--query-length", type=int, default=256)
    parser.add_argument("--action-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--select-metric", choices=("overall", "exec_positive"), default="exec_positive")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
