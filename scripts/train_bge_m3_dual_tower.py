#!/usr/bin/env python3
"""Fine-tune independent BGE-M3 Intent and Action towers for ClawGym retrieval."""
from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
import os
from pathlib import Path

from evaluate_bge_clawgym import embed, load_split, sha256
from train_bge_clawgym import load_train
from train_nightowl_action_tower import baseline_metrics, evaluate
from evaluate_bge_m3_local import clear_ids


MODEL = Path(os.environ.get("TA_MODEL_ROOT", "../model_hub")) / Path("bge-m3")
DATASET = Path("data/clawgym/bge_dataset_split_20260930_v2")
QRELS = Path("data/clawgym/bge_dev_qrels_exact_text_20261009/qrels_review_closed.jsonl")
BASELINE = Path("outputs/bge_clawgym_silver_v2_dev_top20_20261009/rankings.jsonl")


def candidate_mask(batch, known_positives):
    """Allow row negatives and other batch positives except this Intent's known positives."""
    positives = [row["pos"][0] for row in batch]
    negatives = [text for row in batch for text in row["neg"]]
    candidates = positives + negatives
    mask = [[index == row_index or text not in known_positives[row["query"]]
             for index, text in enumerate(candidates)]
            for row_index, row in enumerate(batch)]
    return candidates, mask


def epoch_order(records, exec_repeat, seed, epoch):
    order = list(range(len(records)))
    exec_indices = [index for index, row in enumerate(records)
                    if json.loads(row["pos"][0].removeprefix("[ACTION] "))["function"] == "exec"]
    order.extend(exec_indices * (exec_repeat - 1))
    random.Random(seed + epoch).shuffle(order)
    return order


def run(args):
    if (args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1
            or args.accumulation < 1 or args.exec_repeat < 1
            or args.query_length < 8 or args.action_length < 8
            or args.top_k < 1 or args.log_every < 1 or args.temperature <= 0
            or min(args.query_learning_rate, args.action_learning_rate,
                   args.projection_learning_rate) <= 0
            or args.weight_decay < 0 or not 0 <= args.warmup_ratio <= 1):
        raise ValueError("invalid training parameters")
    required = ("config.json", "tokenizer.json", "pytorch_model.bin")
    if args.model.is_dir():
        missing = [name for name in required if not (args.model / name).is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete BGE-M3 model: {', '.join(missing)}")
    train_path = args.dataset / "train" / "train.jsonl"
    records, known_positives = load_train(train_path)
    exec_rows = sum(json.loads(row["pos"][0].removeprefix("[ACTION] "))["function"] == "exec"
                    for row in records)
    epoch_rows = len(records) + (args.exec_repeat - 1) * exec_rows
    queries, actions, qrels = load_split(args.dataset, "dev", qrels_path=args.qrels)
    if bool(args.clear_dev_screen_input) != bool(args.clear_dev_decisions):
        raise ValueError("both clear-dev screen input and decisions are required")
    if args.clear_dev_screen_input:
        from evaluate_bge_clawgym import rows
        clear = clear_ids(queries, rows(args.clear_dev_screen_input), rows(args.clear_dev_decisions))
        queries = [row for row in queries if row["query_id"] in clear]
        qrels = {qid: positive for qid, positive in qrels.items() if qid in clear}
        if not queries:
            raise ValueError("no clear dev queries remain")
        from train_nightowl_action_tower import rank_metrics, action_tool
        by_id = {row["action_id"]: row for row in actions}
        baseline_rows = {row["query_id"]: row for row in rows(args.baseline_rankings)}
        if not set(qrels) <= set(baseline_rows):
            raise ValueError("baseline rankings lack clear dev queries")
        all_ranks, exec_ranks = [], []
        for query in queries:
            qid = query["query_id"]
            rank = next((i for i, item in enumerate(baseline_rows[qid]["top_actions"][:args.top_k], 1)
                         if item["action_id"] in qrels[qid]), None)
            all_ranks.append(rank)
            if any(action_tool(by_id[aid]["action"]) == "exec" for aid in qrels[qid]):
                exec_ranks.append(rank)
        baseline = {"overall": rank_metrics(all_ranks, args.top_k),
                    "exec_positive": rank_metrics(exec_ranks, args.top_k)}
    else:
        baseline = baseline_metrics(args.baseline_rankings, actions, qrels, args.top_k)
    if args.dry_run:
        print(json.dumps({"train_rows": len(records),
                          "exec_positive_rows": exec_rows, "sampled_rows_per_epoch": epoch_rows,
                          "sampled_exec_rows_per_epoch": exec_rows * args.exec_repeat,
                          "explicit_negatives": sum(len(row["neg"]) for row in records),
                          "dev_queries": len(queries), "dev_actions": len(actions),
                          "model": str(args.model), "bge_baseline": baseline,
                          "mode": "validated_no_model_loaded"}), flush=True)
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as error:
        raise RuntimeError("install torch and transformers on the GPU machine") from error
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.model.is_dir())
    query_model = AutoModel.from_pretrained(args.model, local_files_only=args.model.is_dir()).to(device)
    action_model = AutoModel.from_pretrained(args.model, local_files_only=args.model.is_dir()).to(device)
    if query_model.config.hidden_size != action_model.config.hidden_size:
        raise ValueError("tower dimensions differ")
    if args.gradient_checkpointing:
        query_model.gradient_checkpointing_enable()
        action_model.gradient_checkpointing_enable()
    projection = torch.nn.Linear(action_model.config.hidden_size,
                                 query_model.config.hidden_size, bias=False).to(device)
    torch.nn.init.eye_(projection.weight)
    optimizer = torch.optim.AdamW([
        {"params": query_model.parameters(), "lr": args.query_learning_rate},
        {"params": action_model.parameters(), "lr": args.action_learning_rate},
        {"params": projection.parameters(), "lr": args.projection_learning_rate},
    ], weight_decay=args.weight_decay)
    batches_per_epoch = math.ceil(epoch_rows / args.batch_size)
    total_steps = math.ceil(batches_per_epoch / args.accumulation) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)
    args.output.mkdir(parents=True)
    history = []
    best, best_epoch = -1.0, None
    autocast = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if device == "cuda" else nullcontext

    for epoch in range(1, args.epochs + 1):
        order = epoch_order(records, args.exec_repeat, args.seed, epoch)
        query_model.train()
        action_model.train()
        projection.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(batches_per_epoch):
            batch = [records[index] for index in order[micro * args.batch_size:(micro + 1) * args.batch_size]]
            candidates, mask = candidate_mask(batch, known_positives)
            query_tokens = tokenizer([row["query"] for row in batch], padding=True,
                                     truncation=True, max_length=args.query_length,
                                     return_tensors="pt").to(device)
            action_tokens = tokenizer(candidates, padding=True, truncation=True,
                                      max_length=args.action_length,
                                      return_tensors="pt").to(device)
            with autocast():
                query_vectors = torch.nn.functional.normalize(
                    query_model(**query_tokens).last_hidden_state[:, 0].float(), dim=-1)
                action_vectors = torch.nn.functional.normalize(
                    action_model(**action_tokens).last_hidden_state[:, 0].float(), dim=-1)
                action_vectors = torch.nn.functional.normalize(projection(action_vectors), dim=-1)
                logits = (query_vectors @ action_vectors.T) / args.temperature
                logits = logits.masked_fill(~torch.tensor(mask, dtype=torch.bool, device=device),
                                            torch.finfo(logits.dtype).min)
                labels = torch.arange(len(batch), device=device)
                loss = torch.nn.functional.cross_entropy(logits, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite training loss")
            loss_sum += float(loss.detach().cpu())
            group_size = min(args.accumulation,
                             batches_per_epoch - (micro // args.accumulation) * args.accumulation)
            (loss / group_size).backward()
            if (micro + 1) % args.accumulation == 0 or micro + 1 == batches_per_epoch:
                torch.nn.utils.clip_grad_norm_(query_model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(action_model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (micro + 1) % args.log_every == 0 or micro + 1 == batches_per_epoch:
                print(json.dumps({"epoch": epoch, "micro_batch": micro + 1,
                                  "micro_batches": batches_per_epoch,
                                  "mean_loss_so_far": loss_sum / (micro + 1)}), flush=True)

        query_model.eval()
        action_model.eval()
        projection.eval()
        raw_actions = embed([row["action"] for row in actions], torch=torch,
                            tokenizer=tokenizer, model=action_model, device=device,
                            batch_size=args.eval_batch_size, max_length=args.action_length)
        with torch.inference_mode():
            dev_vectors = torch.nn.functional.normalize(
                projection(raw_actions.to(device)), dim=-1).cpu()
        metrics, rankings = evaluate(queries, actions, qrels, dev_vectors,
                                     query_model=query_model, query_tokenizer=tokenizer,
                                     torch=torch, device=device, batch_size=args.eval_batch_size,
                                     query_length=args.query_length, top_k=args.top_k)
        score = metrics[args.select_metric][f"known_positive_mrr_at_{args.top_k}"]
        history.append({"epoch": epoch, "mean_loss": loss_sum / batches_per_epoch,
                        "dev": metrics})
        print(json.dumps({"epoch": epoch, **metrics}), flush=True)
        if score > best:
            best, best_epoch = score, epoch
            checkpoint = args.output / "best"
            checkpoint.mkdir(exist_ok=True)
            query_model.save_pretrained(checkpoint / "query_model")
            action_model.save_pretrained(checkpoint / "action_model")
            tokenizer.save_pretrained(checkpoint / "query_model")
            tokenizer.save_pretrained(checkpoint / "action_model")
            torch.save({key: value.detach().cpu() for key, value in projection.state_dict().items()},
                       checkpoint / "projection.pt")
            with (args.output / "best_dev_rankings.jsonl").open("w", encoding="utf-8") as stream:
                for row in rankings:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"schema_version": "bge_m3_dual_tower_v1", "best_epoch": best_epoch,
              "best_dev_mrr_at_k": best, "selection_metric": args.select_metric,
              "query_and_action_towers_independent": True,
              "projection": "identity_initialized_linear_1024_to_1024",
              "pooling": "normalized_cls", "bge_baseline": baseline, "history": history,
              "train_exec_positive_rows": exec_rows,
              "sampled_rows_per_epoch": epoch_rows,
              "sampled_exec_rows_per_epoch": exec_rows * args.exec_repeat,
              "label_limit": "Silver known positives only", "test_evaluated": False,
              "dev_clear_only": bool(args.clear_dev_screen_input),
              "config": {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()},
              "input_sha256": {"train": sha256(train_path), "qrels": sha256(args.qrels),
                               "baseline_rankings": sha256(args.baseline_rankings)}}
    (args.output / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_checkpoint": str(args.output / "best"),
                      "best_epoch": best_epoch, "dev_score": best}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--qrels", type=Path, default=QRELS)
    parser.add_argument("--baseline-rankings", type=Path, default=BASELINE)
    parser.add_argument("--clear-dev-screen-input", type=Path)
    parser.add_argument("--clear-dev-decisions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--exec-repeat", type=int, default=1,
                        help="Repeat each exec-positive training row this many times per epoch")
    parser.add_argument("--query-learning-rate", type=float, default=1e-5)
    parser.add_argument("--action-learning-rate", type=float, default=1e-5)
    parser.add_argument("--projection-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--query-length", type=int, default=256)
    parser.add_argument("--action-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--select-metric", choices=("overall", "exec_positive"),
                        default="exec_positive")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
