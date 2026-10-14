#!/usr/bin/env python3
"""Train a shared BGE encoder on ClawGym Silver query/pos/neg examples."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

from evaluate_bge_clawgym import evaluate_loaded, load_split


def load_train(path):
    records = []
    known_positives = defaultdict(set)
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            query, pos, neg = row.get("query"), row.get("pos"), row.get("neg")
            if not isinstance(query, str) or not query.startswith("[INTENT] ") or not isinstance(pos, list) or len(pos) != 1 or not isinstance(neg, list) or not 1 <= len(neg) <= 3:
                raise ValueError(f"bad training row {number}")
            if any(not isinstance(text, str) or not text.startswith("[ACTION] ") for text in pos + neg):
                raise ValueError(f"bad action serialization at row {number}")
            if pos[0] in neg or len(set(neg)) != len(neg):
                raise ValueError(f"positive collision or duplicate negative at row {number}")
            records.append(row)
            known_positives[query].add(pos[0])
    if not records:
        raise ValueError("empty train data")
    for number, row in enumerate(records, 1):
        if set(row["neg"]) & known_positives[row["query"]]:
            raise ValueError(f"negative collides with another known positive for query at row {number}")
    if len({(row["query"], row["pos"][0], neg) for row in records for neg in row["neg"]}) != sum(len(row["neg"]) for row in records):
        raise ValueError("duplicate training triple")
    return records, known_positives


def encode_train(texts, *, tokenizer, model, torch, device, max_length):
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    outputs = model(**tokens)
    return torch.nn.functional.normalize(outputs.last_hidden_state[:, 0].float(), dim=-1)


def batch_loss(batch, known_positives, *, tokenizer, model, torch, device, max_length, temperature):
    queries = [row["query"] for row in batch]
    positives = [row["pos"][0] for row in batch]
    negatives = [neg for row in batch for neg in row["neg"]]
    query_vectors = encode_train(queries, tokenizer=tokenizer, model=model, torch=torch, device=device, max_length=max_length)
    action_vectors = encode_train(positives + negatives, tokenizer=tokenizer, model=model, torch=torch, device=device, max_length=max_length)
    scores = (query_vectors @ action_vectors.T) / temperature
    # Each query's other known positives are masked out of its denominator.
    # This also handles duplicate or equivalent positives across batch rows.
    masks = []
    for index, query in enumerate(queries):
        allowed = [other == index or action not in known_positives[query]
                   for other, action in enumerate(positives)]
        allowed += [action not in known_positives[query] for action in negatives]
        masks.append(allowed)
    mask = torch.tensor(masks, dtype=torch.bool, device=device)
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    labels = torch.arange(len(batch), device=device)
    return torch.nn.functional.cross_entropy(scores, labels)


def train(args):
    if (args.batch_size < 1 or args.eval_batch_size < 1 or args.epochs < 1 or args.accumulation < 1
            or args.temperature <= 0 or args.max_length < 8 or args.log_every < 1
            or args.learning_rate <= 0 or args.weight_decay < 0 or not 0 <= args.warmup_ratio <= 1
            or args.max_grad_norm <= 0):
        raise ValueError("invalid training parameters")
    train_path = args.dataset / "train" / "train.jsonl"
    records, known_positives = load_train(train_path)
    dev_queries, dev_actions, _ = load_split(args.dataset, "dev")
    if args.dry_run:
        print(json.dumps({"train_rows": len(records), "explicit_negatives": sum(len(row["neg"]) for row in records),
                          "dev_queries": len(dev_queries), "dev_actions": len(dev_actions),
                          "base_model": args.base_model, "mode": "validated_no_model_loaded"}))
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise RuntimeError("install torch and transformers on the training machine") from exc
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModel.from_pretrained(args.base_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(math.ceil(len(records) / args.batch_size) / args.accumulation)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps)
    args.output.mkdir(parents=True)
    history = []
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(records)))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        micro_batches = math.ceil(len(order) / args.batch_size)
        for micro in range(micro_batches):
            indices = order[micro * args.batch_size:(micro + 1) * args.batch_size]
            batch = [records[index] for index in indices]
            loss = batch_loss(batch, known_positives, tokenizer=tokenizer, model=model, torch=torch,
                              device=device, max_length=args.max_length, temperature=args.temperature)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite training loss")
            losses.append(float(loss.detach().cpu()))
            group_start = (micro // args.accumulation) * args.accumulation
            group_size = min(args.accumulation, micro_batches - group_start)
            (loss / group_size).backward()
            if (micro + 1) % args.accumulation == 0 or micro + 1 == micro_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (micro + 1) % args.log_every == 0 or micro + 1 == micro_batches:
                print(json.dumps({"epoch": epoch, "micro_batch": micro + 1, "micro_batches": micro_batches,
                                  "mean_loss_so_far": sum(losses) / len(losses)}), flush=True)
        epoch_dir = args.output / f"epoch_{epoch}"
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        model.eval()
        report, _ = evaluate_loaded(epoch_dir, args.dataset, "dev", torch=torch, tokenizer=tokenizer,
                                    model=model, device=device, batch_size=args.eval_batch_size,
                                    max_length=args.max_length, top_k=10)
        score = report["metrics"]["known_positive_mrr"]
        history.append({"epoch": epoch, "mean_loss": sum(losses) / len(losses), "dev": report})
        print(json.dumps({"epoch": epoch, "dev_known_positive_mrr": score}), flush=True)
        if score > best:
            best = score
            best_epoch = epoch
            shutil.copytree(epoch_dir, args.output / "best", dirs_exist_ok=True)
    summary = {"schema_version": "bge_clawgym_silver_training_v1", "base_model": args.base_model,
               "best_epoch": best_epoch, "best_dev_known_positive_mrr": best,
               "test_evaluated": False, "label_limit": "dev is Silver known-positive retrieval; test remains unopened",
               "config": vars(args) | {"dataset": str(args.dataset), "output": str(args.output)},
               "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(), "history": history}
    (args.output / "training_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_checkpoint": str(args.output / "best"), "best_epoch": best_epoch, "dev_score": best}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
