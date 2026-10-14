#!/usr/bin/env python3
"""Align frozen NightOwl Action vectors to the BGE retrieval space."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from evaluate_bge_clawgym import embed, load_split, sha256
from train_bge_clawgym import load_train
from train_nightowl_action_tower import (
    BGE, DATASET, NIGHTOWL, NIGHTOWL_REVISION, action_model_load_kwargs,
    baseline_metrics, encode_actions, evaluate, prepare_groups,
)


def run(args):
    if (args.epochs < 1 or args.batch_size < 1 or args.encode_batch_size < 1
            or args.accumulation < 1 or args.learning_rate <= 0 or args.temperature <= 0
            or args.teacher_weight < 0 or args.query_learning_rate <= 0
            or args.action_length < 8 or args.query_length < 8 or args.log_every < 1):
        raise ValueError("invalid training parameters")
    train_path = args.dataset / "train" / "train.jsonl"
    records, known_positives = load_train(train_path)
    groups = prepare_groups(records)
    queries, actions, qrels = load_split(args.dataset, "dev", qrels_path=args.qrels)
    baseline = baseline_metrics(args.baseline_rankings, actions, qrels, args.top_k)
    train_actions = list(dict.fromkeys(text for _, positive, negatives in groups
                                        for text in (positive, *negatives)))
    if args.dry_run:
        print(json.dumps({"groups": len(groups), "train_actions": len(train_actions),
                          "dev_queries": len(queries), "dev_actions": len(actions),
                          "train_query_tower": args.train_query_tower,
                          "bge_baseline": baseline, "mode": "validated_no_model_loaded"}), flush=True)
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as error:
        raise RuntimeError("install torch and transformers>=4.56.2 on the GPU machine") from error
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    query_tokenizer = AutoTokenizer.from_pretrained(args.query_model)
    query_model = AutoModel.from_pretrained(args.query_model).to(device)
    action_kwargs = action_model_load_kwargs(args.action_model, args.action_revision)
    action_tokenizer = AutoTokenizer.from_pretrained(args.action_model, **action_kwargs)
    action_model = AutoModel.from_pretrained(args.action_model, **action_kwargs).to(device).eval()
    if query_model.config.hidden_size != action_model.config.hidden_size:
        raise ValueError("tower dimensions differ")
    all_actions = list(dict.fromkeys(train_actions + [row["action"] for row in actions]))
    print(json.dumps({"stage": "encode_actions", "unique_actions": len(all_actions)}), flush=True)
    nightowl_vectors = encode_actions(all_actions, model=action_model, tokenizer=action_tokenizer,
                                      torch=torch, device=device, batch_size=args.encode_batch_size,
                                      max_length=args.action_length)
    del action_model
    if device == "cuda":
        torch.cuda.empty_cache()
    # BGE Action embeddings supply direct coordinates for the projection to learn.
    query_model.eval()
    teacher_vectors = embed(train_actions, torch=torch, tokenizer=query_tokenizer,
                            model=query_model, device=device, batch_size=args.encode_batch_size,
                            max_length=args.action_length)
    action_index = {text: index for index, text in enumerate(all_actions)}
    teacher_index = {text: index for index, text in enumerate(train_actions)}
    dev_indices = [action_index[row["action"]] for row in actions]
    projection = torch.nn.Linear(query_model.config.hidden_size, query_model.config.hidden_size,
                                 bias=False).to(device)
    torch.nn.init.eye_(projection.weight)
    if not args.train_query_tower:
        query_model.requires_grad_(False)
    parameters = [{"params": projection.parameters(), "lr": args.learning_rate}]
    if args.train_query_tower:
        parameters.append({"params": query_model.parameters(), "lr": args.query_learning_rate})
    optimizer = torch.optim.AdamW(parameters, weight_decay=args.weight_decay)
    micro_batches = math.ceil(len(groups) / args.batch_size)
    steps = math.ceil(micro_batches / args.accumulation) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * args.warmup_ratio), steps)
    args.output.mkdir(parents=True)
    history = []
    best = -1.0
    best_epoch = None
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(groups)))
        random.Random(args.seed + epoch).shuffle(order)
        projection.train()
        query_model.train(args.train_query_tower)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(micro_batches):
            batch = [groups[index] for index in order[micro * args.batch_size:(micro + 1) * args.batch_size]]
            tokens = query_tokenizer([row[0] for row in batch], padding=True, truncation=True,
                                     max_length=args.query_length, return_tensors="pt").to(device)
            with torch.set_grad_enabled(args.train_query_tower):
                query_vectors = torch.nn.functional.normalize(
                    query_model(**tokens).last_hidden_state[:, 0].float(), dim=-1)
            candidates = list(dict.fromkeys(text for _, positive, negatives in batch
                                              for text in (positive, *negatives)))
            candidate_index = {text: index for index, text in enumerate(candidates)}
            source = nightowl_vectors[[action_index[text] for text in candidates]].to(device)
            projected = torch.nn.functional.normalize(projection(source), dim=-1)
            scores = (query_vectors @ projected.T) / args.temperature
            # Each row uses only its validated explicit negatives in this first experiment.
            mask = torch.tensor([[text == row[1] or
                                  (text in row[2] and text not in known_positives[row[0]])
                                  for text in candidates] for row in batch], device=device)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            targets = torch.tensor([candidate_index[row[1]] for row in batch], device=device)
            retrieval_loss = torch.nn.functional.cross_entropy(scores, targets)
            teacher = teacher_vectors[[teacher_index[text] for text in candidates]].to(device)
            alignment_loss = (1 - (projected * teacher).sum(dim=-1)).mean()
            loss = retrieval_loss + args.teacher_weight * alignment_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite loss")
            loss_sum += float(loss.detach().cpu())
            group_size = min(args.accumulation, micro_batches - (micro // args.accumulation) * args.accumulation)
            (loss / group_size).backward()
            if (micro + 1) % args.accumulation == 0 or micro + 1 == micro_batches:
                torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
                if args.train_query_tower:
                    torch.nn.utils.clip_grad_norm_(query_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (micro + 1) % args.log_every == 0 or micro + 1 == micro_batches:
                print(json.dumps({"epoch": epoch, "micro_batch": micro + 1,
                                  "micro_batches": micro_batches,
                                  "mean_loss_so_far": loss_sum / (micro + 1)}), flush=True)
        projection.eval()
        query_model.eval()
        with torch.inference_mode():
            dev_vectors = torch.nn.functional.normalize(
                projection(nightowl_vectors[dev_indices].to(device)), dim=-1).cpu()
        report, rankings = evaluate(queries, actions, qrels, dev_vectors,
                                    query_model=query_model, query_tokenizer=query_tokenizer,
                                    torch=torch, device=device, batch_size=args.encode_batch_size,
                                    query_length=args.query_length, top_k=args.top_k)
        score = report[args.select_metric][f"known_positive_mrr_at_{args.top_k}"]
        history.append({"epoch": epoch, "mean_loss": loss_sum / micro_batches, "dev": report})
        print(json.dumps({"epoch": epoch, **report}), flush=True)
        if score > best:
            best, best_epoch = score, epoch
            checkpoint = args.output / "best"
            checkpoint.mkdir(exist_ok=True)
            torch.save({key: value.detach().cpu() for key, value in projection.state_dict().items()},
                       checkpoint / "projection.pt")
            if args.train_query_tower:
                query_model.save_pretrained(checkpoint / "query_model")
                query_tokenizer.save_pretrained(checkpoint / "query_model")
            with (args.output / "best_dev_rankings.jsonl").open("w", encoding="utf-8") as stream:
                for row in rankings:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"schema_version": "nightowl_bge_projection_v1", "best_epoch": best_epoch,
               "best_dev_mrr_at_k": best, "bge_baseline": baseline, "history": history,
               "query_tower_trained": args.train_query_tower, "action_tower_frozen": True,
               "projection": {"type": "linear", "bias": False,
                              "input_dim": query_model.config.hidden_size,
                              "output_dim": query_model.config.hidden_size,
                              "normalization": "l2_after_projection"},
               "test_evaluated": False, "label_limit": "Silver known positives only",
               "config": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
               "input_sha256": {"train": sha256(train_path), "qrels": sha256(args.qrels)}}
    (args.output / "training_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_checkpoint": str(args.output / "best"), "best_epoch": best_epoch,
                      "dev_score": best}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--qrels", type=Path, default=Path(
        "data/clawgym/bge_dev_qrels_exact_text_20261009/qrels_review_closed.jsonl"))
    parser.add_argument("--baseline-rankings", type=Path, default=Path(
        "outputs/bge_clawgym_silver_v2_dev_top20_20261009/rankings.jsonl"))
    parser.add_argument("--query-model", type=Path, default=BGE)
    parser.add_argument("--action-model", type=Path, default=NIGHTOWL)
    parser.add_argument("--action-revision", default=NIGHTOWL_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--query-learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-query-tower", action="store_true")
    parser.add_argument("--teacher-weight", type=float, default=1.0)
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
