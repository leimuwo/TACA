#!/usr/bin/env python3
"""Fine-tune BGE-M3 with a new pair-classification head on weak TA labels."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
import os
from pathlib import Path

from evaluate_bge_clawgym import rows, sha256
from train_reranker_clawgym import prioritize_action

DEFAULT_MODEL = Path(os.environ.get("TA_MODEL_ROOT", "../model_hub")) / Path("bge-m3")
DEFAULT_DATA = Path("data/clawgym/ta_pair_weak_train_v1_20261013")


def validate_data(train, validation):
    if not train or not validation:
        raise ValueError("empty train or internal validation")
    ids = set()
    trajectories = []
    for part in (train, validation):
        current = set()
        for row in part:
            if row["pair_id"] in ids or row["pair_id"] in current:
                raise ValueError("duplicate pair ID")
            current.add(row["pair_id"])
            if row["split"] != "train" or row["label"] not in (0, 1):
                raise ValueError("invalid split or label")
            if not row["intent"] or not row["action"].startswith("[ACTION] "):
                raise ValueError("invalid pair text")
        ids.update(current)
        trajectories.append({row["trajectory_id"] for row in part})
    if trajectories[0] & trajectories[1]:
        raise ValueError("trajectory leakage between train and validation")
    return {"train_pairs": len(train), "validation_pairs": len(validation),
            "train_labels": dict(Counter(x["label"] for x in train)),
            "validation_labels": dict(Counter(x["label"] for x in validation))}


def classification_metrics(logits, labels):
    predictions = [int(x >= 0) for x in logits]
    tp = sum(p == y == 1 for p, y in zip(predictions, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(predictions, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(predictions, labels))
    tn = len(labels) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / len(labels), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}


def main(*, default_model=DEFAULT_MODEL, architecture="bge_m3_joint_pair_classifier",
         classification_head="newly_initialized_one_logit"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1
            or args.gradient_accumulation < 1 or args.learning_rate <= 0 or args.max_length < 32):
        raise ValueError("invalid training arguments")
    train_path = args.data / "train.jsonl"
    val_path = args.data / "internal_validation.jsonl"
    train = list(rows(train_path))
    validation = list(rows(val_path))
    setup = validate_data(train, validation)
    if args.dry_run:
        print(json.dumps({**setup, "mode": "dry_run", "model": str(args.model)}))
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise RuntimeError("install torch and transformers in the GPU environment") from exc
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    # BGE-M3 initializes this head; BGE reranker loads its pretrained head.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, local_files_only=True).to(args.device)
    if model.config.num_labels != 1:
        raise ValueError("pair classifier must have one output logit")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    updates_per_epoch = math.ceil(math.ceil(len(train) / args.batch_size) / args.gradient_accumulation)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, round(.1 * updates_per_epoch * args.epochs)),
                                                num_training_steps=updates_per_epoch * args.epochs)
    use_bf16 = args.device == "cuda" and torch.cuda.is_bf16_supported()
    args.output.mkdir(parents=True)

    def tokenize(batch):
        return tokenizer([x["intent"] for x in batch],
                         [prioritize_action(x["action"]) for x in batch],
                         padding=True, truncation=True, max_length=args.max_length,
                         return_tensors="pt").to(args.device)

    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        started = time.monotonic()
        batches = math.ceil(len(train) / args.batch_size)
        for batch_index, start in enumerate(range(0, len(train), args.batch_size), 1):
            batch = train[start:start + args.batch_size]
            targets = torch.tensor([x["label"] for x in batch], dtype=torch.float32, device=args.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**tokenize(batch)).logits[:, 0]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), targets)
            (loss / args.gradient_accumulation).backward()
            loss_sum += float(loss.detach()) * len(batch)
            if batch_index % args.gradient_accumulation == 0 or batch_index == batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if batch_index % 20 == 0 or batch_index == batches:
                print(json.dumps({"epoch": epoch, "stage": "train", "batches_done": batch_index,
                                  "batches_total": batches, "mean_loss": loss_sum / min(len(train), batch_index * args.batch_size),
                                  "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
        model.eval()
        scores = []
        labels = []
        with torch.inference_mode():
            for start in range(0, len(validation), args.eval_batch_size):
                batch = validation[start:start + args.eval_batch_size]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    output = model(**tokenize(batch)).logits[:, 0]
                scores.extend(output.float().cpu().tolist())
                labels.extend(x["label"] for x in batch)
        metrics = classification_metrics(scores, labels)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train), **metrics})
        print(json.dumps(history[-1]), flush=True)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best = args.output / "best"
            best.mkdir(exist_ok=True)
            model.save_pretrained(best)
            tokenizer.save_pretrained(best)
    report = {**setup, "model": str(args.model), "architecture": architecture,
              "classification_head": classification_head, "best_internal_validation_f1": best_f1,
              "history": history, "label_status": "GQQ weak supervision; train trajectory holdout only",
              "input_sha256": {"train": sha256(train_path), "internal_validation": sha256(val_path)}}
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"best_checkpoint": str(args.output / "best"),
                      "best_internal_validation_f1": best_f1}), flush=True)


if __name__ == "__main__":
    main()
