#!/usr/bin/env python3
"""Train NightOwl/BGE retrieval in three successive, comparable stages."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from evaluate_bge_clawgym import embed, load_split, sha256
from train_bge_clawgym import load_train
from train_nightowl_action_tower import (
    BGE, DATASET, NIGHTOWL, NIGHTOWL_REVISION, action_input,
    action_model_load_kwargs, action_tool, baseline_metrics, encode_actions,
    evaluate, prepare_groups,
)

URL = re.compile(r"https?://[^\s\"'<>]+")
FILE = re.compile(r"(?<![\w])(?:[\w.~-]+/)+[\w.~-]+\.[a-zA-Z0-9]{1,8}(?![\w])")
SCRIPT = re.compile(r"(?<![\w/])([\w.-]+\.py)(?![\w])", re.IGNORECASE)
RUN_SCRIPT = re.compile(r"\b(?:python|python3|python\d\.\d)\s+(?:\S+/)?([\w.-]+\.py)(?![\w])")


@lru_cache(maxsize=None)
def action_fields(text):
    action = json.loads(text.removeprefix("[ACTION] "))
    args = action.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("Action args must be an object")
    return action["function"], args


def explicit_targets(query, positive):
    """Keep only targets literally named by the Intent and present in its positive."""
    _, args = action_fields(positive)
    command = str(args.get("command", ""))
    targets = []
    for field in ("path", "file", "file_path", "url", "targetUrl"):
        value = args.get(field)
        if isinstance(value, str) and value and value.lower() in query.lower():
            targets.append((field, value))
    for url in URL.findall(query):
        if url in command:
            targets.append(("command_url", url))
    for path in FILE.findall(query):
        if path in command:
            targets.append(("command_file", path))
    running = RUN_SCRIPT.search(command)
    if running and running.group(1).lower() in {name.lower() for name in SCRIPT.findall(query)}:
        targets.append(("script", running.group(1).lower()))
    return list(dict.fromkeys(targets))


def is_confirmed_target_mismatch(query, positive, candidate):
    """Do not label unjudged Actions negative without an explicit target conflict."""
    tool, _ = action_fields(positive)
    other_tool, args = action_fields(candidate)
    if tool != other_tool or positive == candidate:
        return False
    for field, target in explicit_targets(query, positive):
        if field == "command_url":
            other = set(URL.findall(str(args.get("command", ""))))
            if other and target not in other and not any(value.lower() in query.lower() for value in other):
                return True
        elif field == "command_file":
            other = set(FILE.findall(str(args.get("command", ""))))
            if other and target not in other and not any(value.lower() in query.lower() for value in other):
                return True
        elif field == "script":
            running = RUN_SCRIPT.search(str(args.get("command", "")))
            if (running and running.group(1).lower() != target
                    and running.group(1).lower() not in {name.lower() for name in SCRIPT.findall(query)}):
                return True
        else:
            other = args.get(field)
            if (isinstance(other, str) and other and other != target
                    and other.lower() not in query.lower()):
                return True
    return False


def build_safe_pool(groups, action_texts, known_positives):
    by_tool = defaultdict(list)
    for index, text in enumerate(action_texts):
        by_tool[action_tool(text)].append(index)
    safe = []
    for query, positive, negatives in groups:
        blocked = known_positives[query] | set(negatives)
        safe.append([index for index in by_tool[action_tool(positive)]
                     if action_texts[index] not in blocked
                     and is_confirmed_target_mismatch(query, positive, action_texts[index])])
    return safe


def select_pool_negatives(eligible, scores, *, hard_count, random_count, rng):
    if not eligible:
        return []
    order = sorted(eligible, key=lambda index: (-scores[index], index))
    hard = order[:hard_count]
    rest = order[hard_count:]
    random_items = rng.sample(rest, min(random_count, len(rest)))
    return hard + random_items


def set_action_trainable(model, layer_count):
    model.requires_grad_(False)
    if layer_count == -1:
        model.requires_grad_(True)
    elif layer_count > 0:
        layers = getattr(model, "layers", None)
        if layers is None or layer_count > len(layers):
            raise ValueError("ModernBERT layers not found or unfreeze count too large")
        for layer in layers[-layer_count:]:
            layer.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def run(args):
    if (args.stage not in (1, 2, 3) or args.epochs < 1 or args.batch_size < 1
            or args.encode_batch_size < 1 or args.accumulation < 1
            or args.hard_negatives < 0 or args.random_negatives < 0
            or args.hard_negatives + args.random_negatives < 1
            or args.action_unfreeze_layers < -1
            or (args.stage == 3 and args.action_unfreeze_layers == 0)
            or (args.stage != 3 and args.action_unfreeze_layers != 0)
            or min(args.learning_rate, args.query_learning_rate, args.action_learning_rate,
                   args.temperature) <= 0 or args.teacher_weight < 0
            or args.weight_decay < 0 or not 0 <= args.warmup_ratio <= 1
            or args.query_length < 8 or args.action_length < 8 or args.top_k < 1
            or args.log_every < 1):
        raise ValueError("invalid stage or training parameters")
    if not args.init_checkpoint.is_dir() or not (args.init_checkpoint / "projection.pt").is_file():
        raise FileNotFoundError("--init-checkpoint must contain projection.pt")
    if args.stage == 3 and not (args.init_checkpoint / "query_model").is_dir():
        raise FileNotFoundError("stage 3 requires the stage 2 query_model checkpoint")
    train_path = args.dataset / "train" / "train.jsonl"
    records, known_positives = load_train(train_path)
    groups = prepare_groups(records)
    queries, actions, qrels = load_split(args.dataset, "dev", qrels_path=args.qrels)
    baseline = baseline_metrics(args.baseline_rankings, actions, qrels, args.top_k)
    train_actions = list(dict.fromkeys(text for _, positive, negatives in groups
                                        for text in (positive, *negatives)))
    safe = build_safe_pool(groups, train_actions, known_positives)
    stats = {"groups": len(groups), "train_actions": len(train_actions),
             "groups_with_safe_pool_negatives": sum(bool(row) for row in safe),
             "exec_groups_with_safe_pool_negatives": sum(bool(candidates) and action_tool(group[1]) == "exec"
                                                          for group, candidates in zip(groups, safe)),
             "safe_pool_pairs": sum(map(len, safe)), "dev_queries": len(queries),
             "dev_actions": len(actions), "stage": args.stage}
    if args.dry_run:
        print(json.dumps({**stats, "bge_baseline": baseline,
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
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    query_path = args.init_checkpoint / "query_model" if args.stage == 3 else args.query_model
    query_tokenizer = AutoTokenizer.from_pretrained(query_path)
    query_model = AutoModel.from_pretrained(query_path).to(device)
    action_path = args.init_checkpoint / "action_model" if (args.init_checkpoint / "action_model").is_dir() else args.action_model
    action_kwargs = action_model_load_kwargs(action_path, args.action_revision)
    action_tokenizer = AutoTokenizer.from_pretrained(action_path, **action_kwargs)
    action_model = AutoModel.from_pretrained(action_path, **action_kwargs).to(device)
    if query_model.config.hidden_size != action_model.config.hidden_size:
        raise ValueError("tower dimensions differ")
    projection = torch.nn.Linear(action_model.config.hidden_size, query_model.config.hidden_size,
                                 bias=False).to(device)
    projection.load_state_dict(torch.load(args.init_checkpoint / "projection.pt",
                                          map_location="cpu", weights_only=True))
    if args.stage == 1:
        query_model.requires_grad_(False)
    action_parameters = set_action_trainable(action_model, args.action_unfreeze_layers)

    # The BGE teacher always stays at its original checkpoint, including stages 2 and 3.
    teacher_model = AutoModel.from_pretrained(args.query_model).to(device).eval()
    teacher_model.requires_grad_(False)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.query_model)
    teacher_vectors = embed(train_actions, torch=torch, tokenizer=teacher_tokenizer,
                            model=teacher_model, device=device, batch_size=args.encode_batch_size,
                            max_length=args.action_length)
    del teacher_model
    if device == "cuda":
        torch.cuda.empty_cache()

    all_actions = list(dict.fromkeys(train_actions + [row["action"] for row in actions]))
    action_index = {text: index for index, text in enumerate(all_actions)}
    teacher_index = {text: index for index, text in enumerate(train_actions)}
    train_queries = list(dict.fromkeys(row[0] for row in groups))
    query_index = {text: index for index, text in enumerate(train_queries)}
    cached_actions = None
    if args.stage != 3:
        cached_actions = encode_actions(all_actions, model=action_model.eval(), tokenizer=action_tokenizer,
                                        torch=torch, device=device, batch_size=args.encode_batch_size,
                                        max_length=args.action_length)
        del action_model
        if device == "cuda":
            torch.cuda.empty_cache()
    optimizer_groups = [{"params": projection.parameters(), "lr": args.learning_rate}]
    if args.stage >= 2:
        optimizer_groups.append({"params": query_model.parameters(), "lr": args.query_learning_rate})
    if action_parameters:
        optimizer_groups.append({"params": action_parameters, "lr": args.action_learning_rate})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
    micro_batches = math.ceil(len(groups) / args.batch_size)
    steps = math.ceil(micro_batches / args.accumulation) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * args.warmup_ratio), steps)
    args.output.mkdir(parents=True)
    history = []
    best, best_epoch = -1.0, None

    for epoch in range(1, args.epochs + 1):
        query_model.eval()
        if args.stage == 3:
            action_model.eval()
            mining_actions = encode_actions(all_actions, model=action_model,
                                            tokenizer=action_tokenizer, torch=torch,
                                            device=device, batch_size=args.encode_batch_size,
                                            max_length=args.action_length)
        else:
            mining_actions = cached_actions
        mining_queries = embed(train_queries, torch=torch, tokenizer=query_tokenizer,
                               model=query_model, device=device, batch_size=args.encode_batch_size,
                               max_length=args.query_length)
        with torch.inference_mode():
            projected_pool = torch.nn.functional.normalize(
                projection(mining_actions[:len(train_actions)].to(device)), dim=-1).cpu()
        mining_scores = mining_queries @ projected_pool.T
        rng = random.Random(args.seed + epoch)
        extra = [select_pool_negatives(safe[index], mining_scores[query_index[row[0]]],
                                       hard_count=args.hard_negatives,
                                       random_count=args.random_negatives, rng=rng)
                 for index, row in enumerate(groups)]
        order = list(range(len(groups)))
        rng.shuffle(order)
        query_model.train(args.stage >= 2)
        projection.train()
        if args.stage == 3:
            action_model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(micro_batches):
            indices = order[micro * args.batch_size:(micro + 1) * args.batch_size]
            batch = [groups[index] for index in indices]
            per_row = [list(row[2]) + [train_actions[index] for index in extra[group_index]]
                       for group_index, row in zip(indices, batch)]
            candidates = list(dict.fromkeys(text for row, negatives in zip(batch, per_row)
                                              for text in [row[1], *negatives]))
            candidate_index = {text: index for index, text in enumerate(candidates)}
            tokens = query_tokenizer([row[0] for row in batch], padding=True, truncation=True,
                                     max_length=args.query_length, return_tensors="pt").to(device)
            with torch.set_grad_enabled(args.stage >= 2):
                query_vectors = torch.nn.functional.normalize(
                    query_model(**tokens).last_hidden_state[:, 0].float(), dim=-1)
            if args.stage == 3:
                action_tokens = action_tokenizer([action_input(text) for text in candidates],
                                                 padding=True, truncation=True,
                                                 max_length=args.action_length,
                                                 return_tensors="pt").to(device)
                source = torch.nn.functional.normalize(
                    action_model(**action_tokens).last_hidden_state[:, 0].float(), dim=-1)
            else:
                source = cached_actions[[action_index[text] for text in candidates]].to(device)
            projected = torch.nn.functional.normalize(projection(source), dim=-1)
            scores = (query_vectors @ projected.T) / args.temperature
            mask = torch.tensor([[text == row[1] or
                                  (text in negatives and text not in known_positives[row[0]])
                                  for text in candidates]
                                 for row, negatives in zip(batch, per_row)], device=device)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            targets = torch.tensor([candidate_index[row[1]] for row in batch], device=device)
            retrieval_loss = torch.nn.functional.cross_entropy(scores, targets)
            teacher = teacher_vectors[[teacher_index[text] for text in candidates]].to(device)
            alignment_loss = (1 - (projected * teacher).sum(dim=-1)).mean()
            loss = retrieval_loss + args.teacher_weight * alignment_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite loss")
            loss_sum += float(loss.detach().cpu())
            group_size = min(args.accumulation, micro_batches -
                             (micro // args.accumulation) * args.accumulation)
            (loss / group_size).backward()
            if (micro + 1) % args.accumulation == 0 or micro + 1 == micro_batches:
                trainable = [parameter for group in optimizer.param_groups
                             for parameter in group["params"]]
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (micro + 1) % args.log_every == 0 or micro + 1 == micro_batches:
                print(json.dumps({"epoch": epoch, "micro_batch": micro + 1,
                                  "micro_batches": micro_batches,
                                  "mean_loss_so_far": loss_sum / (micro + 1)}), flush=True)
        projection.eval()
        query_model.eval()
        if args.stage == 3:
            action_model.eval()
            dev_source = encode_actions([row["action"] for row in actions],
                                        model=action_model, tokenizer=action_tokenizer,
                                        torch=torch, device=device,
                                        batch_size=args.encode_batch_size,
                                        max_length=args.action_length)
        else:
            dev_source = cached_actions[[action_index[row["action"]] for row in actions]]
        with torch.inference_mode():
            dev_vectors = torch.nn.functional.normalize(
                projection(dev_source.to(device)), dim=-1).cpu()
        report, rankings = evaluate(queries, actions, qrels, dev_vectors,
                                    query_model=query_model, query_tokenizer=query_tokenizer,
                                    torch=torch, device=device, batch_size=args.encode_batch_size,
                                    query_length=args.query_length, top_k=args.top_k)
        score = report[args.select_metric][f"known_positive_mrr_at_{args.top_k}"]
        history.append({"epoch": epoch, "mean_loss": loss_sum / micro_batches,
                        "sampled_pool_negatives": sum(map(len, extra)), "dev": report})
        print(json.dumps({"epoch": epoch, **report}), flush=True)
        if score > best:
            best, best_epoch = score, epoch
            checkpoint = args.output / "best"
            checkpoint.mkdir(exist_ok=True)
            torch.save({key: value.detach().cpu() for key, value in projection.state_dict().items()},
                       checkpoint / "projection.pt")
            if args.stage >= 2:
                query_model.save_pretrained(checkpoint / "query_model")
                query_tokenizer.save_pretrained(checkpoint / "query_model")
            if args.stage == 3:
                action_model.save_pretrained(checkpoint / "action_model")
                action_tokenizer.save_pretrained(checkpoint / "action_model")
            with (args.output / "best_dev_rankings.jsonl").open("w", encoding="utf-8") as stream:
                for row in rankings:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"schema_version": "nightowl_progressive_v1", "stage": args.stage,
               "best_epoch": best_epoch, "best_dev_mrr_at_k": best,
               "bge_baseline": baseline, "sampling": stats, "history": history,
               "test_evaluated": False, "label_limit": "Silver known positives only",
               "config": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
               "input_sha256": {"train": sha256(train_path), "qrels": sha256(args.qrels)}}
    (args.output / "training_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_checkpoint": str(args.output / "best"),
                      "best_epoch": best_epoch, "dev_score": best}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--qrels", type=Path, default=Path(
        "data/clawgym/bge_dev_qrels_exact_text_20261009/qrels_review_closed.jsonl"))
    parser.add_argument("--baseline-rankings", type=Path, default=Path(
        "outputs/bge_clawgym_silver_v2_dev_top20_20261009/rankings.jsonl"))
    parser.add_argument("--query-model", type=Path, default=BGE)
    parser.add_argument("--action-model", type=Path, default=NIGHTOWL)
    parser.add_argument("--action-revision", default=NIGHTOWL_REVISION)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--hard-negatives", type=int, default=4)
    parser.add_argument("--random-negatives", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--query-learning-rate", type=float, default=2e-5)
    parser.add_argument("--action-learning-rate", type=float, default=5e-6)
    parser.add_argument("--action-unfreeze-layers", type=int, default=0)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
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
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
