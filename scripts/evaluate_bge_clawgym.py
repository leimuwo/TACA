#!/usr/bin/env python3
"""Evaluate a BGE encoder on fixed ClawGym Silver retrieval pools."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_split(dataset, split, qrels_path=None):
    directory = dataset / split
    queries = list(rows(directory / "queries.jsonl"))
    actions = list(rows(directory / "actions.jsonl"))
    qrels = defaultdict(set)
    relations = list(rows(qrels_path or directory / "qrels.jsonl"))
    if len({row["query_id"] for row in queries}) != len(queries):
        raise ValueError("duplicate query IDs")
    if len({row["action_id"] for row in actions}) != len(actions):
        raise ValueError("duplicate action IDs")
    if any(not isinstance(row.get("query"), str) or not row["query"].startswith("[INTENT] ") for row in queries):
        raise ValueError("invalid query serialization")
    if any(not isinstance(row.get("action"), str) or not row["action"].startswith("[ACTION] ") for row in actions):
        raise ValueError("invalid action serialization")
    query_ids = {row["query_id"] for row in queries}
    action_ids = {row["action_id"] for row in actions}
    for row in relations:
        if row.get("query_id") not in query_ids or row.get("action_id") not in action_ids:
            raise ValueError("qrel references unknown query or action")
        if row.get("relevance") != 1:
            raise ValueError("only known positive qrels are supported")
        qrels[row["query_id"]].add(row["action_id"])
    if not queries or not actions or any(not qrels[row["query_id"]] or not qrels[row["query_id"]] <= action_ids for row in queries):
        raise ValueError("every query needs a known positive in the candidate pool")
    return queries, actions, qrels


def known_positive_metrics(ranks):
    if not ranks:
        raise ValueError("empty rank list")
    return {"query_count": len(ranks), "known_positive_at_1": sum(rank <= 1 for rank in ranks) / len(ranks),
            "known_positive_at_5": sum(rank <= 5 for rank in ranks) / len(ranks),
            "known_positive_at_10": sum(rank <= 10 for rank in ranks) / len(ranks),
            "known_positive_mrr": sum(1 / rank for rank in ranks) / len(ranks),
            "known_positive_mean_rank": sum(ranks) / len(ranks)}


def load_encoder(model_path, device):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install torch and transformers in the training environment") from exc
    actual = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if actual == "auto":
        actual = "cpu"
    if actual == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(actual).eval()
    return torch, tokenizer, model, actual


def embed(texts, *, torch, tokenizer, model, device, batch_size, max_length):
    chunks = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)
        with torch.inference_mode():
            vector = model(**encoded).last_hidden_state[:, 0]
            vector = torch.nn.functional.normalize(vector.float(), dim=-1)
        chunks.append(vector.cpu())
    return torch.cat(chunks)


def evaluate_loaded(model_path, dataset, split, *, torch, tokenizer, model, device, batch_size=32, max_length=512, top_k=10, qrels_path=None):
    if batch_size < 1 or max_length < 8 or top_k < 1:
        raise ValueError("invalid evaluation batch, length or top-k")
    queries, actions, qrels = load_split(dataset, split, qrels_path=qrels_path)
    action_vectors = embed([row["action"] for row in actions], torch=torch, tokenizer=tokenizer, model=model,
                           device=device, batch_size=batch_size, max_length=max_length)
    query_vectors = embed([row["query"] for row in queries], torch=torch, tokenizer=tokenizer, model=model,
                          device=device, batch_size=batch_size, max_length=max_length)
    action_ids = [row["action_id"] for row in actions]
    ranked = []
    ranks = []
    for index, query in enumerate(queries):
        scores = (action_vectors @ query_vectors[index]).tolist()
        order = sorted(range(len(actions)), key=lambda idx: (-scores[idx], action_ids[idx]))
        positive = qrels[query["query_id"]]
        rank = next(place for place, idx in enumerate(order, 1) if action_ids[idx] in positive)
        ranks.append(rank)
        ranked.append({"query_id": query["query_id"], "known_positive_rank": rank,
                       "known_positive_action_ids": sorted(positive),
                       "top_actions": [{"action_id": action_ids[idx], "score": scores[idx],
                                        "is_known_positive": action_ids[idx] in positive}
                                       for idx in order[:top_k]]})
    report = {"schema_version": "bge_clawgym_silver_retrieval_v1", "split": split,
              "model": str(model_path), "device": device, "pool_actions": len(actions),
              "known_positive_edges": sum(len(value) for value in qrels.values()),
              "metrics": known_positive_metrics(ranks),
              "label_limit": "Silver known positives only. Unlisted actions are unjudged; these are not Gold recall/MRR.",
              "input_sha256": {"queries": sha256(dataset / split / "queries.jsonl"),
                               "actions": sha256(dataset / split / "actions.jsonl"),
                               "qrels": sha256(qrels_path or dataset / split / "qrels.jsonl")},
              "qrels_source": str(qrels_path or dataset / split / "qrels.jsonl"),
              "max_length": max_length, "pooling": "normalized_cls"}
    return report, ranked


def evaluate(model_path, dataset, split, *, device="auto", batch_size=32, max_length=512, top_k=10, qrels_path=None):
    torch, tokenizer, model, actual = load_encoder(model_path, device)
    return evaluate_loaded(model_path, dataset, split, torch=torch, tokenizer=tokenizer,
                           model=model, device=actual, batch_size=batch_size,
                           max_length=max_length, top_k=top_k, qrels_path=qrels_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--qrels", type=Path, default=None, help="Optional alternate positive qrels JSONL")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    queries, actions, qrels = load_split(args.dataset, args.split, qrels_path=args.qrels)
    if args.validate_only:
        print(json.dumps({"split": args.split, "queries": len(queries), "actions": len(actions),
                          "known_positive_edges": sum(map(len, qrels.values()))}))
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    report, ranked = evaluate(args.model, args.dataset, args.split, device=args.device,
                              batch_size=args.batch_size, max_length=args.max_length, top_k=args.top_k,
                              qrels_path=args.qrels)
    args.output.mkdir(parents=True)
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output / "rankings.jsonl").open("w", encoding="utf-8") as stream:
        for row in ranked:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
