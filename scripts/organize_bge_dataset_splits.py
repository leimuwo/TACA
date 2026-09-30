#!/usr/bin/env python3
"""Package Silver BGE training data and split-local retrieval evaluation inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical(record) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_text(action):
    return "[ACTION] " + canonical({"function": action["function"], "args": action.get("args", {})})


def intent_for(match, intent_steps, intent_id):
    revised = {item["intent_id"]: item for item in match.get("revised_intents", [])}
    original = {item["intent_id"]: item for item in intent_steps[(match["trajectory_id"], match["thought_step"])].get("execution_intents", [])}
    return revised.get(intent_id) or original.get(intent_id)


def query_id(trajectory_id, thought_step, intent_id):
    raw = canonical([trajectory_id, thought_step, intent_id]).encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def run(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    actions = {row["action_id"]: row for row in read_jsonl(args.actions)}
    intents = {(row["trajectory_id"], row["thought_step"]): row for row in read_jsonl(args.intents)}
    matches = list(read_jsonl(args.matches))
    by_split = defaultdict(list)
    split_trajectories = defaultdict(set)
    for row in matches:
        by_split[row["split"]].append(row)
        split_trajectories[row["split"]].add(row["trajectory_id"])
    if set(by_split) != {"train", "dev", "test"}:
        raise ValueError("expected train, dev and test in source matches")
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if split_trajectories[left] & split_trajectories[right]:
            raise ValueError(f"trajectory leakage between {left} and {right}")
    train_ids = {row["action_id"] for row in by_split["train"]}
    for split in ("dev", "test"):
        if train_ids & {row["action_id"] for row in by_split[split]}:
            raise ValueError(f"action ID leakage from train into {split}")

    output.mkdir(parents=True)
    train_dir = output / "train"
    train_dir.mkdir()
    source_train = list(read_jsonl(args.train_rows))
    source_grouped = list(read_jsonl(args.train_grouped))
    if len(source_train) != len(source_grouped):
        raise ValueError("train and grouped provenance row counts differ")
    known_positives = defaultdict(set)
    for row in source_train:
        known_positives[row["query"]].update(row["pos"])
    train = []
    grouped = []
    removed_collision = 0
    removed_empty = 0
    seen_triples = set()
    for row, detail in zip(source_train, source_grouped, strict=True):
        if row["query"] != detail["query"] or row["pos"] != detail["pos"] or row["neg"] != detail["neg"]:
            raise ValueError("train/grouped provenance mismatch")
        kept_indices = []
        for index, negative in enumerate(row["neg"]):
            key = (row["query"], row["pos"][0], negative)
            if negative in known_positives[row["query"]] or key in seen_triples:
                removed_collision += 1
            else:
                seen_triples.add(key)
                kept_indices.append(index)
        if not kept_indices:
            removed_empty += 1
            continue
        train.append({**row, "neg": [row["neg"][index] for index in kept_indices]})
        grouped.append({**detail, "neg": [detail["neg"][index] for index in kept_indices],
                        "candidate_ids": [detail["candidate_ids"][index] for index in kept_indices],
                        "negative_categories": [detail["negative_categories"][index] for index in kept_indices]})
    write_jsonl(train_dir / "train.jsonl", train)
    write_jsonl(train_dir / "train_grouped.jsonl", grouped)
    if not train or any(not row.get("query", "").startswith("[INTENT] ") or len(row.get("pos", [])) != 1 or not 1 <= len(row.get("neg", [])) <= 3 for row in train):
        raise ValueError("invalid train query/pos/neg format")
    if len({(row["query"], row["pos"][0], negative) for row in train for negative in row["neg"]}) != sum(len(row["neg"]) for row in train):
        raise ValueError("duplicate train triplet")

    counts = {"train": {"rows": len(train), "triplets": sum(len(row["neg"]) for row in train),
                        "removed_positive_collisions_or_duplicates": removed_collision,
                        "removed_rows_without_negatives": removed_empty}}
    for split in ("dev", "test"):
        split_dir = output / split
        split_dir.mkdir()
        rows = by_split[split]
        candidates = []
        seen_action_ids = set()
        query_map = {}
        qrels = set()
        for row in rows:
            action_id = row["action_id"]
            if action_id not in seen_action_ids:
                action = actions[action_id]["action"]
                candidates.append({"action_id": action_id, "action": action_text(action),
                                   "trajectory_id": row["trajectory_id"], "thought_step": row["thought_step"],
                                   "raw_action_sha256": row["raw_action_sha256"]})
                seen_action_ids.add(action_id)
            if row["label"] != "exact_match":
                continue
            for intent_id in row.get("selected_intent_ids", []):
                intent = intent_for(row, intents, intent_id)
                if not intent or not intent.get("intent_text"):
                    raise ValueError(f"missing exact-match intent {row['trajectory_id']}:{row['thought_step']}:{intent_id}")
                qid = query_id(row["trajectory_id"], row["thought_step"], intent_id)
                query = {"query_id": qid, "query": "[INTENT] " + intent["intent_text"],
                         "trajectory_id": row["trajectory_id"], "thought_step": row["thought_step"],
                         "intent_id": intent_id, "supervision_tier": "model_preliminary"}
                if qid in query_map and query_map[qid]["query"] != query["query"]:
                    raise ValueError(f"conflicting text for query {qid}")
                query_map[qid] = query
                qrels.add((qid, action_id))
        queries = sorted(query_map.values(), key=lambda row: row["query_id"])
        candidates.sort(key=lambda row: row["action_id"])
        relations = [{"query_id": qid, "action_id": aid, "relevance": 1,
                      "source_label": "exact_match", "supervision_tier": "model_preliminary"}
                     for qid, aid in sorted(qrels)]
        write_jsonl(split_dir / "queries.jsonl", queries)
        write_jsonl(split_dir / "actions.jsonl", candidates)
        write_jsonl(split_dir / "qrels.jsonl", relations)
        counts[split] = {"trajectories": len(split_trajectories[split]), "queries": len(queries),
                         "candidate_actions": len(candidates), "known_positive_edges": len(relations)}
        if not queries or not candidates or not relations:
            raise ValueError(f"empty {split} evaluation partition")
        if any(rel["action_id"] not in seen_action_ids for rel in relations):
            raise ValueError(f"{split} qrel references missing action")

    readme = """# BGE Silver 数据划分与运行方式

从 Project 根目录运行。`train/train.jsonl` 是 `query` / `pos` / `neg` 训练数据；`train_grouped.jsonl` 保留来源和负例类别。dev 用于每轮选择最佳 checkpoint；test 仅在训练方案固定后评一次。两个评测 split 的 `queries.jsonl`、`actions.jsonl` 和 `qrels.jsonl` 分别是意图、固定 Action 候选池和已知正例关系。

## 安装依赖

在支持训练的机器上创建 Python 3.10+ 环境，安装 `pip install -e '.[training]'`。GPU 机器须安装与本机 CUDA 匹配的 PyTorch；本项目基线使用本地模型 `/inspire/hdd/project/control-technology/public/model_hub/bge-base-en-v1.5`；当前数据的意图和 Action 主要为英文。

## 离线检查

```bash
python scripts/train_bge_clawgym.py --dataset data/clawgym/bge_dataset_split_20260930_v2 --output outputs/bge_clawgym_silver_v2 --base-model /inspire/hdd/project/control-technology/public/model_hub/bge-base-en-v1.5 --dry-run
python scripts/evaluate_bge_clawgym.py --dataset data/clawgym/bge_dataset_split_20260930_v2 --split dev --model /inspire/hdd/project/control-technology/public/model_hub/bge-base-en-v1.5 --output outputs/bge_base_dev --validate-only
python scripts/evaluate_bge_clawgym.py --dataset data/clawgym/bge_dataset_split_20260930_v2 --split test --model /inspire/hdd/project/control-technology/public/model_hub/bge-base-en-v1.5 --output outputs/bge_base_test --validate-only
```

## 训练、选择 checkpoint、最终评测

```bash
python scripts/train_bge_clawgym.py --dataset data/clawgym/bge_dataset_split_20260930_v2 --output outputs/bge_clawgym_silver_v2 --base-model /inspire/hdd/project/control-technology/public/model_hub/bge-base-en-v1.5 --device auto
python scripts/evaluate_bge_clawgym.py --dataset data/clawgym/bge_dataset_split_20260930_v2 --split test --model outputs/bge_clawgym_silver_v2/best --output outputs/bge_clawgym_silver_v2_test --device auto
```

训练默认 3 轮、batch 8、梯度累积 4、学习率 2e-5、最大长度 512。每轮评 dev，按 `known_positive_mrr` 保存 `best/`。训练日志和 dev 分数写入 `training_report.json`；test 的 `metrics.json`、`rankings.jsonl` 写入指定评测目录。所有输出目录须事先不存在。若显存不足，可调低 `--batch-size` 或 `--eval-batch-size`，并适当增加 `--accumulation`。训练前的 dev 基线已保存在 `outputs/bge_base_dev/metrics.json`；与训练后的 dev 指标比较时须使用同一模型、数据、长度及池化设置。

训练使用显式负例和 batch 内 Action 负例，并屏蔽同一 query 的已知正例。`qrels` 仅列出已有模型初标的 `exact_match` 正例；未列出的 Action 是**未判定**。`known_positive_*` 指标只能用于 Silver 诊断与模型比较，不能作为 Gold Recall/MRR。正式评测需要补齐人工相关性标注。`manifest.json` 记录划分数量和输入、输出 SHA-256。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    manifest = {"schema_version": "bge_silver_split_dataset_v1", "date": "2026-09-30",
                "counts": counts, "supervision": "silver_model_preliminary_and_validated",
                "evaluation_ready": "silver_retrieval_diagnostic_only",
                "known_positive_only": True,
                "input_sha256": {name: sha256(path) for name, path in {"train_rows": args.train_rows, "train_grouped": args.train_grouped, "matches": args.matches, "actions": args.actions, "intents": args.intents}.items()},
                "output_sha256": {path.relative_to(output).as_posix(): sha256(path) for path in sorted(output.rglob("*.jsonl"))}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical(counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rows", type=Path, required=True)
    parser.add_argument("--train-grouped", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--intents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
