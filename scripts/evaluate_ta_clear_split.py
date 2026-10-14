#!/usr/bin/env python3
"""Evaluate a TA pair classifier on clear-Intent ClawGym dev or test windows."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from build_ta_coverage_pilot import INTENTS, ROOT, SPLITS, raw_index, split_index, window
from evaluate_bge_clawgym import rows, sha256
from evaluate_ta_coverage_pilot import evaluate
from train_reranker_clawgym import prioritize_action
from ta_evaluation_scope import exclusion_reason, window_exclusions, POLICY_VERSION

DEV_CLEAR = Path("data/clawgym/bge_m3_clear_dev_20261013/intent_clear.jsonl")
TEST_QUERIES = Path("data/clawgym/bge_dataset_split_20260930_v2/test/queries.jsonl")
TEST_AMBIGUOUS = Path("../data/clawgym/ambiguous/ambiguous_test.jsonl")
DEFAULT_CHECKPOINT = Path("outputs/ta_pair_cross_encoder_v1/best")


def query_keys(queries):
    keys = [(x["trajectory_id"], x["thought_step"], x["intent_id"]) for x in queries]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate trajectory/step/Intent query")
    return set(keys)


def clear_keys(split, dev_queries, test_queries, ambiguous):
    if split == "dev":
        if any(x.get("trajectory_id") is None or x.get("intent_id") is None for x in dev_queries):
            raise ValueError("invalid clear dev query")
        return query_keys(dev_queries)
    by_id = {x["query_id"]: x for x in test_queries}
    if len(by_id) != len(test_queries):
        raise ValueError("duplicate test query ID")
    excluded = set()
    for item in ambiguous:
        qid = item.get("source_query_id")
        if item.get("split") != "test" or item.get("ambiguous") is not True or qid not in by_id:
            raise ValueError("invalid test ambiguity record")
        query = by_id[qid]
        if ((item["trajectory_id"], item["thought_step"], item["intent_id"])
                != (query["trajectory_id"], query["thought_step"], query["intent_id"])):
            raise ValueError(f"test ambiguity mapping disagrees with query: {qid}")
        if qid in excluded:
            raise ValueError("duplicate test ambiguity record")
        excluded.add(qid)
    return query_keys([x for x in test_queries if x["query_id"] not in excluded])


def build_clear_windows(split, intent_rows, splits, raw, clear):
    selected = []
    excluded = Counter()
    for source in intent_rows:
        tid = source["trajectory_id"]
        if splits.get(tid) != split:
            continue
        current = [x for x in source.get("execution_intents", [])
                   if x.get("temporal_scope") == "current"]
        if not current:
            excluded["no_current_intent"] += 1
            continue
        if any((tid, source["thought_step"], x["intent_id"]) not in clear for x in current):
            excluded["ambiguous_or_unscreened_current_intent"] += 1
            continue
        if any(exclusion_reason(x.get('intent_text', '')) for x in current):
            excluded['non_atomic_bulk_file_read'] += 1
            continue
        if tid not in raw:
            raise ValueError(f"missing raw trajectory: {tid}")
        record = window(source, raw[tid], split)
        if not record["actions"] or not record["pairs"]:
            excluded["no_action_or_pair"] += 1
            continue
        selected.append(record)
    selected.sort(key=lambda x: x["window_id"])
    if len({x["window_id"] for x in selected}) != len(selected):
        raise ValueError("duplicate selected window")
    return selected, excluded


def predict(windows, checkpoint, device, batch_size, max_length, threshold):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install torch and transformers for model evaluation") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True).to(device).eval()
    if model.config.num_labels != 1:
        raise ValueError("checkpoint must have one output logit")
    output = []
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    with torch.inference_mode():
        for index, w in enumerate(windows, 1):
            intents = {x["intent_id"]: x for x in w["intents"]}
            actions = {x["action_id"]: x for x in w["actions"]}
            scores = {}
            for start in range(0, len(w["pairs"]), batch_size):
                batch = w["pairs"][start:start + batch_size]
                tokens = tokenizer([intents[x["intent_id"]]["text"] for x in batch],
                                   [prioritize_action(actions[x["action_id"]]["action_text"])
                                    for x in batch], padding=True, truncation=True,
                                   max_length=max_length, return_tensors="pt").to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    logits = model(**tokens).logits[:, 0]
                scores.update(zip((x["pair_id"] for x in batch), logits.float().cpu().tolist()))
            due = [x["intent_id"] for x in w["intents"] if x["temporal_scope"] == "current"]
            covered = {x["intent_id"] for x in w["pairs"] if scores[x["pair_id"]] >= threshold}
            matched = sum(iid in covered for iid in due)
            label = "faithful" if matched == len(due) else "partial" if matched else "not_executed"
            output.append({"window_id": w["window_id"], "trajectory_id": w["trajectory_id"],
                           "predicted_label": label, "pair_logits": scores,
                           "threshold": threshold, "covered_intents": matched,
                           "current_intents": len(due)})
            if index % 50 == 0 or index == len(windows):
                print(json.dumps({"stage": "predict", "windows_done": index,
                                  "windows_total": len(windows)}), flush=True)
    return output


def write_jsonl(path, items):
    with path.open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(split):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--intents", type=Path, default=INTENTS)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--raw", type=Path, default=ROOT)
    parser.add_argument("--dev-clear", type=Path, default=DEV_CLEAR)
    parser.add_argument("--test-queries", type=Path, default=TEST_QUERIES)
    parser.add_argument("--test-ambiguous", type=Path, default=TEST_AMBIGUOUS)
    parser.add_argument("--human-gold", type=Path,
                        help="Complete human window Gold for every selected clear window")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument('--reuse-predictions', type=Path,
                        help='Reuse matching pair logits from an earlier run; no GPU inference')
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1 or args.max_length < 32:
        raise ValueError("output must not exist; batch size and max length must be positive")
    if args.prepare_only and args.human_gold:
        raise ValueError("cannot score human Gold in prepare-only mode")
    clear = clear_keys(split,
                       list(rows(args.dev_clear)) if split == "dev" else [],
                       list(rows(args.test_queries)) if split == "test" else [],
                       list(rows(args.test_ambiguous)) if split == "test" else [])
    splits_for_audit = split_index(rows(args.splits))
    windows, excluded = build_clear_windows(split, list(rows(args.intents)),
                                            splits_for_audit, raw_index(args.raw), clear)
    if not windows:
        raise ValueError("no clear windows selected")
    gold = None
    if args.human_gold:
        gold = list(rows(args.human_gold))
        expected = {x["window_id"] for x in windows}
        if (len(gold) != len(expected) or len({x["window_id"] for x in gold}) != len(gold)
                or {x["window_id"] for x in gold} != expected):
            raise ValueError("human Gold must cover every selected clear window exactly once")
        if any(not x.get("gold_label") for x in gold):
            raise ValueError("human Gold has unlabeled windows")
    if args.reuse_predictions and args.prepare_only:
        raise ValueError('reuse-predictions cannot be combined with prepare-only')
    if args.reuse_predictions:
        previous = list(rows(args.reuse_predictions))
        indexed = {p['window_id']: p for p in previous}
        if len(indexed) != len(previous):
            raise ValueError('duplicate reused prediction')
        predictions = []
        for w in windows:
            p = indexed.get(w['window_id'])
            if p is None or set(p['pair_logits']) != {x['pair_id'] for x in w['pairs']}:
                raise ValueError('reused predictions do not match selected window pairs')
            if p['threshold'] != args.threshold:
                raise ValueError('reuse requires the original threshold')
            predictions.append(p)
    else:
        predictions = None if args.prepare_only else predict(
            windows, args.checkpoint, args.device, args.batch_size, args.max_length, args.threshold)
    report = {"split": split, "selection": "all current Intents clear under frozen retrieval screen",
              "windows": len(windows), "trajectories": len({x["trajectory_id"] for x in windows}),
              "pairs": sum(len(x["pairs"]) for x in windows), "excluded": dict(excluded),
              "checkpoint": None if args.prepare_only else str(args.checkpoint),
              "metrics_ready": gold is not None,
              "gold_source": "human_overall" if gold is not None else None,
              'scope_policy': POLICY_VERSION,
              'reused_predictions': str(args.reuse_predictions) if args.reuse_predictions else None,
              "source_sha256": {"intents": sha256(args.intents), "splits": sha256(args.splits),
                                "clear_screen": sha256(args.dev_clear if split == "dev" else args.test_ambiguous)}}
    if gold is not None:
        report["metrics"] = evaluate(windows, gold, predictions, gold_source="human-overall")
        report["human_gold_sha256"] = sha256(args.human_gold)
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "windows.jsonl", windows)
    excluded_windows = []
    for source in rows(args.intents):
        if splits_for_audit.get(source['trajectory_id']) != split:
            continue
        reasons = window_exclusions({'intents': source.get('execution_intents', [])})
        if reasons:
            excluded_windows.append({'trajectory_id': source['trajectory_id'],
                                     'thought_step': source['thought_step'], 'intents': reasons})
    write_jsonl(args.output / 'bulk_read_exclusions.jsonl', excluded_windows)
    write_jsonl(args.output / "ta_gold_template.jsonl",
                [{"window_id": x["window_id"], "gold_label": None,
                  "review_reason": ""} for x in windows])
    if predictions is not None:
        write_jsonl(args.output / "predictions.jsonl", predictions)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "windows": len(windows),
                      "metrics_ready": gold is not None}, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit("run evaluate_ta_clear_dev.py or evaluate_ta_clear_test.py")
