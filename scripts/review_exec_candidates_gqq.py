#!/usr/bin/env python3
"""Ask GQQ to audit unseen ClawGym train exec pairs; never modify training data."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from build_bge_m3_exec_supervision import normalized_action
from build_reviewed_exec_train import BASE, POOL, QUEUE, action
from evaluate_bge_clawgym import rows
from screen_ambiguous_intents_gqq import load_env
from train_bge_clawgym import load_train

SYSTEM = """You audit retrieval-training labels. Decide in this order:
1. Read ONLY the Intent text. Identify the requested operation, target, and any required scope/parameters. Return target_evidence as an exact substring of the Intent that identifies the target; use null if no identifiable target is present. A generic reference is NOT identifying evidence. Examples: 'Check if the input files exist' is ambiguous (which files?); 'Explore the input directory structure' is ambiguous without a path; 'Check external/usb.ids' has an identified target. Never use the positive Action, negatives, source_quote, or trajectory to fill a missing Intent target. Mark ambiguous=true if multiple plausible Actions could satisfy the same underspecified Intent.
2. Compare the proposed positive Action with ALL explicit Intent requirements. Mark positive_match=false for preparation only, different operation/target, or missing required outputs or parameters. If ambiguous=true, positive_match may describe a plausible match, but no negative is safe for training.
3. For EACH negative, ask whether it could achieve the same requested result, even with a different command or tool. Equivalent methods are NOT negatives: cat -A, xxd, and od -c can all reveal whitespace; python and python3 may run the same script; grep, sed, and head may inspect the same file; searching and direct fetching may both find a resource. Only mark safe_negative=true when an explicit Intent requirement is clearly violated, such as wrong operation, wrong named path/URL/object, or missing named parameter/output. Do not call something wrong merely because it is a different implementation, narrower display, or contains an extra harmless option. If uncertain, mark false with category uncertain. If ambiguous=true, mark every negative false.
Return one JSON object only with fields: ambiguous (boolean), target_evidence (exact Intent substring or null), positive_match (boolean), positive_reason (string), ambiguity_reason (string), negatives (array of objects with candidate_edge_id, safe_negative (boolean), category (different_operation, wrong_target, missing_required_parameter, other, or uncertain), reason (string))."""


def select(pool, queue, train, held_out, limit):
    by_id = {item["edge_id"]: item for item in pool}
    existing = {(item["query"], normalized_action(item["pos"][0])) for item in train}
    known = defaultdict(set)
    for item in pool:
        if item["split"] == "train" and item["eligibility"] == "candidate":
            known["[INTENT] " + item["intent_text"]].add(normalized_action(action(item)))
    groups = defaultdict(list)
    for item in queue:
        groups[item["source_edge_id"]].append(item)
    result = []
    for source_id in sorted(groups):
        source = by_id.get(source_id)
        if (source is None or source["split"] != "train" or source["eligibility"] != "candidate"
                or source["positive_action"]["function"] != "exec"
                or source["trajectory_id"] in held_out):
            continue
        query = "[INTENT] " + source["intent_text"]
        pos = action(source)
        if (query, normalized_action(pos)) in existing:
            continue
        negatives = []
        for item in groups[source_id]:
            candidate = by_id.get(item["candidate_edge_id"])
            if (candidate is None or candidate["split"] != "train"
                    or candidate["trajectory_id"] != source["trajectory_id"]
                    or normalized_action(item["positive_action"]) != normalized_action(pos)
                    or normalized_action(item["candidate_action"]) != normalized_action(action(candidate))
                    or normalized_action(action(candidate)) in known[query]):
                continue
            negatives.append({"candidate_edge_id": candidate["edge_id"], "action": action(candidate)})
        if negatives:
            result.append({"source_edge_id": source_id, "trajectory_id": source["trajectory_id"],
                           "source_quote": source.get("source_quote"), "intent": source["intent_text"],
                           "positive_action": pos, "negatives": negatives[:5]})
            if len(result) >= limit:
                break
    return result


def audit(item, endpoint, key, model, timeout):
    payload = {"model": model, "temperature": 0, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(item, ensure_ascii=False)},
    ]}
    request = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(payload, ensure_ascii=False).encode(),
                                     headers={"Authorization": "Bearer " + key,
                                              "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        raw = json.loads(response.read())["choices"][0]["message"]["content"]
    verdict = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    evidence = verdict.get("target_evidence")
    if (not isinstance(verdict.get("ambiguous"), bool)
            or (evidence is not None and (not isinstance(evidence, str) or evidence not in item["intent"]))
            or not isinstance(verdict.get("positive_match"), bool)
            or not isinstance(verdict.get("positive_reason"), str)
            or not isinstance(verdict.get("ambiguity_reason"), str)
            or not isinstance(verdict.get("negatives"), list)
            or {n.get("candidate_edge_id") for n in verdict["negatives"]}
            != {n["candidate_edge_id"] for n in item["negatives"]}):
        raise ValueError("invalid GQQ verdict schema or negative IDs")
    for negative in verdict["negatives"]:
        if (not isinstance(negative.get("safe_negative"), bool)
                or negative.get("category") not in {"different_operation", "wrong_target",
                                                        "missing_required_parameter", "other", "uncertain"}
                or not isinstance(negative.get("reason"), str)):
            raise ValueError("invalid GQQ negative verdict")
    if verdict["ambiguous"] and any(negative["safe_negative"] for negative in verdict["negatives"]):
        raise ValueError("ambiguous Intent cannot have safe training negatives")
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=BASE)
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--queue", type=Path, default=QUEUE)
    parser.add_argument("--env", type=Path, default=Path("../.env"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model", default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or args.retries < 1:
        raise ValueError("limit and retries must be positive")
    train, _ = load_train(args.dataset / "train/train.jsonl")
    held_out = {item["trajectory_id"] for split in ("dev", "test")
                for item in rows(args.dataset / split / "queries.jsonl")}
    candidates = select(list(rows(args.pool)), list(rows(args.queue)), train, held_out, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        for item in candidates:
            print(json.dumps(item, ensure_ascii=False))
        return
    load_env(args.env)
    key = os.environ.get("INF_API_KEY") or os.environ.get("INTENT_API_KEY")
    endpoint = args.endpoint or os.environ.get("INF_API_ENDPOINT") or os.environ.get("INTENT_API_BASE_URL") or "https://qjkpcombh9jkcecah8jdkgd5d5beqege.openapi-qb-ai.sii.edu.cn"
    model = args.model or os.environ.get("INF_MODEL_NAME") or os.environ.get("INTENT_MODEL_NAME") or "deepseek-v4-flash-gqq"
    if not key:
        raise SystemExit("missing INF_API_KEY/INTENT_API_KEY")
    done = {item["source_edge_id"] for item in rows(args.output)} if args.output.exists() else set()
    with args.output.open("a", encoding="utf-8") as stream:
        for item in candidates:
            if item["source_edge_id"] in done:
                continue
            verdict = None
            for attempt in range(args.retries):
                try:
                    verdict = audit(item, endpoint, key, model, args.timeout)
                    break
                except Exception:
                    if attempt + 1 < args.retries:
                        time.sleep(min(2 ** attempt, 8))
            result = {**item, "model": model, "review_status": "gqq_review_not_execution_verified",
                      "verdict": verdict, "review_failed": verdict is None}
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
    print(json.dumps({"selected": len(candidates), "output": str(args.output)}))


if __name__ == "__main__":
    main()
