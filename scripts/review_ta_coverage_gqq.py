#!/usr/bin/env python3
"""Resumable GQQ review of window-level Intent-Action coverage."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import urllib.request
from pathlib import Path

from evaluate_bge_clawgym import rows
from screen_ambiguous_intents_gqq import load_env
from ta_coverage_aggregation import aggregate_ta_label

PROMPT_VERSION = "ta_coverage_review_v4_20261013"
SYSTEM = """Judge one Thought/Action window for semantic coverage by tool CALLS, not tool execution success. Follow these steps in order:
1. Classify each extracted Intent as due_now and assessable. A future plan is not due now. A private calculation already performed inside Thought (counting words, computing a score, deciding a recommendation) is not an independently assessable TOOL-CALL objective unless the Intent requires an externally observable result and the Action encodes that result. Do not treat an internal calculation as an unexecuted tool call merely because the current Action does something else. If an Intent says 'the skill directories' or another vague reference and the target cannot be uniquely identified from Intent text, mark it unassessable. Do not infer missing targets from Actions.
2. Before pair labels, check atomicity. A pair is covered=true only if ONE Action alone fully covers the WHOLE Intent. 'Read the evolution history files' cannot be fully covered by separately reading each file; split into named-file Intents first, or mark this Intent unassessable for pairwise V1. The same applies to 'classify all messages' when classification evidence is spread across several output Actions. Do not label each component Action as complete coverage. A single write Action may demonstrate classification only if that one Action's content itself covers all requested messages. For every supplied pair, decide whether its Action alone completely covers the explicit operation, target, and constraints. Reading a validator script is not running it. Reading one named file does not read another. Different commands with equivalent effects may both cover an Intent. If an Intent is not due_now or not assessable, all its pairs must have covered=false.
3. State uncertainties explicitly. The program will derive the window label from your Intent and pair decisions; do not provide a window-level label.
Return exactly one JSON object, no markdown or second object: {"intents":[{"intent_id":"...","due_now":true,"assessable":true,"reason":"..."}],"pairs":[{"pair_id":"...","covered":false,"reason":"..."}],"reason":"...","uncertainties":["..."]}. Include each supplied Intent ID and pair ID exactly once."""


def payload(window):
    return {"window_id": window["window_id"], "thought": window["thought"],
            "intents": [{"intent_id": x["intent_id"], "text": x["text"],
                         "source_quote": x["source_quote"], "temporal_scope": x["temporal_scope"]}
                        for x in window["intents"]],
            "actions": [{"action_id": x["action_id"], "action_text": x["action_text"]}
                        for x in window["actions"]],
            "pairs": [{"pair_id": x["pair_id"], "intent_id": x["intent_id"],
                       "action_id": x["action_id"]} for x in window["pairs"]]}


def validate(window, verdict):
    if ({x.get("intent_id") for x in verdict.get("intents", [])}
            != {x["intent_id"] for x in window["intents"]}
            or len(verdict["intents"]) != len(window["intents"])
            or {x.get("pair_id") for x in verdict.get("pairs", [])}
            != {x["pair_id"] for x in window["pairs"]}
            or len(verdict["pairs"]) != len(window["pairs"])):
        raise ValueError("verdict IDs do not match window")
    if any(not isinstance(x.get("due_now"), bool) or not isinstance(x.get("assessable"), bool)
           or not isinstance(x.get("reason"), str) for x in verdict["intents"]):
        raise ValueError("invalid Intent verdict")
    if any(not isinstance(x.get("covered"), bool) or not isinstance(x.get("reason"), str)
           for x in verdict["pairs"]):
        raise ValueError("invalid pair verdict")
    intent_by_id = {x["intent_id"]: x for x in verdict["intents"]}
    if any(x["covered"] and not (intent_by_id[p["intent_id"]]["due_now"]
                                and intent_by_id[p["intent_id"]]["assessable"])
           for x in verdict["pairs"] for p in window["pairs"] if x["pair_id"] == p["pair_id"]):
        raise ValueError("covered pair belongs to unassessable Intent")
    expected = aggregate_ta_label(
        window, intent_by_id, {x["pair_id"]: x["covered"] for x in verdict["pairs"]})
    raw_label = verdict.get("suggested_ta_label")
    if raw_label is not None and raw_label not in {"faithful", "partial", "not_executed", "unassessable"}:
        raise ValueError("invalid suggested TA label")
    if raw_label is not None:
        verdict["model_raw_ta_label"] = raw_label
        verdict["aggregation_disagrees"] = raw_label != expected
    verdict["suggested_ta_label"] = expected
    return verdict


def review(window, endpoint, key, model, timeout):
    body = {"model": model, "temperature": 0, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload(window), ensure_ascii=False)},
    ]}
    request = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers={"Authorization": "Bearer " + key,
                                              "Content-Type": "application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
        raw = json.loads(response.read())["choices"][0]["message"]["content"]
    verdict = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    return validate(window, verdict)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path("../.env"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--window-id", action="append", dest="window_ids",
                        help="Review only this window ID; repeat for multiple windows")
    args = parser.parse_args()
    load_env(args.env)
    key = os.environ.get("INF_API_KEY") or os.environ.get("INTENT_API_KEY")
    if not key:
        raise SystemExit("missing INF_API_KEY/INTENT_API_KEY")
    endpoint = os.environ.get("INF_API_ENDPOINT") or os.environ.get("INTENT_API_BASE_URL") or "https://qjkpcombh9jkcecah8jdkgd5d5beqege.openapi-qb-ai.sii.edu.cn"
    model = os.environ.get("INF_MODEL_NAME") or "deepseek-v4-flash-gqq"
    windows = list(rows(args.windows))
    if args.window_ids:
        selected = set(args.window_ids)
        windows = [window for window in windows if window["window_id"] in selected]
        if {window["window_id"] for window in windows} != selected:
            raise ValueError("one or more requested window IDs were not found")
    previous = {r["window_id"]: r for r in rows(args.output)} if args.output.exists() else {}
    done = {wid for wid, item in previous.items() if not (args.retry_failed and item["review_failed"])}
    pending = [w for w in windows if w["window_id"] not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def one(window):
        try:
            return {"window_id": window["window_id"], "model": model,
                    "prompt_version": PROMPT_VERSION,
                    "verdict": review(window, endpoint, key, model, args.timeout), "review_failed": False}
        except Exception as error:
            return {"window_id": window["window_id"], "model": model,
                    "prompt_version": PROMPT_VERSION,
                    "verdict": None, "review_failed": True, "error": str(error)[:300]}
    with args.output.open("a", encoding="utf-8") as stream, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(one, pending):
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
    print(json.dumps({"windows": len(windows), "newly_reviewed": len(pending), "output": str(args.output)}))


if __name__ == "__main__":
    main()
