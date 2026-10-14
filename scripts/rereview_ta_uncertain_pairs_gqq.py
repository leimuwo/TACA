#!/usr/bin/env python3
"""Resume targeted GQQ review of TA pairs blocked by free-text uncertainty."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import urllib.request
from pathlib import Path

from evaluate_bge_clawgym import rows
from review_ta_coverage_gqq import payload
from screen_ambiguous_intents_gqq import load_env

PROMPT_VERSION = "ta_pair_targeted_review_v1_20261013"
SYSTEM = """Review only the supplied Intent-Action pairs for training-label reliability. Judge TOOL CALL coverage, not execution success. Each Action alone must fully cover the whole due-now Intent, including operation, target, and explicit constraints. Never infer a missing target from the Action. A plural or multi-operation Intent needs a single Action covering all parts; otherwise mark it unassessable if its target set cannot be determined. A preparatory Action is not complete coverage. A clearly unrelated Action may be a reliable negative even when some other aspect of the window is ambiguous.

The prior reviewer wrote uncertainty notes. Re-examine each requested pair independently. Return exactly one JSON object with a `pairs` array. Each item must have: `pair_id`, `decision` ("covered", "not_covered", or "uncertain"), `evidence_sufficient` (boolean), `reason` (string), and `remaining_doubt` (string; empty only when no doubt affects this pair). Use `uncertain` whenever target, temporal status, atomicity, tool semantics, or action effect prevents a reliable binary label. A binary decision requires evidence_sufficient=true and remaining_doubt="". Include every requested pair ID exactly once. No markdown or second JSON object."""


def parse_response(raw):
    value = raw.strip()
    if value.startswith("```json"):
        value = value[7:].strip()
    elif value.startswith("```"):
        value = value[3:].strip()
    if value.endswith("```"):
        value = value[:-3].strip()
    result, end = json.JSONDecoder().raw_decode(value)
    if value[end:].strip():
        raise ValueError("extra content after JSON object")
    if not isinstance(result, dict):
        raise ValueError("response is not an object")
    return result


def validate(review, blocked_ids):
    decisions = review.get("pairs")
    if not isinstance(decisions, list) or len(decisions) != len(blocked_ids):
        raise ValueError("pair count mismatch")
    if {p.get("pair_id") for p in decisions} != blocked_ids:
        raise ValueError("pair IDs mismatch")
    for pair in decisions:
        if (pair.get("decision") not in {"covered", "not_covered", "uncertain"}
                or not isinstance(pair.get("evidence_sufficient"), bool)
                or not isinstance(pair.get("reason"), str)
                or not isinstance(pair.get("remaining_doubt"), str)):
            raise ValueError("invalid pair decision")
        if pair["decision"] == "uncertain":
            consistent = not pair["evidence_sufficient"] and bool(pair["remaining_doubt"].strip())
        else:
            consistent = pair["evidence_sufficient"] and not pair["remaining_doubt"].strip()
        if not consistent:
            raise ValueError("decision and evidence conflict")
    return review


def review_one(window, resolution, original, endpoint, key, model, timeout):
    requested = set(resolution["blocked_pair_ids"])
    data = payload(window)
    data["pairs"] = [p for p in data["pairs"] if p["pair_id"] in requested]
    data["prior_uncertainties"] = original["verdict"]["uncertainties"]
    body = {"model": model, "temperature": 0, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
    ]}
    request = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers={"Authorization": "Bearer " + key,
                                              "Content-Type": "application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
        raw = json.loads(response.read())["choices"][0]["message"]["content"]
    return validate(parse_response(raw), requested)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--resolution", type=Path, required=True)
    parser.add_argument("--review", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path("../.env"))
    parser.add_argument("--provider", choices=("gqq", "glm"), default="gqq")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    load_env(args.env)
    key = (os.environ.get("GLM5_2_API") if args.provider == "glm" else
           os.environ.get("INF_API_KEY") or os.environ.get("INTENT_API_KEY"))
    if not key:
        raise SystemExit("missing provider API key")
    endpoint = (os.environ.get("GLM5_2_URL") if args.provider == "glm" else
                os.environ.get("INF_API_ENDPOINT") or os.environ.get("INTENT_API_BASE_URL") or
                "https://qjkpcombh9jkcecah8jdkgd5d5beqege.openapi-qb-ai.sii.edu.cn")
    if not endpoint:
        raise SystemExit("missing provider API endpoint")
    model = "glm-5.2" if args.provider == "glm" else os.environ.get("INF_MODEL_NAME") or "deepseek-v4-flash-gqq"
    windows = {w["window_id"]: w for w in rows(args.windows)}
    resolutions = {r["window_id"]: r for r in rows(args.resolution)
                   if r["blocked_pair_ids"]}
    originals = {}
    for path in args.review:
        for item in rows(path):
            if not item.get("review_failed") and item.get("verdict") is not None:
                originals[item["window_id"]] = item
    if set(resolutions) - set(windows) or set(resolutions) - set(originals):
        raise ValueError("missing windows or original reviews")
    previous = {r["window_id"]: r for r in rows(args.output)} if args.output.exists() else {}
    pending = [wid for wid in sorted(resolutions)
               if wid not in previous or (args.retry_failed and previous[wid]["review_failed"])]
    if args.limit is not None:
        pending = pending[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def one(wid):
        try:
            verdict = review_one(windows[wid], resolutions[wid], originals[wid],
                                 endpoint, key, model, args.timeout)
            return {"window_id": wid, "model": model, "prompt_version": PROMPT_VERSION,
                    "review_failed": False, "verdict": verdict}
        except Exception as error:
            return {"window_id": wid, "model": model, "prompt_version": PROMPT_VERSION,
                    "review_failed": True, "error": str(error)[:300], "verdict": None}

    with args.output.open("a", encoding="utf-8") as stream, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for item in pool.map(one, pending):
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
    print(json.dumps({"windows_requested": len(resolutions), "reviewed_now": len(pending),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
