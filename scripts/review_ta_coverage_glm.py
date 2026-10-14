#!/usr/bin/env python3
"""Resumable GLM-5.2 review of held-out clear-Intent TA windows."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

from evaluate_bge_clawgym import rows
from review_ta_coverage_gqq import PROMPT_VERSION, review
from screen_ambiguous_intents_gqq import load_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path("../.env"))
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    load_env(args.env)
    key = os.environ.get("GLM5_2_API")
    endpoint = os.environ.get("GLM5_2_URL")
    if not key or not endpoint:
        raise SystemExit("missing GLM5_2_API or GLM5_2_URL in environment")
    windows = list(rows(args.windows))
    if len({w["window_id"] for w in windows}) != len(windows):
        raise ValueError("duplicate window ID")
    previous = {r["window_id"]: r for r in rows(args.output)} if args.output.exists() else {}
    pending = [w for w in windows if w["window_id"] not in previous
               or (args.retry_failed and previous[w["window_id"]]["review_failed"])]
    if args.limit is not None:
        pending = pending[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def one(window):
        try:
            verdict = review(window, endpoint, key, args.model, args.timeout)
            return {"window_id": window["window_id"], "model": args.model,
                    "prompt_version": PROMPT_VERSION,
                    "review_failed": False, "verdict": verdict}
        except Exception as error:
            return {"window_id": window["window_id"], "model": args.model,
                    "prompt_version": PROMPT_VERSION,
                    "review_failed": True, "verdict": None, "error": str(error)[:300]}

    with args.output.open("a", encoding="utf-8") as stream, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for item in pool.map(one, pending):
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
    print(json.dumps({"selected_windows": len(windows), "reviewed_now": len(pending),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
