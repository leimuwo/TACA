#!/usr/bin/env python3
"""Queue every pair in a window with GLM uncertainty for structured re-review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_bge_clawgym import rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reviews = {r["window_id"]: r for r in rows(args.reviews)}
    queued = []
    for w in rows(args.windows):
        r = reviews.get(w["window_id"])
        if r and not r.get("review_failed") and r["verdict"].get("uncertainties"):
            queued.append({"window_id": w["window_id"], "blocked_pair_ids":
                           [p["pair_id"] for p in w["pairs"]], "cleared_pair_ids": []})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for item in queued:
            stream.write(json.dumps(item) + "\n")
    print(json.dumps({"windows": len(queued),
                      "pairs": sum(len(r["blocked_pair_ids"]) for r in queued)}))


if __name__ == "__main__":
    main()
