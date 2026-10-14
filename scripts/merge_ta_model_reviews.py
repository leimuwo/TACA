#!/usr/bin/env python3
"""Build one current GQQ verdict per TA window from versioned review runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_bge_clawgym import rows


def merge(windows, sources):
    expected = {window["window_id"] for window in windows}
    if len(expected) != len(windows):
        raise ValueError("duplicate window IDs")
    merged = {}
    for source in sources:
        for review in rows(source):
            wid = review["window_id"]
            if wid not in expected:
                raise ValueError(f"review outside frozen windows: {wid}")
            if review.get("review_failed") or review.get("verdict") is None:
                continue
            merged[wid] = {**review, "review_source": str(source)}
    if set(merged) != expected:
        raise ValueError(f"missing valid review for {len(expected - set(merged))} windows")
    return [merged[window["window_id"]] for window in windows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--review", type=Path, action="append", required=True,
                        help="Repeat in increasing precedence order")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reviews = merge(list(rows(args.windows)), args.review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for item in reviews:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"windows": len(reviews), "null_verdicts": 0,
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
