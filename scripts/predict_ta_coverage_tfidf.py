#!/usr/bin/env python3
"""Dependency-free lexical similarity baseline for TA pipeline smoke tests."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from evaluate_bge_clawgym import rows
from predict_ta_coverage_bge import aggregate


def tokens(text):
    return re.findall(r"[a-z0-9_./:-]+", text.lower())


def tfidf_scores(windows):
    documents = {}
    for window in windows:
        documents.update({x["intent_id"]: x["text"] for x in window["intents"]})
        documents.update({x["action_id"]: x["action_text"] for x in window["actions"]})
    counts = {id: Counter(tokens(text)) for id, text in documents.items()}
    df = Counter(token for row in counts.values() for token in row)
    total = len(counts)
    weighted = {id: {token: (1 + math.log(freq)) * math.log((total + 1) / (df[token] + 1))
                     for token, freq in row.items()} for id, row in counts.items()}
    norms = {id: math.sqrt(sum(value * value for value in row.values())) for id, row in weighted.items()}
    scores = {}
    for window in windows:
        for pair in window["pairs"]:
            left, right = pair["intent_id"], pair["action_id"]
            denominator = norms[left] * norms[right]
            scores[pair["pair_id"]] = (sum(value * weighted[right].get(token, 0)
                                           for token, value in weighted[left].items()) / denominator
                                        if denominator else 0.0)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()
    if args.output.exists() or not 0 <= args.threshold <= 1:
        raise ValueError("output must not exist and threshold must be in [0, 1]")
    windows = list(rows(args.windows))
    scores = tfidf_scores(windows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for window in windows:
            local = {p["pair_id"]: scores[p["pair_id"]] for p in window["pairs"]}
            stream.write(json.dumps({"window_id": window["window_id"],
                                     "predicted_label": aggregate(window, local, args.threshold),
                                     "pair_scores": local, "threshold": args.threshold,
                                     "baseline": "tfidf_cosine_smoke_test"}) + "\n")
    print(json.dumps({"windows": len(windows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
