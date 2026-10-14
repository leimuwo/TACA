#!/usr/bin/env python3
"""Produce clear-only dev query/qrel files without altering original splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_bge_clawgym import rows, sha256
from evaluate_bge_m3_local import DECISIONS, SCREEN_INPUT, DATASET, clear_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--screen-input", type=Path, default=SCREEN_INPUT)
    parser.add_argument("--decisions", type=Path, default=DECISIONS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dev = args.dataset / "dev"
    queries = list(rows(dev / "queries.jsonl"))
    clear = clear_ids(queries, rows(args.screen_input), rows(args.decisions))
    args.output.mkdir(parents=True)
    with (args.output / "intent_clear.jsonl").open("w", encoding="utf-8") as stream:
        for row in queries:
            if row["query_id"] in clear:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output / "qrels_clear.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows(dev / "qrels.jsonl"):
            if row["query_id"] in clear:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"queries_total": len(queries), "queries_clear": len(clear),
              "queries_excluded": len(queries) - len(clear),
              "source_sha256": {"queries": sha256(dev / "queries.jsonl"),
                                "qrels": sha256(dev / "qrels.jsonl"),
                                "screen_input": sha256(args.screen_input),
                                "decisions": sha256(args.decisions)},
              "policy": "filter only; no Intent completion or rewriting"}
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
