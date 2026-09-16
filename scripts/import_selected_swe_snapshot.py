#!/usr/bin/env python3
"""Register selected SWE metadata and small samples without copying full data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from thought_action_retrieval.data.paths import resolve_data_root
from thought_action_retrieval.data.snapshot import import_selected_snapshot


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--source-relative", type=Path, required=True)
    parser.add_argument("--raw-shards-relative", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--expected-count", type=int, default=500)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = import_selected_snapshot(
        data_root=resolve_data_root(args.data_root),
        source_relative=args.source_relative,
        repository_root=args.repository_root,
        sample_ids=args.sample_id,
        raw_shards_relative=args.raw_shards_relative,
        expected_count=args.expected_count,
    )
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
