#!/usr/bin/env python3
"""Audit tracked repository files for secrets, large files, and local paths."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from thought_action_retrieval.repository_audit import audit_repository, format_issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-file-mib", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    max_bytes = int(args.max_file_mib * 1024 * 1024)
    issues = audit_repository(args.root, max_file_bytes=max_bytes)
    if issues:
        print(format_issues(issues))
        return 1
    print("repository audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
