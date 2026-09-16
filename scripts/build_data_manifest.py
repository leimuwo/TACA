#!/usr/bin/env python3
"""Build a deterministic manifest for files below TA_DATA_ROOT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from thought_action_retrieval.data.manifests import (
    atomic_write_manifest,
    build_dataset_manifest,
)
from thought_action_retrieval.data.paths import (
    resolve_data_root,
    resolve_under_data_root,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--relative-path", action="append", required=True)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = resolve_data_root(args.data_root)
    files = [
        resolve_under_data_root(data_root, relative)
        for relative in args.relative_path
    ]
    metadata: dict[str, object] = {}
    if args.metadata_json:
        parsed = json.loads(args.metadata_json.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("metadata JSON must contain an object")
        metadata = parsed
    manifest = build_dataset_manifest(
        name=args.name,
        version=args.version,
        data_root=data_root,
        files=files,
        metadata=metadata,
    )
    atomic_write_manifest(args.output, manifest)
    print(f"wrote manifest for {len(files)} files to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
