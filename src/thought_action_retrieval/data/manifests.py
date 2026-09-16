"""Build deterministic manifests for external datasets."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for ``path``."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(data_root: Path, path: Path) -> dict[str, object]:
    """Describe one file using a data-root-relative path."""
    root = data_root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"file is outside data root: {resolved}")
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"missing dataset file: {resolved}")
    return {
        "path": resolved.relative_to(root).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def build_dataset_manifest(
    *,
    name: str,
    version: str,
    data_root: Path,
    files: Sequence[Path],
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Build a deterministic manifest for the supplied files."""
    if not name.strip():
        raise ValueError("manifest name must not be empty")
    if not version.strip():
        raise ValueError("manifest version must not be empty")
    descriptions = sorted(
        (describe_file(data_root, path) for path in files),
        key=lambda item: str(item["path"]),
    )
    return {
        "schema_version": "dataset_manifest_v1",
        "name": name,
        "version": version,
        "file_count": len(descriptions),
        "total_bytes": sum(int(item["bytes"]) for item in descriptions),
        "files": descriptions,
        "metadata": dict(metadata),
    }


def atomic_write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Atomically serialize a manifest using stable JSON formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)
