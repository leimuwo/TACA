"""Verified JSONL IO and atomic stage publication for Matching Pilot v2."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .v2_schema import canonical_json_bytes, validate_unique_ids


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _content_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, Path):
        return value.read_bytes()
    raise TypeError("published file content must be bytes, text, or a Path")


def _normalize_relative_name(name: str) -> str:
    """Normalize a safe relative POSIX artifact name."""
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"invalid artifact path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"invalid artifact path: {name!r}")
    parts = [part for part in path.parts if part not in ("", ".")]
    if not parts:
        raise ValueError(f"invalid artifact path: {name!r}")
    return PurePosixPath(*parts).as_posix()


def _validate_manifest_name(name: str) -> str:
    if not isinstance(name, str) or not name or "/" in name or "\\" in name:
        raise ValueError(f"manifest_name must be a safe basename: {name!r}")
    normalized = _normalize_relative_name(name)
    if normalized != name or name in (".", ".."):
        raise ValueError(f"manifest_name must be a safe basename: {name!r}")
    return normalized


def publish_stage(destination: Path, files: Mapping[str, Any], *, manifest_name: str = "manifest.json") -> dict[str, Any]:
    """Publish a collection of files atomically, writing the manifest last."""
    normalized_manifest_name = _validate_manifest_name(manifest_name)
    if not isinstance(files, Mapping):
        raise TypeError("files must be a mapping of relative path to content")
    normalized_files: list[tuple[str, Any]] = []
    seen_names: set[str] = set()
    for raw_name, value in files.items():
        normalized = _normalize_relative_name(raw_name)
        if normalized == normalized_manifest_name or normalized in seen_names:
            raise ValueError(f"duplicate or reserved artifact path: {raw_name!r}")
        seen_names.add(normalized)
        normalized_files.append((normalized, value))
    normalized_files.sort(key=lambda item: item[0])

    # Keep the final component lexical: resolving here would follow a dangling
    # symlink and could publish into its target.
    destination = Path(destination).expanduser().absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"publication destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        entries: list[dict[str, Any]] = []
        for name, value in normalized_files:
            relative = PurePosixPath(name)
            content = _content_bytes(value)
            target = staging.joinpath(*relative.parts)
            if not target.parent.resolve().is_relative_to(staging.resolve()):
                raise ValueError(f"artifact path escapes staging directory: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            entries.append({"path": name, "bytes": len(content), "sha256": _sha256_bytes(content)})
        manifest = {
            "schema_version": "matching_v2_stage_manifest",
            "file_count": len(entries),
            "files": entries,
        }
        manifest_path = staging / normalized_manifest_name
        manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
        with manifest_path.open("rb") as handle:
            os.fsync(handle.fileno())
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        # rename is atomic and fails instead of replacing an existing directory.
        os.rename(staging, destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        staging = None  # type: ignore[assignment]
        return manifest
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _manifest_entries(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if set(manifest) != {"schema_version", "file_count", "files"}:
        raise ValueError("manifest has incompatible top-level keys")
    if manifest.get("schema_version") != "matching_v2_stage_manifest":
        raise ValueError("invalid stage manifest schema_version")
    file_count = manifest.get("file_count")
    files = manifest.get("files")
    if type(file_count) is not int or file_count < 0:
        raise ValueError("manifest file_count must be a non-negative integer")
    if not isinstance(files, list) or file_count != len(files):
        raise ValueError("manifest file_count does not match files")
    result: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError("invalid manifest file entry")
        raw_path = entry["path"]
        normalized = _normalize_relative_name(raw_path)
        if normalized != raw_path or normalized in result:
            raise ValueError(f"duplicate or non-canonical manifest path: {raw_path!r}")
        byte_count = entry["bytes"]
        digest = entry["sha256"]
        if type(byte_count) is not int or byte_count < 0:
            raise ValueError(f"invalid byte count for {raw_path!r}")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"invalid sha256 for {raw_path!r}")
        result[normalized] = {"bytes": byte_count, "sha256": digest}
    return result


def _stream_file_hash(path: Path) -> tuple[int, str]:
    size = 0
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _read_file_hash(path: Path) -> tuple[bytes, int, str]:
    """Read and hash one requested artifact from the same byte buffer."""
    raw = path.read_bytes()
    return raw, len(raw), _sha256_bytes(raw)


def read_verified_jsonl(path: Path, manifest_path: Path | None = None, id_field: str | None = None) -> list[dict[str, Any]]:
    """Verify a JSONL file against its manifest before decoding any rows."""
    path = Path(path)
    manifest_path = Path(manifest_path) if manifest_path is not None else path.parent / "manifest.json"
    manifest_path = manifest_path.expanduser().resolve()
    manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_value, Mapping):
        raise ValueError("manifest must contain a JSON object")
    entries = _manifest_entries(manifest_value)
    try:
        relative = Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(manifest_path.parent))).as_posix()
    except ValueError as error:
        raise ValueError(f"JSONL path is outside manifest directory: {path}") from error
    if relative not in entries:
        raise ValueError(f"manifest has no hash for {relative}")
    requested_raw: bytes | None = None
    for declared_path, metadata in entries.items():
        sibling = manifest_path.parent.joinpath(*PurePosixPath(declared_path).parts)
        try:
            sibling.resolve().relative_to(manifest_path.parent.resolve())
        except ValueError as error:
            raise ValueError(f"manifest path escapes its directory: {declared_path}") from error
        if not sibling.is_file():
            raise ValueError(f"manifest-declared file is missing: {declared_path}")
        if declared_path == relative:
            requested_raw, byte_count, actual = _read_file_hash(sibling)
        else:
            byte_count, actual = _stream_file_hash(sibling)
        declared_bytes = metadata["bytes"]
        if byte_count != declared_bytes:
            raise ValueError(f"manifest byte-size mismatch for {declared_path}")
        expected_sha = metadata["sha256"]
        if actual != expected_sha:
            raise ValueError(f"manifest hash mismatch for {declared_path}: {actual} != {expected_sha}")
    if requested_raw is None:
        raise ValueError(f"manifest has no hash for {relative}")
    raw = requested_raw
    rows: list[dict[str, Any]] = []
    logical_line = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        logical_line += 1
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}: logical line {logical_line}: invalid JSON ({error})") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}: logical line {logical_line} must contain a JSON object")
        rows.append(value)
    if id_field is not None:
        validate_unique_ids(rows, id_field)
    return rows
