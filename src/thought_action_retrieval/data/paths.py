"""Portable access to the external research data root."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


class DataRootError(ValueError):
    """Raised when the external data root is absent or unsafe."""


def resolve_data_root(
    explicit: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve and validate the external dataset root."""
    environment = os.environ if environ is None else environ
    raw_value: str | Path | None = explicit
    if raw_value is None:
        raw_value = environment.get("TA_DATA_ROOT")
    if raw_value is None or not str(raw_value).strip():
        raise DataRootError(
            "Set TA_DATA_ROOT or pass an explicit data-root directory."
        )
    root = Path(raw_value).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise DataRootError(f"data root must be an existing directory: {root}")
    return root


def resolve_under_data_root(data_root: Path, relative_path: str | Path) -> Path:
    """Resolve a relative path while preventing escape from ``data_root``."""
    root = data_root.expanduser().resolve()
    relative = Path(relative_path)
    if relative.is_absolute():
        raise DataRootError(f"dataset path must be relative: {relative}")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise DataRootError(f"dataset path resolves outside data root: {relative}")
    return candidate
