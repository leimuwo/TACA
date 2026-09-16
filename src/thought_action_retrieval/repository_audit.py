"""Audit tracked repository files for unsafe or non-portable content."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


FORBIDDEN_PARTS = {".codex", ".sii", "__pycache__", "checkpoints", "models", "logs"}
ASSIGNMENT_RE = re.compile(
    r"^\s*(?:export\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*[=:]\s*(?P<value>.*?)\s*$"
)
CREDENTIAL_NAME_RE = re.compile(
    r"api[_-]?key|secret|token|password",
    re.IGNORECASE,
)
MACHINE_PATH_MARKER = "/inspire" + "/hdd/"


@dataclass(frozen=True, order=True)
class AuditIssue:
    """One repository policy violation."""

    code: str
    path: str
    detail: str


def tracked_files(root: Path) -> list[Path]:
    """Return sorted files known to Git below ``root``."""
    root = root.resolve()
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    names = [name for name in result.stdout.decode("utf-8").split("\0") if name]
    return [root / name for name in sorted(names)]


def _is_forbidden_path(relative: Path) -> bool:
    if relative.name == ".env":
        return True
    if relative.name.startswith(".env.") and relative.name != ".env.example":
        return True
    return any(part in FORBIDDEN_PARTS for part in relative.parts)


def _markdown_without_fenced_examples(text: str) -> str:
    kept: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            kept.append(line)
    return "\n".join(kept)


def _contains_secret(text: str, suffix: str) -> bool:
    if suffix.lower() == ".md":
        text = _markdown_without_fenced_examples(text)
    for line in text.splitlines():
        match = ASSIGNMENT_RE.match(line)
        if not match or not CREDENTIAL_NAME_RE.search(match.group("name")):
            continue
        value = match.group("value").strip().strip("\"'")
        if value and "example.invalid" not in value and value not in {"<secret>", "<api-key>"}:
            return True
    return False


def audit_repository(
    root: Path, max_file_bytes: int = 10 * 1024 * 1024
) -> list[AuditIssue]:
    """Return all policy violations among Git-tracked files."""
    root = root.resolve()
    issues: list[AuditIssue] = []
    for path in tracked_files(root):
        relative = path.relative_to(root)
        relative_text = relative.as_posix()
        if _is_forbidden_path(relative):
            issues.append(
                AuditIssue("forbidden_path", relative_text, "path must not be tracked")
            )
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > max_file_bytes:
            issues.append(
                AuditIssue(
                    "file_too_large",
                    relative_text,
                    f"{size} bytes exceeds {max_file_bytes}",
                )
            )
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if _contains_secret(content, path.suffix):
            issues.append(
                AuditIssue(
                    "possible_secret",
                    relative_text,
                    "non-empty credential-like assignment",
                )
            )
        if path.suffix.lower() != ".md" and MACHINE_PATH_MARKER in content:
            issues.append(
                AuditIssue(
                    "machine_absolute_path",
                    relative_text,
                    f"contains {MACHINE_PATH_MARKER}",
                )
            )
    return sorted(issues)


def format_issues(issues: Sequence[AuditIssue]) -> str:
    """Format audit issues as stable one-line records."""
    return "\n".join(
        f"{issue.code}: {issue.path}: {issue.detail}" for issue in issues
    )
