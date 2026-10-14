"""Deterministic raw-action-first Stage 0 structure auditing."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any, Iterable

from .v2_schema import canonical_json_bytes, record_sha256

NORMALIZER_VERSION = "action-structure-auditor-v1"
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_UNSAFE_VARIABLE = re.compile(r"(?<!\\)\$(?:[A-Za-z_][A-Za-z0-9_]*|\{|\(|\?\*|@|#|!)")
_UNSAFE_COMMANDS = frozenset({"cd", "source", ".", "read", "export"})


def _tokens(command: str) -> list[str] | None:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except (ValueError, TypeError):
        return None


def _top_level_operators(value: str) -> list[tuple[int, str]] | None:
    result: list[tuple[int, str]] = []
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(value):
        char = value[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if quote:
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "'\"":
            quote = char
            index += 1
            continue
        if value.startswith("&&", index):
            result.append((index, "&&"))
            index += 2
        elif value.startswith("||", index):
            result.append((index, "||"))
            index += 2
        elif char == "&" and index > 0 and value[index - 1] in "<>":
            index += 1
        elif char in ";&|":
            result.append((index, char))
            index += 1
        else:
            index += 1
    if quote or escaped:
        return None
    return result


def _split_safe_fragments(command: str) -> list[str] | None:
    operators = _top_level_operators(command)
    if operators is None or any(op not in {"&&", ";"} for _, op in operators):
        return None
    if not operators:
        return [command.strip()] if command.strip() else None
    fragments: list[str] = []
    start = 0
    for position, operator in operators:
        fragment = command[start:position].strip()
        if not fragment:
            return None
        fragments.append(fragment)
        start = position + len(operator)
    final = command[start:].strip()
    return fragments + [final] if final else None


def _fragment_unsafe(fragment: str) -> bool:
    tokens = _tokens(fragment)
    if not tokens:
        return True
    if any(_ASSIGNMENT.match(token) for token in tokens):
        return True
    if _UNSAFE_VARIABLE.search(fragment) or "`" in fragment:
        return True
    if "$(" in fragment or "<(" in fragment or ">(" in fragment or "<<" in fragment:
        return True
    command_tokens = [token for token in tokens if token not in {"<", ">", ">>", "&", "|", "||", "&&", ";"}]
    return bool(command_tokens and command_tokens[0] in _UNSAFE_COMMANDS)


def _fragment_outputs(fragment: str) -> set[str]:
    """Return simple file outputs sufficient to reject obvious dependencies."""
    tokens = _tokens(fragment) or []
    outputs: set[str] = set()
    if tokens and tokens[0] == "touch":
        outputs.update(token for token in tokens[1:] if not token.startswith("-"))
    for index, token in enumerate(tokens[:-1]):
        if token in {">", ">>"}:
            outputs.add(tokens[index + 1])
    return outputs


def _fragments_have_dependency(fragments: list[str]) -> bool:
    outputs: set[str] = set()
    for fragment in fragments:
        tokens = _tokens(fragment) or []
        if any(token in outputs for token in tokens):
            return True
        outputs.update(_fragment_outputs(fragment))
    return False


def classify_exec(command: str) -> str:
    """Classify an exec command without rewriting its source text."""
    if not isinstance(command, str) or not command.strip():
        return "composite_unsafe"
    if _tokens(command) is None or _fragment_unsafe(command):
        return "composite_unsafe"
    operators = _top_level_operators(command)
    if operators is None:
        return "composite_unsafe"
    if any(op in {"||", "&"} for _, op in operators):
        return "composite_unsafe"
    if any(op == "|" for _, op in operators):
        return "atomic"
    fragments = _split_safe_fragments(command)
    if fragments is None:
        return "composite_unsafe"
    if len(fragments) == 1:
        return "atomic"
    if any(_fragment_unsafe(fragment) for fragment in fragments) or _fragments_have_dependency(fragments):
        return "composite_unsafe"
    return "split_safe"


def _parts(raw: Any) -> tuple[str, dict[str, Any], str | None, Any, bool]:
    if isinstance(raw, str):
        action = {"function": "exec", "args": {"command": raw}}
        return "exec", action["args"], None, action, False
    if not isinstance(raw, dict):
        raise ValueError("actual_actions entries must be objects or command strings")
    function = raw.get("function")
    malformed_arguments = False
    if isinstance(function, dict):
        name = function.get("name")
        value = function.get("arguments", {})
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                malformed_arguments = True
                value = {}
    else:
        name, value = function, raw.get("args", {})
    if not isinstance(name, str) or not name:
        raise ValueError("action function must be a non-empty string")
    args = value if isinstance(value, dict) else {"value": value}
    action_id = raw.get("id") or raw.get("call_id") or raw.get("action_id")
    if action_id is not None and not isinstance(action_id, str):
        raise ValueError("action id must be a string")
    return name, args, action_id, raw, malformed_arguments


def audit_action(raw: Any, *, trajectory_id: str, thought_step: int, action_index: int) -> dict[str, Any]:
    function, args, explicit_id, action, malformed_arguments = _parts(raw)
    action_id = explicit_id or f"T{thought_step}-A{action_index}"
    raw_hash = record_sha256(action)
    command = args.get("command") if function == "exec" else None
    atomicity = "composite_unsafe" if malformed_arguments else ("atomic" if function != "exec" else classify_exec(command))
    fragments: list[dict[str, Any]] = []
    if atomicity == "split_safe" and isinstance(command, str):
        for index, fragment in enumerate(_split_safe_fragments(command) or [], start=1):
            fragment_action = {"function": "exec", "args": {**args, "command": fragment}}
            fragments.append({
                "fragment_id": f"{action_id}-F{index}",
                "parent_raw_action_id": action_id,
                "fragment_index": index,
                "function": "exec",
                "args": fragment_action["args"],
                "fragment_sha256": record_sha256(fragment_action),
            })
    analysis = {
        "atomicity": atomicity,
        "atomic_fragments": fragments,
        "action_has_extras": atomicity != "atomic",
        "requires_manual_semantic_review": atomicity == "composite_unsafe",
    }
    return {
        "trajectory_id": trajectory_id,
        "thought_step": thought_step,
        "action_id": action_id,
        "action": action,
        **analysis,
        "normalizer_version": NORMALIZER_VERSION,
        "raw_action_sha256": raw_hash,
        "structure_analysis_sha256": record_sha256(analysis),
    }


def _input_files(intent_run_dir: Path) -> list[Path]:
    if not intent_run_dir.is_dir():
        raise ValueError(f"intent-run directory does not exist: {intent_run_dir}")
    files = sorted(
        path
        for path in intent_run_dir.rglob("*.json")
        if path.is_file() and path.name != "manifest.json"
    )
    if not files:
        raise ValueError("intent-run directory contains no trajectory JSON files")
    return files


def audit_intent_run(intent_run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    collected: list[tuple[str, int, str, dict[str, Any]]] = []
    provenance: list[dict[str, Any]] = []
    for path in _input_files(Path(intent_run_dir)):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid trajectory JSON: {path}") from error
        if not isinstance(record, dict) or not isinstance(record.get("trajectory_id"), str):
            raise ValueError(f"trajectory_id missing or invalid: {path}")
        steps = record.get("steps")
        if not isinstance(steps, list):
            raise ValueError(f"steps must be a list: {path}")
        provenance.append({"path": path.relative_to(intent_run_dir).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for step_pos, step in enumerate(steps):
            if not isinstance(step, dict) or type(step.get("thought_step")) is not int:
                raise ValueError(f"invalid thought_step at {path}:{step_pos}")
            actions = step.get("actual_actions")
            if not isinstance(actions, list):
                raise ValueError(f"actual_actions must be a list at {path}:{step_pos}")
            # Hash ordering makes output stable even if source action arrays/files are reversed.
            keyed = sorted(((record_sha256(_parts(action)[3]), action) for action in actions), key=lambda item: item[0])
            for action_index, (_, action) in enumerate(keyed):
                row = audit_action(action, trajectory_id=record["trajectory_id"], thought_step=step["thought_step"], action_index=action_index)
                collected.append((record["trajectory_id"], step["thought_step"], row["raw_action_sha256"], row))
    collected.sort(key=lambda item: item[:3])
    rows = [item[3] for item in collected]
    counts = {
        "raw_action_count": len(rows),
        "atomic": sum(row["atomicity"] == "atomic" for row in rows),
        "split_safe": sum(row["atomicity"] == "split_safe" for row in rows),
        "composite_unsafe": sum(row["atomicity"] == "composite_unsafe" for row in rows),
        "actions_with_extras": sum(row["action_has_extras"] for row in rows),
        "manual_review_required": sum(row["requires_manual_semantic_review"] for row in rows),
        "diagnostic_fragment_count": sum(len(row["atomic_fragments"]) for row in rows),
    }
    return rows, {"schema_version": "matching_v2_action_structure_manifest", "normalizer_version": NORMALIZER_VERSION, **counts, "input_provenance": {"intent_run_dir": str(Path(intent_run_dir).expanduser().resolve()), "files": provenance}}


def rows_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


# Compatibility name for callers of the first Stage 0 draft.
atomicize_intent_run = audit_intent_run
