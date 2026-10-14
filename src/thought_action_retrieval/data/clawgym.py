"""Convert ClawGym JSONL records into Kimi-compatible trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from thought_action_retrieval.data.paths import (
    resolve_data_root,
    resolve_under_data_root,
)


DEFAULT_SOURCE = Path(
    "clawgym/sft_trajectory_with_think_for_megatron_json_toolcall.jsonl"
)
DEFAULT_OUTPUT = Path("clawgym/selected_1000_json")


def parse_tool_arguments(value: object) -> dict[str, Any]:
    """Return one tool-call argument object or fail closed."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ValueError("tool arguments must be a JSON object or string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("tool arguments are not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must decode to a JSON object")
    return parsed


def normalize_tool_call(call: object) -> dict[str, Any]:
    """Normalize OpenAI-compatible and existing repository tool calls."""
    if not isinstance(call, dict):
        raise ValueError("tool call must be a JSON object")
    function = call.get("function")
    if isinstance(function, str) and function:
        normalized = {
            "function": function,
            "args": parse_tool_arguments(call.get("args", {})),
        }
        for key in ("id", "type"):
            if key in call:
                normalized[key] = call[key]
        return normalized
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise ValueError("tool call function must include a name")
    normalized = {
        "function": function["name"],
        "args": parse_tool_arguments(function.get("arguments", {})),
    }
    for key in ("id", "type"):
        if key in call:
            normalized[key] = call[key]
    return normalized


def normalize_message(message: object) -> dict[str, Any]:
    """Normalize one ClawGym conversation message."""
    if not isinstance(message, dict):
        raise ValueError("message must be a JSON object")
    role = message.get("role")
    if not isinstance(role, str) or not role:
        raise ValueError("message role must be a non-empty string")
    normalized: dict[str, Any] = {
        "role": role,
        "content": message.get("content") if isinstance(message.get("content"), str) else "",
    }
    if "tool_call_id" in message:
        normalized["tool_call_id"] = message.get("tool_call_id")
    if role == "assistant":
        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str):
            reasoning = normalized["content"]
        normalized["assistant_thought"] = {"content": reasoning}
    tool_calls = message.get("tool_calls")
    if tool_calls:
        if not isinstance(tool_calls, list):
            raise ValueError("tool_calls must be a list")
        normalized["tool_calls"] = [normalize_tool_call(call) for call in tool_calls]
    return normalized


def parse_reward_summary(value: object) -> dict[str, Any]:
    """Parse the compact ClawGym reward metadata."""
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ValueError("reward_summary must be a JSON object or string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("reward_summary is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("reward_summary must decode to a JSON object")
    return parsed


def convert_record(
    record: dict[str, Any],
    *,
    source_name: str,
    line_number: int,
    source_sha256: str,
    seed: int,
    converted_at: str,
) -> dict[str, Any]:
    """Convert one source record while retaining portable provenance."""
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    users = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    task = users[0].get("content") if users else None
    if not isinstance(task, str) or not task.strip():
        raise ValueError("trajectory must include a non-empty user task")
    normalized_messages = [normalize_message(message) for message in messages]
    trajectory_id = f"clawgym-L{line_number}-{source_sha256[:12]}"
    reward = parse_reward_summary(record.get("reward_summary"))
    return {
        "id": trajectory_id,
        "task": task,
        "messages": normalized_messages,
        "category": "clawgym",
        "subcategory": "tool_use",
        "_source": {
            "path": source_name,
            "line_number": line_number,
            "sha256": source_sha256,
        },
        "_conversion": {
            "schema_version": "clawgym_kimi_compatible_v1",
            "seed": seed,
            "converted_at": converted_at,
        },
        "clawgym_metadata": {
            **reward,
            "prompt_ids_len": record.get("prompt_ids_len"),
            "response_ranges": record.get("response_ranges"),
        },
    }


def select_line_numbers(total: int, count: int, seed: int) -> list[int]:
    """Select stable unique one-based source line numbers."""
    if count <= 0:
        raise ValueError("count must be positive")
    if count > total:
        raise ValueError(f"requested {count} records from source with {total} rows")
    return sorted(random.Random(seed).sample(range(1, total + 1), count))


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Write formatted JSON with atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _decode_record(raw: bytes, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON object at source line {line_number}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"source line {line_number} must be a JSON object")
    return value


def _inspect_source(source_path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with source_path.open("rb") as handle:
        for count, raw in enumerate(handle, start=1):
            digest.update(raw)
            _decode_record(raw, count)
    if count == 0:
        raise ValueError("source JSONL is empty")
    return count, digest.hexdigest()


def _record_bytes(raw: bytes) -> bytes:
    if raw.endswith(b"\r\n"):
        return raw[:-2]
    if raw.endswith(b"\n"):
        return raw[:-1]
    return raw


def convert_jsonl_sample(
    *,
    source_path: Path,
    output_dir: Path,
    source_name: str,
    count: int,
    seed: int,
    converted_at: str | None = None,
) -> dict[str, Any]:
    """Select and convert a deterministic sample from one JSONL source."""
    source_path = source_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source_path.is_file():
        raise ValueError(f"source JSONL does not exist: {source_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = converted_at or datetime.now(timezone.utc).isoformat()
    source_count, source_digest = _inspect_source(source_path)
    selected_lines = select_line_numbers(source_count, count, seed)
    selected = set(selected_lines)
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with source_path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if line_number not in selected:
                continue
            record_bytes = _record_bytes(raw)
            record_digest = hashlib.sha256(record_bytes).hexdigest()
            converted = convert_record(
                _decode_record(raw, line_number),
                source_name=source_name,
                line_number=line_number,
                source_sha256=record_digest,
                seed=seed,
                converted_at=timestamp,
            )
            trajectory_id = converted["id"]
            if trajectory_id in seen_ids:
                raise ValueError(f"duplicate converted trajectory ID: {trajectory_id}")
            seen_ids.add(trajectory_id)
            filename = (
                f"clawgym_L{line_number:05d}_{record_digest[:12]}.json"
            )
            output_path = output_dir / filename
            atomic_write_json(output_path, converted)
            records.append(
                {
                    "trajectory_id": trajectory_id,
                    "source_line": line_number,
                    "source_sha256": record_digest,
                    "filename": filename,
                    "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                }
            )
    if len(records) != count:
        raise ValueError(f"selected {count} rows but converted {len(records)}")
    manifest = {
        "schema_version": "clawgym_selection_manifest_v1",
        "generated_at": timestamp,
        "source": source_name,
        "source_sha256": source_digest,
        "source_count": source_count,
        "selected_count": count,
        "seed": seed,
        "selected_lines": selected_lines,
        "records": records,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-name")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser


def _resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, str]:
    needs_root = args.source is None or args.output_dir is None
    configured_root = args.data_root is not None or bool(os.environ.get("TA_DATA_ROOT"))
    data_root = (
        resolve_data_root(args.data_root) if needs_root or configured_root else None
    )

    def resolve(value: Path | None, default: Path) -> Path:
        selected = value if value is not None else default
        if selected.is_absolute():
            return selected.expanduser().resolve()
        if data_root is not None:
            return resolve_under_data_root(data_root, selected)
        return selected.expanduser().resolve()

    source = resolve(args.source, DEFAULT_SOURCE)
    output = resolve(args.output_dir, DEFAULT_OUTPUT)
    if args.source_name:
        source_name = args.source_name
    elif data_root is not None and source.is_relative_to(data_root):
        source_name = source.relative_to(data_root).as_posix()
    else:
        source_name = source.name
    return source, output, source_name


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source, output, source_name = _resolve_paths(args)
    manifest = convert_jsonl_sample(
        source_path=source,
        output_dir=output,
        source_name=source_name,
        count=args.count,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0
