#!/usr/bin/env python3
"""Generate observable Thought -> execution-intent annotations with an LLM."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from thought_action_retrieval.data.paths import (
    resolve_data_root,
    resolve_under_data_root,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_PATH = REPOSITORY_ROOT / "prompts" / "thought_to_intent.md"


@dataclass(frozen=True)
class PromptBundle:
    """System and user prompt templates loaded from one reviewed file."""

    system: str
    user_template: str


def load_prompt_bundle(path: Path) -> PromptBundle:
    """Load the reviewed prompt sections from Markdown."""
    text = path.read_text(encoding="utf-8")
    system_heading = re.search(r"^## System Prompt\s*$", text, re.MULTILINE)
    if not system_heading:
        raise ValueError(f"prompt file has no System Prompt section: {path}")
    user_heading = re.search(
        r"^## User Prompt\s*$",
        text[system_heading.end() :],
        re.MULTILINE,
    )
    if not user_heading:
        raise ValueError(f"prompt file has no User Prompt section: {path}")
    user_start = system_heading.end() + user_heading.start()
    user_body_start = system_heading.end() + user_heading.end()
    system = text[system_heading.end() : user_start].strip()
    user_template = text[user_body_start:].strip()
    if not system:
        raise ValueError(f"prompt file has no System Prompt section: {path}")
    if not user_template:
        raise ValueError(f"prompt file has no User Prompt section: {path}")
    return PromptBundle(
        system=system,
        user_template=user_template,
    )


PROMPTS = load_prompt_bundle(DEFAULT_PROMPT_PATH)
DEFAULT_KIMI_RELATIVE = Path(
    "SWE-agent-trajectories/test_data/kimi_selected_json"
)
DEFAULT_SWE_RELATIVE = Path(
    "SWE-agent-trajectories/test_data/selected_500_json"
)
DEFAULT_OUTPUT_RELATIVE = Path(
    "SWE-agent-trajectories/test_data/thought_intent"
)


PRINT_LOCK = threading.Lock()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"trajectory must be a JSON object: {path}")
    return data


def normalize_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def portable_source_name(source_path: Path, data_root: Path | None) -> str:
    """Return a portable path below the data root or just the filename."""
    resolved = source_path.expanduser().resolve()
    if data_root is not None:
        root = data_root.expanduser().resolve()
        if resolved.is_relative_to(root):
            return resolved.relative_to(root).as_posix()
    return resolved.name


def truncate_context(text: str, context_chars: int) -> str:
    if len(text) <= context_chars:
        return text
    return "[Earlier context truncated]\n" + text[-context_chars:]


def render_kimi_message(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "unknown").upper()
    content = normalize_text(message.get("content")).strip()
    chunks = [f"{role}: {content}" if content else f"{role}: [empty content]"]
    tool_calls = message.get("tool_calls") or []
    if isinstance(tool_calls, list) and tool_calls:
        rendered = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            rendered.append(
                f"{call.get('function', 'unknown')}({json.dumps(call.get('args') or {}, ensure_ascii=False, sort_keys=True)})"
            )
        if rendered:
            chunks.append("ACTIONS: " + " | ".join(rendered))
    return "\n".join(chunks)


def extract_kimi_steps(trajectory: dict[str, Any], context_chars: int = 6000) -> list[dict[str, Any]]:
    messages = trajectory.get("messages") or []
    user_request = normalize_text(trajectory.get("task"))
    steps: list[dict[str, Any]] = []
    thought_step = 0
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        thought_step += 1
        preceding = "\n\n".join(
            render_kimi_message(previous)
            for previous in messages[:message_index]
            if isinstance(previous, dict) and previous.get("role") != "system"
        )
        tool_calls = message.get("tool_calls") or []
        actual_actions = [call for call in tool_calls if isinstance(call, dict)] if isinstance(tool_calls, list) else []
        steps.append(
            {
                "thought_step": thought_step,
                "source_index": message_index,
                "user_request": user_request,
                "preceding_context": truncate_context(preceding or "[No preceding context]", context_chars),
                "thought": normalize_text(message.get("content")).strip(),
                "actual_actions": actual_actions,
            }
        )
    return steps


def render_swe_step(step: dict[str, Any]) -> str:
    thought = normalize_text(step.get("thought")).strip()
    action = normalize_text(step.get("action")).strip()
    observation = normalize_text(step.get("observation")).strip()
    return "\n".join(
        [
            f"THOUGHT {step.get('step')}: {thought or '[empty thought]'}",
            f"ACTION {step.get('step')}: {action or '[empty action]'}",
            f"OBSERVATION {step.get('step')}: {observation or '[empty observation]'}",
        ]
    )


def extract_swe_steps(trajectory: dict[str, Any], context_chars: int = 6000) -> list[dict[str, Any]]:
    source_steps = trajectory.get("steps") or []
    user_request = normalize_text(trajectory.get("task"))
    extracted: list[dict[str, Any]] = []
    for sequence, step in enumerate(source_steps, start=1):
        if not isinstance(step, dict):
            continue
        source_step = step.get("step") if isinstance(step.get("step"), int) else sequence
        preceding = "\n\n".join(
            render_swe_step(previous)
            for previous in source_steps[: sequence - 1]
            if isinstance(previous, dict)
        )
        action = normalize_text(step.get("action")).strip()
        extracted.append(
            {
                "thought_step": int(source_step),
                "source_index": sequence - 1,
                "user_request": user_request,
                "preceding_context": truncate_context(preceding or "[No preceding context]", context_chars),
                "thought": normalize_text(step.get("thought")).strip(),
                "actual_actions": [action] if action else [],
            }
        )
    return extracted


def kimi_selection_score(path: Path) -> tuple[float, int, int, str]:
    trajectory = read_json(path)
    action_messages = [
        message
        for message in trajectory.get("messages") or []
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and isinstance(message.get("tool_calls"), list)
        and message.get("tool_calls")
    ]
    direct = sum(bool(normalize_text(message.get("content")).strip()) for message in action_messages)
    coverage = direct / len(action_messages) if action_messages else 0.0
    total_actions = sum(len(message.get("tool_calls") or []) for message in action_messages)
    return (-coverage, -direct, -total_actions, path.name)


def select_trajectory_files(
    kimi_dir: Path, swe_dir: Path, kimi_count: int = 25, swe_count: int = 25
) -> list[dict[str, Any]]:
    kimi_files = sorted(kimi_dir.glob("*.json"), key=kimi_selection_score)
    swe_files = sorted(swe_dir.glob("*.json"), key=lambda path: path.name)
    if len(kimi_files) < kimi_count:
        raise ValueError(f"need {kimi_count} Kimi files, found {len(kimi_files)}")
    if len(swe_files) < swe_count:
        raise ValueError(f"need {swe_count} SWE files, found {len(swe_files)}")
    return [
        *({"source_type": "kimi", "path": path} for path in kimi_files[:kimi_count]),
        *({"source_type": "swe", "path": path} for path in swe_files[:swe_count]),
    ]


def build_user_prompt(step: dict[str, Any]) -> str:
    return PROMPTS.user_template.format(
        thought_step=step["thought_step"],
        user_request=step["user_request"] or "[No user request provided]",
        preceding_context=step["preceding_context"] or "[No preceding context]",
        thought=step["thought"] or "[Empty observable Thought]",
    )


def parse_json_response(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1)
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("model response must be a JSON object")
    return parsed


def validate_intent_result(
    parsed: dict[str, Any], thought_step: int, thought: str
) -> dict[str, Any]:
    if parsed.get("thought_step") != thought_step:
        raise ValueError(f"thought_step must equal {thought_step}")
    intents = parsed.get("execution_intents")
    if not isinstance(intents, list):
        raise ValueError("execution_intents must be a list")
    normalized: list[dict[str, str]] = []
    for sequence, intent in enumerate(intents, start=1):
        if not isinstance(intent, dict):
            raise ValueError("each execution intent must be an object")
        expected_id = f"T{thought_step}-I{sequence}"
        if intent.get("intent_id") != expected_id:
            raise ValueError(f"intent_id must equal {expected_id}")
        intent_text = intent.get("intent_text")
        source_quote = intent.get("source_quote")
        if not isinstance(intent_text, str) or not intent_text.strip():
            raise ValueError("intent_text must be a non-empty string")
        if not isinstance(source_quote, str) or not source_quote.strip():
            raise ValueError("source_quote must be a non-empty string")
        if source_quote not in thought:
            raise ValueError("source_quote must be an exact substring of the Thought")
        normalized.append(
            {
                "intent_id": expected_id,
                "intent_text": intent_text.strip(),
                "source_quote": source_quote,
            }
        )
    no_intent_reason = parsed.get("no_intent_reason")
    if normalized:
        if no_intent_reason is not None:
            raise ValueError("no_intent_reason must be null when intents are present")
    elif not isinstance(no_intent_reason, str) or not no_intent_reason.strip():
        raise ValueError("no_intent_reason must explain why no intent was extracted")
    return {
        "thought_step": thought_step,
        "execution_intents": normalized,
        "no_intent_reason": None if normalized else no_intent_reason.strip(),
    }


def build_chat_payload(user_prompt: str, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPTS.system},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 4000,
        "response_format": {"type": "json_object"},
        "stream": False,
    }


def request_chat_completion(
    *, base_url: str, api_key: str, model: str, user_prompt: str, timeout: int
) -> tuple[str, dict[str, Any]]:
    payload = build_chat_payload(user_prompt, model)
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("API response has no choices")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("API response has empty message content")
    return content, body.get("usage") or {}


def extract_with_retries(
    *,
    step: dict[str, Any],
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    max_retries: int,
) -> tuple[dict[str, Any], str, dict[str, Any], int]:
    user_prompt = build_user_prompt(step)
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw, usage = request_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                user_prompt=user_prompt,
                timeout=timeout,
            )
            parsed = parse_json_response(raw)
            result = validate_intent_result(parsed, step["thought_step"], step["thought"])
            return result, raw, usage, attempt
        except Exception as exc:  # network, JSON, and schema errors all retry
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1) + random.random(), 8.0))
    raise RuntimeError(f"intent extraction failed after {max_retries} attempts: {last_error}")


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def trajectory_identity(source_type: str, trajectory: dict[str, Any], path: Path) -> str:
    if source_type == "kimi":
        return normalize_text(trajectory.get("id")) or path.stem
    return normalize_text(trajectory.get("case_id")) or path.stem


def prepare_output_record(
    source_type: str,
    source_path: Path,
    trajectory: dict[str, Any],
    steps: list[dict[str, Any]],
    model: str,
    base_url: str,
    data_root: Path | None,
) -> dict[str, Any]:
    return {
        "schema_version": "thought_intent_v1",
        "source_type": source_type,
        "source_file": portable_source_name(source_path, data_root),
        "trajectory_id": trajectory_identity(source_type, trajectory, source_path),
        "task": normalize_text(trajectory.get("task")),
        "metadata": {
            "category": trajectory.get("category"),
            "subcategory": trajectory.get("subcategory"),
            "case_id": trajectory.get("case_id"),
            "instance_id": trajectory.get("instance_id"),
            "model_name": trajectory.get("model_name"),
            "target": trajectory.get("target"),
        },
        "generation": {
            "provider": "deepseek",
            "base_url": base_url,
            "model": model,
            "prompt_version": "thought_to_intent_user_v1",
            "system_prompt_sha256": hashlib.sha256(
                PROMPTS.system.encode("utf-8")
            ).hexdigest(),
        },
        "status": "pending",
        "steps": [
            {
                **step,
                "status": "pending",
                "intent_result": None,
                "raw_model_response": None,
                "usage": None,
                "attempts": 0,
                "error": None,
            }
            for step in steps
        ],
    }


def merge_resume_record(existing: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    if existing.get("trajectory_id") != fresh.get("trajectory_id"):
        return fresh
    old_steps = {
        (step.get("thought_step"), step.get("thought")): step
        for step in existing.get("steps") or []
        if isinstance(step, dict)
    }
    for step in fresh["steps"]:
        old = old_steps.get((step["thought_step"], step["thought"]))
        if old and old.get("status") in {"success", "skipped_empty"}:
            step.update(
                {
                    key: old.get(key)
                    for key in (
                        "status",
                        "intent_result",
                        "raw_model_response",
                        "usage",
                        "attempts",
                        "error",
                    )
                }
            )
    return fresh


def process_trajectory(
    *,
    source_type: str,
    source_path: Path,
    output_dir: Path,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    max_retries: int,
    context_chars: int,
    dry_run: bool,
    data_root: Path | None,
) -> dict[str, Any]:
    trajectory = read_json(source_path)
    steps = (
        extract_kimi_steps(trajectory, context_chars)
        if source_type == "kimi"
        else extract_swe_steps(trajectory, context_chars)
    )
    output_path = output_dir / f"{source_type}__{source_path.name}"
    record = prepare_output_record(
        source_type,
        source_path,
        trajectory,
        steps,
        model,
        base_url,
        data_root,
    )
    if output_path.exists():
        record = merge_resume_record(read_json(output_path), record)
    atomic_write_json(output_path, record)

    for step in record["steps"]:
        if step["status"] in {"success", "skipped_empty"}:
            continue
        if not step["thought"].strip():
            step["status"] = "skipped_empty"
            step["intent_result"] = {
                "thought_step": step["thought_step"],
                "execution_intents": [],
                "no_intent_reason": "The observable Thought is empty.",
            }
            step["error"] = None
            atomic_write_json(output_path, record)
            continue
        if dry_run:
            step["status"] = "dry_run"
            continue
        try:
            result, raw, usage, attempts = extract_with_retries(
                step=step,
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout=timeout,
                max_retries=max_retries,
            )
            step.update(
                {
                    "status": "success",
                    "intent_result": result,
                    "raw_model_response": raw,
                    "usage": usage,
                    "attempts": attempts,
                    "error": None,
                }
            )
        except Exception as exc:
            step.update(
                {
                    "status": "error",
                    "attempts": max_retries,
                    "error": str(exc),
                }
            )
        atomic_write_json(output_path, record)

    statuses = [step["status"] for step in record["steps"]]
    if all(status in {"success", "skipped_empty"} for status in statuses):
        record["status"] = "completed"
    elif dry_run and all(status in {"dry_run", "skipped_empty"} for status in statuses):
        record["status"] = "dry_run"
    else:
        record["status"] = "partial"
    record["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(output_path, record)
    with PRINT_LOCK:
        completed = sum(status in {"success", "skipped_empty"} for status in statuses)
        print(f"[{record['status']}] {output_path.name}: {completed}/{len(statuses)} steps", flush=True)
    return record


def summarize(
    records: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    args: argparse.Namespace,
    data_root: Path | None,
) -> dict[str, Any]:
    steps = [step for record in records for step in record.get("steps") or []]
    status_counts: dict[str, int] = {}
    for step in steps:
        status = str(step.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    intents = sum(
        len((step.get("intent_result") or {}).get("execution_intents") or [])
        for step in steps
    )
    usage_fields: dict[str, int] = {}
    for step in steps:
        for key, value in (step.get("usage") or {}).items():
            if isinstance(value, int):
                usage_fields[key] = usage_fields.get(key, 0) + value
    return {
        "schema_version": "thought_intent_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "selected_trajectory_count": len(selected),
        "source_counts": {
            "kimi": sum(item["source_type"] == "kimi" for item in selected),
            "swe": sum(item["source_type"] == "swe" for item in selected),
        },
        "step_count": len(steps),
        "step_status_counts": status_counts,
        "execution_intent_count": intents,
        "usage": usage_fields,
        "selected_files": [
            {
                "source_type": item["source_type"],
                "path": portable_source_name(item["path"], data_root),
            }
            for item in selected
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--kimi-dir", type=Path)
    parser.add_argument("--swe-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--kimi-count", type=int, default=25)
    parser.add_argument("--swe-count", type=int, default=25)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("INTENT_API_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("INTENT_MODEL_NAME", "deepseek-v4-flash"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--context-chars", type=int, default=6000)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve_cli_directories(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path | None]:
    needs_root = any(
        value is None for value in (args.kimi_dir, args.swe_dir, args.output_dir)
    )
    configured_root = args.data_root is not None or bool(os.environ.get("TA_DATA_ROOT"))
    data_root = resolve_data_root(args.data_root) if needs_root or configured_root else None
    kimi_dir = (
        args.kimi_dir.expanduser().resolve()
        if args.kimi_dir is not None
        else resolve_under_data_root(data_root, DEFAULT_KIMI_RELATIVE)
    )
    swe_dir = (
        args.swe_dir.expanduser().resolve()
        if args.swe_dir is not None
        else resolve_under_data_root(data_root, DEFAULT_SWE_RELATIVE)
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else resolve_under_data_root(data_root, DEFAULT_OUTPUT_RELATIVE)
    )
    return kimi_dir, swe_dir, output_dir, data_root


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = os.environ.get("INTENT_API_KEY")
    if not args.dry_run and not api_key:
        raise SystemExit("Set INTENT_API_KEY before running")
    kimi_dir, swe_dir, output_dir, data_root = _resolve_cli_directories(args)
    selected = select_trajectory_files(
        kimi_dir,
        swe_dir,
        args.kimi_count,
        args.swe_count,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = [
        {
            "source_type": item["source_type"],
            "source_path": item["path"],
            "output_dir": output_dir,
            "base_url": args.base_url,
            "api_key": api_key or "",
            "model": args.model,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "context_chars": args.context_chars,
            "dry_run": args.dry_run,
            "data_root": data_root,
        }
        for item in selected
    ]
    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_trajectory, **item) for item in kwargs]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
    manifest = summarize(records, selected, args, data_root)
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0 if not manifest["step_status_counts"].get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
