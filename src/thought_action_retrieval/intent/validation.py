"""Validate and publish action-blind temporal Intent records for Pilot v2."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from thought_action_retrieval.intent.extraction import TEMPORAL_PROMPT_PATH, load_prompt_bundle
from thought_action_retrieval.matching.v2_schema import hash_intent, prompt_bundle_sha256


_HIGH_REASONS = (
    "uncertain_scope",
    "multiple_intents",
    "multi_target",
    "multiple_operation_cues",
    "short_quote",
    "action_count_gap",
)
_MEDIUM_REASONS = (
    "conditional_scope",
    "unresolved_target",
    "long_thought",
    "intent_count_threshold",
    "rule_conflict",
)
_OPERATION_CUES = re.compile(
    r"\b(?:list|read|write|create|fetch|download|search|inspect|run|execute|open|edit|update|delete|remove|move|copy|install|parse|extract|save|find|check)\w*\b",
    re.IGNORECASE,
)
_UNRESOLVED_TARGET = re.compile(r"\b(?:it|them|this|that|these|those)\b", re.IGNORECASE)


def approved_temporal_prompt_provenance() -> dict[str, str]:
    """Return the exact hashes for the repository's reviewed temporal prompt."""
    prompts = load_prompt_bundle(TEMPORAL_PROMPT_PATH)
    return {
        "system_prompt_sha256": hashlib.sha256(prompts.system.encode("utf-8")).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(prompts.user_template.encode("utf-8")).hexdigest(),
        "prompt_sha256": prompt_bundle_sha256(prompts.system, prompts.user_template),
    }


def _action_counts(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int], int]:
    counts: dict[tuple[str, int], int] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"action audit row {index} must be an object")
        trajectory_id = row.get("trajectory_id")
        thought_step = row.get("thought_step")
        if not isinstance(trajectory_id, str) or not trajectory_id or type(thought_step) is not int:
            raise ValueError(f"action audit row {index} has invalid trajectory identity")
        key = (trajectory_id, thought_step)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _risk_level(reasons: list[str]) -> str:
    if any(reason in _HIGH_REASONS for reason in reasons):
        return "high"
    if any(reason in _MEDIUM_REASONS for reason in reasons):
        return "medium"
    return "low"


def score_intent_risk(
    intent: Mapping[str, Any],
    *,
    thought: str,
    intent_count: int,
    action_count: int = 0,
    raw_action_count: int | None = None,
    long_thought_threshold: int = 500,
    intent_count_threshold: int = 3,
    rule_conflict: bool = False,
) -> dict[str, Any]:
    """Return deterministic, non-semantic risk metadata for one exported Intent."""
    if not isinstance(intent, Mapping) or not isinstance(thought, str):
        raise ValueError("intent and thought must be valid")
    if raw_action_count is not None:
        action_count = raw_action_count
    if type(intent_count) is not int or type(action_count) is not int:
        raise ValueError("intent_count and action_count must be integers")
    if type(long_thought_threshold) is not int or type(intent_count_threshold) is not int:
        raise ValueError("risk thresholds must be integers")
    scope = intent.get("temporal_scope")
    quote = intent.get("source_quote")
    text = intent.get("intent_text")
    if scope not in {"current", "future", "conditional", "uncertain"}:
        raise ValueError("invalid temporal_scope")
    if not isinstance(quote, str) or not quote or not isinstance(text, str) or not text:
        raise ValueError("intent text and source_quote must be non-empty strings")
    if quote not in thought:
        raise ValueError("source_quote must be an exact substring of the Thought")
    cue_text = f"{text} {quote}"
    operation_count = len(list(_OPERATION_CUES.finditer(cue_text)))
    multi_target = bool(re.search(r"\b\w+(?:\s*,\s*\w+)+(?:\s*,?\s*(?:and|or)\s+\w+)?", cue_text, re.IGNORECASE))
    if not multi_target and _OPERATION_CUES.search(cue_text):
        multi_target = bool(re.search(r"\b(?:and|or)\s+[A-Za-z0-9_./-]+", cue_text, re.IGNORECASE))
    tests = {
        "uncertain_scope": scope == "uncertain",
        "multiple_intents": intent_count > 1,
        "multi_target": multi_target,
        # A second verb in explanatory context ("after listing ...") is not
        # itself a concurrent operation; require an explicit coordination cue.
        "multiple_operation_cues": operation_count > 1 and bool(re.search(r"\b(?:then|and|or)\b", cue_text, re.IGNORECASE)),
        "short_quote": len(re.sub(r"\s+", "", quote)) < 12,
        "action_count_gap": abs(intent_count - action_count) > 1,
        "conditional_scope": scope == "conditional",
        "unresolved_target": bool(_UNRESOLVED_TARGET.search(cue_text)),
        "long_thought": len(thought) > long_thought_threshold,
        "intent_count_threshold": intent_count > intent_count_threshold,
        "rule_conflict": bool(rule_conflict),
    }
    reasons = [reason for reason in (*_HIGH_REASONS, *_MEDIUM_REASONS) if tests[reason]]
    return {"level": _risk_level(reasons), "reasons": reasons}


def _validate_record_provenance(record: Mapping[str, Any], approved: Mapping[str, str]) -> tuple[str, str]:
    if record.get("schema_version") != "thought_intent_temporal_v1":
        raise ValueError("record must use thought_intent_temporal_v1")
    if record.get("status") != "completed":
        raise ValueError("record must be completed")
    trajectory_id = record.get("trajectory_id")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("record trajectory_id must be a non-empty string")
    generation = record.get("generation")
    if not isinstance(generation, Mapping) or generation.get("temporal_scope") is not True:
        raise ValueError("record must have temporal provenance")
    extraction_run_id = generation.get("extraction_run_id")
    if not isinstance(extraction_run_id, str) or not extraction_run_id:
        raise ValueError("record extraction_run_id must be a non-empty string")
    for field, expected in approved.items():
        if generation.get(field) != expected:
            raise ValueError(f"record {field} does not match the approved temporal prompt")
    return trajectory_id, extraction_run_id


def _validate_step(step: Mapping[str, Any], trajectory_id: str) -> tuple[int, str, list[Mapping[str, Any]], str | None]:
    thought_step = step.get("thought_step")
    thought = step.get("thought")
    if type(thought_step) is not int or not isinstance(thought, str):
        raise ValueError(f"invalid Thought step for trajectory {trajectory_id}")
    if step.get("status") not in {"success", "skipped_empty"}:
        raise ValueError(f"Thought step {thought_step} is not complete")
    result = step.get("intent_result")
    if not isinstance(result, Mapping) or result.get("thought_step") != thought_step:
        raise ValueError(f"Thought step {thought_step} has invalid intent_result")
    intents = result.get("execution_intents")
    reason = result.get("no_intent_reason")
    if not isinstance(intents, list):
        raise ValueError(f"Thought step {thought_step} execution_intents must be a list")
    if intents and reason is not None:
        raise ValueError(f"Thought step {thought_step} must not have no_intent_reason with intents")
    if not intents and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError(f"Thought step {thought_step} needs no_intent_reason")
    if not all(isinstance(intent, Mapping) for intent in intents):
        raise ValueError(f"Thought step {thought_step} contains an invalid Intent")
    return thought_step, thought, intents, None if intents else reason.strip()


def export_blind_intents(
    records: Sequence[Mapping[str, Any]],
    action_audit_rows: Iterable[Mapping[str, Any]] = (),
    *,
    long_thought_threshold: int = 500,
    intent_count_threshold: int = 3,
) -> list[dict[str, Any]]:
    """Export completed temporal records without copying any raw Action payload."""
    if not isinstance(records, Sequence):
        raise ValueError("records must be a sequence")
    approved = approved_temporal_prompt_provenance()
    action_counts = _action_counts(action_audit_rows)
    rows: list[dict[str, Any]] = []
    seen_intents: set[tuple[str, int, str]] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("record must be an object")
        trajectory_id, extraction_run_id = _validate_record_provenance(record, approved)
        steps = record.get("steps")
        if not isinstance(steps, list):
            raise ValueError(f"record {trajectory_id} steps must be a list")
        for step in steps:
            if not isinstance(step, Mapping):
                raise ValueError(f"record {trajectory_id} has invalid step")
            thought_step, thought, raw_intents, no_intent_reason = _validate_step(step, trajectory_id)
            exported_intents: list[dict[str, Any]] = []
            for raw_intent in sorted(raw_intents, key=lambda value: str(value.get("intent_id", ""))):
                intent_id = raw_intent.get("intent_id")
                intent_text = raw_intent.get("intent_text")
                source_quote = raw_intent.get("source_quote")
                temporal_scope = raw_intent.get("temporal_scope")
                identity = (trajectory_id, thought_step, intent_id)
                if not isinstance(intent_id, str) or not intent_id or identity in seen_intents:
                    raise ValueError(f"duplicate or invalid intent_id at {trajectory_id}:{thought_step}")
                if not isinstance(intent_text, str) or not intent_text.strip() or not isinstance(source_quote, str) or not source_quote.strip():
                    raise ValueError(f"invalid Intent text at {trajectory_id}:{thought_step}")
                if source_quote not in thought:
                    raise ValueError("source_quote must be an exact substring of the Thought")
                if temporal_scope not in {"current", "future", "conditional", "uncertain"}:
                    raise ValueError("invalid temporal_scope")
                exported = {
                    "trajectory_id": trajectory_id,
                    "thought_step": thought_step,
                    "intent_id": intent_id,
                    "intent_text": intent_text.strip(),
                    "source_quote": source_quote,
                    "temporal_scope": temporal_scope,
                    "prompt_sha256": approved["prompt_sha256"],
                    "extraction_run_id": extraction_run_id,
                }
                digest = hash_intent(exported)
                if "original_intent_sha256" in raw_intent and raw_intent["original_intent_sha256"] != digest:
                    raise ValueError("original_intent_sha256 does not match the frozen Intent contract")
                exported["original_intent_sha256"] = digest
                risk = score_intent_risk(
                    exported,
                    thought=thought,
                    intent_count=len(raw_intents),
                    action_count=action_counts.get((trajectory_id, thought_step), 0),
                    long_thought_threshold=long_thought_threshold,
                    intent_count_threshold=intent_count_threshold,
                    rule_conflict=raw_intent.get("rule_conflict") is True,
                )
                exported["risk_level"] = risk["level"]
                exported["risk_reasons"] = risk["reasons"]
                seen_intents.add(identity)
                exported_intents.append(exported)
            rows.append({
                "trajectory_id": trajectory_id,
                "thought_step": thought_step,
                "user_request": step.get("user_request") if isinstance(step.get("user_request"), str) else record.get("task", ""),
                "preceding_context": step.get("preceding_context") if isinstance(step.get("preceding_context"), str) else "",
                "thought": thought,
                "execution_intents": exported_intents,
                "no_intent_reason": no_intent_reason,
            })
    rows.sort(key=lambda row: (row["trajectory_id"], row["thought_step"]))
    return rows


def load_intent_run_records(intent_run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Load trajectory records deterministically, excluding extractor manifests."""
    root = Path(intent_run_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"intent-run directory does not exist: {root}")
    files = sorted(path for path in root.rglob("*.json") if path.name != "manifest.json")
    if not files:
        raise ValueError("intent-run directory contains no trajectory JSON files")
    import json
    records: list[dict[str, Any]] = []
    provenance: list[dict[str, str]] = []
    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid trajectory JSON: {path}") from error
        if not isinstance(record, dict):
            raise ValueError(f"trajectory JSON must be an object: {path}")
        records.append(record)
        provenance.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return records, provenance
