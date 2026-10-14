"""Frozen schema and deterministic hashing contracts for Matching Pilot v2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


ALIGNMENT_LABELS = frozenset({
    "exact_match",
    "partial_match",
    "object_mismatch",
    "type_mismatch",
    "no_action",
})
VERIFICATION_STATUSES = frozenset({
    "single_verified",
    "repeat_agreed",
    "self_adjudicated",
    "excluded_after_review",
})
RISK_LEVELS = frozenset({"high", "medium", "low"})

INTENT_HASH_FIELDS = (
    "trajectory_id",
    "thought_step",
    "intent_id",
    "intent_text",
    "source_quote",
    "temporal_scope",
    "prompt_sha256",
)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value into the canonical UTF-8 form."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def record_sha256(value: Any) -> str:
    """Return SHA-256 for canonical JSON or already-encoded bytes."""
    payload = value if isinstance(value, bytes) else canonical_json_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def prompt_bundle_sha256(system: str, user_template: str) -> str:
    """Hash the approved Prompt bundle as its exact two-field canonical object."""
    if not isinstance(system, str) or not isinstance(user_template, str):
        raise TypeError("Prompt sections must be strings")
    return record_sha256({"system": system, "user_template": user_template})


def intent_hash_payload(intent: Mapping[str, Any]) -> dict[str, Any]:
    """Select exactly the fields covered by the frozen Intent hash contract.

    Additional record metadata is intentionally ignored; missing contract fields
    are rejected so callers cannot accidentally hash a partially selected record.
    """
    if not isinstance(intent, Mapping):
        raise TypeError("intent must be a mapping")
    missing = [field for field in INTENT_HASH_FIELDS if field not in intent]
    if missing:
        raise ValueError(f"intent hash payload missing fields: {', '.join(missing)}")
    payload = {field: intent[field] for field in INTENT_HASH_FIELDS}
    # Null or non-string identity/text values create ambiguous hashes and are not
    # valid frozen Intent records. thought_step is deliberately an integer.
    for field in ("trajectory_id", "intent_id", "intent_text", "source_quote", "temporal_scope", "prompt_sha256"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise ValueError(f"intent hash field must be a non-empty string: {field}")
    if type(payload["thought_step"]) is not int:
        raise ValueError("intent hash field thought_step must be an integer")
    return payload


def hash_intent(intent: Mapping[str, Any]) -> str:
    """Hash an Intent using only the seven frozen contract fields."""
    return record_sha256(intent_hash_payload(intent))


def validate_unique_ids(rows: Sequence[Mapping[str, Any]], id_field: str = "stable_id") -> set[str]:
    """Validate non-empty stable IDs and reject duplicates."""
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index} must be an object")
        value = row.get(id_field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"row {index} has missing or invalid {id_field}")
        if value in seen:
            raise ValueError(f"duplicate {id_field}: {value}")
        seen.add(value)
    return seen


_PILOT_FIXED_VALUES = {
    "schema_version": "intent_action_matching_pilot_v2",
    "config_version": "pilot_config_v1",
    "run_name": "pilot_100",
    "trajectory_count": 100,
    "selection_seed": 20260927,
    "split_seed": 20260928,
    "review_policy": "single-plus-blind-repeat-v1",
    "intent_schema_version": "intent_v2",
    "action_schema_version": "action_v1",
    "matching_schema_version": "matching_v2",
    "verification_schema_version": "verification_v1",
    "action_atomicizer_version": "action-atomicizer-v1",
    "template_cluster_version": "normalized-task-minhash-v1",
    "split_version": "group-stratified-v1",
    "dataset_version": "dataset-v1",
}


def build_pilot_run_config(prompt_sha256: str) -> dict[str, Any]:
    """Build the deterministic, approved Pilot-100 run configuration."""
    if not isinstance(prompt_sha256, str) or not prompt_sha256:
        raise ValueError("prompt_sha256 must be a non-empty string")
    return {**_PILOT_FIXED_VALUES, "prompt_sha256": prompt_sha256}


def validate_pilot_run_config(config: Mapping[str, Any], prompt_sha256: str | None = None) -> dict[str, Any]:
    """Validate and return a frozen Pilot config, rejecting changed values."""
    if not isinstance(config, Mapping):
        raise ValueError("run config must be an object")
    expected = build_pilot_run_config(prompt_sha256 or str(config.get("prompt_sha256", "")))
    validate_frozen_run_config(config, expected)
    return dict(config)


def validate_frozen_run_config(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Reject reuse of a run root when any frozen config value changed."""
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        raise ValueError("run config must be an object")
    if set(actual) != set(expected):
        actual_only = sorted(set(actual) - set(expected))
        expected_only = sorted(set(expected) - set(actual))
        raise ValueError(
            "run config mismatch for keys: "
            f"unexpected={actual_only!r}, missing={expected_only!r}"
        )
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"run config mismatch for {key}: {actual.get(key)!r} != {value!r}")
