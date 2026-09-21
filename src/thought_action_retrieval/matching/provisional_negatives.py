"""Validation primitives for the provisional negative-case pilot."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from thought_action_retrieval.matching.phase1_dataset import split_by_group


PROVISIONAL_STATUS = "provisional_auto_pair"
PILOT_TIER = "feasibility_only"
PROVISIONAL_PROVENANCE = "provisional_llm_generated"
NEGATIVE_TYPES = frozenset(
    {
        "same_tool_wrong_parameter",
        "wrong_tool_same_object",
        "same_trajectory_unrelated",
    }
)
_ACTION_RE = re.compile(r"^\[ACTION\] ([^\s(),]+)\((.*)\)$", re.DOTALL)
_PATH_FIELDS = frozenset({"path", "command", "args", "query"})
_SUPPORTING_TOOLS = frozenset(
    {
        "open",
        "search_dir",
        "search_file",
        "find",
        "goto",
        "scroll_down",
        "scroll_up",
    }
)


@dataclass(frozen=True)
class ParsedAction:
    """A normalized Action string parsed into typed named fields."""

    tool: str
    fields: tuple[tuple[str, Any], ...]
    serialized: str

    def field_map(self) -> dict[str, Any]:
        return dict(self.fields)

    def value(self, name: str) -> Any:
        return self.field_map().get(name)

    @property
    def target(self) -> str | None:
        values = self.field_map()
        for field in ("path", "query", "command", "args", "line"):
            value = values.get(field)
            if value is None:
                continue
            if field == "line":
                return str(value)
            text = str(value).strip()
            if text:
                return text
        return None


@dataclass(frozen=True)
class SplitIndexes:
    """Candidate indexes used to validate generated Actions."""

    by_id: dict[str, dict[str, Any]]
    actions_by_split: dict[str, tuple[dict[str, Any], ...]]
    actions_by_tool_split: dict[tuple[str, str], tuple[dict[str, Any], ...]]


@dataclass(frozen=True)
class ValidatedNegative:
    """A proposal that passed all local provisional-pilot gates."""

    negative_type: str
    serialized_action: str
    parsed_action: ParsedAction
    source_candidate_id: str | None
    changed_fields: tuple[str, ...]
    target: str | None
    value_origin: str
    risk_flags: tuple[str, ...]
    reason: str
    provenance: str = PROVISIONAL_PROVENANCE
    supervision_status: str = PROVISIONAL_STATUS
    experiment_tier: str = PILOT_TIER
    human_reviewed: bool = False


@dataclass(frozen=True)
class GenerationValidation:
    """Accepted proposals and rejected proposal audit rows for one response."""

    accepted: tuple[ValidatedNegative, ...]
    rejected: tuple[dict[str, Any], ...]


def _split_fields(text: str) -> list[str]:
    fields: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    depth = 0
    for index, character in enumerate(text):
        if escaped:
            escaped = False
            continue
        if quote is not None:
            if character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character == "(" :
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced Action parentheses")
        elif character == "," and depth == 0:
            fields.append(text[start:index].strip())
            start = index + 1
    if quote is not None or depth != 0:
        raise ValueError("unterminated Action field")
    tail = text[start:].strip()
    if tail:
        fields.append(tail)
    return fields


def _parse_value(value: str) -> Any:
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Action value must be JSON or an integer: {value}") from error
    if not isinstance(parsed, (str, int, float, bool)) and parsed is not None:
        raise ValueError("Action fields must be scalar")
    return parsed


def parse_serialized_action(text: str) -> ParsedAction:
    """Parse the deterministic `[ACTION] tool(field=value)` representation."""

    match = _ACTION_RE.fullmatch(text.strip())
    if not match:
        raise ValueError("Action must use the [ACTION] tool(...) format")
    tool, body = match.groups()
    fields: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for raw_field in _split_fields(body):
        if "=" not in raw_field:
            raise ValueError("Action field must use name=value")
        name, raw_value = (part.strip() for part in raw_field.split("=", 1))
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in seen:
            raise ValueError(f"invalid or duplicate Action field: {name}")
        if not raw_value:
            raise ValueError(f"empty Action value: {name}")
        seen.add(name)
        fields.append((name, _parse_value(raw_value)))
    return ParsedAction(tool=tool, fields=tuple(fields), serialized=text.strip())


def _candidate_action(row: dict[str, Any]) -> ParsedAction:
    action = row.get("action") or row.get("positive_action")
    if not isinstance(action, dict) or not isinstance(action.get("serialized"), str):
        raise ValueError(f"candidate has no serialized Action: {row.get('candidate_id')}")
    return parse_serialized_action(action["serialized"])


def build_split_indexes(rows: Iterable[dict[str, Any]]) -> SplitIndexes:
    """Index candidates while rejecting duplicate IDs and missing split data."""

    by_id: dict[str, dict[str, Any]] = {}
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_tool_split: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        candidate_id = str(row.get("candidate_id", "")).strip()
        split = str(row.get("split", "")).strip()
        if not candidate_id or not split:
            raise ValueError("indexed candidate requires candidate_id and split")
        if candidate_id in by_id:
            raise ValueError(f"duplicate indexed candidate: {candidate_id}")
        if row.get("source") != "swe":
            raise ValueError("provisional indexes accept SWE candidates only")
        if row.get("supervision_status") != PROVISIONAL_STATUS:
            raise ValueError("candidate is missing provisional supervision status")
        if row.get("human_reviewed") is not False:
            raise ValueError("provisional candidate cannot be human reviewed")
        parsed = _candidate_action(row)
        materialized = dict(row)
        materialized["_parsed_action"] = parsed
        by_id[candidate_id] = materialized
        by_split[split].append(materialized)
        by_tool_split[(split, parsed.tool)].append(materialized)
    return SplitIndexes(
        by_id=by_id,
        actions_by_split={key: tuple(value) for key, value in by_split.items()},
        actions_by_tool_split={key: tuple(value) for key, value in by_tool_split.items()},
    )


def prepare_provisional_rows(
    candidates: Iterable[dict[str, Any]],
    seed: int = 20260921,
) -> tuple[dict[str, Any], ...]:
    """Watermark automatic pairs and assign isolated deterministic splits."""

    prepared: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.get("source") != "swe":
            raise ValueError("provisional pilot accepts SWE candidates only")
        _candidate_action(candidate)
        row = dict(candidate)
        row.update(
            {
                "supervision_status": PROVISIONAL_STATUS,
                "experiment_tier": PILOT_TIER,
                "human_reviewed": False,
            }
        )
        prepared.append(row)
    split = split_by_group(prepared, seed=seed)
    rows = (*split.train, *split.validation, *split.test)
    return tuple(sorted(rows, key=lambda row: str(row["candidate_id"])))


def _pool_order(parent: dict[str, Any], row: dict[str, Any]) -> tuple[int, int, str]:
    parent_action = _candidate_action(parent)
    candidate_action = row["_parsed_action"]
    same_trajectory = row.get("trajectory_id") == parent.get("trajectory_id")
    same_tool = candidate_action.tool == parent_action.tool
    digest = hashlib.sha256(str(row["candidate_id"]).encode("utf-8")).hexdigest()
    return (0 if same_trajectory else 1, 0 if same_tool else 1, digest)


def build_generation_request(
    parent: dict[str, Any],
    indexes: SplitIndexes,
    max_pool_actions: int = 24,
) -> dict[str, Any]:
    """Build a bounded, stable same-split evidence pool for one model request."""

    if max_pool_actions < 1:
        raise ValueError("max_pool_actions must be positive")
    split = _parent_split(parent, indexes)
    parent_id = str(parent["candidate_id"])
    candidates = [
        row
        for row in indexes.actions_by_split.get(split, ())
        if row["candidate_id"] != parent_id
        and row["_parsed_action"].serialized != _candidate_action(parent).serialized
    ]
    selected = sorted(candidates, key=lambda row: _pool_order(parent, row))[
        :max_pool_actions
    ]
    observed_pool = [
        {
            "candidate_id": row["candidate_id"],
            "trajectory_id": row["trajectory_id"],
            "thought_step": row.get("thought_step"),
            "serialized_action": row["_parsed_action"].serialized,
            "same_trajectory": row["trajectory_id"] == parent["trajectory_id"],
        }
        for row in selected
    ]
    request_body = {
        "candidate_id": parent_id,
        "split": split,
        "trajectory_id": parent["trajectory_id"],
        "thought_step": parent.get("thought_step"),
        "intent_text": parent["intent_text"],
        "positive_action": _candidate_action(parent).serialized,
        "allowed_negative_types": sorted(NEGATIVE_TYPES),
        "observed_action_pool": observed_pool,
        "supervision_status": PROVISIONAL_STATUS,
        "experiment_tier": PILOT_TIER,
    }
    digest = hashlib.sha256(
        json.dumps(request_body, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return {"request_id": f"{parent_id}:R-{digest}", **request_body}


def _parent_split(parent: dict[str, Any], indexes: SplitIndexes) -> str:
    candidate_id = str(parent.get("candidate_id", ""))
    indexed = indexes.by_id.get(candidate_id)
    if indexed is None:
        raise ValueError(f"parent is not present in split indexes: {candidate_id}")
    return str(indexed["split"])


def _same_target(left: ParsedAction, right: ParsedAction) -> bool:
    left_target = left.target
    right_target = right.target
    return bool(left_target and right_target and left_target == right_target)


def _changed_fields(parent: ParsedAction, candidate: ParsedAction) -> tuple[str, ...]:
    if parent.tool != candidate.tool:
        return ()
    left = parent.field_map()
    right = candidate.field_map()
    if set(left) != set(right):
        return ()
    return tuple(sorted(name for name in left if left[name] != right[name]))


def _has_shared_target_token(parent: dict[str, Any], candidate: ParsedAction) -> bool:
    intent = str(parent.get("intent_text", "")).lower()
    positive = str((parent.get("action") or {}).get("serialized", "")).lower()
    target = (candidate.target or "").lower()
    if not target:
        return False
    tokens = {token for token in re.findall(r"[a-z0-9_./-]+", target) if len(token) > 1}
    return bool(tokens & set(re.findall(r"[a-z0-9_./-]+", intent + " " + positive)))


def _validate_source(
    proposal: dict[str, Any],
    indexes: SplitIndexes,
    split: str,
) -> tuple[dict[str, Any] | None, str]:
    source_id = proposal.get("source_candidate_id")
    if source_id is None:
        return None, "synthetic_value"
    if not isinstance(source_id, str) or source_id not in indexes.by_id:
        raise ValueError("source candidate does not exist")
    source = indexes.by_id[source_id]
    if source["split"] != split:
        raise ValueError("source candidate must be in the same split")
    return source, "observed_value"


def validate_negative_proposal(
    parent: dict[str, Any],
    proposal: dict[str, Any],
    indexes: SplitIndexes,
) -> ValidatedNegative:
    """Validate one model proposal without making a semantic LLM judgment."""

    if parent.get("supervision_status") != PROVISIONAL_STATUS:
        raise ValueError("parent is not provisional")
    if parent.get("human_reviewed") is not False:
        raise ValueError("parent must not be human reviewed")
    negative_type = proposal.get("negative_type")
    if negative_type not in NEGATIVE_TYPES:
        raise ValueError("unsupported negative type")
    serialized = proposal.get("serialized_action")
    if not isinstance(serialized, str) or not serialized.strip():
        raise ValueError("serialized_action is required")
    candidate = parse_serialized_action(serialized)
    positive = _candidate_action(parent)
    if candidate.serialized == positive.serialized:
        raise ValueError("negative Action must differ from positive Action")
    split = _parent_split(parent, indexes)
    source, value_origin = _validate_source(proposal, indexes, split)
    source_action = source["_parsed_action"] if source else None
    changed = _changed_fields(positive, candidate)
    risk_flags: list[str] = []

    if negative_type == "same_tool_wrong_parameter":
        if candidate.tool != positive.tool:
            raise ValueError("same tool is required")
        if len(changed) != 1:
            raise ValueError("same-tool negative must change exactly one field")
        field = changed[0]
        if field not in _PATH_FIELDS and field != "line":
            raise ValueError("field is not eligible for parameter corruption")
        if source_action is not None:
            if source_action.tool != positive.tool or source_action.value(field) != candidate.value(field):
                raise ValueError("source Action does not provide the changed value")

    elif negative_type == "wrong_tool_same_object":
        if candidate.tool == positive.tool:
            raise ValueError("wrong-tool negative requires a different tool")
        if not _same_target(positive, candidate):
            raise ValueError("wrong-tool negative requires the exact same target")
        if source_action is not None and source_action.serialized != candidate.serialized:
            raise ValueError("source Action does not match proposed Action")

    else:
        if source is None:
            raise ValueError("same-trajectory negative requires an observed source")
        if source["trajectory_id"] != parent["trajectory_id"]:
            raise ValueError("source candidate must be in the same trajectory")
        distance = abs(int(source.get("thought_step", 0)) - int(parent.get("thought_step", 0)))
        if distance < 2:
            raise ValueError("same-trajectory candidate must be at least two steps away")
        if source_action is None or source_action.serialized != candidate.serialized:
            raise ValueError("source Action does not match proposed Action")
        if candidate.tool in _SUPPORTING_TOOLS:
            raise ValueError("same-trajectory candidate may be a supporting or navigation Action")
        if candidate.tool == positive.tool or _has_shared_target_token(parent, candidate):
            raise ValueError("same-trajectory candidate is not sufficiently unrelated")
        risk_flags.append("higher_false_negative_risk")

    reason = proposal.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason is required")
    return ValidatedNegative(
        negative_type=negative_type,
        serialized_action=candidate.serialized,
        parsed_action=candidate,
        source_candidate_id=proposal.get("source_candidate_id"),
        changed_fields=changed,
        target=candidate.target,
        value_origin=value_origin,
        risk_flags=tuple(risk_flags),
        reason=reason.strip(),
    )


def validate_generation_response(
    parent: dict[str, Any],
    request: dict[str, Any],
    response: dict[str, Any],
    indexes: SplitIndexes,
) -> GenerationValidation:
    """Validate one generated response and retain every rejected proposal."""

    if response.get("request_id") != request.get("request_id"):
        raise ValueError("response request_id does not match generation request")
    proposals = response.get("negative_cases")
    if not isinstance(proposals, list):
        raise ValueError("response negative_cases must be a list")

    accepted: list[ValidatedNegative] = []
    rejected: list[dict[str, Any]] = []
    for proposal in proposals:
        if not isinstance(proposal, dict):
            rejected.append({"proposal": proposal, "error": "proposal must be an object"})
            continue
        try:
            accepted.append(validate_negative_proposal(parent, proposal, indexes))
        except ValueError as error:
            rejected.append({**proposal, "error": str(error)})
    return GenerationValidation(accepted=tuple(accepted), rejected=tuple(rejected))


def select_validated_negatives(
    parent_id: str,
    validated: Iterable[ValidatedNegative],
) -> tuple[dict[str, Any], ...]:
    """Select at most five negatives using the pilot's deterministic 3+1+1 caps."""

    caps = {
        "same_tool_wrong_parameter": 3,
        "wrong_tool_same_object": 1,
        "same_trajectory_unrelated": 1,
    }
    grouped: dict[str, list[ValidatedNegative]] = defaultdict(list)
    seen: set[str] = set()
    for negative in validated:
        if negative.serialized_action in seen:
            continue
        seen.add(negative.serialized_action)
        grouped[negative.negative_type].append(negative)

    selected: list[ValidatedNegative] = []
    for negative_type in caps:
        selected.extend(grouped[negative_type][: caps[negative_type]])

    rows: list[dict[str, Any]] = []
    for ordinal, negative in enumerate(selected, start=1):
        rows.append(
            {
                "negative_id": stable_negative_id(
                    parent_id, ordinal, negative.serialized_action
                ),
                "serialized": negative.serialized_action,
                "negative_type": negative.negative_type,
                "source_candidate_id": negative.source_candidate_id,
                "changed_fields": list(negative.changed_fields),
                "target": negative.target,
                "value_origin": negative.value_origin,
                "risk_flags": list(negative.risk_flags),
                "reason": negative.reason,
                "provenance": negative.provenance,
                "supervision_status": negative.supervision_status,
                "experiment_tier": negative.experiment_tier,
                "human_reviewed": negative.human_reviewed,
            }
        )
    return tuple(rows)


def stable_negative_id(parent_id: str, ordinal: int, serialized_action: str) -> str:
    digest = hashlib.sha256(serialized_action.encode("utf-8")).hexdigest()[:10]
    return f"{parent_id}:N{ordinal}-{digest}"
