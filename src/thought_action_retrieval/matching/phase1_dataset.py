"""Build auditable Phase 1 one-Intent--one-short-Action datasets."""

from __future__ import annotations

import json
import math
import hashlib
import re
import shlex
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable


ALLOWED_REVIEW_LABELS = frozenset(
    {"direct_match", "partial_match", "no_match", "unfulfilled", "ambiguous"}
)
_ASSIGNMENT_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*", re.DOTALL)
_SHELL_CONTROL_TOKENS = frozenset({"&", "&&", "|", "||", ";"})


@dataclass(frozen=True)
class ActionNormalization:
    """A deterministic representation of one raw SWE Action."""

    raw: str
    command: str
    payload: str
    serialized: str


@dataclass(frozen=True)
class ActionAssessment:
    """Short-Action eligibility and diagnostics."""

    normalization: ActionNormalization | None
    action_tokens: int
    payload_tokens: int
    eligible: bool
    exclusion_reasons: tuple[str, ...]


@dataclass(frozen=True)
class BuildConfig:
    """Configuration for Phase 1 candidate construction."""

    max_action_tokens: int = 128
    max_payload_tokens: int = 64


@dataclass(frozen=True)
class BuildResult:
    """Candidate rows and reconciled audit statistics."""

    audit_summary: dict[str, Any]
    annotation_candidates: tuple[dict[str, Any], ...]
    unfulfilled_candidates: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ReviewResult:
    """Validated reviews merged with their source candidates."""

    reviewed_rows: tuple[dict[str, Any], ...]
    unreviewed_candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class SplitResult:
    """Trajectory/template-isolated dataset splits."""

    train: tuple[dict[str, Any], ...]
    validation: tuple[dict[str, Any], ...]
    test: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _token_estimate(value: str) -> int:
    return math.ceil(len(value) / 4) if value else 0


def _call(command: str, fields: tuple[tuple[str, str], ...]) -> str:
    arguments = ", ".join(f"{name}={value}" for name, value in fields)
    return f"[ACTION] {command}({arguments})"


def serialize_swe_action(raw: str) -> ActionNormalization:
    """Parse and deterministically serialize a short SWE Action."""

    stripped = raw.strip()
    if not stripped:
        raise ValueError("Action is empty")

    try:
        parts = shlex.split(stripped)
    except ValueError as error:
        raise ValueError("Action cannot be parsed") from error

    if not parts:
        raise ValueError("Action is empty")

    command, *arguments = parts
    payload = " ".join(arguments)

    if command == "search_dir" and len(arguments) == 2:
        serialized = _call(
            command,
            (("query", _quote(arguments[0])), ("path", _quote(arguments[1]))),
        )
    elif command == "open" and len(arguments) == 1:
        serialized = _call(command, (("path", _quote(payload)),))
    elif command == "open" and len(arguments) == 2 and arguments[1].isdigit():
        serialized = _call(
            command,
            (("path", _quote(arguments[0])), ("line", arguments[1])),
        )
    elif command in {"rm", "create"} and arguments:
        serialized = _call(command, (("path", _quote(payload)),))
    elif command == "goto" and len(arguments) == 1:
        line = arguments[0]
        serialized = _call(
            command,
            (("line", line if line.isdigit() else _quote(line)),),
        )
    elif command == "scroll_down" and not arguments:
        serialized = _call(command, ())
    elif command == "python" and arguments:
        serialized = _call(command, (("command", _quote(payload)),))
    else:
        serialized = _call(command, (("args", _quote(payload)),) if payload else ())

    return ActionNormalization(
        raw=raw,
        command=command,
        payload=payload,
        serialized=serialized,
    )


def _contains_shell_control(raw: str) -> bool:
    lexer = shlex.shlex(raw, posix=True, punctuation_chars="|&;")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return any(token in _SHELL_CONTROL_TOKENS for token in lexer)
    except ValueError:
        return False


def _is_assignment_fragment(raw: str) -> bool:
    try:
        parts = shlex.split(raw)
    except ValueError:
        return False
    if not parts:
        return False
    return all(_ASSIGNMENT_TOKEN_RE.fullmatch(part.rstrip(",")) for part in parts)


def assess_short_action(
    raw: str,
    max_action_tokens: int = 128,
    max_payload_tokens: int = 64,
) -> ActionAssessment:
    """Assess whether an Action is suitable for Phase 1 semantic matching."""

    stripped = raw.strip()
    reasons: list[str] = []
    normalization: ActionNormalization | None = None

    if not stripped:
        reasons.append("empty_action")
    else:
        try:
            normalization = serialize_swe_action(raw)
        except ValueError:
            reasons.append("unparseable_action")

    command = normalization.command if normalization else ""
    payload = normalization.payload if normalization else ""
    serialized = normalization.serialized if normalization else stripped

    if command == "edit":
        reasons.append("edit_action")
    if command == "end_of_edit" or stripped == "end_of_edit":
        reasons.append("end_of_edit")
    if _is_assignment_fragment(stripped):
        reasons.append("assignment_fragment")
    if _contains_shell_control(stripped):
        reasons.append("composite_shell")
    if "<<" in stripped:
        reasons.append("heredoc")
    if "\n" in stripped or "\r" in stripped:
        reasons.append("multiline")

    action_tokens = _token_estimate(serialized)
    payload_tokens = _token_estimate(payload)
    if action_tokens > max_action_tokens:
        reasons.append("action_too_long")
    if payload_tokens > max_payload_tokens:
        reasons.append("payload_too_long")

    return ActionAssessment(
        normalization=normalization,
        action_tokens=action_tokens,
        payload_tokens=payload_tokens,
        eligible=not reasons,
        exclusion_reasons=tuple(dict.fromkeys(reasons)),
    )


def _source_summary(summary: dict[str, Any], source: str) -> dict[str, int]:
    sources = summary.setdefault("sources", {})
    return sources.setdefault(source, {"trajectories": 0, "steps": 0})


def _annotation_fields() -> dict[str, str]:
    return {"label": "", "annotator": "", "notes": ""}


def _intent_values(step: dict[str, Any]) -> list[dict[str, Any]]:
    result = step.get("intent_result")
    if not isinstance(result, dict):
        return []
    intents = result.get("execution_intents", [])
    return [intent for intent in intents if isinstance(intent, dict)] if isinstance(intents, list) else []


def _action_values(step: dict[str, Any]) -> list[str]:
    actions = step.get("actual_actions", [])
    if not isinstance(actions, list):
        return []
    return [action for action in actions if isinstance(action, str)]


def _neighboring_actions(steps: list[dict[str, Any]], index: int) -> dict[str, list[str]]:
    previous = _action_values(steps[index - 1]) if index > 0 else []
    following = _action_values(steps[index + 1]) if index + 1 < len(steps) else []
    return {"previous": previous, "next": following}


def _base_candidate(
    record: dict[str, Any],
    step: dict[str, Any],
    intent: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    intent_text = str(intent.get("intent_text", "")).strip()
    return {
        "candidate_id": candidate_id,
        "source": "swe",
        "trajectory_id": str(record.get("trajectory_id", "")),
        "thought_step": step.get("thought_step"),
        "task": str(record.get("task", "")),
        "thought": str(step.get("thought", "")),
        "source_quote": str(intent.get("source_quote", "")),
        "intent_id": str(intent.get("intent_id", "")),
        "intent_text": intent_text,
        "intent_input": f"[INTENT] {intent_text}",
        "source_file": record.get("source_file"),
        "annotation": _annotation_fields(),
    }


def extract_candidates(
    records: Iterable[dict[str, Any]],
    config: BuildConfig,
) -> BuildResult:
    """Extract auditable SWE candidates from Thought-to-Intent records."""

    summary: dict[str, Any] = {
        "trajectories": 0,
        "steps": 0,
        "sources": {},
        "annotation_candidates": 0,
        "unfulfilled_candidates": 0,
        "excluded": {},
        "action_exclusion_reasons": {},
    }
    excluded: Counter[str] = Counter()
    action_exclusion_reasons: Counter[str] = Counter()
    annotations: list[dict[str, Any]] = []
    unfulfilled: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for record in records:
        source = str(record.get("source_type", "unknown")).lower()
        source_counts = _source_summary(summary, source)
        source_counts["trajectories"] += 1
        summary["trajectories"] += 1

        raw_steps = record.get("steps", [])
        steps = [step for step in raw_steps if isinstance(step, dict)] if isinstance(raw_steps, list) else []
        summary["steps"] += len(steps)
        source_counts["steps"] += len(steps)

        for index, step in enumerate(steps):
            if source != "swe":
                excluded["non_swe_source"] += 1
                continue
            if step.get("status") != "success":
                excluded["step_not_success"] += 1
                continue

            intents = _intent_values(step)
            actions = _action_values(step)
            if len(intents) != 1:
                excluded["intent_count_not_one"] += 1
                continue

            intent = intents[0]
            trajectory_id = str(record.get("trajectory_id", ""))
            intent_id = str(intent.get("intent_id", ""))
            base_id = f"swe:{trajectory_id}:{intent_id}"

            if not actions:
                candidate_id = base_id
                if candidate_id in seen_ids:
                    raise ValueError(f"duplicate candidate_id: {candidate_id}")
                seen_ids.add(candidate_id)
                row = _base_candidate(record, step, intent, candidate_id)
                row["positive_action"] = None
                row["neighboring_actions"] = _neighboring_actions(steps, index)
                unfulfilled.append(row)
                continue

            if len(actions) != 1:
                excluded["action_count_not_one"] += 1
                continue

            assessment = assess_short_action(
                actions[0],
                max_action_tokens=config.max_action_tokens,
                max_payload_tokens=config.max_payload_tokens,
            )
            if not assessment.eligible or assessment.normalization is None:
                excluded["ineligible_action"] += 1
                action_exclusion_reasons.update(assessment.exclusion_reasons)
                continue

            if base_id in seen_ids:
                raise ValueError(f"duplicate candidate_id: {base_id}")
            seen_ids.add(base_id)
            row = _base_candidate(record, step, intent, base_id)
            row["action"] = {
                "action_id": f"T{step.get('thought_step')}-A1",
                "raw": assessment.normalization.raw,
                "command": assessment.normalization.command,
                "serialized": assessment.normalization.serialized,
            }
            row["filter_diagnostics"] = {
                "action_tokens": assessment.action_tokens,
                "payload_tokens": assessment.payload_tokens,
                "exclusion_reasons": list(assessment.exclusion_reasons),
            }
            row["neighboring_actions"] = _neighboring_actions(steps, index)
            annotations.append(row)

    summary["annotation_candidates"] = len(annotations)
    summary["unfulfilled_candidates"] = len(unfulfilled)
    summary["excluded"] = dict(sorted(excluded.items()))
    summary["action_exclusion_reasons"] = dict(
        sorted(action_exclusion_reasons.items())
    )
    return BuildResult(
        audit_summary=summary,
        annotation_candidates=tuple(annotations),
        unfulfilled_candidates=tuple(unfulfilled),
    )


def _candidate_action(candidate: dict[str, Any]) -> dict[str, Any] | None:
    action = candidate.get("action", candidate.get("positive_action"))
    return action if isinstance(action, dict) else None


def validate_reviewed_annotations(
    candidates: Iterable[dict[str, Any]],
    reviewed_rows: Iterable[dict[str, Any]],
) -> ReviewResult:
    """Validate human reviews and merge them into immutable candidate copies."""

    candidate_list = list(candidates)
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidate_list:
        candidate_id = str(candidate.get("candidate_id", ""))
        if candidate_id in candidate_by_id:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        candidate_by_id[candidate_id] = candidate

    review_by_id: dict[str, dict[str, str]] = {}
    for review in reviewed_rows:
        candidate_id = str(review.get("candidate_id", ""))
        if candidate_id not in candidate_by_id:
            raise ValueError(f"unknown candidate_id: {candidate_id}")
        if candidate_id in review_by_id:
            raise ValueError(f"duplicate review: {candidate_id}")

        label = str(review.get("label", "")).strip()
        if not label:
            raise ValueError(f"missing label for {candidate_id}")
        if label not in ALLOWED_REVIEW_LABELS:
            raise ValueError(f"unsupported label for {candidate_id}: {label}")

        action = _candidate_action(candidate_by_id[candidate_id])
        if label == "direct_match" and action is None:
            raise ValueError("direct_match requires an Action")
        if label == "unfulfilled" and action is not None:
            raise ValueError("unfulfilled requires no Action")

        review_by_id[candidate_id] = {
            "label": label,
            "annotator": str(review.get("annotator", "")),
            "notes": str(review.get("notes", "")),
        }

    merged: list[dict[str, Any]] = []
    unreviewed: list[str] = []
    for candidate in candidate_list:
        candidate_id = str(candidate.get("candidate_id", ""))
        annotation = review_by_id.get(candidate_id)
        if annotation is None:
            unreviewed.append(candidate_id)
            continue
        row = deepcopy(candidate)
        row["annotation"] = annotation
        merged.append(row)

    return ReviewResult(
        reviewed_rows=tuple(merged),
        unreviewed_candidate_ids=tuple(unreviewed),
    )


def export_gold_relations(
    review_result: ReviewResult,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Export confirmed training positives and evaluation relations."""

    positives: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    for row in review_result.reviewed_rows:
        annotation = row["annotation"]
        label = annotation["label"]
        action = _candidate_action(row)
        provenance = {
            "candidate_id": row["candidate_id"],
            "thought_step": row.get("thought_step"),
            "source_file": row.get("source_file"),
            "annotation": deepcopy(annotation),
        }

        if label == "direct_match":
            positives.append(
                {
                    "candidate_id": row["candidate_id"],
                    "trajectory_id": row["trajectory_id"],
                    "intent_id": row["intent_id"],
                    "intent_text": row["intent_text"],
                    "intent_input": row["intent_input"],
                    "positive_action": deepcopy(action),
                    "relation_type": "individual",
                    "source": "swe",
                    "task": row.get("task", ""),
                    "provenance": provenance,
                }
            )

        if label in {"direct_match", "no_match", "unfulfilled"}:
            evaluations.append(
                {
                    "candidate_id": row["candidate_id"],
                    "trajectory_id": row["trajectory_id"],
                    "thought_step": row.get("thought_step"),
                    "intent_id": row["intent_id"],
                    "intent_text": row["intent_text"],
                    "action": deepcopy(action),
                    "relation_label": label,
                    "source": "swe",
                    "annotation": deepcopy(annotation),
                }
            )

    return tuple(positives), tuple(evaluations)


_QUOTED_RE = re.compile(r"(['\"])(?:\\.|(?!\1).)*\1")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_PATH_RE = re.compile(r"(?<![\w.-])(?:[a-zA-Z]:)?(?:[\w.-]+/)+[\w.-]+")
_NUMBER_RE = re.compile(r"\b\d+\b")
_SPACE_RE = re.compile(r"\s+")


def task_template_key(task: str) -> str:
    """Normalize issue-specific values while preserving task semantics."""

    normalized = task.lower()
    normalized = _QUOTED_RE.sub("<quoted>", normalized)
    normalized = _UUID_RE.sub("<uuid>", normalized)
    normalized = _PATH_RE.sub("<path>", normalized)
    normalized = _NUMBER_RE.sub("<number>", normalized)
    return _SPACE_RE.sub(" ", normalized).strip()


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _overlap(values: dict[str, set[str]]) -> list[str]:
    names = ("train", "validation", "test")
    overlap: set[str] = set()
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap.update(values[left] & values[right])
    return sorted(overlap)


def split_by_group(
    rows: list[dict[str, Any]],
    seed: int = 20260916,
) -> SplitResult:
    """Split rows without crossing trajectory or normalized-template groups."""

    disjoint = _DisjointSet(len(rows))
    first_trajectory: dict[str, int] = {}
    first_template: dict[str, int] = {}
    templates: list[str] = []
    candidate_ids: list[str] = []
    seen_candidate_ids: set[str] = set()

    for index, row in enumerate(rows):
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not candidate_id:
            provenance = row.get("provenance")
            if isinstance(provenance, dict):
                candidate_id = str(provenance.get("candidate_id", "")).strip()
        if not candidate_id:
            raise ValueError("split row requires a stable candidate_id")
        if candidate_id in seen_candidate_ids:
            raise ValueError(f"duplicate candidate_id in split rows: {candidate_id}")
        seen_candidate_ids.add(candidate_id)
        candidate_ids.append(candidate_id)

        trajectory = str(row.get("trajectory_id", "")) or f"row:{index}"
        template = task_template_key(str(row.get("task", "")))
        if not template:
            template = f"trajectory:{trajectory}"
        templates.append(template)

        if trajectory in first_trajectory:
            disjoint.union(index, first_trajectory[trajectory])
        else:
            first_trajectory[trajectory] = index
        if template in first_template:
            disjoint.union(index, first_template[template])
        else:
            first_template[template] = index

    components: dict[int, list[int]] = {}
    for index in range(len(rows)):
        components.setdefault(disjoint.find(index), []).append(index)

    def group_order(indices: list[int]) -> tuple[int, str]:
        stable_values = sorted(candidate_ids[index] for index in indices)
        digest = hashlib.sha256(
            f"{seed}|{'|'.join(stable_values)}".encode("utf-8")
        ).hexdigest()
        return (-len(indices), digest)

    groups = sorted(components.values(), key=group_order)
    targets = {
        "train": len(rows) * 0.70,
        "validation": len(rows) * 0.15,
        "test": len(rows) * 0.15,
    }
    assignments: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    counts = {name: 0 for name in assignments}
    split_order = ("train", "validation", "test")

    for group in groups:
        def projected_error(split: str) -> tuple[float, int]:
            projected = dict(counts)
            projected[split] += len(group)
            error = sum(abs(projected[name] - targets[name]) for name in split_order)
            return error, split_order.index(split)

        selected = min(split_order, key=projected_error)
        assignments[selected].extend(group)
        counts[selected] += len(group)

    split_rows: dict[str, tuple[dict[str, Any], ...]] = {}
    trajectory_sets: dict[str, set[str]] = {}
    template_sets: dict[str, set[str]] = {}
    for split in split_order:
        ordered_indices = sorted(
            assignments[split],
            key=lambda index: candidate_ids[index],
        )
        materialized: list[dict[str, Any]] = []
        trajectory_sets[split] = set()
        template_sets[split] = set()
        for index in ordered_indices:
            row = deepcopy(rows[index])
            row["split"] = split
            materialized.append(row)
            trajectory_sets[split].add(str(row.get("trajectory_id", "")))
            template_sets[split].add(templates[index])
        split_rows[split] = tuple(materialized)

    manifest = {
        "seed": seed,
        "target_proportions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "row_count": len(rows),
        "group_count": len(groups),
        "split_counts": {name: len(split_rows[name]) for name in split_order},
        "split_group_counts": {
            name: len({disjoint.find(index) for index in assignments[name]})
            for name in split_order
        },
        "trajectory_overlap": _overlap(trajectory_sets),
        "template_overlap": _overlap(template_sets),
    }
    if manifest["trajectory_overlap"] or manifest["template_overlap"]:
        raise AssertionError("group-aware split produced overlap")

    return SplitResult(
        train=split_rows["train"],
        validation=split_rows["validation"],
        test=split_rows["test"],
        manifest=manifest,
    )
