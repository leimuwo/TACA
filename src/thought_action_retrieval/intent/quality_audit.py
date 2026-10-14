"""Heuristic quality screening for temporal Thought-to-Intent annotations."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


PURE_LANGUAGE = re.compile(r"\b(summar(?:y|ize)|report back|respond|answer the user|explain(?: the)? result)\b", re.I)
SPECULATION = re.compile(r"\b(maybe|perhaps|possibly|could|might|or perhaps)\b", re.I)
FUTURE_RESTATEMENT = re.compile(r"\b(complete the task|continue the analysis|process the remaining|finish the task)\b", re.I)
COMPOUND = re.compile(r"\b(search|fetch|read|write|create|parse|extract|download|run|inspect|open|save)\b.*\b(and|then|followed by)\b.*\b(search|fetch|read|write|create|parse|extract|download|run|inspect|open|save)\b", re.I)


def _example(trajectory_id: str, step: dict, intent: dict) -> dict:
    return {
        "trajectory_id": trajectory_id,
        "thought_step": step.get("thought_step"),
        "intent_id": intent.get("intent_id"),
        "intent_text": intent.get("intent_text"),
        "temporal_scope": intent.get("temporal_scope"),
        "source_quote": intent.get("source_quote"),
        "thought": step.get("thought"),
        "actual_actions": step.get("actual_actions") or [],
    }


def audit_records(records: Iterable[dict], *, example_limit: int = 10) -> dict:
    totals = Counter()
    scopes = Counter()
    cardinalities = Counter()
    screening = {
        name: {"count": 0, "examples": []}
        for name in ("pure_language_output", "current_speculation", "future_task_restatement", "compound_granularity")
    }
    per_trajectory = []
    for record in records:
        local = Counter()
        trajectory_id = str(record.get("trajectory_id") or "")
        for step in record.get("steps") or []:
            totals["steps"] += 1
            local["steps"] += 1
            actions = step.get("actual_actions") or []
            intents = ((step.get("intent_result") or {}).get("execution_intents") or [])
            current = [intent for intent in intents if intent.get("temporal_scope") == "current"]
            totals["actions"] += len(actions)
            totals["intents"] += len(intents)
            totals["current_intents"] += len(current)
            local["actions"] += len(actions)
            local["intents"] += len(intents)
            local["current_intents"] += len(current)
            cardinalities[f"{len(current)}_current__{len(actions)}_action"] += 1
            for intent in intents:
                scope = str(intent.get("temporal_scope") or "missing")
                scopes[scope] += 1
                text = str(intent.get("intent_text") or "")
                quote = str(intent.get("source_quote") or "")
                thought = str(step.get("thought") or "")
                checks = {
                    "pure_language_output": bool(PURE_LANGUAGE.search(text)),
                    "current_speculation": scope == "current" and bool(SPECULATION.search(quote or thought)),
                    "future_task_restatement": scope == "future" and bool(FUTURE_RESTATEMENT.search(text)),
                    "compound_granularity": bool(COMPOUND.search(text)),
                }
                for name, matched in checks.items():
                    if not matched:
                        continue
                    screening[name]["count"] += 1
                    if len(screening[name]["examples"]) < example_limit:
                        screening[name]["examples"].append(_example(trajectory_id, step, intent))
        per_trajectory.append({"trajectory_id": trajectory_id, **dict(local)})
    return {
        "schema_version": "temporal_intent_quality_audit_v1",
        "label": "heuristic_not_gold",
        "totals": dict(totals),
        "temporal_scope_counts": dict(sorted(scopes.items())),
        "current_action_cardinalities": dict(sorted(cardinalities.items())),
        "screening": screening,
        "per_trajectory": per_trajectory,
    }


def _markdown(report: dict) -> str:
    lines = ["# Temporal Intent Quality Audit", "", "Screening heuristics only; findings are not Gold labels.", "", "## Totals", ""]
    for key, value in report["totals"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Temporal scopes", ""])
    for key, value in report["temporal_scope_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Current Intent / Action cardinalities", ""])
    for key, value in report["current_action_cardinalities"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Screening candidates", ""])
    for name, result in report["screening"].items():
        lines.append(f"### {name}: {result['count']}")
        lines.append("")
        for example in result["examples"]:
            lines.append(f"- `{example['trajectory_id']}` step {example['thought_step']} `{example['intent_id']}`: {example['intent_text']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def audit_directory(input_dir: Path, *, example_limit: int = 10) -> dict:
    paths = sorted(path for path in input_dir.glob("*.json") if path.name not in {"manifest.json", "quality_report.json"})
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    report = audit_records(records, example_limit=example_limit)
    manifest_path = input_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report["generation_manifest"] = {
            "step_count": manifest.get("step_count"),
            "step_status_counts": manifest.get("step_status_counts"),
            "execution_intent_count": manifest.get("execution_intent_count"),
            "temporal_scope_counts": manifest.get("temporal_scope_counts"),
        }
        if manifest.get("step_count") != report["totals"].get("steps") or manifest.get("execution_intent_count") != report["totals"].get("intents"):
            raise ValueError("quality audit totals do not reconcile with generation manifest")
    (input_dir / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (input_dir / "quality_report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--example-limit", type=int, default=10)
    args = parser.parse_args(argv)
    report = audit_directory(args.input_dir.resolve(), example_limit=args.example_limit)
    print(json.dumps({"totals": report["totals"], "screening_counts": {key: value["count"] for key, value in report["screening"].items()}}, ensure_ascii=False))
    return 0
