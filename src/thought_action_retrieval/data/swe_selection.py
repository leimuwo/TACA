#!/usr/bin/env python3
"""Select representative SWE-agent bug-fix trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


FENCED_BLOCK_RE = re.compile(r"```[^\n`]*\n?(.*?)```", re.DOTALL)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
THOUGHT_MARKER_RE = re.compile(
    r"\b(?:because|indicat(?:e|es|ed)|need|next|plan|should|suggest(?:s|ed)?|"
    r"therefore|to (?:address|debug|fix|investigate|reproduce|verify)|we (?:can|will)|"
    r"let(?:'s| us))\b",
    re.IGNORECASE,
)
BUGFIX_RE = re.compile(
    r"\b(?:bug|broken|cannot|can't|crash(?:es|ed|ing)?|doesn'?t|error|exception|"
    r"fail(?:s|ed|ing|ure)?|fix|hang(?:s|ing)?|incorrect(?:ly)?|invalid|not work(?:ing)?|"
    r"regression|traceback|unexpected|wrong)\b|"
    r"\b(?:attribute|index|key|runtime|type|value)error\b",
    re.IGNORECASE,
)
FEATURE_ONLY_RE = re.compile(
    r"^(?:add|allow|create|document|enhance|feature request|freeze|have an option|"
    r"implement|introduce|log|option to|provide|refactor|report|support|treat)\b",
    re.IGNORECASE,
)
PAIR_MODEL_QUOTAS = {"70b": 87, "8b": 25, "405b": 13}
TOTAL_MODEL_QUOTAS = {"70b": 175, "8b": 50, "405b": 25}
DEFAULT_SEED = 20260814


def is_bugfix_task(task_text: str) -> bool:
    """Return whether task wording describes a bug, failure, or incorrect behavior."""
    normalized = " ".join((task_text or "").split())
    if not normalized:
        return False
    if FEATURE_ONLY_RE.search(normalized):
        return False
    if BUGFIX_RE.search(normalized):
        return True
    return not FEATURE_ONLY_RE.search(normalized) and bool(
        re.search(r"\b(?:issue|problem|unexpected behavior)\b", normalized, re.IGNORECASE)
    )


def classify_action(action: str) -> str:
    """Classify one SWE-agent command into a debugging phase."""
    command = (action or "").strip().lower()
    if re.search(
        r"(?:^|\s)(?:pytest|tox|nosetests|go test|cargo test|npm test|yarn test|"
        r"mvn test|gradle test|rspec|rake test)(?:\s|$)|python\s+[^\n]*test",
        command,
    ):
        return "test"
    if re.search(r"(?:^|\n)(?:edit\b|create\b|apply_patch\b)|end_of_edit", command):
        return "edit"
    if re.search(
        r"(?:^|\s)(?:search_dir|search_file|find_file|open|scroll_up|scroll_down|"
        r"cat|sed|grep|rg|find|ls|head|tail)(?:\s|$)",
        command,
    ):
        return "inspect"
    if re.search(r"(?:^|\s)(?:python|node|ruby|java|bash|sh)\b", command):
        return "run"
    return "other"


def _issue_text(record: dict[str, Any]) -> str:
    trajectory = record.get("trajectory") or []
    for message in trajectory:
        if message.get("role") == "user" and (message.get("text") or "").strip():
            text = message["text"].strip()
            marker = re.search(r"\bISSUE:\s*", text, re.IGNORECASE)
            return text[marker.end() :] if marker else text
    return ""


def _has_explicit_thought(text_before_action: str) -> bool:
    prose = (text_before_action or "").strip()
    return (
        len(prose) >= 20
        and len(WORD_RE.findall(prose)) >= 4
        and bool(THOUGHT_MARKER_RE.search(prose))
    )


def analyze_record(record: dict[str, Any]) -> dict[str, Any]:
    """Compute hard-filter and quality features for one trajectory record."""
    trajectory = record.get("trajectory") or []
    ai_turns = 0
    valid_cycles = 0
    categories: set[str] = set()
    thought_lengths: list[int] = []

    for index, message in enumerate(trajectory):
        if message.get("role") != "ai":
            continue
        ai_turns += 1
        text = (message.get("text") or "").strip()
        blocks = list(FENCED_BLOCK_RE.finditer(text))
        if not blocks:
            continue
        action = blocks[-1].group(1).strip()
        thought = text[: blocks[-1].start()].strip()
        if not action or not _has_explicit_thought(thought):
            continue
        if index + 1 >= len(trajectory):
            continue
        observation = trajectory[index + 1]
        if observation.get("role") != "user" or not (observation.get("text") or "").strip():
            continue
        valid_cycles += 1
        thought_lengths.append(len(thought))
        categories.add(classify_action(action))

    first_system = trajectory[0] if trajectory else {}
    context_complete = bool(
        trajectory
        and first_system.get("role") == "system"
        and ((first_system.get("system_prompt") or first_system.get("text") or "").strip())
        and _issue_text(record)
    )
    issue_text = _issue_text(record)
    patch = (record.get("generated_patch") or "").strip()
    eval_logs = (record.get("eval_logs") or "").strip()
    complete_cycle_ratio = valid_cycles / ai_turns if ai_turns else 0.0
    patch_evidence = len(patch) >= 40 and ("diff --git" in patch or "@@" in patch)
    eval_evidence = len(eval_logs) >= 100
    hard_eligible = bool(
        context_complete
        and is_bugfix_task(_issue_title(issue_text))
        and record.get("model_name", "").startswith("swe-agent-")
        and isinstance(record.get("target"), bool)
        and ai_turns >= 10
        and valid_cycles >= 10
        and complete_cycle_ratio >= 0.8
        and "inspect" in categories
        and "edit" in categories
        and patch_evidence
        and eval_evidence
    )

    return {
        "hard_eligible": hard_eligible,
        "ai_turns": ai_turns,
        "valid_cycles": valid_cycles,
        "complete_cycle_ratio": round(complete_cycle_ratio, 6),
        "action_categories": sorted(categories),
        "mean_thought_chars": round(
            sum(thought_lengths) / len(thought_lengths), 2
        ) if thought_lengths else 0.0,
        "context_complete": context_complete,
        "patch_evidence": patch_evidence,
        "eval_evidence": eval_evidence,
        "issue_text": issue_text,
    }


def quality_score(analysis: dict[str, Any], record: dict[str, Any]) -> float:
    """Return the approved 100-point trajectory quality score."""
    valid_cycles = analysis["valid_cycles"]
    cycle_score = 15.0 + 15.0 * min(max(valid_cycles - 10, 0) / 30, 1.0)
    thought_score = (
        15.0 * analysis["complete_cycle_ratio"]
        + 10.0 * min(analysis["mean_thought_chars"] / 350, 1.0)
    )
    categories = set(analysis["action_categories"])
    richness_score = (
        5.0 * ("inspect" in categories)
        + 5.0 * ("edit" in categories)
        + 6.0 * ("test" in categories)
        + 2.0 * ("run" in categories)
        + 2.0 * ("other" in categories)
    )
    patch = record.get("generated_patch") or ""
    eval_logs = record.get("eval_logs") or ""
    evidence_score = (
        5.0 * analysis["patch_evidence"]
        + 5.0 * analysis["eval_evidence"]
        + 5.0 * ("test" in categories or len(eval_logs) >= 1000)
    )
    length_score = 10.0 if 15 <= valid_cycles <= 50 else 8.0
    return round(cycle_score + thought_score + richness_score + evidence_score + length_score, 3)


def _length_band(valid_cycles: int) -> str:
    if valid_cycles < 20:
        return "10-19"
    if valid_cycles < 40:
        return "20-39"
    return "40+"


def _issue_title(issue_text: str) -> str:
    for line in issue_text.splitlines():
        clean = line.strip().strip("#").strip()
        if clean:
            return clean[:500]
    return ""


def _stable_jitter(uid: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{uid}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def fit_pair_model_quotas(
    candidates: list[dict[str, Any]],
    *,
    preferred_quotas: dict[str, int],
    total_pairs: int,
) -> dict[str, int]:
    """Fit preferred pair quotas to per-model paired-repository capacity."""
    targets_by_instance: dict[tuple[str, str, str], set[bool]] = defaultdict(set)
    for candidate in candidates:
        key = (
            candidate["model_short"],
            candidate["repository"],
            candidate["instance_id"],
        )
        targets_by_instance[key].add(candidate["target"])
    paired_instances: dict[str, set[str]] = defaultdict(set)
    for (model, repository, _instance_id), targets in targets_by_instance.items():
        if targets == {True, False}:
            paired_instances[model].add(_instance_id)
    capacities = {
        model: len(paired_instances.get(model, set()))
        for model in preferred_quotas
    }
    quotas = {
        model: min(preferred, capacities[model])
        for model, preferred in preferred_quotas.items()
    }
    remaining = total_pairs - sum(quotas.values())
    while remaining > 0:
        viable = [
            model for model in quotas if quotas[model] < capacities[model]
        ]
        if not viable:
            raise ValueError(
                f"only {sum(quotas.values())} repository-diverse pairs available; "
                f"requested {total_pairs}"
            )
        model = max(
            viable,
            key=lambda name: (
                capacities[name] - quotas[name],
                preferred_quotas[name],
                name,
            ),
        )
        quotas[model] += 1
        remaining -= 1
    return quotas


def select_representative_candidates(
    candidates: list[dict[str, Any]],
    *,
    pair_model_quotas: dict[str, int],
    total_model_quotas: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    """Select controlled pairs followed by model- and length-diverse extras."""
    by_key: dict[tuple[str, bool, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_key[
            (candidate["instance_id"], candidate["target"], candidate["model_short"])
        ].append(candidate)
    for values in by_key.values():
        values.sort(
            key=lambda item: (item["quality_score"], _stable_jitter(item["uid"], seed)),
            reverse=True,
        )

    selected: list[dict[str, Any]] = []
    used_model_instances: set[tuple[str, str]] = set()
    used_uids: set[str] = set()
    repository_counts: Counter[str] = Counter()
    band_counts: dict[bool, Counter[str]] = {True: Counter(), False: Counter()}
    pair_number = 0

    model_order = sorted(
        pair_model_quotas,
        key=lambda model: (pair_model_quotas[model], model),
    )
    for model in model_order:
        options: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        instance_ids = {key[0] for key in by_key if key[2] == model}
        for instance_id in instance_ids:
            successes = by_key.get((instance_id, True, model), [])
            failures = by_key.get((instance_id, False, model), [])
            if not successes or not failures:
                continue
            success, failure = successes[0], failures[0]
            diversity = 2.0 * (success["length_band"] != failure["length_band"])
            score = (
                success["quality_score"]
                + failure["quality_score"]
                + diversity
                + _stable_jitter(instance_id, seed)
            )
            options.append((score, success, failure))
        options.sort(key=lambda item: item[0], reverse=True)

        picked = 0
        while picked < pair_model_quotas[model]:
            unused_options = [
                option
                for option in options
                if (model, option[1]["instance_id"]) not in used_model_instances
            ]
            preferred = [
                option
                for option in unused_options
                if repository_counts[option[1]["repository"]] <= 2
            ]
            viable = preferred or unused_options
            if not viable:
                raise ValueError(f"insufficient pair candidates for model {model}")
            viable.sort(
                key=lambda option: (
                    option[0]
                    + 3.0 / (1 + band_counts[True][option[1]["length_band"]])
                    + 3.0 / (1 + band_counts[False][option[2]["length_band"]])
                    + 2.0 / (1 + repository_counts[option[1]["repository"]])
                ),
                reverse=True,
            )
            _, success, failure = viable[0]
            pair_number += 1
            pair_id = f"P{pair_number:02d}"
            for candidate in (success, failure):
                item = dict(candidate)
                item.update(selection_group="paired", pair_id=pair_id)
                selected.append(item)
                used_uids.add(item["uid"])
                band_counts[item["target"]][item["length_band"]] += 1
            used_model_instances.add((model, success["instance_id"]))
            repository_counts[success["repository"]] += 2
            picked += 1

    total_per_target = sum(total_model_quotas.values())
    desired_bands = {
        "10-19": round(total_per_target * 0.4),
        "20-39": round(total_per_target * 0.4),
        "40+": total_per_target - 2 * round(total_per_target * 0.4),
    }
    for target in (True, False):
        for model in model_order:
            needed = total_model_quotas[model] - pair_model_quotas[model]
            for _ in range(needed):
                matching = [
                    candidate
                    for candidate in candidates
                    if candidate["target"] is target
                    and candidate["model_short"] == model
                    and candidate["uid"] not in used_uids
                ]
                preferred = [
                    candidate
                    for candidate in matching
                    if (model, candidate["instance_id"]) not in used_model_instances
                    and repository_counts[candidate["repository"]] < 4
                ]
                unused_instances = [
                    candidate
                    for candidate in matching
                    if (model, candidate["instance_id"]) not in used_model_instances
                ]
                viable = preferred or unused_instances or matching
                if not viable:
                    raise ValueError(
                        f"insufficient diversity candidates for target={target}, model={model}"
                    )
                def rank(candidate: dict[str, Any]) -> float:
                    deficit = max(
                        desired_bands[candidate["length_band"]]
                        - band_counts[target][candidate["length_band"]],
                        0,
                    )
                    return (
                        candidate["quality_score"]
                        + min(deficit, 8) * 2.0
                        + 2.0 / (1 + repository_counts[candidate["repository"]])
                        + _stable_jitter(candidate["uid"], seed)
                    )
                chosen = max(viable, key=rank)
                item = dict(chosen)
                item.update(selection_group="diversity", pair_id="")
                selected.append(item)
                used_uids.add(item["uid"])
                used_model_instances.add((model, item["instance_id"]))
                repository_counts[item["repository"]] += 1
                band_counts[target][item["length_band"]] += 1

    return selected


def _candidate_from_record(
    record: dict[str, Any], source_file: str, source_row: int
) -> dict[str, Any] | None:
    analysis = analyze_record(record)
    if not analysis["hard_eligible"]:
        return None
    instance_id = record["instance_id"]
    model_name = record["model_name"]
    return {
        "uid": f"{source_file}:{source_row}",
        "source_file": source_file,
        "source_row": source_row,
        "instance_id": instance_id,
        "repository": instance_id.rsplit("-", 1)[0],
        "model_name": model_name,
        "model_short": model_name.rsplit("-", 1)[-1],
        "target": record["target"],
        "exit_status": record["exit_status"],
        "message_count": len(record["trajectory"]),
        "ai_turns": analysis["ai_turns"],
        "valid_cycles": analysis["valid_cycles"],
        "complete_cycle_ratio": analysis["complete_cycle_ratio"],
        "mean_thought_chars": analysis["mean_thought_chars"],
        "action_categories": analysis["action_categories"],
        "length_band": _length_band(analysis["valid_cycles"]),
        "patch_chars": len(record.get("generated_patch") or ""),
        "eval_log_chars": len(record.get("eval_logs") or ""),
        "issue_title": _issue_title(analysis["issue_text"]),
        "quality_score": quality_score(analysis, record),
    }


def scan_jsonl_candidates(source_dir: Path) -> list[dict[str, Any]]:
    """Scan JSONL shards once and retain metadata for hard-eligible records."""
    candidates: list[dict[str, Any]] = []
    jsonl_paths = sorted(source_dir.glob("*.jsonl"))
    if not jsonl_paths:
        raise ValueError(f"no JSONL shards found in {source_dir}")
    for jsonl_path in jsonl_paths:
        row_count = 0
        with jsonl_path.open("rb") as handle:
            while True:
                source_offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON in {jsonl_path.name} row {row_count}"
                    ) from error
                candidate = _candidate_from_record(
                    record, jsonl_path.name, row_count
                )
                if candidate:
                    candidate["source_offset"] = source_offset
                    candidates.append(candidate)
                row_count += 1
        print(
            f"scanned {jsonl_path.name}: rows={row_count}, candidates={len(candidates)}",
            flush=True,
        )
    return candidates


def load_selected_jsonl_records(
    source_dir: Path, selected: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        by_file[item["source_file"]].append(item)
    for source_file, items in by_file.items():
        with (source_dir / source_file).open("rb") as handle:
            for item in items:
                if "source_offset" in item:
                    handle.seek(item["source_offset"])
                    line = handle.readline()
                else:
                    handle.seek(0)
                    line = b""
                    for source_row, candidate_line in enumerate(handle):
                        if source_row == item["source_row"]:
                            line = candidate_line
                            break
                if not line:
                    raise ValueError(
                        f"missing selected row {item['source_row']} in {source_file}"
                    )
                records[item["uid"]] = json.loads(line)
    if len(records) != len(selected):
        raise ValueError(f"loaded {len(records)} records for {len(selected)} selections")
    return records


def record_to_case(
    record: dict[str, Any], selection: dict[str, Any], case_id: str
) -> dict[str, Any]:
    """Convert one raw SWE-agent record to the Thought/Action CASE structure."""
    trajectory = record.get("trajectory") or []
    system_prompt = ""
    task = ""
    for message in trajectory:
        if message.get("role") == "system" and not system_prompt:
            system_prompt = (
                message.get("system_prompt") or message.get("text") or ""
            ).strip()
        elif message.get("role") == "user" and not task:
            task = (message.get("text") or "").strip()
        if system_prompt and task:
            break

    steps: list[dict[str, Any]] = []
    for index, message in enumerate(trajectory):
        if message.get("role") != "ai" or index + 1 >= len(trajectory):
            continue
        text = (message.get("text") or "").strip()
        blocks = list(FENCED_BLOCK_RE.finditer(text))
        if not blocks:
            continue
        action = blocks[-1].group(1).strip()
        thought = text[: blocks[-1].start()].strip()
        observation_message = trajectory[index + 1]
        observation = (observation_message.get("text") or "").strip()
        if (
            not action
            or not _has_explicit_thought(thought)
            or observation_message.get("role") != "user"
            or not observation
        ):
            continue
        steps.append(
            {
                "step": len(steps) + 1,
                "thought": thought,
                "action": action,
                "observation": observation,
            }
        )

    instance_id = record["instance_id"]
    return {
        "case_id": case_id,
        "selection_group": selection["selection_group"],
        "pair_id": selection.get("pair_id", ""),
        "instance_id": instance_id,
        "repository": selection.get("repository", instance_id.rsplit("-", 1)[0]),
        "model_name": record["model_name"],
        "target": record["target"],
        "exit_status": record.get("exit_status", ""),
        "system_prompt": system_prompt,
        "task": task,
        "steps": steps,
        "generated_patch": record.get("generated_patch", ""),
        "eval_logs": record.get("eval_logs", ""),
    }


def write_case_outputs(
    output_dir: Path,
    selected: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    seed: int,
    eligible_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"selected_{len(selected)}.jsonl"
    manifest_path = output_dir / "manifest.csv"
    summary_path = output_dir / "selection_summary.json"

    ordered = sorted(
        selected,
        key=lambda item: (
            0 if item["selection_group"] == "paired" else 1,
            item["pair_id"],
            0 if item["target"] else 1,
            item["model_short"],
            item["uid"],
        ),
    )
    for index, item in enumerate(ordered, 1):
        item["selection_id"] = f"CASE-{index:04d}"

    cases = [
        record_to_case(records[item["uid"]], item, item["selection_id"])
        for item in ordered
    ]
    expected_case_names = {f"{case['case_id']}.json" for case in cases}
    stale_case_names = {
        path.name for path in output_dir.glob("CASE-*.json")
    } - expected_case_names
    if stale_case_names:
        raise ValueError(
            f"output directory contains stale CASE files: {sorted(stale_case_names)[:3]}"
        )
    for case in cases:
        case_path = output_dir / f"{case['case_id']}.json"
        case_tmp = case_path.with_suffix(".json.tmp")
        case_tmp.write_text(
            json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        case_tmp.replace(case_path)

    jsonl_tmp = jsonl_path.with_suffix(".jsonl.tmp")
    with jsonl_tmp.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    jsonl_tmp.replace(jsonl_path)

    fields = [
        "selection_id", "selection_group", "pair_id", "source_file", "source_row",
        "instance_id", "repository", "model_name", "target", "exit_status",
        "issue_title", "message_count", "ai_turns", "valid_cycles", "length_band",
        "complete_cycle_ratio", "mean_thought_chars", "action_categories", "patch_chars",
        "eval_log_chars", "quality_score", "selection_reason",
    ]
    manifest_tmp = manifest_path.with_suffix(".csv.tmp")
    with manifest_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in ordered:
            row = {key: item.get(key, "") for key in fields}
            row["action_categories"] = "|".join(item["action_categories"])
            row["selection_reason"] = (
                "matched success/failure trajectory for the same task and model"
                if item["selection_group"] == "paired"
                else "high-quality stratified sample adding task/model/length diversity"
            )
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    manifest_tmp.replace(manifest_path)

    summary = {
        "seed": seed,
        "eligible_candidate_count": eligible_count,
        "selected_count": len(ordered),
        "hard_filters": {
            "real_swe_agent_trajectory": True,
            "bugfix_task_language": True,
            "minimum_valid_thought_action_observation_cycles": 10,
            "minimum_complete_cycle_ratio": 0.8,
            "requires_inspection_and_edit_actions": True,
            "requires_nonempty_patch_and_eval_evidence": True,
            "preserves_complete_original_record": True,
        },
        "target_counts": dict(Counter(str(item["target"]).lower() for item in ordered)),
        "selection_group_counts": dict(Counter(item["selection_group"] for item in ordered)),
        "model_counts": dict(Counter(item["model_name"] for item in ordered)),
        "model_counts_by_target": {
            str(target).lower(): dict(Counter(
                item["model_name"] for item in ordered if item["target"] is target
            )) for target in (True, False)
        },
        "length_band_counts": dict(Counter(item["length_band"] for item in ordered)),
        "length_band_counts_by_target": {
            str(target).lower(): dict(Counter(
                item["length_band"] for item in ordered if item["target"] is target
            )) for target in (True, False)
        },
        "unique_instance_ids": len({item["instance_id"] for item in ordered}),
        "unique_repositories": len({item["repository"] for item in ordered}),
        "quality_score": {
            "min": min(item["quality_score"] for item in ordered),
            "max": max(item["quality_score"] for item in ordered),
            "mean": round(sum(item["quality_score"] for item in ordered) / len(ordered), 3),
        },
    }
    summary_tmp = summary_path.with_suffix(".json.tmp")
    summary_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_tmp.replace(summary_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    candidates = scan_jsonl_candidates(args.source)
    pair_model_quotas = fit_pair_model_quotas(
        candidates,
        preferred_quotas=PAIR_MODEL_QUOTAS,
        total_pairs=sum(PAIR_MODEL_QUOTAS.values()),
    )
    print(f"pair model quotas: {pair_model_quotas}", flush=True)
    selected = select_representative_candidates(
        candidates,
        pair_model_quotas=pair_model_quotas,
        total_model_quotas=TOTAL_MODEL_QUOTAS,
        seed=args.seed,
    )
    records = load_selected_jsonl_records(args.source, selected)
    write_case_outputs(args.output, selected, records, args.seed, len(candidates))
    print(f"wrote {len(selected)} cases to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
