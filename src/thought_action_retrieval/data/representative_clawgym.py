"""Deterministic representative selection from converted ClawGym trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence


TASK_FAMILIES = (
    "csv",
    "json",
    "web",
    "download",
    "report",
    "social",
    "pdf",
    "browser",
    "image",
    "email",
    "file",
)


@dataclass(frozen=True)
class TrajectoryFeature:
    path: Path
    trajectory_id: str
    sha256: str
    action_messages: int
    thought_action_steps: int
    action_count: int
    multi_action_steps: int
    coverage: float
    coverage_bucket: str
    length_bucket: str
    tools: tuple[str, ...]
    task_families: tuple[str, ...]

    @property
    def stratum(self) -> str:
        return f"{self.coverage_bucket}__{self.length_bucket}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observable_thought(message: dict) -> str:
    value = message.get("assistant_thought")
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"].strip()
    value = message.get("reasoning_content")
    return value.strip() if isinstance(value, str) else ""


def _tool_name(call: dict) -> str:
    function = call.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "").strip()
    return str(function or "").strip()


def _coverage_bucket(coverage: float) -> str:
    if coverage >= 0.999999:
        return "complete"
    if coverage >= 0.6:
        return "partial_high"
    return "partial_low"


def analyze_trajectory(path: Path) -> TrajectoryFeature:
    data = json.loads(path.read_text(encoding="utf-8"))
    messages = data.get("messages") or []
    action_messages = [
        message
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and isinstance(message.get("tool_calls"), list)
        and message.get("tool_calls")
    ]
    thought_steps = sum(bool(_observable_thought(message)) for message in action_messages)
    action_count = sum(len(message.get("tool_calls") or []) for message in action_messages)
    multi_action_steps = sum(len(message.get("tool_calls") or []) > 1 for message in action_messages)
    tools = sorted(
        {
            name
            for message in action_messages
            for call in message.get("tool_calls") or []
            if (name := _tool_name(call))
        }
    )
    coverage = thought_steps / len(action_messages) if action_messages else 0.0
    task = str(data.get("task") or "").lower()
    families = tuple(name for name in TASK_FAMILIES if name in task) or ("other",)
    return TrajectoryFeature(
        path=path.resolve(),
        trajectory_id=str(data.get("id") or path.stem),
        sha256=_sha256(path),
        action_messages=len(action_messages),
        thought_action_steps=thought_steps,
        action_count=action_count,
        multi_action_steps=multi_action_steps,
        coverage=coverage,
        coverage_bucket=_coverage_bucket(coverage),
        length_bucket="unassigned",
        tools=tuple(tools),
        task_families=families,
    )


def _length_boundaries(features: Sequence[TrajectoryFeature]) -> tuple[int, int, int]:
    values = sorted(item.thought_action_steps for item in features)
    if not values:
        raise ValueError("cannot compute length buckets from an empty pool")
    return tuple(values[min(len(values) - 1, math.floor((len(values) - 1) * q))] for q in (0.25, 0.5, 0.75))  # type: ignore[return-value]


def _assign_length_buckets(features: Sequence[TrajectoryFeature]) -> list[TrajectoryFeature]:
    q1, q2, q3 = _length_boundaries(features)
    assigned = []
    for item in features:
        value = item.thought_action_steps
        bucket = "q1_short" if value <= q1 else "q2_medium" if value <= q2 else "q3_long" if value <= q3 else "q4_very_long"
        assigned.append(replace(item, length_bucket=bucket))
    return assigned


def allocate_quotas(counts: dict[str, int], sample_count: int) -> dict[str, int]:
    if sample_count < 1 or sample_count > sum(counts.values()):
        raise ValueError("sample count must be positive and no larger than the pool")
    nonempty = {key: value for key, value in counts.items() if value > 0}
    quotas = {key: 0 for key in nonempty}
    if sample_count >= len(nonempty):
        quotas = {key: 1 for key in nonempty}
    remaining = sample_count - sum(quotas.values())
    capacity = {key: nonempty[key] - quotas[key] for key in nonempty}
    total_capacity = sum(capacity.values())
    if remaining and total_capacity:
        ideals = {key: remaining * capacity[key] / total_capacity for key in nonempty}
        for key in nonempty:
            quotas[key] += min(capacity[key], math.floor(ideals[key]))
        left = sample_count - sum(quotas.values())
        order = sorted(nonempty, key=lambda key: (-(ideals[key] - math.floor(ideals[key])), key))
        for key in order:
            if not left:
                break
            if quotas[key] < nonempty[key]:
                quotas[key] += 1
                left -= 1
    if sum(quotas.values()) != sample_count:
        raise AssertionError("quota allocation did not reconcile")
    return quotas


def _labels(item: TrajectoryFeature) -> tuple[str, ...]:
    multi = "multi0" if item.multi_action_steps == 0 else "multi1_2" if item.multi_action_steps <= 2 else "multi3plus"
    return tuple([*(f"tool:{name}" for name in item.tools), *(f"task:{name}" for name in item.task_families), multi])


def _stable_tie(item: TrajectoryFeature, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{item.trajectory_id}|{item.path.name}".encode()).hexdigest()


def select_representative(
    paths: Iterable[Path], *, count: int, seed: int, excluded_names: set[str]
) -> list[TrajectoryFeature]:
    eligible = [analyze_trajectory(path) for path in sorted(paths, key=lambda value: value.name) if path.name not in excluded_names]
    if len(eligible) < count:
        raise ValueError(f"need {count} eligible trajectories, found {len(eligible)}")
    eligible = _assign_length_buckets(eligible)
    by_stratum: dict[str, list[TrajectoryFeature]] = defaultdict(list)
    for item in eligible:
        by_stratum[item.stratum].append(item)
    quotas = allocate_quotas({key: len(value) for key, value in by_stratum.items()}, count)
    selected: list[TrajectoryFeature] = []
    label_counts: Counter[str] = Counter()
    for stratum in sorted(by_stratum):
        pool = list(by_stratum[stratum])
        for _ in range(quotas[stratum]):
            chosen = min(
                pool,
                key=lambda item: (
                    -sum(1.0 / (1 + label_counts[label]) for label in _labels(item)),
                    _stable_tie(item, seed),
                ),
            )
            pool.remove(chosen)
            selected.append(chosen)
            label_counts.update(_labels(chosen))
    return sorted(selected, key=lambda item: item.path.name)


def publish_selection(
    paths: Iterable[Path], output_dir: Path, *, count: int, seed: int, excluded_names: set[str]
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    selected = select_representative(paths, count=count, seed=seed, excluded_names=excluded_names)
    source_counts = Counter(item.stratum for item in _assign_length_buckets([analyze_trajectory(path) for path in paths if path.name not in excluded_names]))
    selected_counts = Counter(item.stratum for item in selected)
    records = [
        {
            "trajectory_id": item.trajectory_id,
            "filename": item.path.name,
            "sha256": item.sha256,
            "thought_action_steps": item.thought_action_steps,
            "action_count": item.action_count,
            "multi_action_steps": item.multi_action_steps,
            "coverage": item.coverage,
            "coverage_bucket": item.coverage_bucket,
            "length_bucket": item.length_bucket,
            "stratum": item.stratum,
            "tools": list(item.tools),
            "task_families": list(item.task_families),
        }
        for item in selected
    ]
    manifest = {
        "schema_version": "clawgym_representative_selection_v1",
        "seed": seed,
        "requested_count": count,
        "selected_count": len(selected),
        "excluded_files": sorted(excluded_names),
        "source_stratum_counts": dict(sorted(source_counts.items())),
        "selected_stratum_counts": dict(sorted(selected_counts.items())),
        "records": records,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for item in selected:
            shutil.copy2(item.path, staged / item.path.name)
        (staged / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staged.replace(output_dir)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--exclude-file", action="append", default=[])
    args = parser.parse_args(argv)
    paths = [path for path in args.input_dir.resolve().glob("*.json") if path.name != "manifest.json"]
    manifest = publish_selection(paths, args.output_dir.resolve(), count=args.count, seed=args.seed, excluded_names={Path(name).name for name in args.exclude_file})
    print(json.dumps({"selected_count": manifest["selected_count"], "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False))
    return 0
