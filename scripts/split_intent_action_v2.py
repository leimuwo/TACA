#!/usr/bin/env python3
"""Create a deterministic trajectory and template-cluster parent split."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20260928
THRESHOLD = 0.90
PERMUTATIONS = 128
SPLITS = ("train", "dev", "test")
RATIOS = {"train": 0.70, "dev": 0.15, "test": 0.15}


def normalized_task(task: str) -> str:
    task = task.lower()
    task = re.sub(r"https?://\S+", " URL ", task)
    task = re.sub(r"(?:[a-z]:)?[/\\][\w./\\-]+", " PATH ", task)
    task = re.sub(r"\b\d+(?:[.,]\d+)*\b", " NUM ", task)
    return " ".join(re.findall(r"[a-z_]+", task))


def shingles(text: str) -> set[str]:
    tokens = text.split()
    return {" ".join(tokens[i:i + 3]) for i in range(max(1, len(tokens) - 2))} or {""}


def minhash_signature(items: set[str]) -> tuple[int, ...]:
    values = [int.from_bytes(hashlib.blake2b(item.encode(), digest_size=8).digest(), "big") for item in items]
    prime = (1 << 61) - 1
    return tuple(min(((1103515245 + i * 2654435761) * value + 12345 + i * 97531) % prime for value in values) for i in range(PERMUTATIONS))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b)


def find(parent: dict[str, str], value: str) -> str:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--matches", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    trajectories = {}
    for path in sorted(args.trajectories.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        trajectory_id = record["trajectory_id"]
        if trajectory_id in trajectories:
            raise ValueError(f"duplicate trajectory: {trajectory_id}")
        trajectories[trajectory_id] = record["task"]
    if not trajectories:
        raise ValueError("no trajectories")
    ids = sorted(trajectories)
    sets = {trajectory_id: shingles(normalized_task(trajectories[trajectory_id])) for trajectory_id in ids}
    signatures = {trajectory_id: minhash_signature(sets[trajectory_id]) for trajectory_id in ids}
    parent = {trajectory_id: trajectory_id for trajectory_id in ids}
    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    for trajectory_id in ids:
        signature = signatures[trajectory_id]
        for band in range(32):
            buckets[(band, signature[band * 4:band * 4 + 4])].append(trajectory_id)
    candidates = set()
    for group in buckets.values():
        for index, first in enumerate(group):
            for second in group[index + 1:]:
                candidates.add((first, second))
    for first, second in sorted(candidates):
        if jaccard(sets[first], sets[second]) >= THRESHOLD:
            root_a, root_b = find(parent, first), find(parent, second)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)
    clusters: dict[str, list[str]] = defaultdict(list)
    for trajectory_id in ids:
        clusters[find(parent, trajectory_id)].append(trajectory_id)
    cluster_ids = {root: f"TC-{index:04d}" for index, root in enumerate(sorted(clusters), 1)}
    target = {split: len(ids) * ratio for split, ratio in RATIOS.items()}
    counts = Counter()
    assignment = {}
    roots = sorted(clusters, key=lambda root: (-len(clusters[root]), hashlib.sha256(f"{SEED}:{root}".encode()).hexdigest()))
    for root in roots:
        group = clusters[root]
        split = min(SPLITS, key=lambda name: (counts[name] + len(group) - target[name]) ** 2 - (counts[name] - target[name]) ** 2)
        for trajectory_id in group:
            assignment[trajectory_id] = {"trajectory_id": trajectory_id, "template_cluster_id": cluster_ids[root], "split": split}
        counts[split] += len(group)
    labels = defaultdict(Counter)
    action_count = 0
    for line in args.matches.open(encoding="utf-8"):
        row = json.loads(line)
        trajectory_id = row["trajectory_id"]
        if trajectory_id not in assignment:
            raise ValueError(f"match trajectory not in parent split: {trajectory_id}")
        labels[assignment[trajectory_id]["split"]][row["label"]] += 1
        action_count += 1
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "split_assignment.jsonl").open("w", encoding="utf-8") as handle:
        for trajectory_id in ids:
            handle.write(json.dumps(assignment[trajectory_id], ensure_ascii=False, sort_keys=True) + "\n")
    cluster_manifest = {"cluster_algorithm": "normalized-task-minhash-v1", "similarity_threshold": THRESHOLD, "minhash_permutations": PERMUTATIONS, "records": [{"trajectory_id": trajectory_id, "template_cluster_id": assignment[trajectory_id]["template_cluster_id"]} for trajectory_id in ids], "cluster_sizes": dict(Counter(assignment[trajectory_id]["template_cluster_id"] for trajectory_id in ids))}
    (args.output_dir / "template_cluster_manifest.json").write_text(json.dumps(cluster_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    input_hash = hashlib.sha256(args.matches.read_bytes()).hexdigest()
    manifest = {"split_seed": SEED, "split_algorithm_version": "group-stratified-v1", "input_manifest_sha256": input_hash, "trajectory_count": len(ids), "action_count": action_count, "train_count": counts["train"], "dev_count": counts["dev"], "test_count": counts["test"], "cluster_count": len(clusters), "largest_cluster_count": max(map(len, clusters.values())), "label_counts_by_split": {split: dict(labels[split]) for split in SPLITS}, "supervision_tier": "model_preliminary", "human_gold_review": "skipped_by_user"}
    (args.output_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
