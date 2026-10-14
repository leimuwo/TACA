#!/usr/bin/env python3
"""Build conservative split-local diagnostic counterexamples from exact rows."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thought_action_retrieval.matching.action_atomicization import audit_action

CAP = 30


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def select(rows):
    seen = set()
    for row in sorted(rows, key=lambda r: hashlib.sha256(r["action_id"].encode()).hexdigest()):
        if row["trajectory_id"] not in seen:
            seen.add(row["trajectory_id"])
            yield row


def sample(source, modified, kind, label, *, donor=None, trap_type=None):
    structure = audit_action(modified, trajectory_id=source["trajectory_id"], thought_step=source["thought_step"], action_index=0) if modified is not None else None
    result = {
        "sample_id": digest([source["action_id"], kind, modified])[:24],
        "source_action_id": source["action_id"],
        "source_action_sha256": source["raw_action_sha256"],
        "trajectory_id": source["trajectory_id"],
        "thought_step": source["thought_step"],
        "template_cluster_id": source["template_cluster_id"],
        "split": source["split"],
        "evaluation_view": "adversarial" if trap_type else "perturbed",
        "perturbation_type": kind,
        "trap_type": trap_type,
        "selected_intent_ids": source["selected_intent_ids"],
        "original_action": source["raw_action"],
        "action": modified,
        "raw_action_sha256": structure["raw_action_sha256"] if structure is not None else None,
        "action_structure": {key: structure[key] for key in ("atomicity", "atomic_fragments", "action_has_extras", "requires_manual_semantic_review", "normalizer_version", "structure_analysis_sha256")} if structure is not None else None,
        "target_label": label,
        "changed_dimensions": ["action_presence"] if kind == "action_deletion" else ["operation_family"] if kind == "type_replacement" else ["target_url"],
        "donor_action_id": donor["action_id"] if donor else None,
        "supervision_tier": "model_diagnostic",
        "verification_status": "unreviewed_generated",
        "review_required": True,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    actions = {row["action_id"]: row for row in map(json.loads, args.actions.open(encoding="utf-8"))}
    matches = list(map(json.loads, args.matches.open(encoding="utf-8")))
    exact = [row for row in matches if row["label"] == "exact_match" and len(row.get("selected_intent_ids", [])) == 1]
    for row in exact:
        row["raw_action"] = actions[row["action_id"]]["action"]
    urls_by_step = defaultdict(list)
    for row in exact:
        action = row["raw_action"]
        if action.get("function") == "web_fetch" and isinstance(action.get("args", {}).get("url"), str):
            urls_by_step[(row["trajectory_id"], row["thought_step"])].append(row)
    output = []
    counts = Counter()
    for split in ("train", "dev", "test"):
        group = [row for row in exact if row["split"] == split]
        for row in list(select(group))[:CAP]:
            output.append(sample(row, None, "action_deletion", "no_action"))
            counts[(split, "action_deletion")] += 1
        reads = [row for row in group if row["raw_action"].get("function") == "read" and isinstance(row["raw_action"].get("args", {}).get("path"), str)]
        for row in list(select(reads))[:CAP]:
            modified = copy.deepcopy(row["raw_action"])
            modified["function"] = "delete"
            output.append(sample(row, modified, "type_replacement", "type_mismatch"))
            counts[(split, "type_replacement")] += 1
        candidates = []
        for row in group:
            action = row["raw_action"]
            if action.get("function") != "web_fetch" or not isinstance(action.get("args", {}).get("url"), str):
                continue
            own_url = action["args"]["url"]
            donors = [donor for donor in urls_by_step[(row["trajectory_id"], row["thought_step"])] if donor["action_id"] != row["action_id"] and donor["raw_action"]["args"]["url"] != own_url and donor["selected_intent_ids"] != row["selected_intent_ids"]]
            if donors:
                candidates.append((row, sorted(donors, key=lambda d: d["action_id"])[0]))
        for row, donor in list(select_pairs(candidates))[:CAP]:
            modified = copy.deepcopy(row["raw_action"])
            modified["args"]["url"] = donor["raw_action"]["args"]["url"]
            output.append(sample(row, modified, "object_replacement", "object_mismatch", donor=donor))
            counts[(split, "object_replacement")] += 1
            if split == "test":
                deceptive = copy.deepcopy(modified)
                original_name = Path(urlparse(row["raw_action"]["args"]["url"]).path).name
                if original_name:
                    deceptive["args"]["url"] += "#" + original_name
                    output.append(sample(row, deceptive, "object_replacement", "object_mismatch", donor=donor, trap_type="misleading_url_fragment"))
                    counts[(split, "adversarial_misleading_fragment")] += 1
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "generated_samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(output, key=lambda r: (r["split"], r["evaluation_view"], r["sample_id"])):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {"schema_version": "matching_v2_stage4_model_diagnostics", "source_exact_count": len(exact), "generated_count": len(output), "counts": {f"{split}:{kind}": count for (split, kind), count in sorted(counts.items())}, "cap_per_split_and_type": CAP, "supervision_tier": "model_diagnostic", "gold_review": "skipped_by_user", "note": "Generated labels are hypotheses requiring review; no scope reduction was generated without an independently verifiable target set."}
    (args.output_dir / "generation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def select_pairs(pairs):
    seen = set()
    for pair in sorted(pairs, key=lambda p: hashlib.sha256(p[0]["action_id"].encode()).hexdigest()):
        if pair[0]["trajectory_id"] not in seen:
            seen.add(pair[0]["trajectory_id"])
            yield pair


if __name__ == "__main__":
    main()
