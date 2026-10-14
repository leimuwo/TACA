#!/usr/bin/env python3
"""Generate and GQQ-check 5,000 split-local hard negatives for BGE training."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import posixpath
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thought_action_retrieval.matching.action_atomicization import audit_action

TARGET = {"object_mismatch": 3000, "type_mismatch": 2000}
BASE = "https://qjkpcombh9jkcecah8jdkgd5d5beqege.openapi-qb-ai.sii.edu.cn/v1"
MODEL = "deepseek-v4-flash-gqq"


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(compact(value).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def generate_candidates(matches, actions, intents):
    exact = []
    donors = defaultdict(list)
    for row in matches:
        if row["split"] != "train" or row["label"] != "exact_match" or len(row.get("selected_intent_ids", [])) != 1:
            continue
        action = actions[row["action_id"]]["action"]
        function, args = action.get("function"), action.get("args", {})
        field = "url" if function == "web_fetch" else "path" if function in {"read", "write", "edit"} else None
        if not field or not isinstance(args.get(field), str) or len(compact(action)) > 3000:
            continue
        intent_id = row["selected_intent_ids"][0]
        revised = {item["intent_id"]: item for item in row.get("revised_intents", [])}
        step = intents[(row["trajectory_id"], row["thought_step"])]
        original = {item["intent_id"]: item for item in step.get("execution_intents", [])}
        intent = revised.get(intent_id) or original.get(intent_id)
        if not intent:
            continue
        source = {"row": row, "action": action, "intent": intent, "function": function, "field": field, "target": args[field]}
        exact.append(source)
        donors[(function, field)].append(source)
    pools = {"object_mismatch": [], "type_mismatch": []}
    for source in exact:
        row, action, function, field = source["row"], source["action"], source["function"], source["field"]
        name = posixpath.basename(source["target"].split("?", 1)[0]).lower()
        stem = name.rsplit(".", 1)[0]
        intent_text = source["intent"]["intent_text"].lower()
        # A generic intent such as "read the input files" does not make a
        # different path a clear negative. Require an explicit target mention.
        if (len(name) >= 5 and name in intent_text) or (len(stem) >= 5 and stem in intent_text):
            possible = [donor for donor in donors[(function, field)] if donor["row"]["trajectory_id"] != row["trajectory_id"] and donor["target"] != source["target"]]
            possible.sort(key=lambda donor: sha([row["action_id"], donor["row"]["action_id"]]))
            for donor in possible[:4]:
                modified = copy.deepcopy(action)
                modified["args"][field] = donor["target"]
                pools["object_mismatch"].append(make_candidate(source, modified, "object_mismatch", donor["row"]["action_id"]))
        if function in {"read", "write", "edit"}:
            modified = copy.deepcopy(action)
            modified["function"] = "write" if function == "read" else "read"
            modified["args"] = {"path": source["target"], "content": ""} if function == "read" else {"path": source["target"]}
            pools["type_mismatch"].append(make_candidate(source, modified, "type_mismatch", None))
    for label in pools:
        pools[label].sort(key=lambda item: sha([label, item["source_action_id"], item["candidate_id"]]))
    return pools


def make_candidate(source, modified, label, donor_id):
    row = source["row"]
    candidate = {
        "candidate_id": sha([row["action_id"], label, modified])[:24],
        "source_action_id": row["action_id"],
        "source_action_sha256": row["raw_action_sha256"],
        "donor_action_id": donor_id,
        "trajectory_id": row["trajectory_id"],
        "thought_step": row["thought_step"],
        "template_cluster_id": row["template_cluster_id"],
        "split": "train",
        "intent_id": source["intent"]["intent_id"],
        "intent_text": source["intent"]["intent_text"],
        "source_quote": source["intent"]["source_quote"],
        "positive_action": source["action"],
        "negative_action": modified,
        "target_label": label,
        "changed_dimensions": ["target"] if label == "object_mismatch" else ["operation_family"],
        "supervision_tier": "silver_model_generated",
    }
    return candidate


def check_batch(batch, key, timeout, retries):
    concise = [{"candidate_id": row["candidate_id"], "intent": row["intent_text"], "original_action": {"function": row["positive_action"]["function"], "args": row["positive_action"]["args"]}, "modified_action": {"function": row["negative_action"]["function"], "args": row["negative_action"]["args"]}, "target_label": row["target_label"]} for row in batch]
    prompt = "For each candidate, verify that only the intended operation family or target changed, the modified Action is plausible, and it clearly fails to fulfill the Intent with the target_label. Return a JSON object with results array of {candidate_id, accept, label, reason}. Reject unclear cases, extra changes, and cases where the modified Action still fulfills the Intent. Do not infer hidden state."
    payload = {"model": MODEL, "temperature": 0, "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": compact(concise)}]}
    request = urllib.request.Request(BASE + "/chat/completions", data=compact(payload).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last = None
    for _ in range(retries):
        try:
            with opener.open(request, timeout=timeout) as response:
                content = json.loads(response.read())["choices"][0]["message"]["content"].strip()
            content = content.removeprefix("```json").removesuffix("```").strip()
            parsed = json.loads(content)
            results = parsed.get("results") if isinstance(parsed, dict) else parsed
            if not isinstance(results, list):
                raise ValueError("missing results")
            by_id = {row["candidate_id"]: row for row in results if isinstance(row, dict) and isinstance(row.get("candidate_id"), str)}
            if set(by_id) != {row["candidate_id"] for row in batch}:
                raise ValueError("candidate IDs missing or unexpected")
            return by_id
        except Exception as error:
            last = str(error)
    return {row["candidate_id"]: {"accept": False, "label": "model_error", "reason": last} for row in batch}


def mutate_one_dimension(candidate):
    before, after = candidate["positive_action"], candidate["negative_action"]
    if candidate["target_label"] == "type_mismatch":
        return (before["function"] != after["function"]
                and before["args"].get("path") == after["args"].get("path")
                and after["function"] in {"read", "write"}
                and set(after["args"]) == ({"path"} if after["function"] == "read" else {"path", "content"}))
    field = "url" if before["function"] == "web_fetch" else "path"
    return before["function"] == after["function"] and set(before["args"]) == set(after["args"]) and all(before["args"][key] == after["args"][key] for key in before["args"] if key != field) and before["args"].get(field) != after["args"].get(field)


def training_text_key(candidate):
    def action_text(action):
        return {"function": action["function"], "args": action.get("args", {})}
    return sha([candidate["intent_text"], action_text(candidate["positive_action"]), action_text(candidate["negative_action"])])


def run(args):
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records = list(read_jsonl(args.matches))
    actions = {row["action_id"]: row for row in read_jsonl(args.actions)}
    intents = {(row["trajectory_id"], row["thought_step"]): row for row in read_jsonl(args.intents)}
    pools = generate_candidates(records, actions, intents)
    if len(pools["object_mismatch"]) < TARGET["object_mismatch"] or len(pools["type_mismatch"]) < TARGET["type_mismatch"]:
        raise ValueError("insufficient candidates")
    path = root / "validated_generated_negatives.jsonl"
    accepted = {row["candidate_id"]: row for row in read_jsonl(path)} if path.exists() else {}
    counts = Counter(row["target_label"] for row in accepted.values())
    used = set(accepted)
    excluded_path = root / "excluded_duplicate_candidate_ids.txt"
    if excluded_path.exists():
        used.update(line.strip() for line in excluded_path.read_text().splitlines() if line.strip())
    seen_training_text = {training_text_key(row) for row in accepted.values()}
    positions = {label: 0 for label in TARGET}
    with path.open("a", encoding="utf-8") as out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            while any(counts[label] < TARGET[label] for label in TARGET):
                tasks = []
                for label in TARGET:
                    if counts[label] >= TARGET[label]:
                        continue
                    batch = []
                    while positions[label] < len(pools[label]) and len(batch) < args.batch_size:
                        candidate = pools[label][positions[label]]
                        positions[label] += 1
                        if candidate["candidate_id"] not in used and training_text_key(candidate) not in seen_training_text and mutate_one_dimension(candidate):
                            used.add(candidate["candidate_id"])
                            batch.append(candidate)
                    if batch:
                        tasks.append((batch, executor.submit(check_batch, batch, args.key, args.timeout, args.retries)))
                if not tasks:
                    break
                for batch, future in tasks:
                    decisions = future.result()
                    for candidate in batch:
                        decision = decisions[candidate["candidate_id"]]
                        label = candidate["target_label"]
                        if counts[label] >= TARGET[label] or decision.get("accept") is not True or decision.get("label") != label:
                            continue
                        key = training_text_key(candidate)
                        if key in seen_training_text:
                            continue
                        audit = audit_action(candidate["negative_action"], trajectory_id=candidate["trajectory_id"], thought_step=candidate["thought_step"], action_index=0)
                        candidate["negative_action_sha256"] = audit["raw_action_sha256"]
                        candidate["negative_action_structure"] = {key: audit[key] for key in ("atomicity", "action_has_extras", "requires_manual_semantic_review", "normalizer_version", "structure_analysis_sha256")}
                        candidate["gqq_validation"] = {"model": MODEL, "label": label, "reason": decision.get("reason", ""), "accepted": True}
                        out.write(compact(candidate) + "\n")
                        out.flush()
                        os.fsync(out.fileno())
                        seen_training_text.add(key)
                        counts[label] += 1
                if sum(counts.values()) % 250 < args.batch_size * args.workers:
                    print(compact({"accepted": sum(counts.values()), "labels": dict(counts)}), flush=True)
    if any(counts[label] < TARGET[label] for label in TARGET):
        raise RuntimeError(f"target not reached: {dict(counts)}; pool sizes: { {key: len(value) for key, value in pools.items()} }")
    print(compact({"accepted": sum(counts.values()), "labels": dict(counts), "pool_sizes": {key: len(value) for key, value in pools.items()}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--intents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    args.key = os.environ["INTENT_API_KEY"]
    run(args)


if __name__ == "__main__":
    main()
