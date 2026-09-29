#!/usr/bin/env python3
"""Build conservative P1/P2 hard negatives, with GQQ generation for shortages."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import posixpath
import shlex
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thought_action_retrieval.matching.action_atomicization import audit_action

BASE = os.environ.get("INTENT_API_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
MODEL = os.environ.get("INTENT_MODEL_NAME", "deepseek-v4-flash-gqq")
MAX_TEXT = 1600


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(compact(value).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def append(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(compact(row) + "\n")


def action_view(action):
    return {"function": action.get("function"), "args": action.get("args", {})}


def action_text(action):
    return "[ACTION] " + compact(action_view(action))


def intent_for(match, intent_steps, intent_id):
    revised = {item["intent_id"]: item for item in match.get("revised_intents", [])}
    original = {item["intent_id"]: item for item in intent_steps[(match["trajectory_id"], match["thought_step"])].get("execution_intents", [])}
    return revised.get(intent_id) or original.get(intent_id)


def build_positive_pool(matches, actions, intent_steps):
    pool = []
    for row in matches:
        if row["split"] != "train" or row["label"] != "exact_match":
            continue
        action = actions[row["action_id"]]["action"]
        for intent_id in row.get("selected_intent_ids", []):
            intent = intent_for(row, intent_steps, intent_id)
            if not intent or not intent.get("intent_text"):
                continue
            reason = None
            if row.get("review_required") or row.get("atomicity") != "atomic":
                reason = "source_requires_review"
            elif len(action_text(action)) > MAX_TEXT:
                reason = "training_text_too_long"
            elif not isinstance(action.get("args"), dict):
                reason = "invalid_action_args"
            pool.append({
                "edge_id": digest([row["trajectory_id"], row["thought_step"], intent_id, row["action_id"]])[:24],
                "trajectory_id": row["trajectory_id"], "thought_step": row["thought_step"],
                "template_cluster_id": row["template_cluster_id"], "split": "train",
                "intent_id": intent_id, "intent_text": intent["intent_text"],
                "source_quote": intent.get("source_quote"), "action_id": row["action_id"],
                "raw_action_sha256": row["raw_action_sha256"], "positive_action": action,
                "source_tier": row.get("source_tier"), "positive_status": "model_preliminary",
                "eligibility": "candidate" if reason is None else "review_queue",
                "eligibility_reason": reason,
            })
    return pool


def target_field(action):
    function, args = action.get("function"), action.get("args", {})
    fields = ("url",) if function == "web_fetch" else ("path", "file", "file_path") if function in {"read", "write", "edit"} else ()
    return next((field for field in fields if isinstance(args.get(field), str) and args[field]), None)


def explicitly_named(intent, value):
    name = posixpath.basename(value.split("?", 1)[0]).lower()
    stem = name.rsplit(".", 1)[0]
    intent = intent.lower()
    return (len(name) >= 5 and name in intent) or (len(stem) >= 5 and stem in intent)


def target_constrained(intent, original, replacement=None):
    if explicitly_named(intent, original):
        return True
    intent = intent.lower()
    original = original.lower().replace("\\", "/")
    replacement = (replacement or "").lower().replace("\\", "/")
    for segment in ("input", "output", "config", "assets"):
        if segment in intent and ("/" + segment + "/") in ("/" + original.lstrip("/")):
            if replacement and ("/" + segment + "/") not in ("/" + replacement.lstrip("/")):
                return True
    return False


def candidate(edge, action, category, origin, donor_id=None, reason=None):
    candidate_id = digest([edge["edge_id"], category, action_view(action)])[:24]
    return {"candidate_id": candidate_id, "edge_id": edge["edge_id"],
            "source_action_id": edge["action_id"], "intent_id": edge["intent_id"],
            "trajectory_id": edge["trajectory_id"], "thought_step": edge["thought_step"],
            "split": "train", "category": category, "candidate_action": action,
            "origin": origin, "donor_action_id": donor_id, "generation_reason": reason}


def check_candidate(edge, row, all_positive_views):
    before, after = edge["positive_action"], row["candidate_action"]
    if not isinstance(after, dict) or not isinstance(after.get("args"), dict):
        return "bad_action_schema"
    if action_view(after) == action_view(before) or compact(action_view(after)) in all_positive_views[edge["edge_id"]]:
        return "positive_collision"
    if len(action_text(after)) > MAX_TEXT:
        return "training_text_too_long"
    category = row["category"]
    if category == "P1":
        if before.get("function") != after.get("function"):
            return "p1_changed_function"
        a, b = before["args"], after["args"]
        if set(a) != set(b) or sum(a[key] != b[key] for key in a) != 1:
            return "p1_not_one_parameter"
        field = next(key for key in a if a[key] != b[key])
        if field not in {"path", "file", "file_path", "url", "command"} or not isinstance(a[field], str) or not isinstance(b[field], str):
            return "p1_unsupported_parameter"
        if field == "command":
            try:
                old_tokens, new_tokens = shlex.split(a[field]), shlex.split(b[field])
            except ValueError:
                return "p1_invalid_command"
            if len(old_tokens) != len(new_tokens) or not old_tokens or sum(x != y for x, y in zip(old_tokens, new_tokens)) != 1:
                return "p1_command_not_one_token"
            changed_at = next(i for i, (x, y) in enumerate(zip(old_tokens, new_tokens)) if x != y)
            if changed_at == 0 or not target_constrained(edge["intent_text"], old_tokens[changed_at], new_tokens[changed_at]):
                return "p1_command_target_not_explicit"
            if posixpath.basename(old_tokens[changed_at]).lower() == posixpath.basename(new_tokens[changed_at]).lower():
                return "p1_command_same_target_name"
        else:
            if not target_constrained(edge["intent_text"], a[field], b[field]):
                return "p1_target_not_explicit"
            if posixpath.basename(a[field].split("?", 1)[0]).lower() == posixpath.basename(b[field].split("?", 1)[0]).lower():
                return "p1_same_target_name"
    elif category == "P2":
        old, new = before.get("function"), after.get("function")
        if old == new == "exec":
            if set(before["args"]) != set(after["args"]) or any(before["args"][key] != after["args"][key] for key in before["args"] if key != "command"):
                return "p2_exec_changed_other_args"
            try:
                old_tokens, new_tokens = shlex.split(before["args"]["command"]), shlex.split(after["args"]["command"])
            except (KeyError, ValueError):
                return "p2_exec_invalid_command"
            if len(old_tokens) < 2 or len(old_tokens) != len(new_tokens) or old_tokens[1:] != new_tokens[1:] or (old_tokens[0], new_tokens[0]) not in {("python3", "cat"), ("python", "cat"), ("cat", "python3"), ("mkdir", "ls"), ("ls", "mkdir")}:
                return "p2_exec_not_single_operation_change"
        else:
            if old == new or (old, new) not in {("read", "write"), ("write", "read"), ("edit", "read")}:
                return "p2_unsupported_operation"
            old_field = target_field(before)
            if not old_field or after["args"].get(old_field) != before["args"].get(old_field):
                return "p2_changed_target"
            expected = {old_field} if new == "read" else {old_field, "content"}
            if set(after["args"]) != expected or (new == "write" and not isinstance(after["args"].get("content"), str)):
                return "p2_invalid_args"
    else:
        return "unknown_category"
    return None


def model_call(system, payload, key, retries=2, timeout=35):
    body = {"model": MODEL, "temperature": 0, "messages": [{"role": "system", "content": system}, {"role": "user", "content": compact(payload)}]}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last = None
    for _ in range(retries):
        try:
            request = urllib.request.Request(BASE + "/chat/completions", data=compact(body).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            with opener.open(request, timeout=timeout) as response:
                content = json.loads(response.read())["choices"][0]["message"]["content"].strip()
            content = content.removeprefix("```json").removesuffix("```").strip()
            return json.loads(content)
        except Exception as error:
            last = str(error)
    raise RuntimeError(last)


VALIDATOR = "Evaluate each Intent and proposed Action using only visible semantics. Accept only a valid P1 wrong target or P2 wrong operation which clearly cannot fully satisfy the Intent. Reject partial fulfillment, supporting/intermediate steps, equivalent positives, ambiguous broad intents, or hidden-state assumptions. Return JSON object results:[{candidate_id,accept,category,reason}]. Every ID exactly once."
GENERATOR = "Propose at most 3 clear hard-negative Actions for this Intent/positive Action. P1: keep function and every args key; change exactly one string path/file/file_path/url to a clearly wrong target; for exec, change exactly one command target token only if that target is explicit in Intent. P2: only change read->write or write/edit->read, keeping exactly the same target; for simple exec, change only the first command token among python3->cat, python->cat, cat->python3, mkdir->ls, ls->mkdir and preserve all later tokens. All tool arguments must be INSIDE the args object. The entire candidate_action must have function and args fields. Never invent a tool, change content alone, return a positive equivalent, or use partial/supporting steps. If constraints cannot give a clear negative, return an empty list. Return JSON object candidates:[{category,candidate_action,changed_fields,why_incorrect}] and nothing else."


def validate_batch(batch, edges, key):
    payload = [{"candidate_id": row["candidate_id"], "intent": edges[row["edge_id"]]["intent_text"],
                "positive": action_view(edges[row["edge_id"]]["positive_action"]),
                "negative": action_view(row["candidate_action"]), "category": row["category"]} for row in batch]
    try:
        result = model_call(VALIDATOR, payload, key)
        decisions = result["results"] if isinstance(result, dict) else result
        by_id = {x["candidate_id"]: x for x in decisions}
        if set(by_id) != {x["candidate_id"] for x in batch}:
            raise ValueError("validation IDs differ")
        return by_id
    except Exception as error:
        return {row["candidate_id"]: {"accept": False, "category": "model_error", "reason": str(error)} for row in batch}


def generate_for_edge(edge, needed, key):
    payload = {"intent": edge["intent_text"], "positive_action": action_view(edge["positive_action"]),
               "needed_categories": needed, "allowed_operations": ["read", "write", "edit", "web_fetch", "exec"],
               "argument_patterns": {"read": {"path": "string"}, "write": {"path": "string", "content": "string"}, "edit": {"path": "string", "edits": "array"}, "web_fetch": {"url": "string"}}}
    try:
        result = model_call(GENERATOR, payload, key)
        return result.get("candidates", []) if isinstance(result, dict) else []
    except Exception:
        return []


def generation_worth_trying(edge, counts):
    action = edge["positive_action"]
    function = action["function"]
    field = target_field(action)
    if field and counts[(edge["edge_id"], "P1")] < 2 and explicitly_named(edge["intent_text"], action["args"][field]):
        return True
    if counts[(edge["edge_id"], "P2")] < 1 and function in {"read", "write", "edit"} and field:
        return True
    if function == "exec" and counts[(edge["edge_id"], "P2")] < 1:
        try:
            tokens = shlex.split(action["args"]["command"])
            return len(tokens) == 2 and tokens[0] in {"python3", "python", "cat", "mkdir", "ls"} and not tokens[1].startswith("-")
        except (KeyError, ValueError):
            return False
    return False


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    matches = list(read_jsonl(args.matches))
    actions = {x["action_id"]: x for x in read_jsonl(args.actions)}
    intents = {(x["trajectory_id"], x["thought_step"]): x for x in read_jsonl(args.intents)}
    pool = build_positive_pool(matches, actions, intents)
    edges = {x["edge_id"]: x for x in pool}
    with (args.output / "positive_pool.jsonl").open("w", encoding="utf-8") as handle:
        for edge in pool:
            handle.write(compact(edge) + "\n")
    valid_edges = [x for x in pool if x["eligibility"] == "candidate"]
    if args.limit:
        valid_edges = valid_edges[:args.limit]
    by_action_intent = {(x["action_id"], x["intent_id"]): x for x in valid_edges}
    positives_by_intent = defaultdict(set)
    for edge in pool:
        key = (edge["trajectory_id"], edge["thought_step"], edge["intent_id"])
        positives_by_intent[key].add(compact(action_view(edge["positive_action"])))
    all_positive_views = {edge["edge_id"]: positives_by_intent[(edge["trajectory_id"], edge["thought_step"], edge["intent_id"])] for edge in pool}
    donors = defaultdict(list)
    exec_donors = defaultdict(list)
    for edge in valid_edges:
        field = target_field(edge["positive_action"])
        if field:
            donors[(edge["positive_action"]["function"], field)].append(edge)
        elif edge["positive_action"].get("function") == "exec":
            try:
                tokens = shlex.split(edge["positive_action"]["args"]["command"])
            except (KeyError, ValueError):
                tokens = []
            if len(tokens) == 2 and tokens[0] in {"python3", "python", "cat", "mkdir", "ls"} and not tokens[1].startswith("-"):
                exec_donors[tokens[0]].append((edge, tokens[1]))
    old = defaultdict(list)
    if args.old_negatives:
        for row in read_jsonl(args.old_negatives):
            old[(row["source_action_id"], row["intent_id"])].append(row)
    path = args.output / "validated_candidates.jsonl"
    accepted = {x["candidate_id"]: x for x in read_jsonl(path)} if path.exists() else {}
    seen = {digest([x["edge_id"], x["category"], action_view(x["candidate_action"])]) for x in accepted.values()}
    counts = Counter((x["edge_id"], x["category"]) for x in accepted.values())
    status = Counter()
    pending = []
    for edge in valid_edges:
        action = edge["positive_action"]
        source = []
        for old_row in old[(edge["action_id"], edge["intent_id"])]:
            cat = "P1" if old_row["target_label"] == "object_mismatch" else "P2"
            prior = candidate(edge, old_row["negative_action"], cat, "previous_gqq_validated")
            prior["prior_validation"] = old_row.get("gqq_validation")
            source.append(prior)
        field = target_field(action)
        if field:
            possible = [donor for donor in donors[(action["function"], field)] if donor["trajectory_id"] != edge["trajectory_id"] and donor["positive_action"]["args"][field] != action["args"][field]]
            possible.sort(key=lambda donor: digest([edge["edge_id"], donor["edge_id"]]))
            for donor in possible[:5]:
                changed = copy.deepcopy(action)
                changed["args"][field] = donor["positive_action"]["args"][field]
                source.append(candidate(edge, changed, "P1", "rule_donor", donor["action_id"]))
        if action["function"] in {"read", "write", "edit"} and field:
            new = "write" if action["function"] == "read" else "read"
            changed = copy.deepcopy(action)
            changed["function"] = new
            changed["args"] = {field: action["args"][field], **({"content": ""} if new == "write" else {})}
            source.append(candidate(edge, changed, "P2", "rule_operation"))
        if action["function"] == "exec" and isinstance(action.get("args", {}).get("command"), str):
            try:
                tokens = shlex.split(action["args"]["command"])
            except ValueError:
                tokens = []
            replacement = {"python3": "cat", "python": "cat", "cat": "python3", "mkdir": "ls", "ls": "mkdir"}.get(tokens[0]) if len(tokens) == 2 and not tokens[1].startswith("-") else None
            if replacement:
                changed = copy.deepcopy(action)
                changed["args"]["command"] = shlex.join([replacement, tokens[1]])
                source.append(candidate(edge, changed, "P2", "rule_exec_operation"))
            if len(tokens) == 2 and tokens[0] in exec_donors and explicitly_named(edge["intent_text"], tokens[1]):
                possible = [(donor, target) for donor, target in exec_donors[tokens[0]] if donor["trajectory_id"] != edge["trajectory_id"] and target != tokens[1]]
                possible.sort(key=lambda item: digest([edge["edge_id"], item[0]["edge_id"]]))
                for donor, target in possible[:5]:
                    changed = copy.deepcopy(action)
                    changed["args"]["command"] = shlex.join([tokens[0], target])
                    source.append(candidate(edge, changed, "P1", "rule_exec_donor", donor["action_id"]))
        for row in source:
            limit = 2 if row["category"] == "P1" else 1
            if counts[(edge["edge_id"], row["category"])] >= limit:
                continue
            key = digest([edge["edge_id"], row["category"], action_view(row["candidate_action"])])
            if key in seen:
                continue
            if row["candidate_id"] in accepted:
                seen.add(key)
                continue
            reason = check_candidate(edge, row, all_positive_views)
            if reason:
                status[reason] += 1
                continue
            seen.add(key)
            if row["origin"] == "previous_gqq_validated" and row["prior_validation"] and row["prior_validation"].get("accepted") is True:
                audit = audit_action(row["candidate_action"], trajectory_id=row["trajectory_id"], thought_step=row["thought_step"], action_index=0)
                if audit["atomicity"] != "atomic":
                    status["non_atomic_prior"] += 1
                    continue
                row["candidate_sha256"] = audit["raw_action_sha256"]
                row["validation"] = {"model": MODEL, "accepted": True, "reason": row["prior_validation"].get("reason", ""), "prompt_version": "v1-prior-validator-reused"}
                row["supervision_tier"] = "silver_model_validated"
                accepted[row["candidate_id"]] = row
                counts[(edge["edge_id"], row["category"])] += 1
                append(path, row)
            else:
                pending.append(row)
    pending = [row for row in pending if counts[(row["edge_id"], row["category"])] < (2 if row["category"] == "P1" else 1)]
    print(compact({"positive_edges": len(pool), "eligible_edges": len(valid_edges), "rule_candidates": len(pending), "previously_accepted": len(accepted), "rule_exclusions": dict(status)}), flush=True)

    def accept_rows(batch, decisions):
        for row in batch:
            d = decisions[row["candidate_id"]]
            cat = row["category"]
            if d.get("accept") is not True or d.get("category") != cat:
                status["model_reject_" + cat] += 1
                continue
            limit = 2 if cat == "P1" else 1
            if counts[(row["edge_id"], cat)] >= limit:
                continue
            audit = audit_action(row["candidate_action"], trajectory_id=row["trajectory_id"], thought_step=row["thought_step"], action_index=0)
            if audit["atomicity"] != "atomic":
                status["non_atomic_candidate"] += 1
                continue
            row["candidate_sha256"] = audit["raw_action_sha256"]
            row["validation"] = {"model": MODEL, "accepted": True, "reason": d.get("reason", ""), "prompt_version": "v2-validator-1"}
            row["supervision_tier"] = "silver_model_validated"
            accepted[row["candidate_id"]] = row
            counts[(row["edge_id"], cat)] += 1
            append(path, row)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for start in range(0, len(pending), args.batch_size * args.workers):
            work = []
            for offset in range(start, min(start + args.batch_size * args.workers, len(pending)), args.batch_size):
                batch = pending[offset:offset + args.batch_size]
                work.append((batch, executor.submit(validate_batch, batch, edges, args.key)))
            for batch, future in work:
                accept_rows(batch, future.result())
            if start % 600 == 0:
                print(compact({"rule_checked": min(start + args.batch_size * args.workers, len(pending)), "accepted": len(accepted)}), flush=True)

        attempted_path = args.output / "gqq_generation_attempted_edge_ids.txt"
        attempted = set(attempted_path.read_text(encoding="utf-8").splitlines()) if attempted_path.exists() else set()
        deficits = [edge for edge in valid_edges if edge["edge_id"] not in attempted and generation_worth_trying(edge, counts)]
        if args.gqq_generate_limit:
            deficits = deficits[:args.gqq_generate_limit]
        print(compact({"deficit_edges": len(deficits), "accepted_before_generation": len(accepted)}), flush=True)
        for start in range(0, len(deficits), args.workers):
            jobs = []
            for edge in deficits[start:start + args.workers]:
                need = {"P1": max(0, 2 - counts[(edge["edge_id"], "P1")]), "P2": max(0, 1 - counts[(edge["edge_id"], "P2")])}
                jobs.append((edge, executor.submit(generate_for_edge, edge, need, args.key)))
            for edge, future in jobs:
                generated = future.result()
                with attempted_path.open("a", encoding="utf-8") as handle:
                    handle.write(edge["edge_id"] + "\n")
                batch = []
                for item in generated[:3]:
                    if not isinstance(item, dict) or item.get("category") not in {"P1", "P2"} or not isinstance(item.get("candidate_action"), dict):
                        status["invalid_generation_response"] += 1
                        continue
                    proposed = item["candidate_action"]
                    if isinstance(proposed.get("args"), dict):
                        for misplaced in ("path", "file", "file_path", "url", "content", "command"):
                            if misplaced in proposed and misplaced not in proposed["args"]:
                                proposed = copy.deepcopy(proposed)
                                proposed["args"][misplaced] = proposed.pop(misplaced)
                    row = candidate(edge, proposed, item["category"], "gqq_generated", reason=item.get("why_incorrect"))
                    row["changed_fields"] = item.get("changed_fields")
                    key = digest([edge["edge_id"], row["category"], action_view(row["candidate_action"])])
                    if key in seen or counts[(edge["edge_id"], row["category"])] >= (2 if row["category"] == "P1" else 1):
                        continue
                    reason = check_candidate(edge, row, all_positive_views)
                    if reason:
                        status["generated_" + reason] += 1
                        continue
                    seen.add(key)
                    batch.append(row)
                if batch:
                    accept_rows(batch, validate_batch(batch, edges, args.key))
            if start % 100 == 0:
                print(compact({"generated_edges_checked": min(start + args.workers, len(deficits)), "accepted": len(accepted)}), flush=True)

    selected = defaultdict(list)
    for row in accepted.values():
        selected[row["edge_id"]].append(row)
    train, positive_only = [], []
    for edge in valid_edges:
        neg = sorted(selected[edge["edge_id"]], key=lambda x: (x["category"], x["candidate_id"]))
        if len(neg) < 3:
            positive_only.append({"edge_id": edge["edge_id"], "action_id": edge["action_id"], "intent_id": edge["intent_id"], "accepted_negative_count": len(neg), "missing": {"P1": max(0, 2 - counts[(edge["edge_id"], "P1")]), "P2": max(0, 1 - counts[(edge["edge_id"], "P2")])}})
            continue
        train.append({"query": "[INTENT] " + edge["intent_text"], "pos": [action_text(edge["positive_action"])],
                      "neg": [action_text(x["candidate_action"]) for x in neg],
                      "negative_categories": [x["category"] for x in neg],
                      "candidate_ids": [x["candidate_id"] for x in neg], "edge_id": edge["edge_id"],
                      "action_id": edge["action_id"], "intent_id": edge["intent_id"],
                      "trajectory_id": edge["trajectory_id"], "split": "train",
                      "supervision_tier": "silver_model_validated"})
    # The same semantic Intent/Action can occur in several trajectories. Keep
    # one training row and retain every contributing source edge for provenance.
    merged = {}
    for row in train:
        key = (row["query"], row["pos"][0])
        if key not in merged:
            merged[key] = {**row, "source_edge_ids": [row["edge_id"]]}
        else:
            merged[key]["source_edge_ids"].append(row["edge_id"])
    collapsed = len(train) - len(merged)
    train = list(merged.values())
    for name, rows in (("train_grouped", train), ("positive_only", positive_only)):
        with (args.output / (name + ".jsonl")).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(compact(row) + "\n")
    with (args.output / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in train:
            handle.write(compact({key: row[key] for key in ("query", "pos", "neg")}) + "\n")
    def file_sha256(path):
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    manifest = {"schema_version": "bge_negatives_v2", "model": MODEL, "positive_edges": len(pool),
                "eligible_edges": len(valid_edges), "accepted_candidates": len(accepted),
                "accepted_by_category": dict(Counter(x["category"] for x in accepted.values())),
                "accepted_by_origin": dict(Counter(x["origin"] for x in accepted.values())),
                "eligible_by_tool": dict(Counter(x["positive_action"]["function"] for x in valid_edges)),
                "train_by_tool": dict(Counter(edges[x["edge_id"]]["positive_action"]["function"] for x in train)),
                "train_grouped_rows": len(train), "collapsed_duplicate_training_rows": collapsed,
                "train_triplets": 3 * len(train),
                "positive_only_rows": len(positive_only),
                "last_run_exclusions": dict(status), "supervision_tier": "silver_model_validated",
                "input_sha256": {name: file_sha256(path) for name, path in {"matches": args.matches, "actions": args.actions, "intents": args.intents, "old_negatives": args.old_negatives}.items() if path},
                "note": "P4 is sampled dynamically during training; no human Gold review has been claimed."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(compact(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--intents", type=Path, required=True)
    parser.add_argument("--old-negatives", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gqq-generate-limit", type=int, default=0)
    args = parser.parse_args()
    args.key = os.environ["INTENT_API_KEY"]
    run(args)


if __name__ == "__main__":
    main()
