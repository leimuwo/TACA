#!/usr/bin/env python3
"""Audit high-ranked unlabeled dev candidates and close clear same-target equivalents."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def action_obj(row):
    return json.loads(row["action"][len("[ACTION] "):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--rankings", type=Path, required=True)
    ap.add_argument("--qrels", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    queries = {x["query_id"]: x for x in read_jsonl(args.split_dir / "queries.jsonl")}
    actions = {x["action_id"]: x for x in read_jsonl(args.split_dir / "actions.jsonl")}
    known = defaultdict(set)
    for x in read_jsonl(args.qrels):
        known[x["query_id"]].add(x["action_id"])

    # These are manually audited clear equivalents: same operation and same target,
    # with only an accepted path/file spelling or a concise equivalent request.
    clear = {
        "018f12755bd0b0b2d8a706ce": "same read target; file/path spelling only",
        "1c149c2327a599e44e8883ba": "same read target; relative versus absolute path",
        "20af613f7bf919d9aab39e8b": "same read target; relative versus absolute path",
        "21e6a2a504d5476da0994621": "same read target and file",
        "55fceeee9efd993dfcfc3e9a": "same read target; file/path field spelling only",
        "5b43424e37b00fc88439a81d": "same read target and file",
        "a0a053da49de5d25e369f190": "same read target; file/path field spelling only",
        "f79b3f80167bba91dc21340a": "same read target; relative versus absolute path",
        "bbe18b77b5f5b7dc8043bad9": "same write target and constraints; concise rewrite remains under 120 words",
    }
    different_edit = {
        "0142e49853f9496ce3d92658", "1efdfa2bfc18bd733cf3c412",
        "30ecf9d42305f2faf9b15e20", "777c00ca86fc69c792b71437",
        "7f0ad64d37b867aeae5f6a73", "a7ef7b097e4ab5c37e9facb4",
        "b176be594f1acca399aeadfa", "bb5fe5e3c721d28b9bc07598",
        "ca814282b9dd37a633764e91", "ee71bafbb40298daf35f7ada",
    }
    different_write = {
        "62ca6127a0c247955c5dbd1d", "6a95a798629bcb8cbdae345d",
        "8a05ce574b608914b64a6fd3", "de2b61f71b54180bd6100e90",
        "ef69a79b9a3dd8b9ff41fbc8",
    }
    partial_read = {"369af7ade70f0f7fc86eb1f1", "c638a9ae425084d0fe6c9a21"}
    reviewed_ids = set(clear) | different_edit | different_write | partial_read
    if len(reviewed_ids) != 26:
        raise ValueError("review ID overlap or omission")
    rows = []
    for ranking in read_jsonl(args.rankings):
        qid = ranking["query_id"]
        if not ranking["top_actions"]:
            continue
        top = ranking["top_actions"][0]
        if top["is_known_positive"] or qid not in reviewed_ids:
            continue
        aid = top["action_id"]
        if aid not in actions:
            continue
        if qid in clear:
            label, reason = "clear_equivalent", clear[qid]
        elif qid in different_edit:
            label, reason = "different_edit", "same file but different edit or stage; does not perform the specified change"
        elif qid in different_write:
            label, reason = "intent_missing_context", "same output path but unrelated document content across trajectories; query omits identifying task context"
        else:
            label, reason = "partial_read", "top action reads only a limited range; full-file intent is not established"
        positives = sorted(known[qid])
        rows.append({
            "query_id": qid,
            "query": queries[qid]["query"],
            "known_positive_action_ids": positives,
            "known_positive_actions": [actions[x]["action"] for x in positives],
            "top1_action_id": aid,
            "top1_score": top["score"],
            "top1_rank": 1,
            "top1_action": actions[aid]["action"],
            "top1_trajectory_id": actions[aid].get("trajectory_id"),
            "top1_thought_step": actions[aid].get("thought_step"),
            "review_label": label,
            "review_reason": reason,
            "add_qrel": qid in clear,
        })
    rows.sort(key=lambda x: x["query_id"])
    if {x["query_id"] for x in rows} != reviewed_ids:
        raise ValueError("the frozen rankings do not match the reviewed ID list")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_path = args.output_dir / "high_rank_unlabeled_review.jsonl"
    review_path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
    additions = []
    for x in rows:
        if x["add_qrel"] and x["top1_action_id"] not in known[x["query_id"]]:
            additions.append({"query_id": x["query_id"], "action_id": x["top1_action_id"], "relevance": 1,
                              "source_label": "manual_audit_equivalent", "supervision_tier": "dev_reviewed_equivalent"})
    closed = read_jsonl(args.qrels) + additions
    closed.sort(key=lambda x: (x["query_id"], x["action_id"]))
    (args.output_dir / "qrels_review_closed.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in closed), encoding="utf-8")
    summary = {"audited_candidates": len(rows), "clear_equivalents": len(additions),
               "remaining_unjudged": len(rows) - len(additions), "source_qrels": str(args.qrels)}
    (args.output_dir / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # A reproducible spot check of the two dominant exec-related error groups.
    pools = defaultdict(list)
    for ranking in read_jsonl(args.rankings):
        qid = ranking["query_id"]
        top_aid = ranking["top_actions"][0]["action_id"]
        if top_aid in known[qid] or qid in reviewed_ids:
            continue
        top_fn = action_obj(actions[top_aid])["function"]
        positive_ids = sorted(known[qid])
        if any(action_obj(actions[positive_id])["function"] == "exec" for positive_id in positive_ids):
            if top_fn in {"exec", "web_fetch"}:
                pools[top_fn].append((ranking, top_aid, positive_ids))
    rng = random.Random(20261009)
    spot = []
    for tool, size in (("exec", 15), ("web_fetch", 10)):
        for ranking, aid, positives in rng.sample(pools[tool], size):
            qid = ranking["query_id"]
            spot.append({"query_id": qid, "query": queries[qid]["query"],
                         "known_positive_action_ids": positives,
                         "known_positive_actions": [actions[x]["action"] for x in positives],
                         "top1_action_id": aid, "top1_action": actions[aid]["action"],
                         "top1_score": ranking["top_actions"][0]["score"],
                         "top1_rank": 1, "top1_trajectory_id": actions[aid].get("trajectory_id"),
                         "top1_thought_step": actions[aid].get("thought_step"),
                         "comparison": f"exec_to_{tool}", "review_label": "spot_check_unadjudicated",
                         "add_qrel": False})
    spot.sort(key=lambda x: x["query_id"])
    spot_reviews = {
        "1c98bee0b1e871dc46992135": ("wrong_parameters", "counts keywords rather than log entries"),
        "1e844068c118aecc6eeee080": ("clear_equivalent", "tee saves the same validator stdout and stderr to the required file"),
        "2671efe3d322c9d04eb95a9e": ("wrong_operation", "creates directories rather than verifying deliverables"),
        "33131e2dabeb14be6c788390": ("clear_equivalent", "same script, input flags and output directory; absolute paths and interpreter spelling differ"),
        "3607e4bd66b42e759bdf3b0f": ("wrong_operation", "validates a JSON file rather than creating output directory"),
        "41a3090a41e914879cbfe4b8": ("wrong_query", "searches Dudley Town Deal instead of Habitat for Humanity"),
        "512a2b101ffcfb33b0cbb5ac": ("partial_or_different_operation", "web_fetch reads a landing page and does not download the PDF to disk"),
        "5bb6d9378e7facab9f1013d7": ("wrong_source", "fetches CDC instead of California Department of Public Health"),
        "60ead738af43e5f9bc890176": ("wrong_time_or_scope", "opens a 2026 article rather than searching next-month fixtures in the source task"),
        "6329e69ae271522e02929859": ("wrong_operation", "processes data rather than creating directory structure"),
        "6823efd52be30326f98d5ff5": ("wrong_query", "Google query concerns Julius Erving rather than Bun"),
        "77c38b71a9cfe877527bbb2c": ("partial_or_different_operation", "reads a UNESCO overview page rather than locating a working report PDF URL"),
        "8007575e1189397c52754ff9": ("wrong_operation", "creates directories rather than processing data"),
        "897f368b6100abdf76508737": ("wrong_url", "fetches base-rate-neglect instead of testing the spaced base-rate URL"),
        "96a09409f367a36e680e11a5": ("wrong_source", "checks HHS rather than California Department of Health URL"),
        "bb24055ae3af9d396cf8ed05": ("needs_context", "similar scripts differ in additional filtering; requires full code comparison before equivalence"),
        "bf378b8653aedac96dea40b6": ("clear_equivalent", "same APA dictionary target; web_fetch directly fetches the definition"),
        "bf95ae7b358e04c917dff923": ("conditional_equivalent", "installs and runs speedtest-cli; conditional on installation and network succeeding"),
        "c55cb72f4c9ea9144e42f025": ("clear_equivalent", "same build_report.py script and all required input/output flags"),
        "ca4e29b1b31328c747c0e67c": ("partial_or_different_operation", "reads WMO publication page rather than checking the report URL"),
        "e7a44f8e5454ac9052a24292": ("partial_or_different_operation", "reads one Dudley council page rather than searching official sources broadly"),
        "f299cf4febdc833913377781": ("clear_equivalent", "find lists workspace files and directories, satisfying the broad inspection intent"),
        "f6264e2f507810e392b2b462": ("wrong_script", "runs compliance_check.py instead of tools/validate.py"),
        "fc07b7d66848befe439381ca": ("partial_or_different_operation", "fetches page text but does not save raw HTML using curl"),
        "fe45e632eb3d35323bf1c187": ("wrong_script", "runs run.py rather than extract_episodes.py and does not capture baseline output"),
    }
    if {x["query_id"] for x in spot} != set(spot_reviews):
        raise ValueError("spot check sample changed; audit decisions must be refreshed")
    for x in spot:
        label, reason = spot_reviews[x["query_id"]]
        x["review_label"] = label
        x["review_reason"] = reason
        x["add_qrel"] = label == "clear_equivalent"
    spot_additions = [
        {"query_id": x["query_id"], "action_id": x["top1_action_id"], "relevance": 1,
         "source_label": "manual_audit_equivalent", "supervision_tier": "dev_reviewed_equivalent"}
        for x in spot if x["add_qrel"]
    ]
    closed.extend(spot_additions)
    closed.sort(key=lambda x: (x["query_id"], x["action_id"]))
    (args.output_dir / "qrels_review_closed.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in closed), encoding="utf-8")
    (args.output_dir / "exec_high_rank_spot_check.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in spot), encoding="utf-8")
    summary["exec_spot_check"] = {"exec_to_exec": 15, "exec_to_web_fetch": 10,
                                   "clear_equivalents": len(spot_additions),
                                   "status": "reviewed; clear equivalents added to reviewed qrels"}
    summary["total_reviewed_additions"] = len(additions) + len(spot_additions)
    summary["reviewed_qrel_edges"] = len(closed)
    (args.output_dir / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
