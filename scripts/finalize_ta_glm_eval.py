#!/usr/bin/env python3
"""Build dev/test GLM weak-label evaluation subsets with explicit exclusions."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluate_bge_clawgym import rows, sha256
from review_ta_coverage_gqq import validate
from resolve_ta_uncertainties import NO_UNCERTAINTY
from merge_ta_targeted_reviews import POSITIVE_SCOPE_RISK
from ta_coverage_aggregation import aggregate_ta_label


def text_key(intent, action):
    return (" ".join(intent.casefold().split()), " ".join(action.casefold().split()))


def finalize(windows, reviews, train_rows, other_keys=()):
    latest = {r["window_id"]: r for r in reviews}
    train_ids = {r["trajectory_id"] for r in train_rows}
    train_keys = {text_key(r["intent"], r["action"]) for r in train_rows}
    selected_keys = set(other_keys)
    accepted, pairs, queue = [], [], []
    reasons = Counter()
    for w in windows:
        wid = w["window_id"]
        reason = None
        review = latest.get(wid)
        if w["trajectory_id"] in train_ids:
            raise ValueError(f"train trajectory in held-out sample: {wid}")
        if review is None or review.get("review_failed"):
            reason = "missing_or_failed_review"
        else:
            try:
                v = validate(w, review["verdict"])
            except (ValueError, KeyError, TypeError) as exc:
                reason = "invalid_review:" + str(exc)[:100]
        if reason is None:
            unresolved = [u for u in (v.get("uncertainties") or [])
                          if not NO_UNCERTAINTY.match(" ".join(u.split()))]
            if unresolved:
                reason = "uncertain_review"
            decisions = {i["intent_id"]: i for i in v["intents"]}
            if reason is None and any(
                decisions[i["intent_id"]]["due_now"] != (i["temporal_scope"] == "current")
                for i in w["intents"]):
                reason = "temporal_disagreement"
            if reason is None and any(
                not decisions[i["intent_id"]]["assessable"]
                for i in w["intents"] if i["temporal_scope"] == "current"):
                reason = "unassessable_current_intent"
            intents = {i["intent_id"]: i for i in w["intents"]}
            actions = {a["action_id"]: a for a in w["actions"]}
            keys = {text_key(intents[p["intent_id"]]["text"],
                             actions[p["action_id"]]["action_text"]) for p in w["pairs"]}
            if reason is None and keys & (train_keys | selected_keys):
                reason = "duplicate_text_pair_across_splits"
        if reason is not None:
            queue.append({"window_id": wid, "split": w["split"],
                          "trajectory_id": w["trajectory_id"], "reason": reason})
            reasons[reason.split(":")[0]] += 1
            continue
        selected_keys.update(keys)
        accepted.append({"window_id": wid, "split": w["split"],
                         "trajectory_id": w["trajectory_id"],
                         "label": v["suggested_ta_label"],
                         "label_source": "glm-5.2_weak_supervision",
                         "model": review["model"], "prompt_version": review["prompt_version"]})
        by_pair = {p["pair_id"]: p for p in v["pairs"]}
        for p in w["pairs"]:
            result = by_pair[p["pair_id"]]
            pairs.append({"pair_id": p["pair_id"], "window_id": wid,
                          "trajectory_id": w["trajectory_id"], "split": w["split"],
                          "intent_id": p["intent_id"], "action_id": p["action_id"],
                          "intent": intents[p["intent_id"]]["text"],
                          "action": actions[p["action_id"]]["action_text"],
                          "tool": actions[p["action_id"]]["raw_call"]["function"],
                          "label": int(result["covered"]),
                          "label_source": "glm-5.2_weak_supervision",
                          "review_reason": result["reason"]})
    report = {"selected_windows": len(windows), "accepted_windows": len(accepted),
              "accepted_pairs": len(pairs), "window_labels": dict(Counter(x["label"] for x in accepted)),
              "pair_labels": dict(Counter(x["label"] for x in pairs)),
              "pair_tools": dict(Counter(x["tool"] for x in pairs)),
              "excluded_windows": dict(reasons)}
    return accepted, pairs, queue, selected_keys, report


def finalize_pairs(windows, reviews, pair_reviews, train_rows, other_keys=()):
    latest = {r["window_id"]: r for r in reviews}
    targeted = {r["window_id"]: r for r in pair_reviews}
    train_ids = {r["trajectory_id"] for r in train_rows}
    used_keys = {text_key(r["intent"], r["action"]) for r in train_rows} | set(other_keys)
    accepted, pairs, queue = [], [], []
    reasons = Counter()
    for w in windows:
        wid = w["window_id"]
        if w["trajectory_id"] in train_ids:
            raise ValueError(f"train trajectory in held-out sample: {wid}")
        review = latest.get(wid)
        if review is None or review.get("review_failed"):
            queue.append({"window_id": wid, "reason": "missing_or_failed_review"})
            reasons["missing_or_failed_review"] += 1
            continue
        try:
            v = validate(w, review["verdict"])
        except (ValueError, KeyError, TypeError):
            queue.append({"window_id": wid, "reason": "invalid_review"})
            reasons["invalid_review"] += 1
            continue
        intents = {i["intent_id"]: i for i in w["intents"]}
        actions = {a["action_id"]: a for a in w["actions"]}
        intent_decisions = {i["intent_id"]: i for i in v["intents"]}
        original_pairs = {p["pair_id"]: p for p in v["pairs"]}
        scope_risk = any(POSITIVE_SCOPE_RISK.search(note)
                         for note in v.get("uncertainties") or [])
        uncertain = bool(v.get("uncertainties"))
        second = targeted.get(wid)
        second_pairs = ({p["pair_id"]: p for p in second["verdict"]["pairs"]}
                        if second and not second.get("review_failed") else {})
        complete = True
        for p in w["pairs"]:
            pid, iid = p["pair_id"], p["intent_id"]
            original = original_pairs[pid]
            key = text_key(intents[iid]["text"], actions[p["action_id"]]["action_text"])
            decision = intent_decisions[iid]
            reason = None
            if not (decision["due_now"] and decision["assessable"]
                    and intents[iid]["temporal_scope"] == "current"):
                reason = "intent_not_eligible"
            elif key in used_keys:
                reason = "duplicate_text_pair_across_splits"
            elif uncertain:
                new = second_pairs.get(pid)
                if new is None:
                    reason = "missing_pair_rereview"
                elif new["decision"] == "uncertain" or not new["evidence_sufficient"]:
                    reason = "pair_still_uncertain"
                elif new["remaining_doubt"].strip():
                    reason = "pair_still_uncertain"
                elif new["decision"] != ("covered" if original["covered"] else "not_covered"):
                    reason = "pair_label_conflict"
                elif original["covered"] and (scope_risk or POSITIVE_SCOPE_RISK.search(new["reason"])):
                    reason = "ambiguous_positive"
            if reason:
                complete = False
                reasons[reason] += 1
                queue.append({"window_id": wid, "pair_id": pid, "reason": reason})
                continue
            used_keys.add(key)
            pairs.append({"pair_id": pid, "window_id": wid,
                          "trajectory_id": w["trajectory_id"], "split": w["split"],
                          "intent_id": iid, "action_id": p["action_id"],
                          "intent": intents[iid]["text"],
                          "action": actions[p["action_id"]]["action_text"],
                          "tool": actions[p["action_id"]]["raw_call"]["function"],
                          "label": int(original["covered"]),
                          "label_source": ("glm-5.2_two_pass_agreement" if uncertain
                                           else "glm-5.2_single_pass"),
                          "review_reason": original["reason"],
                          "pair_rereview_reason": second_pairs.get(pid, {}).get("reason")})
        if complete and w["pairs"]:
            aggregate_label = aggregate_ta_label(
                w, intent_decisions,
                {pid: original_pairs[pid]["covered"] for pid in original_pairs})
            accepted.append({"window_id": wid, "split": w["split"],
                             "trajectory_id": w["trajectory_id"],
                             "label": aggregate_label,
                             "model_suggested_label": v["suggested_ta_label"],
                             "label_source": "program_aggregation_of_glm-5.2_weak_pairs",
                             "model": review["model"], "prompt_version": review["prompt_version"]})
    report = {"selected_windows": len(windows), "accepted_windows": len(accepted),
              "accepted_pairs": len(pairs), "window_labels": dict(Counter(x["label"] for x in accepted)),
              "pair_labels": dict(Counter(x["label"] for x in pairs)),
              "pair_tools": dict(Counter(x["tool"] for x in pairs)),
              "pair_label_sources": dict(Counter(x["label_source"] for x in pairs)),
              "excluded_pairs_or_windows": dict(reasons)}
    return accepted, pairs, queue, used_keys, report


def write(path, records):
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--pair-reviews", action="store_true",
                        help="Use split/pair_reviews.jsonl for uncertain windows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train = list(rows(args.train))
    args.output.mkdir(parents=True)
    keys = set()
    report = {"schema_version": "ta_glm52_weak_eval_v1",
              "label_status": "model_weak_supervision_not_human_gold",
              "model": "glm-5.2", "splits": {},
              "input_sha256": {"train": sha256(args.train)}}
    trajectories = set()
    for split in ("dev", "test"):
        source = args.root / split
        windows_path, reviews_path = source / "windows.jsonl", source / "model_reviews.jsonl"
        windows, reviews = list(rows(windows_path)), list(rows(reviews_path))
        tids = {w["trajectory_id"] for w in windows}
        if trajectories & tids or any(w["split"] != split for w in windows):
            raise ValueError("dev/test trajectory or split violation")
        trajectories.update(tids)
        pair_review_path = source / "pair_reviews.jsonl"
        accepted, pairs, queue, keys, stats = (
            finalize_pairs(windows, reviews, list(rows(pair_review_path)), train, keys)
            if args.pair_reviews else finalize(windows, reviews, train, keys))
        out = args.output / split
        out.mkdir()
        write(out / "window_labels.jsonl", accepted)
        write(out / "pairs.jsonl", pairs)
        write(out / "review_queue.jsonl", queue)
        stats["trajectories"] = len({x["trajectory_id"] for x in accepted})
        report["splits"][split] = stats
        report["input_sha256"][split] = {"windows": sha256(windows_path),
                                         "reviews": sha256(reviews_path)}
        if args.pair_reviews:
            report["input_sha256"][split]["pair_reviews"] = sha256(pair_review_path)
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
