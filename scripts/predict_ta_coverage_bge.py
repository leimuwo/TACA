#!/usr/bin/env python3
"""Independent BGE similarity/threshold baseline for TA window labels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from evaluate_bge_clawgym import embed, load_encoder, rows

MODEL = Path(os.environ.get("TA_MODEL_ROOT", "../model_hub")) / Path("bge-m3")


def aggregate(window, scores, threshold):
    due = [item for item in window["intents"] if item["temporal_scope"] == "current"]
    if not due:
        return "unassessable"
    covered = sum(any(scores.get(pair["pair_id"], -1) >= threshold for pair in window["pairs"]
                      if pair["intent_id"] == intent["intent_id"]) for intent in due)
    return "faithful" if covered == len(due) else "partial" if covered else "not_executed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--threshold", type=float, required=True,
                        help="Fixed externally or calibrated on separate dev Gold; never tune on test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    windows = list(rows(args.windows))
    intents = {x["intent_id"]: x["text"] for w in windows for x in w["intents"]}
    actions = {x["action_id"]: x["action_text"] for w in windows for x in w["actions"]}
    torch, tokenizer, model, device = load_encoder(args.model, args.device)
    intent_ids, action_ids = list(intents), list(actions)
    q = embed(["[INTENT] " + intents[x] for x in intent_ids], torch=torch, tokenizer=tokenizer,
              model=model, device=device, batch_size=args.batch_size, max_length=256)
    a = embed([actions[x] for x in action_ids], torch=torch, tokenizer=tokenizer,
              model=model, device=device, batch_size=args.batch_size, max_length=512)
    qi, ai = {id: n for n, id in enumerate(intent_ids)}, {id: n for n, id in enumerate(action_ids)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for window in windows:
            scores = {p["pair_id"]: float(q[qi[p["intent_id"]]] @ a[ai[p["action_id"]]])
                      for p in window["pairs"]}
            stream.write(json.dumps({"window_id": window["window_id"],
                                     "predicted_label": aggregate(window, scores, args.threshold),
                                     "pair_scores": scores, "threshold": args.threshold,
                                     "model": str(args.model), "baseline": "normalized_cls_cosine"}) + "\n")
    print(json.dumps({"windows": len(windows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
