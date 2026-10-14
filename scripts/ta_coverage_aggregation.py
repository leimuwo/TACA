"""Derive a window label from due-now Intent and pair coverage decisions."""
from __future__ import annotations


def aggregate_ta_label(window, intent_decisions, pair_covered):
    due = [intent["intent_id"] for intent in window["intents"]
           if intent_decisions[intent["intent_id"]]["due_now"]]
    if not due or any(not intent_decisions[intent_id]["assessable"] for intent_id in due):
        return "unassessable"
    covered = {pair["intent_id"] for pair in window["pairs"]
               if pair_covered[pair["pair_id"]]}
    count = sum(intent_id in covered for intent_id in due)
    return "faithful" if count == len(due) else "partial" if count else "not_executed"
