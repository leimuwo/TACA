#!/usr/bin/env python3
"""Create the deterministic, machine-validated Stage 1.5 Intent artifact."""
from __future__ import annotations
import argparse, json
from pathlib import Path

SCOPES = {"current", "future", "conditional", "uncertain"}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    rows_out = []
    counts = {"due_now": 0, "not_due": 0, "conditional_unresolved": 0, "uncertain": 0}
    with args.input.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            intents = []
            for intent in row.get("execution_intents", []):
                text = intent.get("intent_text")
                quote = intent.get("source_quote")
                scope = intent.get("temporal_scope")
                valid = isinstance(text, str) and bool(text.strip())
                supported = isinstance(quote, str) and bool(quote) and quote in row.get("thought", "")
                granularity_ok = len(text.split()) >= 2 if valid else False
                scope_ok = scope in SCOPES
                eligibility = {"current":"due_now", "future":"not_due", "conditional":"conditional_unresolved", "uncertain":"uncertain"}.get(scope, "uncertain")
                if not (valid and supported and granularity_ok and scope_ok):
                    eligibility = "uncertain"
                counts[eligibility] += 1
                intents.append({**intent, "intent_valid": valid, "source_supported": supported, "granularity_ok": granularity_ok, "temporal_scope_correct": scope_ok, "match_eligibility": eligibility, "validation_status": "auto_validated", "adjudication_reason": None if valid and supported and granularity_ok and scope_ok else "automatic_validation_failed"})
            rows_out.append({**row, "execution_intents": intents})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"thought_rows": len(rows_out), "intent_counts": counts}, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
