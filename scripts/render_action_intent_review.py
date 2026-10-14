#!/usr/bin/env python3
"""Render model flagged Action–Intent matches as a reviewable Markdown file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error


def indented(value: str) -> str:
    return "\n".join("    " + line for line in value.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--intents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matches = list(read_jsonl(args.matches))
    if any(row.get("review_required") is not True for row in matches):
        raise ValueError("matches contains a record without review_required=true")
    actions = {row["action_id"]: row for row in read_jsonl(args.actions)}
    steps = {
        (row["trajectory_id"], row["thought_step"]): row
        for row in read_jsonl(args.intents)
    }
    labels = Counter(row["label"] for row in matches)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        output.write("# 需要人工审核的 Intent–Action 匹配\n\n")
        output.write(f"总数：{len(matches)}。初步标签分布：")
        output.write("、".join(f"{label} {count}" for label, count in sorted(labels.items())))
        output.write("。以下标签均为模型初判。\n\n")
        for index, match in enumerate(matches, 1):
            action = actions.get(match["action_id"])
            step = steps.get((match["trajectory_id"], match["thought_step"]))
            if action is None or step is None:
                raise ValueError(f"missing Action or Intent step for {match['action_id']}")
            candidates = [
                intent for intent in step.get("execution_intents", [])
                if intent.get("match_eligibility") == "due_now"
            ]
            output.write(f"## {index}. {match['trajectory_id']} / step {match['thought_step']} / {match['action_id']}\n\n")
            output.write(f"- **模型初判**：{match['label']}；置信度 {match['confidence']}\n")
            output.write(f"- **选择的 Intent**：{match.get('selected_intent_id') or '无'}\n")
            output.write(f"- **其他候选**：{', '.join(match.get('alternative_intent_ids') or []) or '无'}\n")
            output.write(f"- **Action 结构**：{action['atomicity']}；额外操作 {action['action_has_extras']}；需语义审核 {action['requires_manual_semantic_review']}\n")
            output.write(f"- **模型理由**：{match.get('reason') or '无'}\n\n")
            output.write("**Thought 原文**\n\n")
            output.write(indented(step.get("thought", "")) + "\n\n")
            output.write("**当前候选 Intent**\n\n")
            if candidates:
                for intent in candidates:
                    output.write(f"- `{intent['intent_id']}`：{intent['intent_text']}\n")
                    output.write(f"  - 原文依据：{intent['source_quote']}\n")
            else:
                output.write("无 `due_now` Intent。\n")
            output.write("\n**完整 raw Action**\n\n")
            output.write(indented(json.dumps(action["action"], ensure_ascii=False, indent=2)) + "\n\n")
            output.write("**人工结论**：待审核\n\n---\n\n")
    print(f"Rendered {len(matches)} cases to {args.output.resolve()}")


if __name__ == "__main__":
    main()
