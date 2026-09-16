# Intent–Action Phase 1 Data Design

## Goal

Build an auditable first-stage dataset for training a shared bi-encoder on one atomic execution Intent mapped to one short raw Action. The dataset is derived from the existing `thought_intent` annotations, uses SWE trajectories as the primary source, and never treats an automatically paired Intent–Action example as confirmed training gold before human review.

## Scope

Phase 0–1 includes:

- auditing Thought → Intent extraction outputs;
- selecting SWE steps with exactly one execution Intent and exactly one Action;
- filtering out long code/text payloads and unsuitable composite Actions;
- deterministically serializing short Actions without LLM paraphrasing;
- generating a human annotation queue;
- importing reviewed labels;
- exporting only `direct_match` pairs as v1 positive examples;
- retaining `no_match` and `unfulfilled` for evaluation;
- splitting by trajectory, with near-duplicate task templates grouped together.

Phase 0–1 does not include model training, Kimi multi-positive training, collective Action-set fulfillment, or automatic promotion of candidates to gold labels.

## Inputs

- `data/SWE-agent-trajectories/test_data/thought_intent/swe__*.json`
- `data/SWE-agent-trajectories/test_data/thought_intent/kimi__*.json` for audit statistics only
- `data/SWE-agent-trajectories/test_data/thought_intent/manifest.json`

Each step supplies the observable Thought, extracted execution Intents, raw current-step Actions, and trajectory metadata.

## Candidate Rules

A SWE step is a one-to-one candidate when:

- the step status is `success`;
- it contains exactly one extracted execution Intent;
- it contains exactly one non-empty Action;
- the Action passes the short-Action filter.

The filter rejects:

- `edit` Actions unconditionally in v1;
- code/text payloads over 64 approximate tokens;
- total serialized Actions over 128 approximate tokens;
- heredocs, multiline scripts, or `end_of_edit` payloads;
- large patches and long document writes;
- truncated or unparseable Actions.

Short commands such as `search_dir`, `open`, `goto`, `scroll_down`, `python test.py`, `rm file`, and `create file` remain candidates when their payloads fit the limits.

Token estimates use a deterministic conservative approximation of `ceil(character_count / 4)`, so dataset construction does not depend on a tokenizer package.

## Action Serialization

Training text uses an explicit type prefix:

- Intent: `[INTENT] <intent_text>`
- Action: `[ACTION] <normalized_action>`

SWE Actions are normalized into stable function-like forms when parsing is reliable:

- `search_dir "class Symbol" hy` → `[ACTION] search_dir(query="class Symbol", path="hy")`
- `open hy/models.py` → `[ACTION] open(path="hy/models.py")`
- `goto 550` → `[ACTION] goto(line=550)`
- `scroll_down` → `[ACTION] scroll_down()`
- `python test_symbol.py` → `[ACTION] python(command="test_symbol.py")`
- `rm test_symbol.py` → `[ACTION] rm(path="test_symbol.py")`
- `create test_symbol.py` → `[ACTION] create(path="test_symbol.py")`

Unknown but short commands are retained in a deterministic generic representation with the command name and raw argument string. The raw Action is always preserved next to the normalized form.

## Annotation Queue

Every candidate is exported as one JSONL row containing:

- stable `candidate_id`;
- trajectory and thought-step identifiers;
- original Thought and source quote;
- `intent_id`, raw `intent_text`, and prefixed Intent text;
- raw and serialized Action;
- filter diagnostics;
- nearby previous and next Actions for human context;
- empty annotation fields.

Allowed human labels are:

- `direct_match`;
- `partial_match`;
- `no_match`;
- `unfulfilled`;
- `ambiguous`.

The exporter fails closed: a row without an allowed human label never enters the training positives.

## Unfulfilled Intents

Steps with one explicit execution Intent and no Action are exported to a separate unfulfilled review queue. They are not used by InfoNCE training. After human confirmation, they are retained for threshold calibration and fulfillment evaluation.

The current data format records current-step Actions. It does not by itself prove whether a future-plan Intent is fulfilled in a later step. Therefore, later-step matching is not automatically labeled in Phase 1.

## Split Policy

All rows from the same trajectory go to the same split. A normalized task-template key additionally groups near-duplicate cases before deterministic assignment.

Target proportions are 70% train, 15% validation, and 15% test by group. Assignment uses a fixed seed and a stable hash, then greedily balances example counts without splitting groups. The resulting manifest records group and example counts and asserts zero trajectory overlap.

The manually reviewed test set is frozen after export. Any later dataset rebuild must write a new versioned directory rather than silently overwriting it.

## Outputs

Default external output directory below `TA_DATA_ROOT`:

`SWE-agent-trajectories/test_data/intent_action_phase1_v1/`

Files:

- `audit_summary.json`
- `annotation_candidates.jsonl`
- `unfulfilled_candidates.jsonl`
- `annotation_template.csv`
- `reviewed_annotations.jsonl` supplied by a human
- `positives.jsonl`
- `evaluation_relations.jsonl`
- `splits/train.jsonl`
- `splits/validation.jsonl`
- `splits/test.jsonl`
- `split_manifest.json`

## Acceptance Criteria

- every positive has exactly one Intent and one short Action;
- no v1 positive contains `edit`, `end_of_edit`, multiline code, or a payload over the configured limits;
- every training positive has the human label `direct_match`;
- all output IDs are stable and unique;
- raw and normalized Actions are both retained;
- no trajectory or task-template group crosses splits;
- audit counts reconcile with source records;
- tests cover filtering, serialization, label validation, and split isolation.
