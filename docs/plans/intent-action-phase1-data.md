# Intent–Action Phase 1 Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable SWE one-Intent–one-short-Action candidate pipeline, human-review interface, and trajectory-isolated dataset exporter.

**Architecture:** A focused Python module owns Action parsing/filtering, candidate extraction, annotation validation, and group-aware splitting. A thin CLI reads the existing `thought_intent` files and writes versioned JSONL/CSV artifacts. Automated tests use synthetic trajectories plus the real Phase 0 corpus for reconciliation checks.

**Tech Stack:** Python 3 standard library, `unittest`, JSONL/CSV artifacts.

**Spec:** `docs/designs/intent-action-phase1-data-design.md`

## Global Constraints

- SWE is the only v1 positive-training source; Kimi is audit-only.
- A v1 candidate has exactly one execution Intent and one Action.
- `edit`, multiline scripts, heredocs, `end_of_edit`, and oversized payloads are excluded.
- Action serialization is deterministic and never uses an LLM paraphrase.
- Only human-reviewed `direct_match` rows may enter `positives.jsonl`.
- Splits are isolated by trajectory and normalized task-template group.
- Existing `thought_intent` files are read-only.

---

### Task 1: Short Action normalization and filtering

**Files:**
- Create: `src/thought_action_retrieval/matching/phase1_dataset.py`
- Test: `tests/test_intent_action_phase1.py`

**Interfaces:**
- Produces: `serialize_swe_action(raw: str) -> ActionNormalization`
- Produces: `assess_short_action(raw: str, max_action_tokens: int = 128, max_payload_tokens: int = 64) -> ActionAssessment`

- [ ] **Step 1: Write failing unit tests** covering `search_dir`, `open`, `goto`, `scroll_down`, `python`, `rm`, and `create`, plus rejection of `edit`, `end_of_edit`, heredocs, multiline scripts, and payloads over the limits.
- [ ] **Step 2: Run** `python -m unittest tests.test_intent_action_phase1.ActionNormalizationTests -v` and verify failure because the module is absent.
- [ ] **Step 3: Implement immutable dataclasses** carrying raw text, command name, normalized Action text, token estimates, eligibility, and exact exclusion reasons.
- [ ] **Step 4: Implement deterministic parsing** with `shlex.split`; use named fields for known SWE commands and a generic `command(args="...")` fallback for unknown short commands.
- [ ] **Step 5: Run the normalization tests** and verify all pass.

### Task 2: Candidate extraction and Phase 0 audit

**Files:**
- Modify: `src/thought_action_retrieval/matching/phase1_dataset.py`
- Modify: `tests/test_intent_action_phase1.py`

**Interfaces:**
- Consumes: `assess_short_action`
- Produces: `extract_candidates(records: Iterable[dict], config: BuildConfig) -> BuildResult`
- Produces: `BuildResult.audit_summary`, `BuildResult.annotation_candidates`, and `BuildResult.unfulfilled_candidates`

- [ ] **Step 1: Write failing tests** with synthetic SWE and Kimi records proving that only SWE one-intent–one-action short steps enter the annotation queue, while Kimi and rejected Actions appear only in audit counts.
- [ ] **Step 2: Test stable IDs** using the literal form `<source>:<trajectory_id>:T<step>-I<sequence>` and verify duplicate IDs raise an error.
- [ ] **Step 3: Implement candidate rows** containing Thought, source quote, raw/serialized inputs, neighboring Actions, filter diagnostics, and empty human-label fields.
- [ ] **Step 4: Implement unfulfilled rows** for exactly one Intent and zero current-step Actions without treating future fulfillment as known.
- [ ] **Step 5: Add real-corpus reconciliation test** asserting 25 SWE and 25 Kimi input trajectories, 1,166 steps, and internally consistent audit totals without hard-coding a final eligible count.
- [ ] **Step 6: Run** `python -m unittest tests.test_intent_action_phase1.CandidateExtractionTests -v` and verify pass.

### Task 3: Human annotation import and gold export

**Files:**
- Modify: `src/thought_action_retrieval/matching/phase1_dataset.py`
- Modify: `tests/test_intent_action_phase1.py`

**Interfaces:**
- Produces: `validate_reviewed_annotations(candidates, reviewed_rows) -> ReviewResult`
- Produces: `export_gold_relations(review_result) -> tuple[list[dict], list[dict]]`

- [ ] **Step 1: Write failing tests** proving unknown IDs, duplicate reviews, missing labels, and unsupported labels are rejected.
- [ ] **Step 2: Test fail-closed behavior**: unlabeled candidates and `partial_match`/`ambiguous` never enter positives.
- [ ] **Step 3: Implement allowed labels** `direct_match`, `partial_match`, `no_match`, `unfulfilled`, and `ambiguous`, preserving annotator and notes.
- [ ] **Step 4: Export positives** only for `direct_match`, with exact v1 schema fields `trajectory_id`, `intent_id`, `intent_text`, `positive_action`, `relation_type`, and `source` plus provenance.
- [ ] **Step 5: Export evaluation relations** for reviewed `direct_match`, `no_match`, and `unfulfilled` rows.
- [ ] **Step 6: Run annotation tests** and verify pass.

### Task 4: Group-aware dataset splitting

**Files:**
- Modify: `src/thought_action_retrieval/matching/phase1_dataset.py`
- Modify: `tests/test_intent_action_phase1.py`

**Interfaces:**
- Produces: `task_template_key(task: str) -> str`
- Produces: `split_by_group(rows: list[dict], seed: int = 20260916) -> SplitResult`

- [ ] **Step 1: Write failing tests** proving all rows from one trajectory remain together and near-duplicate task templates such as line-number variants remain together.
- [ ] **Step 2: Implement task-template normalization** by lowercasing, collapsing whitespace, replacing quoted strings, integers, UUIDs, paths, and issue-specific identifiers with stable placeholders.
- [ ] **Step 3: Implement deterministic grouped assignment** targeting 70/15/15 example proportions while never splitting a group.
- [ ] **Step 4: Add manifest assertions** for zero trajectory overlap, zero template overlap, row counts, group counts, and fixed seed.
- [ ] **Step 5: Run split tests** and verify pass.

### Task 5: CLI artifacts and real Phase 0–1 build

**Files:**
- Create: `scripts/build_intent_action_dataset.py`
- Modify: `tests/test_intent_action_phase1.py`
- Create at runtime below `TA_DATA_ROOT`: `SWE-agent-trajectories/test_data/intent_action_phase1_v1/*`

**Interfaces:**
- Consumes all Task 1–4 functions.
- Produces audit and annotation artifacts in build mode.
- Produces positives, evaluation relations, and splits only when a reviewed annotation file is supplied.

- [ ] **Step 1: Write a failing CLI integration test** using temporary source/output directories and assert exact filenames plus JSONL/CSV row parity.
- [ ] **Step 2: Implement `build` mode** with flags for input directory, output directory, token limits, and deterministic overwrite protection.
- [ ] **Step 3: Implement `finalize` mode** requiring `reviewed_annotations.jsonl`; refuse to emit training splits if any included positive lacks `direct_match` review.
- [ ] **Step 4: Run the full test suite** with `python -m unittest tests.test_intent_action_phase1 tests.test_generate_thought_intents -v`.
- [ ] **Step 5: Run the real build** against `data/SWE-agent-trajectories/test_data/thought_intent` and write the Phase 0 audit, candidate JSONL, unfulfilled JSONL, annotation CSV, and build manifest.
- [ ] **Step 6: Verify artifacts** by parsing every JSON/JSONL file, checking unique IDs, confirming zero eligible `edit` Actions, and reconciling counts with the audit summary.
- [ ] **Step 7: Do not run finalize or model training** until human-reviewed annotations exist.

### Task 6: Phase 2–3 follow-on boundary

**Files:**
- Create after annotation review: `docs/plans/intent-action-retrieval-training.md`

**Interfaces:**
- Consumes frozen `positives.jsonl`, evaluation relations, and split manifest from Task 5.
- Produces TF-IDF, BM25, frozen embedding, and fine-tuned bi-encoder evaluations.

- [ ] **Step 1: Confirm annotation coverage** reaches the experiment minimum and the frozen test set has at least 200 reviewed Intents, or explicitly approve a smaller feasibility run.
- [ ] **Step 2: Write a separate implementation plan** for hard-negative generation, baselines, shared-encoder training, threshold calibration, and retrieval/fulfillment metrics.
- [ ] **Step 3: Keep Kimi multi-positive and collective Action-set modeling outside that first training implementation.**
