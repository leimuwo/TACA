# Provisional Negative And Training Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate validated LLM-proposed hard negatives for 609 provisional SWE pairs, run retrieval baselines, and train a feasibility-only shared bi-encoder.

**Architecture:** A standard-library negative module validates model proposals independently from the network runner. A CLI performs preflight, resumable generation, atomic dataset publication, and split reconciliation. Separate baseline and training modules consume the frozen provisional dataset and emit uniformly shaped feasibility metrics.

**Tech Stack:** Python 3.10+, standard library, NumPy, scikit-learn, rank-bm25, PyTorch, Transformers, Sentence Transformers.

**Spec:** `docs/superpowers/specs/2026-09-21-provisional-negative-training-pilot-design.md`

## Global Constraints

- Every relation is marked `provisional_auto_pair`, `feasibility_only`, and `human_reviewed: false`.
- Model proposals never bypass local structural, semantic-category, provenance, deduplication, or split checks.
- Kimi input and output counts are zero.
- Negative sources never cross train, validation, and test boundaries.
- No record contains more than five explicit negatives.
- API credentials and authorization headers are never persisted or logged.
- Generated data, checkpoints, and model weights remain outside Git.

---

### Task 1: Action parsing and proposal validation

**Files:**
- Create: `src/thought_action_retrieval/matching/provisional_negatives.py`
- Create: `tests/test_provisional_negatives.py`

**Interfaces:**
- Produces: `parse_serialized_action(text: str) -> ParsedAction`
- Produces: `validate_negative_proposal(parent, proposal, indexes) -> ValidatedNegative`
- Produces: `build_split_indexes(rows) -> SplitIndexes`

- [ ] Write failing table-driven tests for known Action fields, exact one-field parameter changes, wrong-tool same-target cases, and invalid/matching/multi-field cases.
- [ ] Run `PYTHONPATH=src python -m unittest tests.test_provisional_negatives.ActionParsingTests tests.test_provisional_negatives.ProposalValidationTests -v` and verify the missing module failure.
- [ ] Implement immutable parsed/validated dataclasses and deterministic parsing.
- [ ] Implement category-specific validation and mandatory provisional provenance.
- [ ] Run the focused tests and verify pass.

### Task 2: Split indexes and prompt construction

**Files:**
- Modify: `src/thought_action_retrieval/matching/provisional_negatives.py`
- Modify: `tests/test_provisional_negatives.py`
- Create: `prompts/provisional_negative_generation.md`

**Interfaces:**
- Produces: `prepare_provisional_rows(candidates, seed=20260921)`
- Produces: `build_generation_request(parent, indexes, max_pool_actions=24)`
- Produces: stable request and negative IDs.

- [ ] Write failing tests proving zero split overlap, same-split pools, bounded same-trajectory context, stable IDs, and exclusion of the positive Action.
- [ ] Implement split preparation by reusing `split_by_group` and build indexes by tool, target, trajectory, and serialized Action.
- [ ] Implement the reviewed strict-JSON generation prompt and prompt hash.
- [ ] Run focused tests and verify deterministic output under reversed input order.

### Task 3: Inference client and resumable generation CLI

**Files:**
- Create: `src/thought_action_retrieval/inference/__init__.py`
- Create: `src/thought_action_retrieval/inference/client.py`
- Create: `scripts/generate_provisional_negatives.py`
- Create: `tests/test_inference_client.py`
- Modify: `.env.example`
- Modify: `README.md`

**Interfaces:**
- Produces: `InferenceClient.preflight()` and `InferenceClient.generate_json()`
- Produces CLI subcommands `prepare`, `generate`, and `finalize`.

- [ ] Write failing tests with a local HTTP server for root-path OpenAI-style responses, HTTP errors, malformed JSON, authentication fail-fast, proxy bypass, and secret-free diagnostics.
- [ ] Implement configurable endpoint, path, optional model, timeout, retries, and no-proxy handling.
- [ ] Implement atomic request/result state, hash-qualified resume, and global consecutive-failure circuit breaker.
- [ ] Implement `prepare` without credentials, `generate` requiring `INF_API_KEY`, and `finalize` refusing incomplete or inconsistent inputs.
- [ ] Run integration tests and a real-data prepare dry run over 609 candidates.

### Task 4: Provisional dataset finalization

**Files:**
- Modify: `src/thought_action_retrieval/matching/provisional_negatives.py`
- Modify: `scripts/generate_provisional_negatives.py`
- Modify: `tests/test_provisional_negatives.py`

**Interfaces:**
- Produces `provisional_training_records.jsonl`, split JSONL files, `negative_audit.json`, and `pilot_manifest.json`.

- [ ] Write failing tests for selection priority, five-negative cap, byte stability, missing watermark rejection, and atomic rollback.
- [ ] Implement deterministic selection and audit/rejection aggregation.
- [ ] Build the real 609-row prepared dataset and verify no Kimi or cross-split records.
- [ ] After online generation is available, finalize validated negative records and report coverage by type.

### Task 5: Sparse baselines and shared metrics

**Files:**
- Create: `src/thought_action_retrieval/training/metrics.py`
- Create: `src/thought_action_retrieval/training/baselines.py`
- Create: `scripts/run_retrieval_baselines.py`
- Create: `tests/test_retrieval_metrics.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces common Recall@1, Recall@3, MRR, Mean Rank, pairwise accuracy, and margin metrics.
- Produces TF-IDF and BM25 results with identical report schema.

- [ ] Write failing hand-calculated metric tests and tiny corpus ranking tests.
- [ ] Implement dependency-independent metrics, then TF-IDF and BM25 adapters.
- [ ] Add a `training` optional dependency group and install it into `.venv`.
- [ ] Run sparse baselines on frozen validation/test splits and save feasibility-only reports.

### Task 6: Frozen encoder and bi-encoder training

**Files:**
- Create: `src/thought_action_retrieval/training/biencoder.py`
- Create: `scripts/train_biencoder.py`
- Create: `tests/test_biencoder.py`

**Interfaces:**
- Produces frozen BGE evaluation, one-batch smoke training, resumable epochs, checkpoint selection, and final report.

- [ ] Write failing tests around dependency diagnostics, dataset collation, configuration validation, checkpoint metadata, and report watermarks without downloading a model.
- [ ] Implement lazy imports so ordinary tests do not require training dependencies.
- [ ] Download `BAAI/bge-small-en-v1.5`, run frozen evaluation, and run one-batch CPU smoke training.
- [ ] Train up to three epochs with validation-MRR early stopping, then evaluate the selected checkpoint once on test.
- [ ] Compare TF-IDF, BM25, frozen BGE, and fine-tuned BGE in one feasibility-only report.

### Task 7: Final verification and handoff

**Files:**
- Modify: `docs/project-progress-2026-09-21.md`
- Modify: `README.md`

- [ ] Run the full offline test suite, compileall, repository audit, and diff check.
- [ ] Reconcile all dataset and experiment manifest counts and hashes.
- [ ] Confirm every output and report is visibly provisional and feasibility-only.
- [ ] Commit implementation and preserve the feature worktree for review.
