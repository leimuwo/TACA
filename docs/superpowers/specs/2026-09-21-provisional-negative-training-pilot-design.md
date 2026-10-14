# Provisional Negative Generation And Bi-Encoder Pilot Design

## Decision

Run a clearly isolated feasibility experiment using the 609 automatically paired
SWE Intent--Action candidates as provisional positives. Use the configured
Inspire inference endpoint to propose hard-negative cases, apply strict local
structural and provenance validation, establish lexical and frozen-embedding
baselines, and fine-tune a small shared bi-encoder. No output from this pilot is
Gold data or a publishable final result; the full experiment must later be
repeated from human-reviewed `direct_match` positives.

## Goal

Answer two feasibility questions quickly:

1. Can deterministic parameter- and tool-sensitive negatives be constructed at
   useful coverage from the current SWE candidate pool?
2. Does a small fine-tuned shared encoder improve Intent--Action retrieval and
   parameter discrimination over lexical and frozen-embedding baselines?

The pilot consumes the current 609 candidates only. It does not wait for the
full 500-trajectory Thought-to-Intent extraction.

## Non-Gold Boundary

Every record and artifact must carry all of these fields:

```json
{
  "supervision_status": "provisional_auto_pair",
  "experiment_tier": "feasibility_only",
  "human_reviewed": false
}
```

The words `gold`, `confirmed`, and `direct_match` must not describe these
relations. Pilot outputs use a separate versioned directory and cannot be read
by the formal Phase 1 `finalize` path. Training and evaluation reports must
repeat that metrics may contain positive-label noise and false negatives.

## Inputs

Primary input:

```text
$TA_DATA_ROOT/SWE-agent-trajectories/test_data/
  intent_action_phase1_v1/annotation_candidates.jsonl
```

The builder verifies:

- exactly 609 input candidates for the current snapshot;
- every candidate source is SWE;
- each candidate contains one Intent and one short normalized Action;
- candidate IDs are unique;
- the source build manifest and candidate file hashes match the recorded pilot
  manifest.

The count is recorded, not permanently built into reusable library code. A
future pilot dataset version may use a different count.

## Architecture

```text
609 provisional Intent--Action pairs
                |
                v
trajectory/template-aware split
                |
                v
per-split Action index
                |
                v
deterministic hard-negative generation
                |
                v
provisional training records + audit manifest
                |
       +--------+---------+
       |        |         |
       v        v         v
    TF-IDF    BM25    frozen encoder
                           |
                           v
                  shared bi-encoder fine-tune
                           |
                           v
                  retrieval + pairwise report
```

The implementation is divided into four responsibilities:

1. `provisional_negatives.py`: Action field parsing, per-split indexing,
   candidate parsing, risk filtering, deterministic selection, and dataset
   serialization.
2. `generate_provisional_negatives.py`: inference requests, strict response
   validation, resume state, manifest reconciliation, atomic publication, and
   dry-run reporting.
3. `retrieval_baselines.py`: TF-IDF, BM25, frozen-encoder scoring, and shared
   retrieval metrics.
4. `train_biencoder.py`: dependency checks, shared-encoder training,
   checkpointing, early stopping, and final evaluation.

## Split Policy

The existing trajectory/template-aware splitter assigns provisional positives
to train, validation, and test before any negative is generated. Target
proportions remain 70/15/15. Every negative Action and substituted value must be
drawn from the same split as its parent.

No candidate Action, parameter value, trajectory context, or hard-negative
record may cross split boundaries. Validation and test assignments are frozen
inside the pilot manifest after the first successful build.

## Inference Contract

The configured endpoint defaults to:

```text
https://qjkpcombh9jkcecah8jdkgd5d5beqege.openapi-qb-ai.sii.edu.cn
```

It is treated as a model-specific endpoint, so the request path and optional
model name remain configurable. The API key is read only from `INF_API_KEY`.
The client bypasses inherited HTTP proxies for this host because the current
proxy path cannot complete TLS negotiation while direct access reaches the
Inspire authentication gateway.

For each provisional positive, the model receives the Intent, positive Action,
allowed negative categories, and a bounded list of eligible same-split and
same-trajectory observed Actions. It returns strict JSON containing three to
five proposed cases. Each proposal includes `negative_type`, `serialized_action`, a
short semantic reason, and an optional observed source candidate ID.

The model proposal is never trusted as a label. Local validation must confirm
the claimed category, parse and normalize the Action, enforce split provenance,
reject duplicates and matching Actions, and attach the permanent provenance
value `provisional_llm_generated`. Invalid proposals remain in an audit file
and never enter training.

The endpoint protocol is detected with one preflight request before batch
generation. Authentication, unsupported request schema, or repeated transport
failures stop the run instead of being retried for every record.

## Negative Types

### Same Tool, Wrong Parameter

This is the highest-priority and lowest-risk category. The model proposes an
Action, and the validator parses known normalized fields and verifies that
exactly one important value differs from the positive.

Examples:

```text
[ACTION] goto(line=550)
→ [ACTION] goto(line=450)

[ACTION] open(path="hy/models.py")
→ [ACTION] open(path="hy/reader.py")
```

Supported v1 fields are:

- `goto.line`;
- `open.path`;
- `search_dir.query` and `search_dir.path`, changing one at a time;
- `python.command` when the command is a single short script invocation;
- `rm.path`;
- `create.path`;
- generic `args` only when both source and replacement contain one scalar
  argument.

The replacement must preserve field type and must produce a normalized Action
different from the positive. Observed same-split values are preferred and are
marked `observed_value`; model-created scalar values are marked
`synthetic_value`. Multi-field
mutations, invented strings, code payloads, and invalid round trips are rejected.

### Wrong Tool, Same Object

This category accepts a proposed Action with a different tool but the same exact
normalized target. The Action may be selected from the supplied same-split pool
or proposed by the model, but the target must be preserved exactly.

Example:

```text
Intent:   Run test_symbol.py.
Positive: [ACTION] python(command="test_symbol.py")
Negative: [ACTION] open(path="test_symbol.py")
```

The target must be an exact filename, path basename, quoted search object, or
single command argument after normalization. Fuzzy semantic matching is not
used. Tool pairs with ambiguous equivalence are excluded.

### Strict Same-Trajectory Unrelated

This category has the greatest false-negative risk. The model must select an
observed Action by candidate ID from the supplied same-trajectory pool; free-form
generation is not allowed for this category. It is used only when all of
the following hold:

- the Action is an observed eligible short Action in the same trajectory and
  split;
- it is at least two steps away from the parent;
- its tool differs from the positive tool;
- it shares no exact path, filename, query, line number, or identifier token
  with the Intent or positive Action;
- it is not an `open`, `search`, `find`, `goto`, or navigation Action that could
  reasonably support the Intent;
- it is not adjacent to another known Action involving the positive target;
- it is not duplicated elsewhere in the selected negatives.

In v1 this leaves mainly obviously different execution, cleanup, or inspection
Actions. The category contributes at most one negative per parent. Every such
record retains a `higher_false_negative_risk` flag for audit, despite passing
the deterministic gate.

## Selection Policy

The generator proposes at most 12 candidates and retains at most 5 negatives
per provisional positive. Stable hashes with seed `20260921` break ties.

Selection priority is:

1. up to three same-tool wrong-parameter negatives;
2. up to one wrong-tool same-object negative;
3. up to one strict same-trajectory unrelated negative.

The target is 3--5 explicit negatives, not a quota. A positive remains usable
with fewer negatives because in-batch negatives are also available. The
generator never relaxes a safety rule merely to reach the target count.

## Dataset Schema

Each grouped training record has this shape:

```json
{
  "candidate_id": "swe:CASE-064:T10-I1",
  "trajectory_id": "CASE-064",
  "split": "train",
  "intent_text": "Go to line 550.",
  "intent_input": "[INTENT] Go to line 550.",
  "positive_action": {
    "raw": "goto 550",
    "serialized": "[ACTION] goto(line=550)"
  },
  "negative_actions": [
    {
      "negative_id": "swe:CASE-064:T10-I1:N1",
      "serialized": "[ACTION] goto(line=450)",
      "negative_type": "same_tool_wrong_parameter",
      "source_candidate_id": "swe:CASE-064:T8-I1",
      "changed_fields": ["line"],
      "risk_flags": []
    }
  ],
  "supervision_status": "provisional_auto_pair",
  "experiment_tier": "feasibility_only",
  "human_reviewed": false,
  "source": "swe"
}
```

## Baselines

All methods use the same frozen splits and candidate Action pools.

### TF-IDF

Use scikit-learn word and character n-gram TF-IDF with cosine similarity. This
tests whether filename, command, and parameter overlap already solve the task.

### BM25

Use `rank-bm25` over serialized Action tokens. This is a sparse lexical ranking
baseline, not a learned model.

### Frozen Encoder

Use `BAAI/bge-small-en-v1.5` as the default CPU-capable pilot encoder. Inputs
retain `[INTENT]` and `[ACTION]` prefixes. Mean pooling and normalized cosine
similarity are used through Sentence Transformers.

## Bi-Encoder Training

The model is a shared-parameter bi-encoder initialized from
`BAAI/bge-small-en-v1.5`. The same encoder processes Intent and Action text.

Initial pilot configuration:

```text
Max sequence length: 192
Similarity: cosine
Temperature scale: 20.0
Learning rate: 2e-5
Epochs: maximum 3
Effective batch size: 32
Warmup ratio: 0.1
Weight decay: 0.01
Seed: 20260921
Early stopping: validation MRR, patience 1 epoch
```

Sentence Transformers `MultipleNegativesRankingLoss` consumes one anchor,
positive, and available explicit hard negatives per row. Other positives in the
batch serve as in-batch negatives. Training rows with no explicit negative are
still valid positive pairs.

The initial run is CPU-compatible because no GPU is currently visible. The CLI
supports `--device cuda` when a GPU becomes available. A one-batch smoke run is
mandatory before a full epoch.

## Metrics

Retrieval metrics on validation and test:

- Recall@1;
- Recall@3;
- MRR;
- Mean Rank.

Hard-negative metrics:

- pairwise accuracy;
- mean and median similarity margin
  `score(intent, positive) - score(intent, negative)`;
- accuracy by negative type;
- coverage: positives with 0, 1, 2, 3, 4, or 5 explicit negatives.

Reports compare TF-IDF, BM25, frozen BGE, and fine-tuned BGE. Test metrics are
computed once after selecting the checkpoint by validation MRR.

Because labels are provisional, the report title and every result summary must
state `FEASIBILITY ONLY — NOT GOLD EVALUATION`.

## Dependencies And Environment

Training dependencies are declared as a `training` optional dependency group in
`pyproject.toml` with bounded minimum versions for:

- `numpy`;
- `scikit-learn`;
- `rank-bm25`;
- `torch`;
- `transformers`;
- `sentence-transformers`.

They are installed into the repository-local ignored `.venv`, not implicitly
into the base environment. Model downloads use the normal Hugging Face cache;
checkpoints and generated data remain outside Git.

## Outputs

Default external dataset directory:

```text
$TA_DATA_ROOT/SWE-agent-trajectories/test_data/
  provisional_intent_action_pilot_v1/
```

It contains:

- `negative_candidates.jsonl`;
- `negative_responses.jsonl`;
- `negative_failures.jsonl`;
- `provisional_training_records.jsonl`;
- `splits/train.jsonl`;
- `splits/validation.jsonl`;
- `splits/test.jsonl`;
- `pilot_manifest.json`;
- `negative_audit.json`.

Default external experiment directory:

```text
$TA_DATA_ROOT/SWE-agent-trajectories/experiments/
  provisional_biencoder_pilot_v1/
```

It contains:

- baseline metrics;
- training configuration and logs;
- checkpoint metadata;
- the selected model checkpoint;
- validation and test predictions;
- final feasibility report.

## Failure Handling

Dataset building fails closed on duplicate IDs, malformed Actions, cross-split
provenance, input hash mismatches, or missing provisional watermarks. Artifacts
are written to a temporary directory and atomically published.

Generation supports interruption and resume. Cache reuse requires the same
candidate ID, prompt hash, endpoint, request schema, model setting, and source
manifest hash. The API key and authorization headers are never persisted.

Training stops with an actionable error when dependencies, input manifests, or
model files are unavailable. Checkpoints are written after each epoch, so an
interrupted CPU run can resume. NaN loss, empty splits, or metric reconciliation
failures stop training and preserve diagnostic logs.

## Testing

Unit tests cover:

- parsing each supported normalized Action field;
- single-field substitutions and rejection of multi-field mutations;
- exact-target wrong-tool lookup;
- strict same-trajectory exclusions;
- split isolation, deterministic IDs, ordering, caps, and deduplication;
- mandatory provisional watermarks;
- TF-IDF, BM25, retrieval metrics, and pairwise metrics on hand-checked fixtures;
- dependency errors and one-batch training smoke behavior without downloading a
  model in ordinary unit tests.

An integration test builds a small provisional dataset end to end. Real-data
verification builds all 609 records, parses every artifact, reports negative
coverage, and checks that no Kimi rows or cross-split sources appear.

## Acceptance Criteria

The pilot is ready to run when:

- all 609 current input candidates are represented exactly once;
- every output row is marked provisional and non-human-reviewed;
- Kimi row count is zero;
- trajectory and template overlap across splits is zero;
- every negative is distinct from its positive and derived from the same split;
- no positive has more than five explicit negatives;
- generation is byte-stable for identical inputs and seed;
- the one-batch model smoke run completes;
- all four evaluation methods emit the same metric schema;
- the final report is visibly labeled feasibility-only.

The pilot is considered informative, not successful, if the fine-tuned model
fails to beat frozen BGE. The result is still useful for deciding whether data
quality, negative construction, or model capacity is the limiting factor.

## Transition To Gold Training

After human review, the same interfaces are reused with these mandatory
changes:

- input changes from provisional candidates to finalized `direct_match`
  positives;
- strict LLM or human validation is added for risky negatives;
- provisional watermarks are replaced with Gold provenance;
- validation and test relations are manually confirmed and frozen;
- models are retrained from the base checkpoint rather than continued from the
  provisional model.

The provisional checkpoint must never initialize the formal Gold experiment,
because it may encode noise from automatically paired relations.
