# SWE Hard-Negative Data Design

## Decision

Build Phase 1 explicit hard negatives automatically from SWE data only. The
pipeline generates structurally valid candidate Actions with deterministic
rules, then uses a strict LLM judge to reject false negatives. The LLM does not
freely invent Actions.

Kimi trajectories, multi-positive relations, and collective Action fulfillment
remain outside this phase.

## Goal

For every human-confirmed one-Intent--one-short-Action `direct_match` positive,
produce up to five trustworthy hard negatives for shared bi-encoder training.
The negatives should force the model to distinguish important parameters,
tools, and targets instead of learning only broad tool categories.

The target is three to five explicit negatives per positive when that many pass
all checks. A positive with only one or two trustworthy negatives remains valid.
A positive with no trustworthy explicit negative remains available for
in-batch-negative training.

## Scope And Prerequisites

The pipeline consumes the frozen outputs of the Phase 1 Intent--Action dataset
builder:

- SWE examples only;
- human-confirmed `direct_match` positives only;
- exactly one atomic Intent and one short Action per positive;
- the frozen trajectory/template-aware split assignment;
- raw and normalized Action representations;
- stable trajectory, step, Intent, and Action identifiers.

Negative generation runs after splitting. Candidate pools are constructed
separately inside `train`, `validation`, and `test`; no Action, parameter value,
or contextual record may cross a split boundary.

The pipeline does not use `partial_match`, `no_match`, `unfulfilled`,
`ambiguous`, unreviewed relations, long code Actions, patches, or `edit`
payloads as negative training examples.

## Semantic Boundary

The embedding model learns whether an Action semantically fulfills an Intent.
It does not learn whether that Action occurred at an allowed time.

An Action is therefore not a negative merely because it:

- occurred before the Intent;
- was not selected as the current-step positive;
- appears in another step;
- has not been annotated as a positive.

The following must be rejected as negative candidates:

- any semantically matching Action, regardless of temporal position;
- a possible partial fulfillment;
- a supporting or intermediate Action;
- a reasonable prerequisite for the Intent;
- another confirmed correct Action for the same Intent;
- a candidate whose relationship cannot be determined confidently.

Temporal validity is handled by a later sequence-matching component, not by the
embedding space.

## Architecture And Data Flow

```text
human-confirmed SWE direct_match positives
                    |
                    v
frozen trajectory/template split assignment
                    |
                    v
per-split Action and parameter indexes
                    |
                    v
deterministic candidate generation
                    |
                    v
structural and provenance exclusion checks
                    |
                    v
strict LLM semantic judgment
                    |
                    v
confidence/risk gates and deterministic selection
                    |
                    v
versioned train/validation/test hard-negative artifacts
```

The implementation has four isolated responsibilities:

1. **Indexing:** build per-split indexes by tool, typed argument, normalized
   target, trajectory, and Action ID.
2. **Candidate generation:** propose parameter, wrong-tool, and same-trajectory
   candidates without deciding their semantic label.
3. **Judgment:** call the configured LLM using a frozen prompt and validate its
   structured response.
4. **Finalization:** accept only candidates that pass every deterministic and
   LLM gate, then create immutable manifests and training records.

## Candidate Types

### Parameter Corruption

`parameter_corruption` is the highest-priority category. It keeps the tool and
Action structure fixed while changing exactly one semantically important
argument.

Examples:

```text
Intent:   Go to line 550.
Positive: goto 550
Negative: goto 450

Intent:   Open hy/models.py.
Positive: open hy/models.py
Negative: open hy/reader.py
```

Alternate values come from valid Actions in the same split and must retain the
original argument type. The generator prefers an observed Action with the
desired alternate value. When it reconstructs an Action, it may only substitute
a value drawn from that same-tool, same-field pool; it may not invent arbitrary
values. Exactly one critical field changes, and parsing plus round-trip
normalization must succeed.

### Wrong Tool, Same Object

`wrong_tool_same_object` keeps an observable target such as a path, filename,
query object, or test name while selecting an existing Action that uses a
different tool.

Example:

```text
Intent:   Run test_symbol.py.
Positive: python test_symbol.py
Negative: open test_symbol.py
```

These candidates must be observed Actions from the same split. The pipeline
does not rewrite arguments across incompatible tool schemas. Exact target
overlap is required in v1; fuzzy target inference is deferred.

### Same-Trajectory Unrelated

`same_trajectory_unrelated` selects an existing short Action elsewhere in the
same trajectory. This category is useful but carries the greatest false-negative
risk, so every candidate requires both deterministic exclusion checks and LLM
judgment.

The generator excludes the current positive, duplicate normalized Actions,
known correct Actions for the same Intent, and any Action rejected by the
Phase 1 short-Action filter. Adjacency and temporal direction may be recorded as
metadata but may not determine the semantic label.

## Candidate Ranking And Limits

For each positive, the generator proposes at most 12 unique candidates. It
prioritizes candidates in this order:

1. same-tool single-parameter differences;
2. different-tool Actions with exact target overlap;
3. same-trajectory Actions with weaker lexical overlap.

Within a category, stable hashes and a configured seed break ties so repeated
runs over identical inputs produce identical candidate order.

After judgment, finalization keeps at most five negatives, preferring category
diversity while retaining this priority:

1. one or two `parameter_corruption` negatives;
2. one `wrong_tool_same_object` negative;
3. zero to two `same_trajectory_unrelated` negatives;
4. additional accepted parameter negatives when another category is absent.

The counts are targets, not quotas. The system never lowers its acceptance
threshold or introduces random strings merely to reach three negatives.

## LLM Judgment Contract

The judge receives:

- the self-contained `intent_text`;
- the confirmed positive Action as a semantic reference;
- one candidate Action;
- the candidate type and provenance;
- a small local Thought/Action context window for same-trajectory candidates.

The prompt instructs the judge to evaluate semantic fulfillment independently
of temporal position. String inequality with the positive is not sufficient.
Partial fulfillment, supporting work, intermediate work, reasonable
prerequisites, and uncertain relationships must not be labeled as safe
negatives.

The judge returns strict JSON with exactly these semantic labels:

- `safe_negative`;
- `semantic_match`;
- `partial_or_supporting`;
- `ambiguous`;
- `invalid_candidate`.

Required response fields are:

```json
{
  "label": "safe_negative",
  "negative_type": "wrong_tool_same_object",
  "confidence": 0.97,
  "reason": "The Action opens the file but does not execute it.",
  "risk_flags": []
}
```

The configured temperature is `0`. The model name, base URL, request
parameters, prompt version, and prompt SHA-256 are recorded in the output
manifest. The API key is read only from an environment variable and is never
written to requests captured on disk, output artifacts, logs, or Git.

## Acceptance Gates

A candidate enters the final dataset only when all of the following hold:

- its source is SWE;
- its parent positive is human-confirmed `direct_match`;
- it belongs to the same split as the parent positive;
- its normalized Action is valid, short, and distinct from the positive;
- it is not a known positive for the same Intent ID or exact normalized Intent
  text in the same split;
- it passes category-specific structural validation;
- the judge label is `safe_negative`;
- judge confidence is at least `0.90`;
- `risk_flags` is empty;
- its normalized Action is unique within the training record.

All other candidates remain in the audit artifacts with their rejection reason
but never enter embedding training.

## Training Record Schema

Each output row groups a confirmed positive with its accepted explicit
negatives:

```json
{
  "trajectory_id": "CASE-064",
  "split": "train",
  "intent_id": "T10-I1",
  "intent_text": "Go to line 550.",
  "positive_action": {
    "action_id": "T10-A1",
    "raw": "goto 550",
    "serialized": "[ACTION] goto(line=550)"
  },
  "negative_actions": [
    {
      "negative_id": "CASE-064:T10-I1:N1",
      "raw": "goto 450",
      "serialized": "[ACTION] goto(line=450)",
      "negative_type": "parameter_corruption",
      "candidate_source": "same_split",
      "source_action_id": "T8-A1",
      "changed_fields": ["line"],
      "judge": {
        "label": "safe_negative",
        "confidence": 0.97,
        "reason": "The requested line is 550, not 450.",
        "risk_flags": []
      }
    }
  ],
  "source": "swe"
}
```

Judge explanations and provenance are retained for audit but are not embedding
inputs. Training consumes only prefixed `intent_text`, the serialized positive
Action, and serialized accepted negative Actions.

## Outputs And Versioning

The default external output directory is:

`$TA_DATA_ROOT/SWE-agent-trajectories/test_data/intent_action_hard_negatives_v1/`

It contains:

- `negative_candidates.jsonl`: all structurally valid candidates;
- `negative_judgments.jsonl`: validated LLM responses and rejected responses;
- `hard_negatives.jsonl`: grouped final training records;
- `splits/train.jsonl`;
- `splits/validation.jsonl`;
- `splits/test.jsonl`;
- `negative_manifest.json`: inputs, hashes, prompt/model settings, counts, and
  rejection statistics;
- `failures.jsonl`: exhausted API, transport, and response-validation failures.

Large generated artifacts remain outside Git. Git tracks the prompt, schema,
configuration example, small fixtures, source, tests, and concise manifests or
reports approved for publication.

Validation and test artifacts are frozen after their first accepted build. A
new prompt, model, threshold, source manifest, or generator configuration
requires a new versioned output directory rather than an in-place overwrite.

## Failure Handling And Resume Behavior

Every candidate has a stable ID derived from the input dataset version, split,
Intent ID, candidate type, candidate Action, and generator version. Completed
judgments are cached by candidate ID plus prompt/model/configuration hashes.

The runner is idempotent and supports interruption:

- successful cached judgments are reused only when all hashes match;
- malformed JSON is retried up to three times;
- transient API failures use bounded exponential backoff;
- exhausted failures are recorded and excluded, not treated as negatives;
- partial output is written atomically;
- duplicate IDs, cross-split provenance, or manifest inconsistencies stop the
  build;
- an individual positive with no accepted negatives does not stop the build.

Finalization refuses to run while candidate generation is still active or when
the judgment files fail count, schema, or checksum reconciliation.

## Testing

Unit tests cover:

- typed single-field parameter substitution;
- rejection of multi-field and invalid substitutions;
- exact-target wrong-tool lookup;
- same-trajectory lookup and deterministic exclusions;
- stable candidate IDs, ranking, deduplication, and caps;
- strict judgment-schema validation and every semantic label;
- confidence and risk-flag gates;
- prevention of cross-split candidates;
- cache-key invalidation when prompt, model, or configuration changes;
- retry, exhausted-failure, resume, and atomic-output behavior.

Integration tests use a fake OpenAI-compatible judge and small synthetic SWE
fixtures. A real-data dry run reports candidate and acceptance counts without
modifying frozen artifacts. A sampled quality audit reads 50--100 accepted
pairs and estimates the false-negative rate; this audit measures the automatic
pipeline but is not a per-record admission gate.

## Acceptance Criteria

The implementation is accepted when:

- Kimi input and output counts are exactly zero;
- every parent is a reviewed SWE `direct_match` positive;
- every candidate and substituted value is derived from the same split;
- trajectory and task-template split isolation remains intact;
- final records contain zero duplicate or structurally invalid Actions;
- no record contains more than five explicit negatives;
- every accepted negative has `safe_negative`, confidence at least `0.90`, and
  no risk flags;
- all non-safe and failed judgments are absent from training output;
- output counts and checksums reconcile with the manifest;
- a repeated run with identical inputs and cached judgments is byte-stable;
- the sampled false-negative rate is at most 5 percent.

If the sampled false-negative rate exceeds 5 percent, the dataset is not used
for training. Candidate rules or the judgment prompt must be revised and the
dataset rebuilt under a new version.

## Out Of Scope

This design does not include:

- Kimi negatives or unconfirmed Kimi Action groups;
- multi-positive or collective Action-set training;
- free-form LLM generation of Actions;
- random cross-corpus explicit negatives beyond in-batch negatives;
- temporal fulfillment modeling;
- bi-encoder implementation, loss functions, or model evaluation.

Those concerns belong to later dataset or training designs after the SWE
one-to-one hard-negative pipeline is validated.
