# Thought–Intent–Action Retrieval

## Scope

This private research repository studies whether an agent's short tool Action
matches an atomic execution Intent extracted from its observable Thought. The
first training target is one atomic Intent mapped to one short Action through a
shared bi-encoder retrieval model.

Only observable trajectory text is used. Hidden chain-of-thought is outside the
project scope.

## Repository Boundary

Source code, tests, prompts, configuration examples, manifests, and small review
samples are stored in Git. Raw trajectories, the full selected trajectory set,
model weights, and bulk experiment outputs remain in an external data pool.

This repository is independent and has no code, history, or runtime dependency
on TACA.

## Setup

Python 3.10 or newer is required.

```bash
python -m pip install -e .
cp .env.example .env
```

Set `TA_DATA_ROOT` in the untracked `.env` or shell environment to the shared
data directory. Intent extraction credentials belong in `INTENT_API_KEY`, and
provisional-negative generation credentials belong in `INF_API_KEY`; never add
either value to tracked files.

The optional training stack is installed into a repository-local venv:

```bash
python -m venv .venv
python -m pip install -e '.[training]'
```

On the current Inspire image, clear the inherited `LD_LIBRARY_PATH` when using
the venv so its CPU PyTorch libraries are not shadowed by system Python 3.12
CUDA libraries.

## Data

See the [data policy](data/README.md) and [data catalog](docs/data-catalog.md).
The registered 500-case SWE snapshot is described by
[its dataset manifest](data/manifests/swe-selected-500/dataset_manifest.json).

## Commands

Run all offline tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Audit staged and tracked files:

```bash
PYTHONPATH=src python scripts/audit_repository.py --root .
```

Select the reproducible 500-case SWE set:

```bash
PYTHONPATH=src python scripts/select_swe_trajectories.py \
  --source "$TA_DATA_ROOT/SWE-agent-trajectories/data_jsonl" \
  --output "$TA_DATA_ROOT/SWE-agent-trajectories/test_data/selected_500_json"
```

Register external selected-set metadata and three small samples:

```bash
PYTHONPATH=src python scripts/import_selected_swe_snapshot.py \
  --source-relative SWE-agent-trajectories/test_data/selected_500_json \
  --raw-shards-relative SWE-agent-trajectories/data_jsonl \
  --sample-id CASE-0001 --sample-id CASE-0251 --sample-id CASE-0500
```

Inspect Thought-to-Intent inputs without making API calls:

```bash
PYTHONPATH=src python scripts/extract_thought_intents.py \
  --swe-count 3 --kimi-count 0 --dry-run
```

Build the Phase 1 SWE annotation queue from completed Thought-to-Intent files:

```bash
PYTHONPATH=src python scripts/build_intent_action_dataset.py build \
  --input-dir "$TA_DATA_ROOT/SWE-agent-trajectories/test_data/thought_intent" \
  --output-dir "$TA_DATA_ROOT/SWE-agent-trajectories/test_data/intent_action_phase1_v1"
```

Review `annotation_template.csv`, save the labels as
`reviewed_annotations.jsonl`, and then export gold relations and
trajectory/template-isolated splits:

```json
{"candidate_id": "swe:CASE-0001:T1-I1", "label": "direct_match", "annotator": "reviewer-name", "notes": "Action directly fulfills the Intent."}
```

The JSONL file contains one JSON object per reviewed candidate. Allowed labels
are `direct_match`, `partial_match`, `no_match`, `unfulfilled`, and
`ambiguous`; `annotator` and `notes` may be empty strings.

```bash
PYTHONPATH=src python scripts/build_intent_action_dataset.py finalize \
  --dataset-dir "$TA_DATA_ROOT/SWE-agent-trajectories/test_data/intent_action_phase1_v1"
```

Build mode never promotes automatic pairs to gold. Finalize mode exports only
human-reviewed `direct_match` rows as training positives.

Prepare the separate feasibility-only provisional pilot from the current SWE
annotation candidates:

```bash
PYTHONPATH=src python scripts/generate_provisional_negatives.py prepare \
  --input-dir "$TA_DATA_ROOT/SWE-agent-trajectories/test_data/intent_action_phase1_v1" \
  --output-dir "$TA_DATA_ROOT/SWE-agent-trajectories/test_data/provisional_intent_action_pilot_v1"
```

After exporting `INF_API_KEY` securely, generate and locally validate LLM
proposals. The command supports hash-qualified resume and bypasses the inherited
proxy for the configured endpoint:

```bash
PYTHONPATH=src python scripts/generate_provisional_negatives.py generate \
  --dataset-dir "$TA_DATA_ROOT/SWE-agent-trajectories/test_data/provisional_intent_action_pilot_v1"

PYTHONPATH=src python scripts/generate_provisional_negatives.py finalize \
  --dataset-dir "$TA_DATA_ROOT/SWE-agent-trajectories/test_data/provisional_intent_action_pilot_v1"
```

Run dependency-free sparse baselines and validate the bi-encoder training plan:

```bash
PYTHONPATH=src python scripts/run_retrieval_baselines.py \
  --dataset-dir "$TA_DATA_ROOT/SWE-agent-trajectories/test_data/provisional_intent_action_pilot_v1" \
  --output-dir "$TA_DATA_ROOT/SWE-agent-trajectories/experiments/provisional_sparse_pilot_v1"

env -u LD_LIBRARY_PATH PYTHONPATH=src .venv/bin/python scripts/train_biencoder.py \
  --dataset-dir "$TA_DATA_ROOT/SWE-agent-trajectories/test_data/provisional_intent_action_pilot_v1" \
  --output-dir "$TA_DATA_ROOT/SWE-agent-trajectories/experiments/provisional_biencoder_pilot_v1" \
  --dry-run
```

All provisional outputs are permanently labeled
`FEASIBILITY ONLY — NOT GOLD EVALUATION` and must not be reported as Gold
evaluation results.

## Research Roadmap

Phase 1 builds human-reviewed one-Intent–one-short-Action data, retrieval
baselines, and a shared bi-encoder with parameter-sensitive hard negatives.
Later phases add Kimi multi-positive examples and collective Action-set
fulfillment without mixing those supervision types into the initial model.

See the [Phase 1 data design](docs/designs/intent-action-phase1-data-design.md)
and [implementation plan](docs/plans/intent-action-phase1-data.md).
