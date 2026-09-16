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
data directory. API credentials belong only in the environment variable
`INTENT_API_KEY`; never add them to tracked files.

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

## Research Roadmap

Phase 1 builds human-reviewed one-Intent–one-short-Action data, retrieval
baselines, and a shared bi-encoder with parameter-sensitive hard negatives.
Later phases add Kimi multi-positive examples and collective Action-set
fulfillment without mixing those supervision types into the initial model.

See the [Phase 1 data design](docs/designs/intent-action-phase1-data-design.md)
and [implementation plan](docs/plans/intent-action-phase1-data.md).
