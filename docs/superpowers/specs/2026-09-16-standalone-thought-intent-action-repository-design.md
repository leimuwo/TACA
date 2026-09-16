# Standalone Thought–Intent–Action Research Repository Design

## Decision

Create a new, independent Git repository at:

`/inspire/hdd/project/security-defense-and-attack/public/wx/Project`

The repository starts with a new Git history. It does not copy, import, depend on,
or preserve any source code, documents, generated artifacts, configuration, Git
history, or remote settings from `TACA/`.

The existing directories outside `Project/`, including `TACA/` and the shared
`wx/data/` tree, remain unchanged.

## Goal

Provide a clean and reproducible home for the current research workflow:

1. select valid Thought–Action trajectories;
2. extract atomic execution Intents from observable Thoughts;
3. construct reviewed one-Intent–one-short-Action training data;
4. train and evaluate Intent–Action retrieval models;
5. extend later to multi-Action fulfillment and consistency evaluation.

Git manages source code, tests, prompts, configuration, documentation, data
manifests, and small samples. Large datasets, model weights, and generated
experiment outputs remain outside Git.

## Repository Boundary

### Included

- maintained Python source and command-line entry points;
- unit and integration tests;
- model-independent prompts and schemas;
- experiment configuration without secrets or machine-specific paths;
- dataset manifests, checksums, licenses, statistics, and small test fixtures;
- reviewed design documents, plans, and concise experiment reports;
- environment and dependency declarations;
- commands needed to reproduce dataset selection and training.

### Excluded

- every file under the existing `TACA/` directory;
- the previous TACA Git history and GitHub remote;
- complete raw or processed trajectory datasets;
- the full 500-case selected SWE export;
- API keys, `.env` files, credentials, and private endpoints;
- caches, logs, checkpoints, downloaded model weights, and bulk outputs;
- unrelated projects under `wx/`, including `WSIC/`, `reference/`, and other
  nested repositories.

## Proposed Layout

```text
Project/
├── README.md
├── LICENSE
├── .gitignore
├── .gitattributes
├── .env.example
├── pyproject.toml
├── configs/
│   ├── paths.example.toml
│   ├── selection/
│   ├── extraction/
│   └── training/
├── prompts/
│   └── thought_to_intent.md
├── src/
│   └── thought_action_retrieval/
│       ├── data/
│       ├── intent/
│       ├── matching/
│       ├── training/
│       └── evaluation/
├── scripts/
│   ├── select_swe_trajectories.py
│   ├── extract_thought_intents.py
│   └── build_intent_action_dataset.py
├── tests/
│   ├── fixtures/
│   └── ...
├── docs/
│   ├── data-catalog.md
│   ├── annotation-guideline.md
│   ├── experiments/
│   ├── plans/
│   └── superpowers/specs/
├── data/
│   ├── README.md
│   ├── manifests/
│   ├── samples/
│   └── local/                 # ignored local links or mount points
└── outputs/
    ├── README.md
    └── manifests/
```

## Initial Source Migration

Only the following current `wx/` materials are candidates for migration. Each
file is copied into the new repository and then revised to remove absolute paths,
embedded credentials, and assumptions about the old workspace:

- `scripts/select_representative_swe_cases.py`;
- `tests/test_select_representative_swe_cases.py`;
- `scripts/generate_thought_intents.py`;
- `tests/test_generate_thought_intents.py`;
- `docs/superpowers/specs/2026-09-16-intent-action-phase1-data-design.md`;
- `docs/superpowers/plans/2026-09-16-intent-action-phase1-data.md`.

No other files are migrated implicitly. Additional code must be explicitly
reviewed before inclusion.

## Data Architecture

The existing shared data pool remains at:

`/inspire/hdd/project/security-defense-and-attack/public/wx/data`

Programs discover it through an environment variable rather than a hard-coded
path:

```text
TA_DATA_ROOT=/inspire/hdd/project/security-defense-and-attack/public/wx/data
```

`configs/paths.example.toml` documents relative dataset locations below that
root. A developer may override the root for another machine without changing
tracked source code.

Each managed dataset receives a manifest under `data/manifests/` containing:

- stable dataset name and version;
- source and license information;
- local relative path below `TA_DATA_ROOT`;
- creation or download command;
- file count and total byte count;
- SHA-256 checksums for important source and derived artifacts;
- schema identifier and record count;
- generation seed and configuration;
- parent dataset/version references.

The initial SWE manifest records the 12 source JSONL shards and the selected
500-case derivative. Git stores the selected set's `manifest.csv`,
`selection_summary.json`, and a few small representative cases, but not the
complete 111 MB export.

## Code Architecture

Reusable logic lives in the `src/thought_action_retrieval/` package. Files under
`scripts/` are thin command-line wrappers and must not contain core business
logic that cannot be imported and tested.

The initial package boundaries are:

- `data`: trajectory schemas, JSONL readers, selectors, manifests, and short
  Action filtering;
- `intent`: Thought-to-Intent request construction, response validation, and
  extraction schemas;
- `matching`: Action serialization, candidate construction, negatives, and
  retrieval interfaces;
- `training`: bi-encoder dataset adapters, loss configuration, and training;
- `evaluation`: retrieval, fulfillment, hard-negative, and cross-source metrics.

This migration first preserves verified behavior. Refactoring scripts into the
package happens incrementally with tests, rather than as an unverified bulk
rewrite.

## Configuration and Secrets

Tracked configuration contains only portable defaults and placeholders.

`.env.example` documents variable names such as:

```text
TA_DATA_ROOT=/path/to/shared/data
INTENT_API_BASE_URL=https://example.invalid/v1
INTENT_API_KEY=
INTENT_MODEL_NAME=
```

`.env`, `.env.*` except `.env.example`, credential stores, and machine-local
configuration are ignored. Scripts must read secrets from environment variables
or explicit secret-management facilities. They must never accept a committed API
key as a default.

## Git and Artifact Policy

The repository uses a new `main` branch and initially has no remote. A remote is
added only after the local content has passed secret and large-file audits.

Normal Git tracks source and text metadata. Git LFS is not required for the
initial repository because large data and model files are excluded entirely.
Files above a configurable threshold (initially 10 MiB) fail the repository
audit unless explicitly allowlisted as small research fixtures.

Generated outputs are ignored by default. A concise report or output manifest
may be promoted into Git only when it is required to understand or reproduce a
result.

## Data Flow

```text
shared external data pool
        |
        v
trajectory selection + manifest generation
        |
        v
small reviewed samples / external derived dataset
        |
        v
Thought -> atomic Intent extraction
        |
        v
human-reviewed Intent-Action relations
        |
        v
training splits -> baselines -> bi-encoder -> evaluation reports
```

Every derived artifact records its input manifest, configuration, seed, and
producing Git commit. This permits reproduction without storing large artifacts
inside Git.

## Error Handling and Safety

- Dataset builders fail if required inputs or manifest versions are missing.
- Existing versioned datasets are not overwritten without an explicit flag.
- Partial output is written to temporary files and atomically renamed.
- Invalid JSON, duplicate IDs, inconsistent counts, and checksum mismatches stop
  the build.
- LLM extraction failures are recorded separately and never silently treated as
  empty Intents.
- Human review is required before automatically paired relations become gold
  training positives.
- Repository setup never deletes or changes existing external datasets.

## Testing and Verification

The initial repository setup is accepted when:

- the new Git history contains no TACA-derived content or remote;
- secret scanning finds no credentials in tracked files;
- tracked files contain no current-machine absolute paths except explanatory
  examples in documentation;
- no tracked file exceeds the agreed size threshold;
- migrated selector and Intent extraction tests pass;
- test fixtures use synthetic or small explicitly approved samples;
- all checked-in manifests parse and reconcile their declared counts;
- a clean checkout can run the unit tests without access to the full datasets;
- integration checks can use `TA_DATA_ROOT` when full local data is available.

## Implementation Sequence

1. Create repository safety files and dependency metadata.
2. Establish package, script, test, documentation, data, and output directories.
3. Copy only the approved current `wx/` files listed in this design.
4. Run secret, absolute-path, and file-size audits.
5. Add portable data-root configuration and dataset manifest tooling.
6. Migrate tests and preserve current selector/extractor behavior.
7. Add small synthetic fixtures and selected-data metadata.
8. Run the full test and repository audit suite.
9. Create the first implementation commit; add a private remote only afterward.

## Non-Goals for Initial Setup

- copying or rewriting TACA;
- committing full raw or processed datasets;
- introducing DVC or Git LFS before multi-machine data synchronization is
  required;
- training the bi-encoder during repository setup;
- reorganizing unrelated projects in the parent workspace;
- deleting duplicate or obsolete files outside `Project/`.
