# Publishing the TACA Data to ModelScope

This procedure publishes the external data independently from the GitHub code
repository. The ModelScope dataset must remain private unless the data owner
has explicitly approved public redistribution.

## Published scope

Upload this source directory:

```text
/inspire/hdd/project/security-defense-and-attack/public/wx/data/SWE-agent-trajectories
```

The publication contains 2,016 files and 7,529,577,161 bytes. It includes the
original Parquet and JSONL shards, selected trajectories, Thought-to-Intent
results, provisional negative cases, Phase 1 candidates, and experiment
reports. It excludes `.cache/` and all credentials.

The tracked integrity files are:

```text
data/manifests/modelscope-full-v1/inventory.json
data/manifests/modelscope-full-v1/SHA256SUMS
```

## Authentication

Install the official client outside the project environment if necessary:

```bash
python -m venv /tmp/taca-modelscope-client
/tmp/taca-modelscope-client/bin/python -m pip install \
  --index-url https://pypi.org/simple 'modelscope>=1.20,<2'
```

Set a newly generated token in the shell. Never add it to `.env`, Git, a
command transcript, or this document.

```bash
export MODELSCOPE_API_TOKEN='<set-in-shell>'
/tmp/taca-modelscope-client/bin/modelscope whoami
```

## Create and upload

Choose the authenticated ModelScope namespace and export the canonical ID:

```bash
export TACA_MODELSCOPE_DATASET='<owner>/TACA-data'
```

Create a private dataset repository:

```bash
/tmp/taca-modelscope-client/bin/modelscope create \
  "$TACA_MODELSCOPE_DATASET" \
  --repo-type dataset \
  --visibility private \
  --description 'TACA Thought-Intent-Action research data' \
  --exist-ok
```

Upload the dataset with resumable transfer. Do not use `--sync` because a
mistyped source path could delete remote files.

```bash
/tmp/taca-modelscope-client/bin/modelscope upload \
  "$TACA_MODELSCOPE_DATASET" \
  /inspire/hdd/project/security-defense-and-attack/public/wx/data/SWE-agent-trajectories \
  SWE-agent-trajectories \
  --repo-type dataset \
  --exclude '.cache/**' \
  --use-cache \
  --max-workers 4 \
  --commit-message 'data: publish TACA full dataset v1'
```

Upload the tracked manifests to the repository root:

```bash
/tmp/taca-modelscope-client/bin/modelscope upload \
  "$TACA_MODELSCOPE_DATASET" \
  data/manifests/modelscope-full-v1/inventory.json \
  inventory.json \
  --repo-type dataset

/tmp/taca-modelscope-client/bin/modelscope upload \
  "$TACA_MODELSCOPE_DATASET" \
  data/manifests/modelscope-full-v1/SHA256SUMS \
  SHA256SUMS \
  --repo-type dataset
```

After upload, record the final dataset ID and immutable revision in the handoff
message. The receiving agent should pin that revision instead of relying on a
moving `master` branch.

## Local pre-upload verification

Run from the parent of `SWE-agent-trajectories`:

```bash
cd /inspire/hdd/project/security-defense-and-attack/public/wx/data
sha256sum --check \
  /inspire/hdd/project/security-defense-and-attack/public/wx/Project/.worktrees/bootstrap/data/manifests/modelscope-full-v1/SHA256SUMS
```

All 2,016 entries must report `OK`. If any entry fails, regenerate and commit a
new versioned manifest instead of editing the published v1 manifest in place.
