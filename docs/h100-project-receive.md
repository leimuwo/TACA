# H100 Project Reception Guide

This document is the handoff contract for an agent receiving the TACA code and
data on an H100 host. Code comes from GitHub; data comes from a private
ModelScope Dataset repository.

## Required handoff values

The sender must provide these non-secret values separately after publication:

```bash
export TACA_GITHUB_REPO='git@github.com:leimuwo/TACA.git'
export TACA_GITHUB_REF='feature/phase1-dataset'
export TACA_MODELSCOPE_DATASET='MongTsai/TACA-data'
export TACA_MODELSCOPE_REVISION='master'
```

Dataset page: <https://modelscope.cn/datasets/MongTsai/TACA-data>

The receiver must configure its own GitHub SSH access and ModelScope token.
Tokens and private keys must never be copied from the source machine.

The ModelScope dataset is private. The receiving account must be granted
access by `MongTsai`. ModelScope's dataset tag endpoint returned 404 during
publication, so this handoff uses `master` plus the tracked full-file SHA-256
manifest as the immutable content contract. If `master` changes, verification
will fail before training.

## Capacity preflight

The dataset contains 7,529,577,161 bytes. Reserve at least 30 GiB for the
dataset, Python environment, downloaded base model, checkpoints, and reports.

```bash
df -h "$PWD"
nvidia-smi
python --version
```

Stop if the destination filesystem has less than 30 GiB available or if the
H100 is not visible.

## Receive the code

```bash
mkdir -p workspace/wx
cd workspace/wx
git clone --branch "$TACA_GITHUB_REF" --single-branch \
  "$TACA_GITHUB_REPO" Project
cd Project
export TACA_PROJECT_ROOT="$PWD"
git rev-parse HEAD
git status --short
```

The final status output must be empty. Record the commit hash in the experiment
notes.

## Create clean environments

Create the training environment from the repository instead of copying the
source machine's `.venv`:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[training]'
```

Install the ModelScope client in a separate temporary environment:

```bash
python -m venv /tmp/taca-modelscope-client
/tmp/taca-modelscope-client/bin/python -m pip install \
  --index-url https://pypi.org/simple 'modelscope>=1.20,<2'
export MODELSCOPE_API_TOKEN='<set-in-shell>'
/tmp/taca-modelscope-client/bin/modelscope whoami
```

## Download the data

Choose a data location with enough capacity. The repository expects
`TA_DATA_ROOT` to be the directory containing `SWE-agent-trajectories`.

```bash
export TA_DATA_ROOT="${TA_DATA_ROOT:-$TACA_PROJECT_ROOT/../TACA-data}"
mkdir -p "$TA_DATA_ROOT"

/tmp/taca-modelscope-client/bin/modelscope download \
  "$TACA_MODELSCOPE_DATASET" \
  --repo-type dataset \
  --revision "$TACA_MODELSCOPE_REVISION" \
  --local-dir "$TA_DATA_ROOT" \
  --include 'SWE-agent-trajectories/**' \
  --max-workers 4
```

The resulting layout must start with:

```text
$TA_DATA_ROOT/
└── SWE-agent-trajectories/
    ├── data/
    ├── data_jsonl/
    ├── test_data/
    └── experiments/
```

## Verify every downloaded file

Run the tracked SHA-256 manifest from the data root:

```bash
cd "$TA_DATA_ROOT"
sha256sum --check \
  "$TACA_PROJECT_ROOT/data/manifests/modelscope-full-v1/SHA256SUMS"
```

The command must report 2,016 successful entries and zero failures. Also verify
the manifest itself:

```bash
cd "$TACA_PROJECT_ROOT"
printf '%s  %s\n' \
  '5562c08a3a734801991017b356bf44640ef1dba88a1b692762d83f8b36fa82f0' \
  'data/manifests/modelscope-full-v1/SHA256SUMS' | sha256sum --check
```

Do not start training when any checksum fails. Re-download the failed file or
the pinned snapshot first.

Publication verification on 2026-09-21 compared every local source file with
the ModelScope file tree: 2,016 matched paths, 7,529,577,161 matched bytes,
zero missing files, zero size mismatches, and zero SHA-256/LFS blob mismatches.

## Validate the project

```bash
cd "$TACA_PROJECT_ROOT"
TA_DATA_ROOT="$TA_DATA_ROOT" PYTHONPATH=src \
  .venv/bin/python -m unittest discover -s tests -v

TA_DATA_ROOT="$TA_DATA_ROOT" PYTHONPATH=src \
  .venv/bin/python scripts/train_biencoder.py \
  --dataset-dir "$TA_DATA_ROOT/SWE-agent-trajectories/test_data/provisional_intent_action_pilot_v1" \
  --output-dir "$TA_DATA_ROOT/SWE-agent-trajectories/experiments/h100-dry-run" \
  --device cuda \
  --precision bf16 \
  --dry-run
```

The dry-run output directory must not already exist. Remove or rename an old
dry-run directory before repeating the command.

## Start the provisional H100 pilot

```bash
TA_DATA_ROOT="$TA_DATA_ROOT" PYTHONPATH=src \
  .venv/bin/python scripts/train_biencoder.py \
  --dataset-dir "$TA_DATA_ROOT/SWE-agent-trajectories/test_data/provisional_intent_action_pilot_v1" \
  --output-dir "$TA_DATA_ROOT/SWE-agent-trajectories/experiments/provisional_biencoder_h100_v1" \
  --device cuda \
  --precision bf16 \
  --epochs 3 \
  --batch-size 32 \
  --gradient-accumulation-steps 4
```

This run is explicitly `FEASIBILITY ONLY — NOT GOLD EVALUATION`. Preserve the
Git commit, ModelScope dataset revision, `training_plan.json`,
`training_report.json`, `checkpoint-best`, and `checkpoint-last` together.

If CUDA or BF16 validation fails, fix the H100 PyTorch environment instead of
silently switching a formal GPU run to CPU.
