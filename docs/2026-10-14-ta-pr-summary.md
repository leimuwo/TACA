# Thought–Action coverage improvements

The project now supports both Action retrieval experiments and applying a
trained classifier to actual tool calls. The coverage task asks whether a
single Action completely covers one current atomic Intent, then aggregates
Intent coverage into faithful, partial, or not_executed. Unassessable current
targets remain explicit and prevent a strict step-level judgment.

## Included workflow

- Temporal extraction preserves observable evidence, distinguishes current
  goals from future plans, and validates exact source quotes and resume identity.
- ClawGym parsing, matching, and hard-negative preparation preserve trajectory
  provenance; BGE, BGE-M3, reranker, and NightOwl experiment scripts retain the
  retrieval baselines and diagnostics.
- Coverage preparation adds train-only expansion, targeted GQQ review,
  uncertainty resolution/quarantine, frozen dev/test sampling, GLM weak labels,
  and independent human review/export/finalization.
- Cross-Encoder training uses a train-trajectory internal holdout. Evaluation
  rejects training trajectory and text-pair overlaps, prints ACC/PRE/REC/F1,
  and retains pair scores and reproducibility hashes.
- Bulk multi-file reading goals are excluded from strict evaluation. Whole
  windows are excluded rather than deleting a goal and labeling the remainder
  faithful; directory enumeration and single-file reading remain eligible.
- A local JSONL/HTTP adapter loads the checkpoint once, accepts explicit
  due-now/assessability decisions and actual calls, and returns pair scores,
  Intent coverage, and a step label. It does not yet modify an Agent loop.

## Reported pair results

At logit threshold 0 on the v4 stratified GLM-labeled subset:

| Split | Model | ACC | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| dev, 200 pairs | pretrained reranker | .8200 | .8873 | .6923 | .7778 |
| dev, 200 pairs | trained coverage model | .9650 | .9773 | .9451 | .9609 |
| test, 250 pairs | pretrained reranker | .8080 | .8280 | .7064 | .7624 |
| test, 250 pairs | trained coverage model | .9480 | .9444 | .9358 | .9401 |

These are binary pair comparisons against model-generated weak labels, not
human Gold or online step-level accuracy. The test subset is stratified using
tool groups and weak labels. Baseline and trained models use the same threshold.
Compact reports and original input hashes are retained under
`results/ta_coverage_20261014/`; local absolute model paths in reports are
normalized to a relative cache path. Full trajectories, annotation exports,
predictions, generated HTML, and checkpoints stay in the external data pool.

## Run

Install the training extra in a GPU environment and set `TA_MODEL_ROOT` to the
local directory containing `bge-reranker-v2-m3` and other baseline models.
Dataset directories below are external and must be supplied locally.

```bash
python scripts/train_ta_pair_cross_encoder.py \
  --data data/clawgym/ta_pair_weak_train_10k_pool_targeted_v3_20261013 \
  --output outputs/ta_pair_cross_encoder_10k_v3 --device cuda

python scripts/evaluate_ta_labeled_pairs.py \
  --checkpoint outputs/ta_pair_cross_encoder_10k_v3/best \
  --train-data data/clawgym/ta_pair_weak_train_10k_pool_targeted_v3_20261013 \
  --pairs data/clawgym/ta_glm52_stratified_pair_eval_v4_20261013/dev/pairs.jsonl \
  --split dev --output outputs/ta_pair_dev --device cuda

python scripts/use_ta_coverage.py --serve --device cuda --port 8765
curl -s http://127.0.0.1:8765/judge \
  -H 'Content-Type: application/json' \
  --data-binary @configs/ta_coverage_example.jsonl
```

Use the finalized test pair file with `--split test` after fixing the model and
threshold on dev. Existing output directories are rejected. Deployment inputs
and JSONL usage are described in `2026-10-14-ta-model-usage.md`.

## Validation and limits

The isolated PR checkout runs 215 unittest cases successfully, with seven
skipped because optional GPU dependencies or external corpus configuration
are unavailable. The tracked-file repository audit passes. New GPU inference
and HTTP startup were not run in this CPU environment; retained experiment
reports come from the existing GPU runs. Human Gold completion, independent
annotation agreement, and connecting the adapter to a specific Agent remain
next-stage work.

This branch includes the Phase 1 training/inference foundation absent from
current main, followed by the ClawGym and TA coverage improvements. Existing
BGE dataset publication PRs remain separate; this PR keeps bulk ClawGym data
external rather than importing their data artifacts.
