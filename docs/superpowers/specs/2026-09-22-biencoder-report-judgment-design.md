# Bi-encoder Training Report Judgment Design

## Purpose

Add a deterministic command-line evaluator for provisional bi-encoder
`training_report.json` files. The command summarizes frozen and fine-tuned
metrics, evaluates fixed feasibility thresholds, and returns an automation-
friendly exit status.

All output remains labeled `FEASIBILITY ONLY — NOT GOLD EVALUATION`.

## Interface

The evaluator lives at `scripts/judge_biencoder_report.py` and accepts exactly
one positional argument:

```bash
python scripts/judge_biencoder_report.py /path/to/training_report.json
```

It prints:

- frozen and fine-tuned validation/test metrics;
- fine-tuned minus frozen changes;
- a pass/fail result for every criterion;
- one final classification: `STRONG_PASS`, `PASS`, or `FAIL`.

Exit statuses are:

- `0` for `PASS` or `STRONG_PASS`;
- `1` for `FAIL`;
- `2` when the file cannot be read, JSON is invalid, or required report fields
  are missing or non-numeric.

## Judgment Rules

`PASS` requires every condition below:

1. Fine-tuned validation MRR is greater than or equal to frozen validation
   MRR.
2. Fine-tuned test MRR improves over frozen test MRR by at least `0.03`.
3. Fine-tuned test Recall@1 is greater than or equal to frozen test Recall@1.
4. Fine-tuned test pairwise accuracy is greater than or equal to frozen test
   pairwise accuracy.
5. Fine-tuned test mean rank is strictly lower than frozen test mean rank.

`STRONG_PASS` requires all `PASS` conditions plus these absolute fine-tuned
test thresholds:

- MRR is at least `0.60`;
- Recall@1 is at least `0.45`;
- pairwise accuracy is at least `0.80`.

Any valid report that does not meet every `PASS` condition is `FAIL`.

## Structure

The script separates report parsing from judgment:

- a loader validates the required nested fields and extracts numeric metrics;
- a pure judgment function calculates deltas, individual checks, and the final
  classification;
- a formatter emits a compact text table and criterion summary;
- `main` maps the classification or input error to the documented exit status.

The evaluator does not modify reports or checkpoints.

## Testing

Unit tests cover:

- a report that produces `STRONG_PASS`;
- a report that produces ordinary `PASS`;
- each mandatory pass condition contributing to `FAIL`;
- malformed or incomplete input returning exit status `2`;
- CLI output containing the feasibility label and final classification.

Tests use synthetic report dictionaries so they remain independent of GPU,
model files, and external datasets.
