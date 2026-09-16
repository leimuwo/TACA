# Data Policy

This directory contains only small, reviewable Git artifacts:

- `manifests/`: tracked provenance, checksums, source statistics, and selection
  metadata;
- `samples/`: a few small examples for tests and human inspection;
- `local/`: ignored optional links or mount points for external data.

Complete raw or derived datasets must remain below the external
`TA_DATA_ROOT`. Do not copy full JSONL, Parquet, model, or checkpoint files into
this repository.

The currently registered dataset is documented in the
[data catalog](../docs/data-catalog.md) and its
[manifest](manifests/swe-selected-500/dataset_manifest.json).
