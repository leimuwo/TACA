# Data Catalog

## SWE Selected 500

The source corpus is stored externally relative to `TA_DATA_ROOT`:

```text
SWE-agent-trajectories/data_jsonl
```

The deterministic selected derivative is stored at:

```text
SWE-agent-trajectories/test_data/selected_500_json
```

Selection statistics:

- 500 trajectories;
- 16,197 observable Thought–Action steps;
- 250 successful and 250 failed trajectories;
- model distribution: 350 70B, 100 8B, and 50 405B;
- 320 unique task instances;
- 269 repositories;
- fixed seed `20260814`.

Tracked provenance and examples:

- [dataset manifest](../data/manifests/swe-selected-500/dataset_manifest.json);
- [selection summary](../data/manifests/swe-selected-500/selection_summary.json);
- [CASE-0001](../data/samples/swe-selected-500/CASE-0001.json);
- [CASE-0251](../data/samples/swe-selected-500/CASE-0251.json);
- [CASE-0500](../data/samples/swe-selected-500/CASE-0500.json).

Reproduce the derivative without writing into Git:

```bash
PYTHONPATH=src python scripts/select_swe_trajectories.py \
  --source "$TA_DATA_ROOT/SWE-agent-trajectories/data_jsonl" \
  --output "$TA_DATA_ROOT/SWE-agent-trajectories/test_data/selected_500_json" \
  --seed 20260814
```

The selection requires a complete task context, at least ten valid observable
Thought–Action–Observation cycles, inspection and edit behavior, patch evidence,
and evaluation evidence. The selected set balances outcome and model while
preferring task, repository, and trajectory-length diversity.
