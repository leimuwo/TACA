# Standalone Research Repository Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the new `Project/` repository into a portable, secret-safe research codebase for SWE trajectory selection and observable Thought-to-Intent extraction while keeping all large datasets outside Git.

**Architecture:** Reusable Python code lives under a small `src/thought_action_retrieval/` package; `scripts/` contains thin CLI wrappers. External datasets are addressed through `TA_DATA_ROOT`, represented in Git by versioned manifests and a few small fixtures, and checked by a repository audit that rejects secrets, absolute machine paths in code/configuration, and tracked files over 10 MiB.

**Tech Stack:** Python 3.10+ standard library, `unittest`, setuptools via `pyproject.toml`, Git.

**Spec:** `docs/superpowers/specs/2026-09-16-standalone-thought-intent-action-repository-design.md`

## Global Constraints

- Do not read from, copy from, modify, depend on, or preserve history from `../TACA/`.
- Do not modify any file outside `Project/`; approved files under the parent `wx/` directory are read-only migration sources.
- Do not commit raw datasets, the complete selected 500-case export, model weights, logs, caches, generated LLM outputs, credentials, or `.env` files.
- Use `TA_DATA_ROOT` to locate external data; tracked Python and configuration files must not contain `/inspire/hdd` paths.
- Keep the repository without a remote until all tests, secret checks, and size checks pass.
- Migrate only the six approved parent-workspace files named in the spec.
- Preserve the current fixed selection seed `20260814` and the verified 500-case quotas: 250 successful, 250 failed, 350 70B, 100 8B, and 50 405B.
- Keep Thought extraction restricted to observable Thought text; the current Action may be retained as evaluation metadata but must not enter the extraction prompt.
- Use test-first development for production behavior and commit after each completed task.

---

## File Structure

The bootstrap creates or modifies these units:

- `src/thought_action_retrieval/repository_audit.py`: tracked-file policy and credential/size/path checks.
- `src/thought_action_retrieval/data/paths.py`: portable data-root resolution and containment checks.
- `src/thought_action_retrieval/data/manifests.py`: deterministic file metadata and dataset manifests.
- `src/thought_action_retrieval/data/swe_selection.py`: migrated SWE analysis, sampling, CASE conversion, and output writing.
- `src/thought_action_retrieval/intent/extraction.py`: migrated trajectory parsing, prompt construction, API handling, validation, resume behavior, and output writing.
- `scripts/*.py`: thin wrappers around package `main()` functions.
- `prompts/thought_to_intent.md`: the approved system and user prompt with explicit section markers.
- `tests/`: synthetic unit/integration tests; no full dataset dependency.
- `data/manifests/` and `data/samples/`: small tracked metadata and fixtures generated from external data.
- `configs/paths.example.toml`: portable relative dataset locations.
- root policy and documentation files: `.gitignore`, `.gitattributes`, `.env.example`, `pyproject.toml`, `README.md`, and `LICENSE`.

---

### Task 1: Repository policy, package skeleton, and self-audit

**Files:**
- Create: `.gitignore`
- Create: `.gitattributes`
- Create: `.env.example`
- Create: `LICENSE`
- Create: `pyproject.toml`
- Create: `src/thought_action_retrieval/__init__.py`
- Create: `src/thought_action_retrieval/repository_audit.py`
- Create: `scripts/audit_repository.py`
- Create: `tests/test_repository_audit.py`

**Interfaces:**
- Produces: `AuditIssue(code: str, path: str, detail: str)`
- Produces: `tracked_files(root: Path) -> list[Path]`
- Produces: `audit_repository(root: Path, max_file_bytes: int = 10 * 1024 * 1024) -> list[AuditIssue]`
- Produces: `format_issues(issues: Sequence[AuditIssue]) -> str`
- Produces CLI: `python scripts/audit_repository.py [--root PATH] [--max-file-mib 10]`

- [ ] **Step 1: Write failing policy tests**

Create `tests/test_repository_audit.py` with temporary Git repositories. The test helper runs `git init`, writes files, and runs `git add` without making a commit.

```python
def test_rejects_tracked_secret_large_file_and_machine_path(tmp_path):
    initialize_git(tmp_path)
    (tmp_path / ".env").write_text("API_KEY=secret-value\n", encoding="utf-8")
    (tmp_path / "large.bin").write_bytes(b"x" * 33)
    (tmp_path / "run.py").write_text(
        'DATA = "/inspire/hdd/private/data"\n', encoding="utf-8"
    )
    git_add_all(tmp_path, force=True)

    issues = audit_repository(tmp_path, max_file_bytes=32)

    assert {issue.code for issue in issues} == {
        "forbidden_path",
        "file_too_large",
        "machine_absolute_path",
        "possible_secret",
    }


def test_allows_empty_secret_example_and_absolute_path_in_markdown(tmp_path):
    initialize_git(tmp_path)
    (tmp_path / ".env.example").write_text("INTENT_API_KEY=\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "Example: `/inspire/hdd/project/data`\n", encoding="utf-8"
    )
    git_add_all(tmp_path)

    assert audit_repository(tmp_path, max_file_bytes=1024) == []
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m unittest tests.test_repository_audit -v
```

Expected: import failure for `thought_action_retrieval.repository_audit`.

- [ ] **Step 3: Add package metadata and ignore policy**

Use this `pyproject.toml` contract:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "thought-action-retrieval"
version = "0.1.0"
description = "Research tools for observable Thought, execution Intent, and Action alignment"
requires-python = ">=3.10"
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]

[tool.unittest]
start-directory = "tests"
```

`.gitignore` must include exactly these policy groups:

```gitignore
# Secrets and local configuration
.env
.env.*
!.env.example

# Python and tooling caches
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/

# External data and generated outputs
data/local/
outputs/*
!outputs/README.md
!outputs/manifests/
!outputs/manifests/**
checkpoints/
models/
logs/

# Editors and operating systems
.DS_Store
Thumbs.db
.idea/
.vscode/
```

`.gitattributes` must normalize text without enabling LFS:

```gitattributes
* text=auto eol=lf
*.py text eol=lf
*.md text eol=lf
*.json text eol=lf
*.jsonl text eol=lf
*.csv text eol=lf
*.toml text eol=lf
```

`.env.example` must contain empty portable variables only:

```dotenv
TA_DATA_ROOT=/path/to/shared/data
INTENT_API_BASE_URL=https://api.deepseek.com
INTENT_API_KEY=
INTENT_MODEL_NAME=deepseek-v4-flash
```

Create a private-research notice in `LICENSE`:

```text
Copyright (c) 2026. All rights reserved.

This repository is maintained for private research use. No permission is
granted to redistribute its source code, data, models, or research artifacts
without explicit authorization from the repository owner.
```

- [ ] **Step 4: Implement the repository audit**

`tracked_files()` must execute `git -C <root> ls-files -z`, decode NUL-separated paths, and return sorted paths. `audit_repository()` must:

1. flag tracked `.env` and `.env.*` except `.env.example` as `forbidden_path`;
2. flag tracked path parts `.codex`, `.sii`, `__pycache__`, `checkpoints`, `models`, or `logs` as `forbidden_path`;
3. flag files larger than `max_file_bytes` as `file_too_large`;
4. scan UTF-8 text files for non-empty assignments to names matching `(api[_-]?key|secret|token|password)` and flag `possible_secret` while allowing empty values and `example.invalid` placeholders;
5. scan non-Markdown text for `/inspire/hdd/` and flag `machine_absolute_path`;
6. skip binary content after a UTF-8 decoding failure.

The CLI returns exit code 1 and prints one line per issue when issues exist; otherwise it prints `repository audit passed` and returns 0.

- [ ] **Step 5: Run tests and the real audit**

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_repository_audit -v
git add .gitignore .gitattributes .env.example LICENSE pyproject.toml \
  src/thought_action_retrieval/__init__.py \
  src/thought_action_retrieval/repository_audit.py \
  scripts/audit_repository.py tests/test_repository_audit.py
PYTHONPATH=src python scripts/audit_repository.py --root .
```

Expected: tests pass; the real audit passes with the design and plan documents allowed to contain explanatory absolute paths.

- [ ] **Step 6: Commit Task 1**

```bash
git add .gitignore .gitattributes .env.example LICENSE pyproject.toml \
  src/thought_action_retrieval/__init__.py \
  src/thought_action_retrieval/repository_audit.py \
  scripts/audit_repository.py tests/test_repository_audit.py
git commit -m "chore: establish standalone repository policy"
```

---

### Task 2: Portable data paths and deterministic manifests

**Files:**
- Create: `configs/paths.example.toml`
- Create: `src/thought_action_retrieval/data/__init__.py`
- Create: `src/thought_action_retrieval/data/paths.py`
- Create: `src/thought_action_retrieval/data/manifests.py`
- Create: `scripts/build_data_manifest.py`
- Create: `tests/test_data_paths.py`
- Create: `tests/test_data_manifests.py`

**Interfaces:**
- Produces: `DataRootError(ValueError)`
- Produces: `resolve_data_root(explicit: Path | None = None, environ: Mapping[str, str] | None = None) -> Path`
- Produces: `resolve_under_data_root(data_root: Path, relative_path: str | Path) -> Path`
- Produces: `sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str`
- Produces: `describe_file(data_root: Path, path: Path) -> dict[str, object]`
- Produces: `build_dataset_manifest(*, name: str, version: str, data_root: Path, files: Sequence[Path], metadata: Mapping[str, object]) -> dict[str, object]`
- Produces CLI: `python scripts/build_data_manifest.py --name NAME --version VERSION --relative-path PATH... --output FILE`

- [ ] **Step 1: Write failing path tests**

```python
def test_resolves_explicit_root_before_environment(tmp_path):
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    result = resolve_data_root(
        explicit=explicit,
        environ={"TA_DATA_ROOT": str(tmp_path / "ignored")},
    )
    assert result == explicit.resolve()


def test_rejects_missing_data_root(tmp_path):
    with self.assertRaisesRegex(DataRootError, "TA_DATA_ROOT"):
        resolve_data_root(environ={})


def test_rejects_path_escape(tmp_path):
    with self.assertRaisesRegex(DataRootError, "outside"):
        resolve_under_data_root(tmp_path, "../secret.json")
```

- [ ] **Step 2: Write failing manifest tests**

```python
def test_manifest_is_sorted_and_uses_relative_paths(tmp_path):
    (tmp_path / "b.jsonl").write_text("b\n", encoding="utf-8")
    (tmp_path / "a.jsonl").write_text("a\n", encoding="utf-8")

    manifest = build_dataset_manifest(
        name="fixture",
        version="v1",
        data_root=tmp_path,
        files=[tmp_path / "b.jsonl", tmp_path / "a.jsonl"],
        metadata={"record_count": 2},
    )

    assert [item["path"] for item in manifest["files"]] == ["a.jsonl", "b.jsonl"]
    assert manifest["total_bytes"] == 4
    assert manifest["metadata"] == {"record_count": 2}
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])
```

- [ ] **Step 3: Run tests and verify RED**

```bash
PYTHONPATH=src python -m unittest tests.test_data_paths tests.test_data_manifests -v
```

Expected: import failures for the new data modules.

- [ ] **Step 4: Implement path containment**

`resolve_data_root()` chooses `explicit` first, then `TA_DATA_ROOT`, expands the path, resolves it, and requires an existing directory. `resolve_under_data_root()` rejects absolute relative paths and verifies the resolved candidate is equal to or below the resolved root using `Path.is_relative_to()`.

Create `configs/paths.example.toml`:

```toml
[datasets]
swe_raw_jsonl = "SWE-agent-trajectories/data_jsonl"
swe_selected_500 = "SWE-agent-trajectories/test_data/selected_500_json"
swe_thought_intent = "SWE-agent-trajectories/test_data/thought_intent"
kimi_selected = "SWE-agent-trajectories/test_data/kimi_selected_json"
```

- [ ] **Step 5: Implement manifest generation**

Every manifest must use this top-level shape:

```json
{
  "schema_version": "dataset_manifest_v1",
  "name": "swe-selected-500",
  "version": "2026-09-16",
  "file_count": 2,
  "total_bytes": 123,
  "files": [
    {"path": "relative/path", "bytes": 100, "sha256": "..."}
  ],
  "metadata": {}
}
```

Do not include timestamps because they make unchanged manifests nondeterministic. Serialize JSON with `ensure_ascii=False`, `sort_keys=True`, and two-space indentation, then atomically replace the destination.

- [ ] **Step 6: Run tests and CLI smoke test**

```bash
PYTHONPATH=src python -m unittest tests.test_data_paths tests.test_data_manifests -v
tmp_dir="$(mktemp -d)"
printf '{"x":1}\n' > "$tmp_dir/sample.jsonl"
TA_DATA_ROOT="$tmp_dir" PYTHONPATH=src python scripts/build_data_manifest.py \
  --name smoke --version v1 --relative-path sample.jsonl \
  --output "$tmp_dir/manifest.json"
python -m json.tool "$tmp_dir/manifest.json" >/dev/null
```

Expected: tests pass and the smoke manifest parses.

- [ ] **Step 7: Commit Task 2**

```bash
git add configs/paths.example.toml src/thought_action_retrieval/data \
  scripts/build_data_manifest.py tests/test_data_paths.py tests/test_data_manifests.py
git commit -m "feat: add portable data paths and manifests"
```

---

### Task 3: Migrate and package the SWE trajectory selector

**Files:**
- Create: `src/thought_action_retrieval/data/swe_selection.py`
- Create: `scripts/select_swe_trajectories.py`
- Create: `tests/test_swe_selection.py`

**Read-only migration sources:**
- `../scripts/select_representative_swe_cases.py`
- `../tests/test_select_representative_swe_cases.py`

**Interfaces:**
- Preserves: `is_bugfix_task(task_text: str) -> bool`
- Preserves: `classify_action(action: str) -> str`
- Preserves: `analyze_record(record: dict[str, Any]) -> dict[str, Any]`
- Preserves: `quality_score(analysis: dict[str, Any], record: dict[str, Any]) -> float`
- Preserves: `fit_pair_model_quotas(...) -> dict[str, int]`
- Preserves: `select_representative_candidates(...) -> list[dict[str, Any]]`
- Preserves: `scan_jsonl_candidates(source_dir: Path) -> list[dict[str, Any]]`
- Preserves: `load_selected_jsonl_records(...) -> dict[str, dict[str, Any]]`
- Preserves: `record_to_case(...) -> dict[str, Any]`
- Preserves: `write_case_outputs(...) -> None`
- Produces: `build_parser() -> argparse.ArgumentParser`
- Produces: `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Establish the migration baseline in the parent workspace**

Run without modifying the parent files:

```bash
cd ..
python -m unittest tests.test_select_representative_swe_cases -v
cd Project
```

Expected: 17 tests pass. Record the count in the Task 3 commit message body.

- [ ] **Step 2: Create migrated tests before production code**

Create `tests/test_swe_selection.py` from the approved parent test behavior, replacing:

```python
from scripts.select_representative_swe_cases import ...
```

with:

```python
from thought_action_retrieval.data.swe_selection import ...
```

Retain all 17 existing assertions. Add one CLI parsing test:

```python
def test_cli_requires_source_and_output():
    parser = build_parser()
    with self.assertRaises(SystemExit):
        parser.parse_args([])
```

- [ ] **Step 3: Run migrated tests and verify RED**

```bash
PYTHONPATH=src python -m unittest tests.test_swe_selection -v
```

Expected: import failure for `thought_action_retrieval.data.swe_selection`.

- [ ] **Step 4: Migrate the selector implementation**

Move the approved implementation behavior into `swe_selection.py` without changing hard filters, scoring, CASE schema, fixed ordering, atomic writes, or soft diversity fallback. Replace only the command entry point:

```python
PAIR_MODEL_QUOTAS = {"70b": 87, "8b": 25, "405b": 13}
TOTAL_MODEL_QUOTAS = {"70b": 175, "8b": 50, "405b": 25}
DEFAULT_SEED = 20260814


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidates = scan_jsonl_candidates(args.source)
    pair_quotas = fit_pair_model_quotas(
        candidates,
        preferred_quotas=PAIR_MODEL_QUOTAS,
        total_pairs=sum(PAIR_MODEL_QUOTAS.values()),
    )
    selected = select_representative_candidates(
        candidates,
        pair_model_quotas=pair_quotas,
        total_model_quotas=TOTAL_MODEL_QUOTAS,
        seed=args.seed,
    )
    records = load_selected_jsonl_records(args.source, selected)
    write_case_outputs(args.output, selected, records, args.seed, len(candidates))
    print(f"wrote {len(selected)} cases to {args.output}", flush=True)
    return 0
```

The wrapper contains only:

```python
#!/usr/bin/env python3
from thought_action_retrieval.data.swe_selection import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run migrated tests and wrapper help**

```bash
PYTHONPATH=src python -m unittest tests.test_swe_selection -v
PYTHONPATH=src python scripts/select_swe_trajectories.py --help >/dev/null
```

Expected: 18 tests pass and help exits 0.

- [ ] **Step 6: Confirm no source drift against the verified output contract**

Assert the constants and expected counts in a test:

```python
def test_verified_500_case_quotas_are_frozen():
    self.assertEqual(DEFAULT_SEED, 20260814)
    self.assertEqual(TOTAL_MODEL_QUOTAS, {"70b": 175, "8b": 50, "405b": 25})
    self.assertEqual(sum(TOTAL_MODEL_QUOTAS.values()) * 2, 500)
```

Run the test file again and require 19 passing tests.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/thought_action_retrieval/data/swe_selection.py \
  scripts/select_swe_trajectories.py tests/test_swe_selection.py
git commit -m "feat: migrate deterministic SWE trajectory selector"
```

---

### Task 4: Migrate Thought-to-Intent extraction with an external prompt

**Files:**
- Create: `prompts/thought_to_intent.md`
- Create: `src/thought_action_retrieval/intent/__init__.py`
- Create: `src/thought_action_retrieval/intent/extraction.py`
- Create: `scripts/extract_thought_intents.py`
- Create: `tests/test_intent_extraction.py`

**Read-only migration sources:**
- `../scripts/generate_thought_intents.py`
- `../tests/test_generate_thought_intents.py`

**Interfaces:**
- Produces: `PromptBundle(system: str, user_template: str)`
- Produces: `load_prompt_bundle(path: Path) -> PromptBundle`
- Preserves: `extract_kimi_steps(...)`, `extract_swe_steps(...)`, `select_trajectory_files(...)`, `build_user_prompt(...)`, `parse_json_response(...)`, `validate_intent_result(...)`, `build_chat_payload(...)`, `extract_with_retries(...)`, `process_trajectory(...)`, and `summarize(...)`
- Produces: `portable_source_name(source_path: Path, data_root: Path | None) -> str`
- Produces: `build_parser() -> argparse.ArgumentParser`
- Produces: `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Establish the parent extraction baseline**

```bash
cd ..
python -m unittest tests.test_generate_thought_intents -v
cd Project
```

Expected: 6 tests pass without making an API request.

- [ ] **Step 2: Create migrated and new failing tests**

Port the six approved tests to package imports, then add prompt and portability tests:

```python
def test_loads_prompt_sections(tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text(
        "# Prompt\n\n## System Prompt\nSystem text.\n\n"
        "## User Prompt\nStep: {thought_step}\nThought: {thought}\n",
        encoding="utf-8",
    )
    bundle = load_prompt_bundle(prompt)
    self.assertEqual(bundle.system, "System text.")
    self.assertIn("{thought_step}", bundle.user_template)


def test_source_name_is_relative_to_data_root(tmp_path):
    source = tmp_path / "SWE" / "CASE-0001.json"
    source.parent.mkdir()
    source.write_text("{}", encoding="utf-8")
    self.assertEqual(
        portable_source_name(source, tmp_path),
        "SWE/CASE-0001.json",
    )


def test_live_run_requires_named_intent_api_key():
    with unittest.mock.patch.dict(os.environ, {}, clear=True):
        with self.assertRaisesRegex(SystemExit, "INTENT_API_KEY"):
            main(["--swe-dir", "x", "--swe-count", "1", "--kimi-count", "0"])
```

Import `os` and `unittest.mock` in the migrated test module for this test.

- [ ] **Step 3: Run tests and verify RED**

```bash
PYTHONPATH=src python -m unittest tests.test_intent_extraction -v
```

Expected: import failure for `thought_action_retrieval.intent.extraction`.

- [ ] **Step 4: Create the approved prompt file**

Copy the complete values of `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE` from the approved parent source into `prompts/thought_to_intent.md`, without including the Python quotes. The file begins with `# Observable Thought to Atomic Execution Intent`; place the System Prompt body after `## System Prompt` and the User Prompt template body after `## User Prompt`.

Verify the copied bodies before continuing:

```python
self.assertEqual(
    hashlib.sha256(bundle.system.encode("utf-8")).hexdigest(),
    "577eb51f44079762c9a9ba090e9cbfa4cf3fedeffd1fa9cc31b06790b96ae2f4",
)
self.assertEqual(
    hashlib.sha256(bundle.user_template.encode("utf-8")).hexdigest(),
    "8ffb169852f30225747bac475c71c928cb7102dbb255f2ab42ad474c106ac5db",
)
```

`load_prompt_bundle()` must locate the two headings, strip their bodies, and reject a missing or empty section with `ValueError`. The production module loads this repository prompt once and uses the System Prompt SHA-256 in output metadata.

- [ ] **Step 5: Migrate extraction behavior and make paths portable**

Preserve the approved parsing, exact-source-quote validation, JSON response format, retry behavior, atomic writes, and resume behavior. Apply these deliberate changes:

1. CLI data-directory defaults are `None`, not repository-relative data paths.
2. When a directory argument is omitted, resolve it below `TA_DATA_ROOT` using `paths.py` and the relative locations in `configs/paths.example.toml`.
3. Read credentials only from `INTENT_API_KEY`; do not fall back to `API_KEY`.
4. CLI `--base-url` defaults to `INTENT_API_BASE_URL` or `https://api.deepseek.com`.
5. CLI `--model` defaults to `INTENT_MODEL_NAME` or `deepseek-v4-flash`.
6. Store source paths relative to `TA_DATA_ROOT`; if a test fixture lies outside it, store only the filename.
7. Accept `argv` in `main()` and return 0 on success.
8. Validate `INTENT_API_KEY` before resolving or reading input directories for every non-dry-run invocation.

The wrapper contains only:

```python
#!/usr/bin/env python3
from thought_action_retrieval.intent.extraction import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run tests and dry-run smoke test**

```bash
PYTHONPATH=src python -m unittest tests.test_intent_extraction -v
PYTHONPATH=src python scripts/extract_thought_intents.py --help >/dev/null
```

Expected: all nine extraction tests pass; help exits 0; no network request occurs.

- [ ] **Step 7: Commit Task 4**

```bash
git add prompts/thought_to_intent.md src/thought_action_retrieval/intent \
  scripts/extract_thought_intents.py tests/test_intent_extraction.py
git commit -m "feat: migrate observable Thought intent extraction"
```

---

### Task 5: Import the selected SWE snapshot metadata and samples

**Files:**
- Create: `src/thought_action_retrieval/data/snapshot.py`
- Create: `scripts/import_selected_swe_snapshot.py`
- Create: `tests/test_swe_snapshot.py`
- Generate: `data/manifests/swe-selected-500/dataset_manifest.json`
- Generate: `data/manifests/swe-selected-500/manifest.csv`
- Generate: `data/manifests/swe-selected-500/selection_summary.json`
- Generate: `data/samples/swe-selected-500/CASE-0001.json`
- Generate: `data/samples/swe-selected-500/CASE-0251.json`
- Generate: `data/samples/swe-selected-500/CASE-0500.json`

**Interfaces:**
- Produces: `validate_selected_snapshot(source_dir: Path, expected_count: int) -> dict[str, object]`
- Produces: `import_selected_snapshot(*, data_root: Path, source_relative: Path, repository_root: Path, sample_ids: Sequence[str], raw_shards_relative: Path, expected_count: int = 500) -> dict[str, object]`
- Produces CLI with `--source-relative`, `--raw-shards-relative`, `--sample-id`, and `--expected-count`.

- [ ] **Step 1: Write a failing synthetic snapshot test**

Build a two-case temporary source with `CASE-0001.json`, `CASE-0002.json`, `selected_2.jsonl`, a two-row `manifest.csv`, `selection_summary.json`, and two raw shard files. Assert:

```python
result = import_selected_snapshot(
    data_root=data_root,
    source_relative=Path("selected"),
    repository_root=repo_root,
    sample_ids=["CASE-0001"],
    raw_shards_relative=Path("raw"),
    expected_count=2,
)

self.assertEqual(result["selected_count"], 2)
self.assertTrue((repo_root / "data/samples/swe-selected-500/CASE-0001.json").exists())
manifest = json.loads(
    (repo_root / "data/manifests/swe-selected-500/dataset_manifest.json").read_text()
)
self.assertEqual(manifest["metadata"]["selected_count"], 2)
self.assertEqual(len(manifest["metadata"]["raw_shards"]), 2)
self.assertFalse((repo_root / "data" / "selected_2.jsonl").exists())
```

Add rejection tests for duplicate CASE IDs, mismatched CSV/JSONL counts, missing sample IDs, and a summary whose `selected_count` differs from `expected_count`.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=src python -m unittest tests.test_swe_snapshot -v
```

Expected: import failure for `thought_action_retrieval.data.snapshot`.

- [ ] **Step 3: Implement strict snapshot validation and atomic import**

Validation must parse every CASE file and JSONL row, enforce four-digit filenames and unique IDs, reconcile manifest rows, and verify these summary fields for the real 500-case dataset:

```python
EXPECTED_TARGET_COUNTS = {"true": 250, "false": 250}
EXPECTED_MODEL_COUNTS = {
    "swe-agent-llama-70b": 350,
    "swe-agent-llama-8b": 100,
    "swe-agent-llama-405b": 50,
}
EXPECTED_SELECTION_GROUP_COUNTS = {"paired": 250, "diversity": 250}
```

For non-500 synthetic tests, require internal reconciliation but skip those frozen distribution constants.

The importer copies only `manifest.csv`, `selection_summary.json`, and requested samples into Git-managed directories. It computes but does not copy the full selected JSONL and raw shards. Record their relative paths, byte sizes, and SHA-256 digests in `dataset_manifest.json`. Write every destination via a temporary sibling and atomic rename.

- [ ] **Step 4: Run synthetic tests**

```bash
PYTHONPATH=src python -m unittest tests.test_swe_snapshot -v
```

Expected: all validation and import tests pass.

- [ ] **Step 5: Import real metadata from the external data pool**

```bash
TA_DATA_ROOT=/inspire/hdd/project/security-defense-and-attack/public/wx/data \
PYTHONPATH=src python scripts/import_selected_swe_snapshot.py \
  --source-relative SWE-agent-trajectories/test_data/selected_500_json \
  --raw-shards-relative SWE-agent-trajectories/data_jsonl \
  --sample-id CASE-0001 --sample-id CASE-0251 --sample-id CASE-0500 \
  --expected-count 500
```

Expected: three CASE samples plus the three manifest/summary files are written. The full selected JSONL and 12 raw shards remain only below `TA_DATA_ROOT`.

- [ ] **Step 6: Verify generated metadata**

```bash
python -m json.tool data/manifests/swe-selected-500/dataset_manifest.json >/dev/null
python -m json.tool data/manifests/swe-selected-500/selection_summary.json >/dev/null
test "$(find data/samples/swe-selected-500 -name 'CASE-*.json' | wc -l)" -eq 3
test ! -e data/manifests/swe-selected-500/selected_500.jsonl
git add src/thought_action_retrieval/data/snapshot.py \
  scripts/import_selected_swe_snapshot.py tests/test_swe_snapshot.py \
  data/manifests/swe-selected-500 data/samples/swe-selected-500
PYTHONPATH=src python scripts/audit_repository.py --root .
```

Expected: JSON parses, exactly three samples exist, no full JSONL exists in the repository, and the audit passes after staging the generated files.

- [ ] **Step 7: Commit Task 5**

```bash
git add src/thought_action_retrieval/data/snapshot.py \
  scripts/import_selected_swe_snapshot.py tests/test_swe_snapshot.py \
  data/manifests/swe-selected-500 data/samples/swe-selected-500
git commit -m "data: register selected SWE 500 snapshot"
```

---

### Task 6: Project documentation, approved research designs, and final verification

**Files:**
- Create: `README.md`
- Create: `data/README.md`
- Create: `outputs/README.md`
- Create: `docs/data-catalog.md`
- Create: `docs/designs/intent-action-phase1-data-design.md`
- Create: `docs/plans/intent-action-phase1-data.md`
- Create: `tests/test_documentation.py`

**Read-only migration sources:**
- `../docs/superpowers/specs/2026-09-16-intent-action-phase1-data-design.md`
- `../docs/superpowers/plans/2026-09-16-intent-action-phase1-data.md`

**Interfaces:**
- Produces documented commands for installation, tests, repository audit, data configuration, SWE selection, snapshot import, and Thought extraction.
- Produces a data catalog entry linking `swe-selected-500` to its tracked manifest.

- [ ] **Step 1: Write documentation contract tests**

Create `tests/test_documentation.py` that loads `README.md` and asserts it contains these literal commands and references:

```python
self.assertIn("python -m unittest discover -s tests -v", readme)
self.assertIn("scripts/audit_repository.py", readme)
self.assertIn("scripts/select_swe_trajectories.py", readme)
self.assertIn("scripts/extract_thought_intents.py", readme)
self.assertIn("TA_DATA_ROOT", readme)
self.assertIn("data/manifests/swe-selected-500/dataset_manifest.json", readme)
```

Also parse every repository-relative Markdown link in `README.md`, `data/README.md`, and `docs/data-catalog.md`; assert each local target exists. Ignore `http://` and `https://` links.

- [ ] **Step 2: Run documentation tests and verify RED**

```bash
PYTHONPATH=src python -m unittest tests.test_documentation -v
```

Expected: failure because `README.md` and catalog files do not exist.

- [ ] **Step 3: Write the root README**

The README must contain these sections in order:

1. `Scope`: observable Thought → execution Intent → short Action retrieval.
2. `Repository boundary`: explicitly states that large datasets are external and that this repository has no dependency on TACA.
3. `Setup`: Python 3.10+, editable install `python -m pip install -e .`, copy `.env.example` to an untracked `.env`, set `TA_DATA_ROOT`.
4. `Data`: link to `data/README.md`, `docs/data-catalog.md`, and the selected-500 dataset manifest.
5. `Commands`: exact test, audit, selector, snapshot import, and dry-run extraction commands.
6. `Research roadmap`: Phase 1 one-Intent–one-short-Action; later Kimi multi-positive and Action-set work.

Do not include a real API key, a private endpoint, or a required machine-specific absolute path.

- [ ] **Step 4: Write data and output policy documentation**

`data/README.md` must define:

- `data/manifests/` as tracked provenance;
- `data/samples/` as small test/review examples;
- `data/local/` as ignored optional links;
- prohibition on copying full external datasets into the repository.

`outputs/README.md` must state that outputs are ignored by default and that only concise reports or manifests are promoted to Git.

`docs/data-catalog.md` must document:

- external SWE raw JSONL path relative to `TA_DATA_ROOT`;
- selected-500 derivative path relative to `TA_DATA_ROOT`;
- 500 trajectories, 16,197 Thought–Action steps, 250/250 target balance, 350/100/50 model distribution, 320 unique instances, and 269 repositories;
- creation seed `20260814`;
- links to the tracked manifest, summary, and three samples;
- the selection command needed to reproduce the derivative.

- [ ] **Step 5: Migrate only the two approved Phase 1 documents**

Copy the exact parent design and implementation plan content into:

- `docs/designs/intent-action-phase1-data-design.md`;
- `docs/plans/intent-action-phase1-data.md`.

Change only repository-relative paths and script names that were established in Tasks 1–5. Do not import any other plan, report, or document.

- [ ] **Step 6: Run all tests and audits**

Stage all intended files first, because the audit examines tracked files:

```bash
git add README.md data/README.md outputs/README.md docs/data-catalog.md \
  docs/designs/intent-action-phase1-data-design.md \
  docs/plans/intent-action-phase1-data.md tests/test_documentation.py
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/audit_repository.py --root .
git diff --check --cached
```

Expected: every test passes, audit exits 0, and `git diff --check --cached` produces no output.

- [ ] **Step 7: Perform explicit boundary checks**

```bash
test -z "$(git remote)"
test -z "$(git ls-files | rg '(^|/)TACA(/|$)' || true)"
test -z "$(git ls-files | rg '(^|/)selected_500\.jsonl$' || true)"
test -z "$(git ls-files | rg '(^|/)\.env$|(^|/)\.codex/|(^|/)\.sii/' || true)"
test -z "$(git ls-files -z | xargs -0 -r du -b | awk '$1 > 10485760 {print $2}')"
```

Expected: all commands exit 0 with no disallowed tracked paths or oversized files.

- [ ] **Step 8: Commit Task 6**

```bash
git commit -m "docs: document standalone research workflow"
```

- [ ] **Step 9: Verify final repository state**

```bash
git status --short
git log --oneline --decorate --max-count=10
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/audit_repository.py --root .
```

Expected: clean worktree, no remote, all tests passing, repository audit passing, and commits for each task after the two planning commits.

Do not add a remote or push. Report that the local repository is ready for the owner to create and attach a private remote.
