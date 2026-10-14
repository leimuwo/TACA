import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from thought_action_retrieval.matching.phase1_dataset import (
    BuildConfig,
    assess_short_action,
    export_gold_relations,
    extract_candidates,
    serialize_swe_action,
    split_by_group,
    task_template_key,
    validate_reviewed_annotations,
)


class ActionNormalizationTests(unittest.TestCase):
    def test_serializes_known_short_actions_with_named_arguments(self):
        cases = {
            'search_dir "class Symbol" hy': (
                "search_dir",
                '[ACTION] search_dir(query="class Symbol", path="hy")',
            ),
            "open hy/models.py": (
                "open",
                '[ACTION] open(path="hy/models.py")',
            ),
            "open hy/models.py 550": (
                "open",
                '[ACTION] open(path="hy/models.py", line=550)',
            ),
            "goto 550": ("goto", "[ACTION] goto(line=550)"),
            "scroll_down": ("scroll_down", "[ACTION] scroll_down()"),
            "python test_symbol.py": (
                "python",
                '[ACTION] python(command="test_symbol.py")',
            ),
            "rm test_symbol.py": (
                "rm",
                '[ACTION] rm(path="test_symbol.py")',
            ),
            "create test_symbol.py": (
                "create",
                '[ACTION] create(path="test_symbol.py")',
            ),
        }

        for raw, (command, serialized) in cases.items():
            with self.subTest(raw=raw):
                result = serialize_swe_action(raw)
                self.assertEqual(result.raw, raw)
                self.assertEqual(result.command, command)
                self.assertEqual(result.serialized, serialized)

    def test_serialization_quotes_and_escapes_string_arguments(self):
        result = serialize_swe_action('search_dir "say \\"hello\\"" src')

        self.assertEqual(
            result.serialized,
            '[ACTION] search_dir(query="say \\\"hello\\\"", path="src")',
        )

    def test_serializes_unknown_short_command_without_losing_arguments(self):
        result = serialize_swe_action("find_file README.md docs")

        self.assertEqual(result.command, "find_file")
        self.assertEqual(
            result.serialized,
            '[ACTION] find_file(args="README.md docs")',
        )

    def test_accepts_short_single_line_actions(self):
        result = assess_short_action("python test_symbol.py")

        self.assertTrue(result.eligible)
        self.assertEqual(result.exclusion_reasons, ())
        self.assertEqual(result.normalization.command, "python")

    def test_rejects_edit_and_end_of_edit_actions(self):
        cases = {
            "edit 10:12": "edit_action",
            "end_of_edit": "end_of_edit",
            'desc="calculate f-statistics only", argstr="--fonly"': "assignment_fragment",
        }

        for raw, reason in cases.items():
            with self.subTest(raw=raw):
                result = assess_short_action(raw)
                self.assertFalse(result.eligible)
                self.assertIn(reason, result.exclusion_reasons)

    def test_rejects_multiline_and_heredoc_actions(self):
        cases = {
            "python - <<'PY'\nprint('x')\nPY": "heredoc",
            "python first.py\npython second.py": "multiline",
        }

        for raw, reason in cases.items():
            with self.subTest(raw=raw):
                result = assess_short_action(raw)
                self.assertFalse(result.eligible)
                self.assertIn(reason, result.exclusion_reasons)

    def test_rejects_composite_shell_actions_but_allows_quoted_operators(self):
        composite = assess_short_action("apt-get update && apt-get install python3")
        quoted = assess_short_action("python -c \"print('a && b')\"")

        self.assertFalse(composite.eligible)
        self.assertIn("composite_shell", composite.exclusion_reasons)
        self.assertTrue(quoted.eligible)

    def test_does_not_treat_environment_prefix_as_an_edit_fragment(self):
        result = assess_short_action("PYTHONPATH=src python test.py")

        self.assertTrue(result.eligible)
        self.assertNotIn("assignment_fragment", result.exclusion_reasons)

    def test_rejects_actions_over_total_or_payload_token_limits(self):
        total = assess_short_action("open " + "a" * 40, max_action_tokens=5)
        payload = assess_short_action(
            "create " + "b" * 40,
            max_action_tokens=128,
            max_payload_tokens=5,
        )

        self.assertFalse(total.eligible)
        self.assertIn("action_too_long", total.exclusion_reasons)
        self.assertFalse(payload.eligible)
        self.assertIn("payload_too_long", payload.exclusion_reasons)

    def test_rejects_empty_or_unparseable_actions(self):
        empty = assess_short_action("   ")
        malformed = assess_short_action('open "unterminated')

        self.assertFalse(empty.eligible)
        self.assertIn("empty_action", empty.exclusion_reasons)
        self.assertFalse(malformed.eligible)
        self.assertIn("unparseable_action", malformed.exclusion_reasons)


def _intent(intent_id="T1-I1", text="Open the target file."):
    return {
        "intent_id": intent_id,
        "intent_text": text,
        "source_quote": "I will open the target file.",
    }


def _step(
    number,
    *,
    thought="I will open the target file.",
    actions=None,
    intents=None,
    status="success",
):
    return {
        "thought_step": number,
        "thought": thought,
        "actual_actions": ["open target.py"] if actions is None else actions,
        "status": status,
        "intent_result": {
            "thought_step": number,
            "execution_intents": [_intent(f"T{number}-I1")] if intents is None else intents,
            "no_intent_reason": None,
        },
    }


def _trajectory(source_type="swe", trajectory_id="CASE-0001", steps=None):
    return {
        "schema_version": "thought_intent_v1",
        "source_type": source_type,
        "trajectory_id": trajectory_id,
        "task": "Fix the parser bug in target.py.",
        "status": "complete",
        "steps": [_step(1)] if steps is None else steps,
    }


class CandidateExtractionTests(unittest.TestCase):
    def test_extracts_only_swe_one_intent_one_short_action_steps(self):
        records = [
            _trajectory(
                steps=[
                    _step(1, actions=["open target.py"]),
                    _step(2, actions=["edit 10:12"]),
                    _step(3, actions=["goto 10", "scroll_down"]),
                    _step(4, intents=[]),
                    _step(5, status="error"),
                ]
            ),
            _trajectory(source_type="kimi", trajectory_id="KIMI-1"),
        ]

        result = extract_candidates(records, BuildConfig())

        self.assertEqual(len(result.annotation_candidates), 1)
        candidate = result.annotation_candidates[0]
        self.assertEqual(candidate["candidate_id"], "swe:CASE-0001:T1-I1")
        self.assertEqual(candidate["intent_text"], "Open the target file.")
        self.assertEqual(candidate["intent_input"], "[INTENT] Open the target file.")
        self.assertEqual(candidate["action"]["raw"], "open target.py")
        self.assertEqual(candidate["action"]["action_id"], "T1-A1")
        self.assertEqual(
            candidate["action"]["serialized"],
            '[ACTION] open(path="target.py")',
        )
        self.assertEqual(candidate["annotation"], {"label": "", "annotator": "", "notes": ""})
        self.assertEqual(result.audit_summary["sources"]["swe"]["trajectories"], 1)
        self.assertEqual(result.audit_summary["sources"]["kimi"]["trajectories"], 1)
        self.assertEqual(result.audit_summary["annotation_candidates"], 1)
        self.assertEqual(result.audit_summary["excluded"]["ineligible_action"], 1)
        self.assertEqual(result.audit_summary["excluded"]["action_count_not_one"], 1)
        self.assertEqual(result.audit_summary["excluded"]["intent_count_not_one"], 1)
        self.assertEqual(result.audit_summary["excluded"]["step_not_success"], 1)
        self.assertEqual(result.audit_summary["excluded"]["non_swe_source"], 1)
        self.assertEqual(
            result.audit_summary["action_exclusion_reasons"]["edit_action"],
            1,
        )

    def test_exports_one_intent_zero_action_as_unfulfilled_candidate(self):
        record = _trajectory(steps=[_step(3, actions=[])])

        result = extract_candidates([record], BuildConfig())

        self.assertEqual(result.annotation_candidates, ())
        self.assertEqual(len(result.unfulfilled_candidates), 1)
        row = result.unfulfilled_candidates[0]
        self.assertEqual(row["candidate_id"], "swe:CASE-0001:T3-I1")
        self.assertEqual(row["intent_id"], "T3-I1")
        self.assertIsNone(row["positive_action"])
        self.assertEqual(result.audit_summary["unfulfilled_candidates"], 1)

    def test_uses_same_stable_id_for_fulfilled_and_unfulfilled_intents(self):
        fulfilled = _trajectory(steps=[_step(3, actions=["open target.py"])])
        unfulfilled = _trajectory(steps=[_step(3, actions=[])])

        with self.assertRaisesRegex(ValueError, "duplicate candidate_id"):
            extract_candidates([fulfilled, unfulfilled], BuildConfig())

    def test_includes_neighboring_actions_for_review_context(self):
        record = _trajectory(
            steps=[
                _step(1, actions=["search_dir target src"]),
                _step(2, actions=["open target.py"]),
                _step(3, actions=["goto 50"]),
            ]
        )

        result = extract_candidates([record], BuildConfig())
        middle = result.annotation_candidates[1]

        self.assertEqual(middle["neighboring_actions"]["previous"], ["search_dir target src"])
        self.assertEqual(middle["neighboring_actions"]["next"], ["goto 50"])

    def test_rejects_duplicate_candidate_ids(self):
        duplicate = _trajectory()

        with self.assertRaisesRegex(ValueError, "duplicate candidate_id"):
            extract_candidates([duplicate, duplicate], BuildConfig())


class AnnotationReviewTests(unittest.TestCase):
    def setUp(self):
        build = extract_candidates(
            [
                _trajectory(
                    steps=[
                        _step(1, actions=["open target.py"]),
                        _step(2, actions=["goto 50"]),
                        _step(3, actions=[]),
                    ]
                )
            ],
            BuildConfig(),
        )
        self.candidates = build.annotation_candidates + build.unfulfilled_candidates

    def test_rejects_unknown_duplicate_missing_and_unsupported_reviews(self):
        cases = [
            (
                [{"candidate_id": "missing", "label": "direct_match"}],
                "unknown candidate_id",
            ),
            (
                [
                    {"candidate_id": "swe:CASE-0001:T1-I1", "label": "direct_match"},
                    {"candidate_id": "swe:CASE-0001:T1-I1", "label": "no_match"},
                ],
                "duplicate review",
            ),
            ([{"candidate_id": "swe:CASE-0001:T1-I1"}], "missing label"),
            (
                [{"candidate_id": "swe:CASE-0001:T1-I1", "label": "maybe"}],
                "unsupported label",
            ),
        ]

        for reviews, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_reviewed_annotations(self.candidates, reviews)

    def test_unreviewed_partial_and_ambiguous_candidates_fail_closed(self):
        review = validate_reviewed_annotations(
            self.candidates,
            [
                {
                    "candidate_id": "swe:CASE-0001:T1-I1",
                    "label": "partial_match",
                    "annotator": "reviewer-a",
                    "notes": "Only locates the file.",
                },
                {
                    "candidate_id": "swe:CASE-0001:T2-I1",
                    "label": "ambiguous",
                },
            ],
        )

        positives, evaluations = export_gold_relations(review)

        self.assertEqual(positives, ())
        self.assertEqual(evaluations, ())
        self.assertEqual(
            review.unreviewed_candidate_ids,
            ("swe:CASE-0001:T3-I1",),
        )
        self.assertEqual(review.reviewed_rows[0]["annotation"]["annotator"], "reviewer-a")

    def test_exports_only_direct_matches_as_individual_positives(self):
        review = validate_reviewed_annotations(
            self.candidates,
            [
                {
                    "candidate_id": "swe:CASE-0001:T1-I1",
                    "label": "direct_match",
                    "annotator": "reviewer-a",
                },
                {
                    "candidate_id": "swe:CASE-0001:T2-I1",
                    "label": "no_match",
                    "notes": "Wrong operation.",
                },
                {
                    "candidate_id": "swe:CASE-0001:T3-I1",
                    "label": "unfulfilled",
                },
            ],
        )

        positives, evaluations = export_gold_relations(review)

        self.assertEqual(len(positives), 1)
        positive = positives[0]
        self.assertEqual(positive["candidate_id"], "swe:CASE-0001:T1-I1")
        self.assertEqual(positive["trajectory_id"], "CASE-0001")
        self.assertEqual(positive["intent_id"], "T1-I1")
        self.assertEqual(positive["relation_type"], "individual")
        self.assertEqual(positive["source"], "swe")
        self.assertEqual(positive["positive_action"]["action_id"], "T1-A1")
        self.assertEqual(len(evaluations), 3)
        self.assertEqual(
            [row["relation_label"] for row in evaluations],
            ["direct_match", "no_match", "unfulfilled"],
        )
        self.assertIsNone(evaluations[2]["action"])

    def test_rejects_direct_match_without_an_action_and_unfulfilled_with_action(self):
        cases = [
            (
                "swe:CASE-0001:T3-I1",
                "direct_match",
                "direct_match requires an Action",
            ),
            (
                "swe:CASE-0001:T1-I1",
                "unfulfilled",
                "unfulfilled requires no Action",
            ),
        ]

        for candidate_id, label, message in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    validate_reviewed_annotations(
                        self.candidates,
                        [{"candidate_id": candidate_id, "label": label}],
                    )


class GroupSplitTests(unittest.TestCase):
    def test_normalizes_issue_specific_values_into_one_template(self):
        first = 'Fix issue 123 at line 550 in "README.md" under src/pkg/a.py'
        second = 'Fix issue 999 at line 450 in "AGENTS.md" under src/pkg/b.py'

        self.assertEqual(task_template_key(first), task_template_key(second))

    def test_keeps_trajectories_and_near_duplicate_templates_in_one_split(self):
        rows = [
            {
                "candidate_id": "A-1",
                "trajectory_id": "A",
                "task": "Fix issue 10 in src/a.py",
            },
            {
                "candidate_id": "A-2",
                "trajectory_id": "A",
                "task": "A differently worded secondary description",
            },
            {
                "candidate_id": "B-1",
                "trajectory_id": "B",
                "task": "Fix issue 20 in src/b.py",
            },
            {
                "candidate_id": "C-1",
                "trajectory_id": "C",
                "task": "Repair parser behavior for alpha inputs",
            },
            {
                "candidate_id": "D-1",
                "trajectory_id": "D",
                "task": "Correct serializer behavior for beta inputs",
            },
            {
                "candidate_id": "E-1",
                "trajectory_id": "E",
                "task": "Resolve cache invalidation for gamma objects",
            },
        ]

        result = split_by_group(rows, seed=17)
        locations = {
            row["candidate_id"]: split
            for split, split_rows in (
                ("train", result.train),
                ("validation", result.validation),
                ("test", result.test),
            )
            for row in split_rows
        }

        self.assertEqual(locations["A-1"], locations["A-2"])
        self.assertEqual(locations["A-1"], locations["B-1"])
        self.assertEqual(result.manifest["trajectory_overlap"], [])
        self.assertEqual(result.manifest["template_overlap"], [])
        self.assertEqual(result.manifest["row_count"], len(rows))
        self.assertEqual(sum(result.manifest["split_counts"].values()), len(rows))

    def test_split_is_deterministic_and_does_not_mutate_input_rows(self):
        rows = [
            {
                "candidate_id": f"CASE-{index}",
                "trajectory_id": f"CASE-{index}",
                "task": f"Unique task word{chr(97 + index)}",
            }
            for index in range(10)
        ]

        first = split_by_group(rows, seed=20260916)
        second = split_by_group(rows, seed=20260916)

        self.assertEqual(first, second)
        self.assertTrue(all("split" not in row for row in rows))
        self.assertEqual(first.manifest["seed"], 20260916)

    def test_production_positive_split_is_invariant_to_input_order(self):
        rows = [
            {
                "candidate_id": f"swe:CASE-{index}:T1-I1",
                "trajectory_id": f"CASE-{index}",
                "task": f"Distinct task token{chr(97 + index)}",
                "intent_id": "T1-I1",
                "positive_action": {"action_id": "T1-A1", "raw": "scroll_down"},
                "provenance": {"candidate_id": f"swe:CASE-{index}:T1-I1"},
            }
            for index in range(12)
        ]

        forward = split_by_group(rows, seed=42)
        reverse = split_by_group(list(reversed(rows)), seed=42)

        def assignments(result):
            return {
                row["candidate_id"]: split
                for split, values in (
                    ("train", result.train),
                    ("validation", result.validation),
                    ("test", result.test),
                )
                for row in values
            }

        self.assertEqual(assignments(forward), assignments(reverse))

    def test_split_rejects_rows_without_a_stable_candidate_id(self):
        with self.assertRaisesRegex(ValueError, "stable candidate_id"):
            split_by_group(
                [{"trajectory_id": "CASE-1", "task": "Fix parser behavior"}],
                seed=1,
            )


class Phase1DatasetCliTests(unittest.TestCase):
    def _write_source(self, directory):
        source = Path(directory) / "thought_intent"
        source.mkdir()
        (source / "swe__CASE-0001.json").write_text(
            json.dumps(
                _trajectory(
                    steps=[
                        _step(1, actions=["open target.py"]),
                        _step(2, actions=[]),
                    ]
                )
            ),
            encoding="utf-8",
        )
        (source / "kimi__CASE-0001.json").write_text(
            json.dumps(_trajectory(source_type="kimi", trajectory_id="KIMI-1")),
            encoding="utf-8",
        )
        (source / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "thought_intent_manifest_v1",
                    "selected_trajectory_count": 2,
                    "source_counts": {"kimi": 1, "swe": 1},
                    "step_count": 3,
                }
            ),
            encoding="utf-8",
        )
        return source

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "scripts/build_intent_action_dataset.py", *args],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_build_writes_audit_annotation_jsonl_csv_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self._write_source(temporary)
            output = Path(temporary) / "phase1"

            result = self._run(
                "build",
                "--input-dir",
                str(source),
                "--output-dir",
                str(output),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "annotation_candidates.jsonl",
                    "annotation_template.csv",
                    "audit_summary.json",
                    "build_manifest.json",
                    "unfulfilled_candidates.jsonl",
                },
            )
            candidates = [
                json.loads(line)
                for line in (output / "annotation_candidates.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            with (output / "annotation_template.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(candidates), 1)
            self.assertEqual(len(csv_rows), len(candidates) + 1)
            self.assertEqual(csv_rows[0]["candidate_id"], candidates[0]["candidate_id"])
            self.assertEqual(csv_rows[1]["candidate_id"], "swe:CASE-0001:T2-I1")
            audit = json.loads((output / "audit_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["sources"]["kimi"]["trajectories"], 1)
            self.assertEqual(audit["annotation_candidates"], 1)

            repeated = self._run(
                "build",
                "--input-dir",
                str(source),
                "--output-dir",
                str(output),
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("already exists", repeated.stderr)

    def test_finalize_requires_reviews_and_writes_gold_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self._write_source(temporary)
            output = Path(temporary) / "phase1"
            build = self._run(
                "build",
                "--input-dir",
                str(source),
                "--output-dir",
                str(output),
            )
            self.assertEqual(build.returncode, 0, build.stderr)

            missing = self._run(
                "finalize",
                "--dataset-dir",
                str(output),
            )
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("reviewed annotations", missing.stderr)

            reviews = [
                {
                    "candidate_id": "swe:CASE-0001:T1-I1",
                    "label": "direct_match",
                    "annotator": "reviewer-a",
                },
                {
                    "candidate_id": "swe:CASE-0001:T2-I1",
                    "label": "unfulfilled",
                },
            ]
            reviewed_path = output / "reviewed_annotations.jsonl"
            reviewed_path.write_text(
                "".join(json.dumps(row) + "\n" for row in reviews),
                encoding="utf-8",
            )

            finalized = self._run(
                "finalize",
                "--dataset-dir",
                str(output),
            )

            self.assertEqual(finalized.returncode, 0, finalized.stderr)
            self.assertTrue((output / "positives.jsonl").is_file())
            self.assertTrue((output / "evaluation_relations.jsonl").is_file())
            self.assertTrue((output / "split_manifest.json").is_file())
            self.assertEqual(
                {path.name for path in (output / "splits").iterdir()},
                {"train.jsonl", "validation.jsonl", "test.jsonl"},
            )
            positive_rows = [
                json.loads(line)
                for line in (output / "positives.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            split_rows = sum(
                len((output / "splits" / name).read_text(encoding="utf-8").splitlines())
                for name in ("train.jsonl", "validation.jsonl", "test.jsonl")
            )
            self.assertEqual(len(positive_rows), 1)
            self.assertEqual(split_rows, len(positive_rows))

    def test_build_fails_on_unexpected_or_wrong_schema_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self._write_source(temporary)
            (source / "broken.json").write_text("{}", encoding="utf-8")

            result = self._run(
                "build",
                "--input-dir",
                str(source),
                "--output-dir",
                str(Path(temporary) / "phase1"),
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid Thought-to-Intent trajectory", result.stderr)

    def test_build_requires_phase0_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self._write_source(temporary)
            (source / "manifest.json").unlink()

            result = self._run(
                "build",
                "--input-dir",
                str(source),
                "--output-dir",
                str(Path(temporary) / "phase1"),
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Phase 0 manifest not found", result.stderr)

    def test_build_rejects_null_or_wrong_schema_steps(self):
        invalid_steps = {
            "null step": None,
            "non-integer thought_step": {
                **_step(1),
                "thought_step": "1",
            },
            "non-string thought": {
                **_step(1),
                "thought": None,
            },
            "non-string status": {
                **_step(1),
                "status": None,
            },
            "non-list actual_actions": {
                **_step(1),
                "actual_actions": "open target.py",
            },
            "non-object intent_result": {
                **_step(1),
                "intent_result": [],
            },
            "non-list execution_intents": {
                **_step(1),
                "intent_result": {
                    **_step(1)["intent_result"],
                    "execution_intents": {},
                },
            },
        }

        for name, invalid_step in invalid_steps.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                source = self._write_source(temporary)
                trajectory_path = source / "swe__CASE-0001.json"
                trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
                trajectory["steps"] = [invalid_step, _step(2, actions=[])]
                trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")

                result = self._run(
                    "build",
                    "--input-dir",
                    str(source),
                    "--output-dir",
                    str(Path(temporary) / "phase1"),
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid Thought-to-Intent step", result.stderr)

    def test_build_reconciles_existing_phase0_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self._write_source(temporary)
            (source / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "thought_intent_manifest_v1",
                        "selected_trajectory_count": 99,
                        "source_counts": {"swe": 1, "kimi": 1},
                        "step_count": 3,
                    }
                ),
                encoding="utf-8",
            )

            result = self._run(
                "build",
                "--input-dir",
                str(source),
                "--output-dir",
                str(Path(temporary) / "phase1"),
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("manifest trajectory count", result.stderr)

    def test_finalize_publication_rolls_back_partial_artifacts(self):
        from scripts.build_intent_action_dataset import publish_finalized_artifacts

        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset"
            staged = dataset / ".finalize-stage"
            dataset.mkdir()
            (staged / "splits").mkdir(parents=True)
            for name in (
                "positives.jsonl",
                "evaluation_relations.jsonl",
                "split_manifest.json",
            ):
                (staged / name).write_text(name, encoding="utf-8")
            for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
                (staged / "splits" / name).write_text(name, encoding="utf-8")

            calls = 0

            def fail_on_second_replace(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected publication failure")
                os.replace(source, destination)

            with self.assertRaisesRegex(OSError, "injected publication failure"):
                publish_finalized_artifacts(
                    staged,
                    dataset,
                    replace=fail_on_second_replace,
                )

            self.assertFalse((dataset / "positives.jsonl").exists())
            self.assertFalse((dataset / "evaluation_relations.jsonl").exists())
            self.assertFalse((dataset / "splits").exists())
            self.assertFalse((dataset / "split_manifest.json").exists())
            self.assertFalse(staged.exists())


class Phase1RealCorpusTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("TA_DATA_ROOT"), "TA_DATA_ROOT is not configured")
    def test_phase0_corpus_reconciles_before_candidate_export(self):
        from scripts.build_intent_action_dataset import _load_records

        input_dir = (
            Path(os.environ["TA_DATA_ROOT"])
            / "SWE-agent-trajectories"
            / "test_data"
            / "thought_intent"
        )
        if not input_dir.is_dir():
            self.skipTest(f"external corpus is unavailable: {input_dir}")

        records, source_paths, manifest_path = _load_records(input_dir)
        result = extract_candidates(records, BuildConfig())

        self.assertEqual(len(source_paths), 50)
        self.assertIsNotNone(manifest_path)
        self.assertEqual(result.audit_summary["trajectories"], 50)
        self.assertEqual(result.audit_summary["steps"], 1166)
        self.assertEqual(
            result.audit_summary["sources"]["swe"],
            {"trajectories": 25, "steps": 1012},
        )
        self.assertEqual(
            result.audit_summary["sources"]["kimi"],
            {"trajectories": 25, "steps": 154},
        )
        self.assertTrue(result.annotation_candidates)
        self.assertTrue(
            all(row["source"] == "swe" for row in result.annotation_candidates)
        )


if __name__ == "__main__":
    unittest.main()
