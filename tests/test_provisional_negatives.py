import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from thought_action_retrieval.matching.provisional_negatives import (
    build_generation_request,
    build_split_indexes,
    parse_serialized_action,
    prepare_provisional_rows,
    select_validated_negatives,
    validate_generation_response,
    validate_negative_proposal,
)


def _row(
    candidate_id,
    serialized,
    *,
    trajectory_id="CASE-0001",
    split="train",
    intent_text="Go to line 550.",
):
    return {
        "candidate_id": candidate_id,
        "trajectory_id": trajectory_id,
        "thought_step": int(candidate_id.rsplit("T", 1)[1].split("-", 1)[0]),
        "intent_text": intent_text,
        "intent_input": f"[INTENT] {intent_text}",
        "action": {
            "raw": serialized,
            "serialized": serialized,
        },
        "split": split,
        "source": "swe",
        "supervision_status": "provisional_auto_pair",
        "experiment_tier": "feasibility_only",
        "human_reviewed": False,
    }


class _GenerationHandler(BaseHTTPRequestHandler):
    request_count = 0

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).request_count += 1
        user_prompt = payload["messages"][-1]["content"]
        if user_prompt == 'Return exactly {"ok": true}.':
            content = '{"ok": true}'
        else:
            request_id = user_prompt.split('"request_id": "', 1)[1].split('"', 1)[0]
            positive = user_prompt.split('"positive_action": "', 1)[1].split('"', 1)[0]
            line = 999 if "line=999" not in positive else 998
            content = json.dumps(
                {
                    "request_id": request_id,
                    "negative_cases": [
                        {
                            "negative_type": "same_tool_wrong_parameter",
                            "serialized_action": f"[ACTION] goto(line={line})",
                            "source_candidate_id": None,
                            "reason": "The line number does not match the Intent.",
                        }
                    ],
                }
            )
        body = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


class ActionParsingTests(unittest.TestCase):
    def test_parses_known_serialized_action_fields(self):
        cases = {
            '[ACTION] goto(line=550)': ("goto", (("line", 550),)),
            '[ACTION] open(path="hy/models.py")': (
                "open",
                (("path", "hy/models.py"),),
            ),
            '[ACTION] search_dir(query="class Symbol", path="hy")': (
                "search_dir",
                (("query", "class Symbol"), ("path", "hy")),
            ),
            '[ACTION] scroll_down()': ("scroll_down", ()),
            '[ACTION] ./login.sh()': ("./login.sh", ()),
        }

        for text, expected in cases.items():
            with self.subTest(text=text):
                parsed = parse_serialized_action(text)
                self.assertEqual((parsed.tool, parsed.fields), expected)
                self.assertEqual(parsed.serialized, text)

    def test_rejects_malformed_or_non_action_text(self):
        for text in ("goto 550", "[ACTION] goto(line=)", "[ACTION] bad(a=one)"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_serialized_action(text)


class ProposalValidationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            _row("swe:CASE-0001:T1-I1", '[ACTION] goto(line=550)'),
            _row("swe:CASE-0001:T3-I1", '[ACTION] goto(line=450)'),
            _row(
                "swe:CASE-0001:T5-I1",
                '[ACTION] open(path="test_symbol.py")',
                intent_text="Open test_symbol.py.",
            ),
            _row(
                "swe:CASE-0001:T7-I1",
                '[ACTION] rm(path="scratch.txt")',
                intent_text="Remove scratch.txt.",
            ),
            _row(
                "swe:CASE-0002:T2-I1",
                '[ACTION] python(command="test_symbol.py")',
                trajectory_id="CASE-0002",
                intent_text="Run test_symbol.py.",
            ),
            _row(
                "swe:CASE-0003:T2-I1",
                '[ACTION] goto(line=300)',
                trajectory_id="CASE-0003",
                split="validation",
            ),
        ]
        self.indexes = build_split_indexes(self.rows)

    def test_accepts_one_field_same_tool_parameter_change(self):
        parent = self.rows[0]
        result = validate_negative_proposal(
            parent,
            {
                "negative_type": "same_tool_wrong_parameter",
                "serialized_action": "[ACTION] goto(line=450)",
                "source_candidate_id": "swe:CASE-0001:T3-I1",
                "reason": "The requested line is 550, not 450.",
            },
            self.indexes,
        )

        self.assertEqual(result.changed_fields, ("line",))
        self.assertEqual(result.value_origin, "observed_value")
        self.assertEqual(result.provenance, "provisional_llm_generated")
        self.assertEqual(result.source_candidate_id, "swe:CASE-0001:T3-I1")

    def test_accepts_wrong_tool_with_exact_same_target(self):
        parent = self.rows[4]
        result = validate_negative_proposal(
            parent,
            {
                "negative_type": "wrong_tool_same_object",
                "serialized_action": '[ACTION] open(path="test_symbol.py")',
                "source_candidate_id": "swe:CASE-0001:T5-I1",
                "reason": "Opening the test does not execute it.",
            },
            self.indexes,
        )

        self.assertEqual(result.negative_type, "wrong_tool_same_object")
        self.assertEqual(result.target, "test_symbol.py")

    def test_rejects_matching_multifield_and_cross_split_proposals(self):
        parent = self.rows[0]
        cases = [
            (
                {
                    "negative_type": "same_tool_wrong_parameter",
                    "serialized_action": "[ACTION] goto(line=550)",
                    "reason": "same",
                },
                "must differ",
            ),
            (
                {
                    "negative_type": "same_tool_wrong_parameter",
                    "serialized_action": '[ACTION] search_dir(query="x", path="y")',
                    "reason": "wrong tool and fields",
                },
                "same tool",
            ),
            (
                {
                    "negative_type": "same_tool_wrong_parameter",
                    "serialized_action": "[ACTION] goto(line=300)",
                    "source_candidate_id": "swe:CASE-0003:T2-I1",
                    "reason": "cross split",
                },
                "same split",
            ),
        ]

        for proposal, message in cases:
            with self.subTest(proposal=proposal):
                with self.assertRaisesRegex(ValueError, message):
                    validate_negative_proposal(parent, proposal, self.indexes)

    def test_same_trajectory_unrelated_requires_observed_distant_action(self):
        parent = self.rows[0]
        accepted = validate_negative_proposal(
            parent,
            {
                "negative_type": "same_trajectory_unrelated",
                "serialized_action": '[ACTION] rm(path="scratch.txt")',
                "source_candidate_id": "swe:CASE-0001:T7-I1",
                "reason": "Different tool and target.",
            },
            self.indexes,
        )

        self.assertEqual(accepted.risk_flags, ("higher_false_negative_risk",))

        navigation = _row(
            "swe:CASE-0001:T9-I1",
            '[ACTION] open(path="other.py")',
            intent_text="Open other.py.",
        )
        navigation_indexes = build_split_indexes([*self.rows, navigation])
        with self.assertRaisesRegex(ValueError, "supporting or navigation"):
            validate_negative_proposal(
                parent,
                {
                    "negative_type": "same_trajectory_unrelated",
                    "serialized_action": '[ACTION] open(path="other.py")',
                    "source_candidate_id": "swe:CASE-0001:T9-I1",
                    "reason": "Different target but potentially supporting.",
                },
                navigation_indexes,
            )

        with self.assertRaisesRegex(ValueError, "same trajectory"):
            validate_negative_proposal(
                parent,
                {
                    "negative_type": "same_trajectory_unrelated",
                    "serialized_action": '[ACTION] python(command="test_symbol.py")',
                    "source_candidate_id": "swe:CASE-0002:T2-I1",
                    "reason": "Different trajectory.",
                },
                self.indexes,
            )


class ProvisionalPreparationTests(unittest.TestCase):
    def test_prepares_stable_split_isolated_provisional_rows(self):
        candidates = [
            {
                **_row("swe:CASE-0001:T1-I1", '[ACTION] goto(line=550)'),
                "task": "Fix issue at line 550 in parser.py",
            },
            {
                **_row("swe:CASE-0001:T3-I1", '[ACTION] goto(line=450)'),
                "task": "Fix issue at line 550 in parser.py",
            },
            {
                **_row(
                    "swe:CASE-0002:T1-I1",
                    '[ACTION] open(path="parser.py")',
                    trajectory_id="CASE-0002",
                ),
                "task": "Fix issue at line 450 in parser.py",
            },
            {
                **_row(
                    "swe:CASE-0003:T1-I1",
                    '[ACTION] python(command="test_parser.py")',
                    trajectory_id="CASE-0003",
                ),
                "task": "Run parser regression tests",
            },
        ]
        raw_candidates = []
        for row in candidates:
            copy = dict(row)
            for key in (
                "split",
                "supervision_status",
                "experiment_tier",
                "human_reviewed",
            ):
                copy.pop(key)
            raw_candidates.append(copy)

        forward = prepare_provisional_rows(raw_candidates, seed=20260921)
        reverse = prepare_provisional_rows(list(reversed(raw_candidates)), seed=20260921)

        assignments = lambda rows: {row["candidate_id"]: row["split"] for row in rows}
        self.assertEqual(assignments(forward), assignments(reverse))
        self.assertEqual(forward[0]["supervision_status"], "provisional_auto_pair")
        self.assertEqual(forward[0]["experiment_tier"], "feasibility_only")
        self.assertFalse(forward[0]["human_reviewed"])
        self.assertEqual(
            assignments(forward)["swe:CASE-0001:T1-I1"],
            assignments(forward)["swe:CASE-0001:T3-I1"],
        )
        self.assertEqual(
            assignments(forward)["swe:CASE-0001:T1-I1"],
            assignments(forward)["swe:CASE-0002:T1-I1"],
        )

    def test_generation_request_has_stable_same_split_bounded_pools(self):
        rows = [
            _row("swe:CASE-0001:T1-I1", '[ACTION] goto(line=550)'),
            _row("swe:CASE-0001:T3-I1", '[ACTION] goto(line=450)'),
            _row(
                "swe:CASE-0001:T5-I1",
                '[ACTION] open(path="other.py")',
                intent_text="Open other.py.",
            ),
            _row(
                "swe:CASE-0002:T1-I1",
                '[ACTION] goto(line=300)',
                trajectory_id="CASE-0002",
            ),
            _row(
                "swe:CASE-0003:T1-I1",
                '[ACTION] goto(line=200)',
                trajectory_id="CASE-0003",
                split="validation",
            ),
        ]
        indexes = build_split_indexes(rows)

        request = build_generation_request(rows[0], indexes, max_pool_actions=2)
        repeated = build_generation_request(rows[0], indexes, max_pool_actions=2)

        self.assertEqual(request, repeated)
        self.assertEqual(request["request_id"], repeated["request_id"])
        self.assertEqual(request["split"], "train")
        pool_ids = [item["candidate_id"] for item in request["observed_action_pool"]]
        self.assertNotIn("swe:CASE-0001:T1-I1", pool_ids)
        self.assertNotIn("swe:CASE-0003:T1-I1", pool_ids)
        self.assertLessEqual(len(pool_ids), 2)
        self.assertIn("swe:CASE-0001:T3-I1", pool_ids)


class GenerationResponseTests(unittest.TestCase):
    def setUp(self):
        self.parent = _row(
            "swe:CASE-0002:T2-I1",
            '[ACTION] python(command="test_symbol.py")',
            trajectory_id="CASE-0002",
            intent_text="Run test_symbol.py.",
        )
        self.rows = [
            self.parent,
            _row(
                "swe:CASE-0002:T4-I1",
                '[ACTION] python(command="other_test.py")',
                trajectory_id="CASE-0002",
            ),
            _row(
                "swe:CASE-0002:T6-I1",
                '[ACTION] rm(path="scratch.txt")',
                trajectory_id="CASE-0002",
            ),
            _row(
                "swe:CASE-0001:T5-I1",
                '[ACTION] open(path="test_symbol.py")',
            ),
        ]
        self.indexes = build_split_indexes(self.rows)
        self.request = build_generation_request(self.parent, self.indexes)

    def test_validates_response_and_retains_rejection_audit(self):
        response = {
            "request_id": self.request["request_id"],
            "negative_cases": [
                {
                    "negative_type": "same_tool_wrong_parameter",
                    "serialized_action": '[ACTION] python(command="other_test.py")',
                    "source_candidate_id": "swe:CASE-0002:T4-I1",
                    "reason": "This runs a different test.",
                },
                {
                    "negative_type": "wrong_tool_same_object",
                    "serialized_action": '[ACTION] open(path="test_symbol.py")',
                    "source_candidate_id": "swe:CASE-0001:T5-I1",
                    "reason": "Opening the file does not run it.",
                },
                {
                    "negative_type": "same_trajectory_unrelated",
                    "serialized_action": '[ACTION] rm(path="scratch.txt")',
                    "source_candidate_id": "swe:CASE-0002:T6-I1",
                    "reason": "Removing a scratch file is unrelated.",
                },
                {
                    "negative_type": "same_tool_wrong_parameter",
                    "serialized_action": '[ACTION] python(command="test_symbol.py")',
                    "source_candidate_id": None,
                    "reason": "Invalid duplicate positive.",
                },
            ],
        }

        result = validate_generation_response(
            self.parent, self.request, response, self.indexes
        )

        self.assertEqual(len(result.accepted), 3)
        self.assertEqual(len(result.rejected), 1)
        self.assertIn("must differ", result.rejected[0]["error"])

    def test_rejects_wrong_request_id_or_non_list_cases(self):
        cases = [
            {"request_id": "wrong", "negative_cases": []},
            {"request_id": self.request["request_id"], "negative_cases": {}},
        ]
        for response in cases:
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    validate_generation_response(
                        self.parent, self.request, response, self.indexes
                    )

    def test_selection_caps_and_prioritizes_negative_types(self):
        response = {
            "request_id": self.request["request_id"],
            "negative_cases": [
                {
                    "negative_type": "same_tool_wrong_parameter",
                    "serialized_action": f'[ACTION] python(command="test_{index}.py")',
                    "source_candidate_id": None,
                    "reason": "Different test.",
                }
                for index in range(6)
            ]
            + [
                {
                    "negative_type": "wrong_tool_same_object",
                    "serialized_action": '[ACTION] open(path="test_symbol.py")',
                    "source_candidate_id": "swe:CASE-0001:T5-I1",
                    "reason": "Opening does not execute.",
                },
                {
                    "negative_type": "same_trajectory_unrelated",
                    "serialized_action": '[ACTION] rm(path="scratch.txt")',
                    "source_candidate_id": "swe:CASE-0002:T6-I1",
                    "reason": "Unrelated cleanup.",
                },
            ],
        }
        validated = validate_generation_response(
            self.parent, self.request, response, self.indexes
        )

        selected = select_validated_negatives(
            self.parent["candidate_id"], validated.accepted
        )

        self.assertEqual(len(selected), 5)
        counts = {}
        for row in selected:
            counts[row["negative_type"]] = counts.get(row["negative_type"], 0) + 1
        self.assertEqual(counts["same_tool_wrong_parameter"], 3)
        self.assertEqual(counts["wrong_tool_same_object"], 1)
        self.assertEqual(counts["same_trajectory_unrelated"], 1)
        self.assertTrue(all(row["negative_id"].startswith(self.parent["candidate_id"]) for row in selected))


class ProvisionalCliTests(unittest.TestCase):
    @staticmethod
    def _write_phase1_source(source: Path) -> None:
        source.mkdir()
        candidates = [
            {
                **_row("swe:CASE-0001:T1-I1", '[ACTION] goto(line=550)'),
                "task": "Fix parser line 550",
            },
            {
                **_row("swe:CASE-0001:T3-I1", '[ACTION] goto(line=450)'),
                "task": "Fix parser line 550",
            },
        ]
        for row in candidates:
            for key in (
                "split",
                "supervision_status",
                "experiment_tier",
                "human_reviewed",
            ):
                row.pop(key)
        (source / "annotation_candidates.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in candidates),
            encoding="utf-8",
        )
        (source / "build_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "intent_action_phase1_build_v1",
                    "counts": {"annotation_candidates": 2},
                }
            ),
            encoding="utf-8",
        )

    def test_prepare_writes_watermarked_pairs_requests_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "phase1"
            output = root / "pilot"
            self._write_phase1_source(source)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/generate_provisional_negatives.py",
                    "prepare",
                    "--input-dir",
                    str(source),
                    "--output-dir",
                    str(output),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "generation_requests.jsonl",
                    "prepare_manifest.json",
                    "provisional_pairs.jsonl",
                },
            )
            pairs = [
                json.loads(line)
                for line in (output / "provisional_pairs.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            requests = [
                json.loads(line)
                for line in (output / "generation_requests.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            manifest = json.loads(
                (output / "prepare_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(pairs), len(requests), 2)
            self.assertTrue(all(not row["human_reviewed"] for row in pairs))
            self.assertEqual(manifest["candidate_count"], 2)
            self.assertEqual(manifest["experiment_tier"], "feasibility_only")

    def test_generate_requires_environment_credential_before_writing_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "phase1"
            dataset = root / "pilot"
            self._write_phase1_source(source)
            repository = Path(__file__).resolve().parents[1]
            subprocess.run(
                [
                    sys.executable,
                    "scripts/generate_provisional_negatives.py",
                    "prepare",
                    "--input-dir",
                    str(source),
                    "--output-dir",
                    str(dataset),
                ],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
            environment = dict(os.environ)
            environment.pop("INF_API_KEY", None)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/generate_provisional_negatives.py",
                    "generate",
                    "--dataset-dir",
                    str(dataset),
                    "--endpoint",
                    "http://127.0.0.1:1",
                ],
                cwd=repository,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("INF_API_KEY", result.stderr)
            self.assertFalse((dataset / ".generation_state").exists())

    def test_generate_resumes_and_finalize_publishes_complete_split_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "phase1"
            dataset = root / "pilot"
            self._write_phase1_source(source)
            repository = Path(__file__).resolve().parents[1]
            subprocess.run(
                [
                    sys.executable,
                    "scripts/generate_provisional_negatives.py",
                    "prepare",
                    "--input-dir",
                    str(source),
                    "--output-dir",
                    str(dataset),
                ],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
            _GenerationHandler.request_count = 0
            server = ThreadingHTTPServer(("127.0.0.1", 0), _GenerationHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                environment = {**os.environ, "INF_API_KEY": "test-only-key"}
                command = [
                    sys.executable,
                    "scripts/generate_provisional_negatives.py",
                    "generate",
                    "--dataset-dir",
                    str(dataset),
                    "--endpoint",
                    f"http://127.0.0.1:{server.server_port}",
                    "--limit",
                    "1",
                ]
                first = subprocess.run(
                    command,
                    cwd=repository,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                second = subprocess.run(
                    command,
                    cwd=repository,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(_GenerationHandler.request_count, 4)
            responses = (dataset / "negative_responses.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(responses), 2)

            finalized = subprocess.run(
                [
                    sys.executable,
                    "scripts/generate_provisional_negatives.py",
                    "finalize",
                    "--dataset-dir",
                    str(dataset),
                ],
                cwd=repository,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(finalized.returncode, 0, finalized.stderr)
            records = (dataset / "provisional_training_records.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(records), 2)
            self.assertTrue((dataset / "splits" / "train.jsonl").is_file())
            manifest = json.loads(
                (dataset / "pilot_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["candidate_count"], 2)
            self.assertEqual(manifest["experiment_tier"], "feasibility_only")


if __name__ == "__main__":
    unittest.main()
