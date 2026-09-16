import hashlib
import json
import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from thought_action_retrieval.intent import extraction


class GenerateThoughtIntentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = extraction

    def test_loads_prompt_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "prompt.md"
            prompt.write_text(
                "# Prompt\n\n## System Prompt\n\nSystem text.\n\n"
                "## User Prompt\n\nStep: {thought_step}\nThought: {thought}\n",
                encoding="utf-8",
            )

            bundle = self.module.load_prompt_bundle(prompt)

        self.assertEqual(bundle.system, "System text.")
        self.assertIn("{thought_step}", bundle.user_template)

    def test_rejects_incomplete_prompt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "prompt.md"
            prompt.write_text("## System Prompt\nOnly system.\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "User Prompt"):
                self.module.load_prompt_bundle(prompt)

    def test_approved_prompt_hashes_are_preserved(self):
        self.assertEqual(
            hashlib.sha256(self.module.PROMPTS.system.encode("utf-8")).hexdigest(),
            "577eb51f44079762c9a9ba090e9cbfa4cf3fedeffd1fa9cc31b06790b96ae2f4",
        )
        self.assertEqual(
            hashlib.sha256(
                self.module.PROMPTS.user_template.encode("utf-8")
            ).hexdigest(),
            "8ffb169852f30225747bac475c71c928cb7102dbb255f2ab42ad474c106ac5db",
        )

    def test_source_name_is_relative_to_data_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "SWE" / "CASE-0001.json"
            source.parent.mkdir()
            source.write_text("{}", encoding="utf-8")

            source_name = self.module.portable_source_name(source, root)

        self.assertEqual(source_name, "SWE/CASE-0001.json")

    def test_source_name_falls_back_to_filename_outside_data_root(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            source = Path(first) / "CASE-0001.json"
            source.write_text("{}", encoding="utf-8")

            source_name = self.module.portable_source_name(source, Path(second))

        self.assertEqual(source_name, "CASE-0001.json")

    def test_live_run_requires_named_intent_api_key_before_reading_inputs(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(SystemExit, "INTENT_API_KEY"):
                self.module.main(
                    [
                        "--swe-dir",
                        "missing-swe",
                        "--swe-count",
                        "1",
                        "--kimi-count",
                        "0",
                    ]
                )

    def test_extracts_kimi_steps_and_excludes_current_action_from_context(self):
        trajectory = {
            "id": "kimi-1",
            "task": "Find the config",
            "category": "Agent Tools",
            "subcategory": "Memory & Context",
            "messages": [
                {"role": "user", "content": "Find the config"},
                {
                    "role": "assistant",
                    "content": "I will search the workspace.",
                    "tool_calls": [
                        {"function": "search_files", "args": {"pattern": "config"}}
                    ],
                },
                {"role": "tool", "content": "Found config.yaml"},
                {
                    "role": "assistant",
                    "content": "I should read it.",
                    "tool_calls": [
                        {"function": "read_file", "args": {"path": "config.yaml"}}
                    ],
                },
            ],
        }

        steps = self.module.extract_kimi_steps(trajectory, context_chars=2000)

        self.assertEqual([step["thought_step"] for step in steps], [1, 2])
        self.assertEqual(steps[0]["actual_actions"][0]["function"], "search_files")
        self.assertNotIn("search_files", steps[0]["preceding_context"])
        self.assertIn("Found config.yaml", steps[1]["preceding_context"])
        self.assertNotIn("read_file", steps[1]["preceding_context"])

    def test_extracts_swe_steps_without_putting_current_action_in_context(self):
        trajectory = {
            "case_id": "CASE-001",
            "task": "Fix the issue",
            "steps": [
                {
                    "step": 1,
                    "thought": "I will locate the file.",
                    "action": "find_file model.py",
                    "observation": "Found model.py",
                },
                {
                    "step": 2,
                    "thought": "Now I should open the file.",
                    "action": "open model.py",
                    "observation": "File contents",
                },
            ],
        }

        steps = self.module.extract_swe_steps(trajectory, context_chars=2000)

        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[1]["actual_actions"], ["open model.py"])
        self.assertIn("Found model.py", steps[1]["preceding_context"])
        self.assertNotIn("open model.py", steps[1]["preceding_context"])

    def test_build_user_prompt_contains_thought_but_not_current_action(self):
        step = {
            "thought_step": 3,
            "user_request": "Find the config",
            "preceding_context": "The workspace was listed.",
            "thought": "I should inspect config.yaml.",
            "actual_actions": ["read_file config.yaml"],
        }

        prompt = self.module.build_user_prompt(step)

        self.assertIn("THOUGHT STEP:\n3", prompt)
        self.assertIn("I should inspect config.yaml.", prompt)
        self.assertNotIn("read_file config.yaml", prompt)

    def test_validate_intent_result_enforces_exact_source_quote(self):
        valid = {
            "thought_step": 2,
            "execution_intents": [
                {
                    "intent_id": "T2-I1",
                    "intent_text": "Read config.yaml.",
                    "source_quote": "I should read config.yaml.",
                }
            ],
            "no_intent_reason": None,
        }

        normalized = self.module.validate_intent_result(
            valid, thought_step=2, thought="I should read config.yaml."
        )
        self.assertEqual(normalized, valid)

        invalid = json.loads(json.dumps(valid))
        invalid["execution_intents"][0]["source_quote"] = "read the file"
        with self.assertRaisesRegex(ValueError, "source_quote"):
            self.module.validate_intent_result(
                invalid, thought_step=2, thought="I should read config.yaml."
            )

    def test_chat_payload_leaves_room_for_reasoning_and_final_json(self):
        payload = self.module.build_chat_payload("prompt", model="deepseek-v4-flash")

        self.assertEqual(payload["max_tokens"], 4000)
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_selects_25_files_from_each_source_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            kimi_dir = base / "kimi"
            swe_dir = base / "swe"
            kimi_dir.mkdir()
            swe_dir.mkdir()
            for index in range(30):
                (kimi_dir / f"{index:04d}.json").write_text(
                    json.dumps(
                        {
                            "id": f"k-{index}",
                            "task": "task",
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": "I will search.",
                                    "tool_calls": [
                                        {"function": "search_files", "args": {"q": "x"}}
                                    ],
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                (swe_dir / f"CASE-{index:03d}.json").write_text(
                    json.dumps(
                        {
                            "case_id": f"CASE-{index:03d}",
                            "task": "task",
                            "steps": [
                                {"step": 1, "thought": "I will search.", "action": "search"}
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            selected = self.module.select_trajectory_files(kimi_dir, swe_dir, 25, 25)

        self.assertEqual(len(selected), 50)
        self.assertEqual(sum(item["source_type"] == "kimi" for item in selected), 25)
        self.assertEqual(sum(item["source_type"] == "swe" for item in selected), 25)
        self.assertEqual(
            [item["path"].name for item in selected if item["source_type"] == "swe"][:2],
            ["CASE-000.json", "CASE-001.json"],
        )


if __name__ == "__main__":
    unittest.main()
