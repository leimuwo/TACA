"""Offline contracts for the asymmetric BGE/NightOwl experiment."""

import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_nightowl_action_tower import action_input, action_model_load_kwargs, baseline_metrics, prepare_groups, rank_metrics


class NightowlActionTowerTests(unittest.TestCase):
    def test_local_model_loads_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for name in ("config.json", "model.safetensors", "tokenizer.json"):
                (path / name).touch()
            self.assertEqual(action_model_load_kwargs(path, "revision"), {"local_files_only": True})
            (path / "tokenizer.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "tokenizer.json"):
                action_model_load_kwargs(path, "revision")

    def test_action_input_prioritizes_command_and_path(self):
        text = '[ACTION] {"args":{"content":"long body","path":"a.py","command":"python a.py"},"function":"exec"}'
        output = action_input(text)
        self.assertLess(output.index('"command"'), output.index('"content"'))
        self.assertLess(output.index('"path"'), output.index('"content"'))
        self.assertEqual(json.loads(output)["args"]["content"], "long body")

    def test_explicit_negatives_stay_grouped(self):
        records = [{"query": "[INTENT] run a", "pos": ["[ACTION] a"],
                    "neg": ["[ACTION] b", "[ACTION] c"]}]
        self.assertEqual(prepare_groups(records), [("[INTENT] run a", "[ACTION] a",
                                                    ["[ACTION] b", "[ACTION] c"])])

    def test_baseline_reports_exec_subset(self):
        actions = [{"action_id": "a", "action": '[ACTION] {"function":"exec","args":{}}'},
                   {"action_id": "b", "action": '[ACTION] {"function":"read","args":{}}'}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rankings.jsonl"
            path.write_text(json.dumps({"query_id": "q", "top_actions": [
                {"action_id": "b"}, {"action_id": "a"}]}) + "\n", encoding="utf-8")
            result = baseline_metrics(path, actions, defaultdict(set, {"q": {"a"}}), 2)
            self.assertEqual(result["overall"]["known_positive_at_1"], 0)
            self.assertEqual(result["exec_positive"]["known_positive_mrr_at_2"], 0.5)

    def test_unreachable_rank_counts_as_zero(self):
        self.assertEqual(rank_metrics([None, 1, 2], 20)["known_positive_mrr_at_20"], 0.5)


if __name__ == "__main__":
    unittest.main()
