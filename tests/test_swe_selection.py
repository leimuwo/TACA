import csv
import json
import tempfile
import unittest
from pathlib import Path

from thought_action_retrieval.data.swe_selection import (
    DEFAULT_SEED,
    TOTAL_MODEL_QUOTAS,
    analyze_record,
    build_parser,
    classify_action,
    fit_pair_model_quotas,
    is_bugfix_task,
    quality_score,
    record_to_case,
    scan_jsonl_candidates,
    select_representative_candidates,
    write_case_outputs,
)


def make_record(*, title="Incorrect result crashes on empty input", cycles=10):
    trajectory = [
        {
            "role": "system",
            "text": "",
            "system_prompt": "You are a software engineering agent.",
            "mask": False,
            "cutoff_date": "2024-01-01",
        },
        {
            "role": "user",
            "text": (
                "We're currently solving the following issue within our repository. "
                f"Here's the issue text:\nISSUE:\n{title}\nThe existing behavior is wrong."
            ),
            "system_prompt": "",
            "mask": False,
            "cutoff_date": "2024-01-01",
        },
    ]
    actions = [
        'search_dir "broken_function"',
        "open src/module.py",
        "python reproduce.py",
        "edit 10:12\nreturn fixed_value\nend_of_edit",
        "pytest tests/test_module.py -q",
    ]
    for index in range(cycles):
        trajectory.extend(
            [
                {
                    "role": "ai",
                    "text": (
                        "The previous result narrows the failure, so I will inspect or "
                        f"verify the next hypothesis in step {index}.\n```\n"
                        f"{actions[index % len(actions)]}\n```"
                    ),
                    "system_prompt": "",
                    "mask": True,
                    "cutoff_date": "2024-01-01",
                },
                {
                    "role": "user",
                    "text": f"Observation for step {index}",
                    "system_prompt": "",
                    "mask": False,
                    "cutoff_date": "2024-01-01",
                },
            ]
        )
    return {
        "instance_id": "owner__repo-123",
        "model_name": "swe-agent-llama-70b",
        "target": True,
        "trajectory": trajectory,
        "exit_status": "submitted",
        "generated_patch": "diff --git a/src/module.py b/src/module.py\n+return fixed_value\n",
        "eval_logs": "pytest completed successfully\n" * 10,
    }


class TaskClassificationTests(unittest.TestCase):
    def test_accepts_bug_fix_language(self):
        self.assertTrue(is_bugfix_task("Parser raises TypeError on empty input"))

    def test_rejects_pure_feature_request(self):
        self.assertFalse(is_bugfix_task("Add support for exporting reports to PDF"))

    def test_feature_title_is_not_rescued_by_error_words_in_description(self):
        record = make_record(title="Feature Request: add PDF export")
        record["trajectory"][1]["text"] += "\nThe current workaround can raise an error."

        self.assertFalse(analyze_record(record)["hard_eligible"])

    def test_rejects_option_and_support_titles(self):
        self.assertFalse(is_bugfix_task("option to skip already included files"))
        self.assertFalse(is_bugfix_task("Support templates in dictionaries"))
        self.assertFalse(
            is_bugfix_task("Support for capturing stack locals with logging.error()")
        )


class ActionClassificationTests(unittest.TestCase):
    def test_classifies_debugging_phases(self):
        self.assertEqual(classify_action('search_dir "needle"'), "inspect")
        self.assertEqual(classify_action("edit 4:5\nvalue = 1\nend_of_edit"), "edit")
        self.assertEqual(classify_action("pytest tests/test_api.py -q"), "test")


class RecordAnalysisTests(unittest.TestCase):
    def test_accepts_complete_ten_cycle_bugfix_trajectory(self):
        analysis = analyze_record(make_record())

        self.assertTrue(analysis["hard_eligible"])
        self.assertEqual(analysis["valid_cycles"], 10)
        self.assertEqual(analysis["complete_cycle_ratio"], 1.0)
        self.assertEqual(
            set(analysis["action_categories"]), {"inspect", "edit", "run", "test"}
        )

    def test_rejects_trajectory_without_explicit_thought(self):
        record = make_record()
        for message in record["trajectory"]:
            if message["role"] == "ai":
                message["text"] = "```\nopen src/module.py\n```"

        analysis = analyze_record(record)

        self.assertFalse(analysis["hard_eligible"])
        self.assertEqual(analysis["valid_cycles"], 0)

    def test_rejects_record_without_patch_evidence(self):
        record = make_record()
        record["generated_patch"] = ""

        self.assertFalse(analyze_record(record)["hard_eligible"])

    def test_quality_score_rewards_test_evidence(self):
        with_test = analyze_record(make_record())
        without_test_record = make_record()
        for message in without_test_record["trajectory"]:
            if message["role"] == "ai":
                message["text"] = message["text"].replace(
                    "pytest tests/test_module.py -q", "python reproduce.py"
                )
        without_test = analyze_record(without_test_record)

        self.assertGreater(
            quality_score(with_test, make_record()),
            quality_score(without_test, without_test_record),
        )


def fake_candidate(uid, instance_id, model, target, score=90.0, band="20-39"):
    return {
        "uid": uid,
        "instance_id": instance_id,
        "repository": instance_id.rsplit("-", 1)[0],
        "model_short": model,
        "model_name": f"swe-agent-llama-{model}",
        "target": target,
        "quality_score": score,
        "length_band": band,
    }


class RepresentativeSelectionTests(unittest.TestCase):
    def test_cli_requires_source_and_output(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_verified_500_case_quotas_are_frozen(self):
        self.assertEqual(DEFAULT_SEED, 20260814)
        self.assertEqual(
            TOTAL_MODEL_QUOTAS,
            {"70b": 175, "8b": 50, "405b": 25},
        )
        self.assertEqual(sum(TOTAL_MODEL_QUOTAS.values()) * 2, 500)

    def test_allows_same_task_to_form_pairs_for_different_models(self):
        candidates = []
        for model in ("405b", "8b"):
            for target in (True, False):
                candidates.append(
                    fake_candidate(
                        f"{model}-{target}", "shared__repo-1", model, target
                    )
                )

        selected = select_representative_candidates(
            candidates,
            pair_model_quotas={"405b": 1, "8b": 1},
            total_model_quotas={"405b": 1, "8b": 1},
            seed=7,
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(
            {(item["model_short"], item["target"]) for item in selected},
            {("405b", True), ("405b", False), ("8b", True), ("8b", False)},
        )

    def test_pair_selection_treats_repository_limit_as_soft_diversity_goal(self):
        candidates = []
        for index in range(3):
            instance_id = f"shared__repo-{index}"
            candidates.append(
                fake_candidate(f"pair-{index}-t", instance_id, "70b", True)
            )
            candidates.append(
                fake_candidate(f"pair-{index}-f", instance_id, "70b", False)
            )

        selected = select_representative_candidates(
            candidates,
            pair_model_quotas={"70b": 3},
            total_model_quotas={"70b": 3},
            seed=7,
        )

        self.assertEqual(len(selected), 6)
        self.assertEqual(len({item["instance_id"] for item in selected}), 3)

    def test_diversity_selection_reuses_task_only_when_quota_requires_it(self):
        candidates = []
        for target, instance_id in ((True, "success__repo-1"), (False, "failure__repo-1")):
            candidates.append(fake_candidate(f"{target}-a", instance_id, "70b", target, score=90))
            candidates.append(fake_candidate(f"{target}-b", instance_id, "70b", target, score=89))

        selected = select_representative_candidates(
            candidates,
            pair_model_quotas={"70b": 0},
            total_model_quotas={"70b": 2},
            seed=7,
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(len({item["uid"] for item in selected}), 4)
        self.assertEqual(len({item["instance_id"] for item in selected}), 2)

    def test_reassigns_pair_quota_when_one_model_has_too_few_repositories(self):
        candidates = []
        for model, repository_count in (("large", 4), ("small", 1)):
            for index in range(repository_count):
                instance_id = f"repo{index}__{model}-1"
                candidates.append(
                    fake_candidate(f"{model}-{index}-t", instance_id, model, True)
                )
                candidates.append(
                    fake_candidate(f"{model}-{index}-f", instance_id, model, False)
                )

        quotas = fit_pair_model_quotas(
            candidates,
            preferred_quotas={"large": 2, "small": 2},
            total_pairs=4,
        )

        self.assertEqual(quotas, {"large": 3, "small": 1})

    def test_builds_balanced_pairs_and_unique_extra_tasks(self):
        candidates = []
        for model in ("70b", "8b"):
            pair_id = f"pair__{model}-1"
            candidates.append(fake_candidate(f"{model}-pair-t", pair_id, model, True))
            candidates.append(fake_candidate(f"{model}-pair-f", pair_id, model, False))
            for target in (True, False):
                for index in range(2):
                    instance_id = f"extra__{model}-{target}-{index}"
                    candidates.append(
                        fake_candidate(
                            f"{model}-{target}-{index}", instance_id, model, target
                        )
                    )

        selected = select_representative_candidates(
            candidates,
            pair_model_quotas={"70b": 1, "8b": 1},
            total_model_quotas={"70b": 2, "8b": 2},
            seed=7,
        )

        self.assertEqual(len(selected), 8)
        self.assertEqual(sum(item["target"] for item in selected), 4)
        self.assertEqual(sum(item["selection_group"] == "paired" for item in selected), 4)
        extras = [item for item in selected if item["selection_group"] == "diversity"]
        self.assertEqual(len({item["instance_id"] for item in extras}), len(extras))
        for target in (True, False):
            counts = {
                model: sum(
                    item["target"] is target and item["model_short"] == model
                    for item in selected
                )
                for model in ("70b", "8b")
            }
            self.assertEqual(counts, {"70b": 2, "8b": 2})


class JsonlSelectionPipelineTests(unittest.TestCase):
    def test_scans_jsonl_and_records_source_location_for_eligible_rows(self):
        eligible = make_record()
        rejected = make_record()
        rejected["generated_patch"] = ""
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp)
            shard = source_dir / "train-00000.jsonl"
            shard.write_text(
                "\n".join(json.dumps(row) for row in (rejected, eligible)) + "\n",
                encoding="utf-8",
            )

            candidates = scan_jsonl_candidates(source_dir)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["source_file"], "train-00000.jsonl")
        self.assertEqual(candidates[0]["source_row"], 1)
        self.assertEqual(candidates[0]["uid"], "train-00000.jsonl:1")

    def test_converts_raw_record_to_case_with_atomic_thought_action_steps(self):
        record = make_record(cycles=10)
        selection = {
            "selection_group": "diversity",
            "pair_id": "",
            "repository": "owner__repo",
        }

        case = record_to_case(record, selection, "CASE-0001")

        self.assertEqual(case["case_id"], "CASE-0001")
        self.assertEqual(case["system_prompt"], "You are a software engineering agent.")
        self.assertIn("Incorrect result crashes", case["task"])
        self.assertEqual(len(case["steps"]), 10)
        self.assertEqual(case["steps"][0]["step"], 1)
        self.assertEqual(case["steps"][0]["action"], 'search_dir "broken_function"')
        self.assertEqual(case["steps"][0]["observation"], "Observation for step 0")
        self.assertNotIn("```", case["steps"][0]["thought"])

    def test_writes_four_digit_case_files_jsonl_manifest_and_summary(self):
        first = make_record()
        second = make_record()
        second["instance_id"] = "other__repo-456"
        second["target"] = False
        selected = [
            {
                "uid": "train.jsonl:0",
                "source_file": "train.jsonl",
                "source_row": 0,
                "instance_id": first["instance_id"],
                "repository": "owner__repo",
                "model_name": first["model_name"],
                "model_short": "70b",
                "target": True,
                "exit_status": first["exit_status"],
                "selection_group": "paired",
                "pair_id": "P0001",
                "quality_score": 90.0,
                "length_band": "10-19",
                "action_categories": ["edit", "inspect"],
            },
            {
                "uid": "train.jsonl:1",
                "source_file": "train.jsonl",
                "source_row": 1,
                "instance_id": second["instance_id"],
                "repository": "other__repo",
                "model_name": second["model_name"],
                "model_short": "70b",
                "target": False,
                "exit_status": second["exit_status"],
                "selection_group": "diversity",
                "pair_id": "",
                "quality_score": 89.0,
                "length_band": "10-19",
                "action_categories": ["edit", "inspect"],
            },
        ]
        records = {"train.jsonl:0": first, "train.jsonl:1": second}
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "selected"

            write_case_outputs(output_dir, selected, records, seed=11, eligible_count=7)

            case_paths = sorted(output_dir.glob("CASE-*.json"))
            self.assertEqual([path.name for path in case_paths], ["CASE-0001.json", "CASE-0002.json"])
            self.assertEqual(
                [json.loads(line)["case_id"] for line in (output_dir / "selected_2.jsonl").read_text().splitlines()],
                ["CASE-0001", "CASE-0002"],
            )
            with (output_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                manifest = list(csv.DictReader(handle))
            self.assertEqual(len(manifest), 2)
            summary = json.loads((output_dir / "selection_summary.json").read_text())
            self.assertEqual(summary["selected_count"], 2)
            self.assertEqual(summary["eligible_candidate_count"], 7)


if __name__ == "__main__":
    unittest.main()
