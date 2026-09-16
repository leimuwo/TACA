import csv
import json
import tempfile
import unittest
from pathlib import Path

from thought_action_retrieval.data.snapshot import (
    import_selected_snapshot,
    validate_selected_snapshot,
)


def make_case(case_id: str, *, target: bool, model: str, group: str) -> dict:
    return {
        "case_id": case_id,
        "selection_group": group,
        "pair_id": "P01" if group == "paired" else "",
        "instance_id": f"owner__repo-{case_id}",
        "repository": "owner__repo",
        "model_name": model,
        "target": target,
        "exit_status": "submitted",
        "system_prompt": "system",
        "task": "Fix incorrect behavior",
        "steps": [
            {
                "step": 1,
                "thought": "I should inspect the file.",
                "action": "open src/module.py",
                "observation": "file contents",
            }
        ],
        "generated_patch": "diff --git a/a b/a\n@@\n+fixed\n",
        "eval_logs": "tests passed",
    }


def write_snapshot(root: Path, *, selected_count: int = 2) -> tuple[Path, Path]:
    source = root / "selected"
    raw = root / "raw"
    source.mkdir(parents=True)
    raw.mkdir(parents=True)
    cases = [
        make_case(
            "CASE-0001",
            target=True,
            model="swe-agent-llama-70b",
            group="paired",
        ),
        make_case(
            "CASE-0002",
            target=False,
            model="swe-agent-llama-8b",
            group="diversity",
        ),
    ]
    for case in cases:
        (source / f"{case['case_id']}.json").write_text(
            json.dumps(case), encoding="utf-8"
        )
    (source / "selected_2.jsonl").write_text(
        "".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8"
    )
    with (source / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["selection_id", "source_file", "source_row"],
        )
        writer.writeheader()
        writer.writerow(
            {"selection_id": "CASE-0001", "source_file": "raw-0.jsonl", "source_row": 0}
        )
        writer.writerow(
            {"selection_id": "CASE-0002", "source_file": "raw-1.jsonl", "source_row": 0}
        )
    summary = {
        "seed": 20260814,
        "eligible_candidate_count": 7,
        "selected_count": selected_count,
        "target_counts": {"true": 1, "false": 1},
        "selection_group_counts": {"paired": 1, "diversity": 1},
        "model_counts": {
            "swe-agent-llama-70b": 1,
            "swe-agent-llama-8b": 1,
        },
        "unique_instance_ids": 2,
        "unique_repositories": 1,
    }
    (source / "selection_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (raw / "raw-0.jsonl").write_text('{"raw":0}\n', encoding="utf-8")
    (raw / "raw-1.jsonl").write_text('{"raw":1}\n', encoding="utf-8")
    return source, raw


class SweSnapshotTests(unittest.TestCase):
    def test_imports_metadata_samples_and_external_digests(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data_root = base / "external"
            repository_root = base / "repository"
            data_root.mkdir()
            repository_root.mkdir()
            write_snapshot(data_root)

            result = import_selected_snapshot(
                data_root=data_root,
                source_relative=Path("selected"),
                repository_root=repository_root,
                sample_ids=["CASE-0001"],
                raw_shards_relative=Path("raw"),
                expected_count=2,
            )

            sample = repository_root / "data/samples/swe-selected-500/CASE-0001.json"
            manifest_path = repository_root / "data/manifests/swe-selected-500/dataset_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            sample_exists = sample.exists()
            full_jsonl_was_copied = (
                repository_root / "data" / "selected_2.jsonl"
            ).exists()

        self.assertEqual(result["selected_count"], 2)
        self.assertTrue(sample_exists)
        self.assertEqual(manifest["metadata"]["selected_count"], 2)
        self.assertEqual(len(manifest["metadata"]["raw_shards"]), 2)
        self.assertFalse(full_jsonl_was_copied)

    def test_rejects_duplicate_case_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = write_snapshot(root)
            duplicate = json.loads((source / "CASE-0001.json").read_text())
            duplicate["case_id"] = "CASE-0001"
            (source / "CASE-0002.json").write_text(json.dumps(duplicate), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate CASE ID"):
                validate_selected_snapshot(source, expected_count=2)

    def test_rejects_jsonl_and_csv_id_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = write_snapshot(root)
            with (source / "manifest.csv").open("a", encoding="utf-8") as handle:
                handle.write("CASE-9999,raw-0.jsonl,9\n")

            with self.assertRaisesRegex(ValueError, "manifest IDs"):
                validate_selected_snapshot(source, expected_count=2)

    def test_rejects_summary_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = write_snapshot(root, selected_count=3)

            with self.assertRaisesRegex(ValueError, "selected_count"):
                validate_selected_snapshot(source, expected_count=2)

    def test_rejects_missing_requested_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data_root = base / "external"
            repository_root = base / "repository"
            data_root.mkdir()
            repository_root.mkdir()
            write_snapshot(data_root)

            with self.assertRaisesRegex(ValueError, "CASE-9999"):
                import_selected_snapshot(
                    data_root=data_root,
                    source_relative=Path("selected"),
                    repository_root=repository_root,
                    sample_ids=["CASE-9999"],
                    raw_shards_relative=Path("raw"),
                    expected_count=2,
                )


if __name__ == "__main__":
    unittest.main()
