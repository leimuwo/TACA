import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from thought_action_retrieval.data.manifests import (
    atomic_write_manifest,
    build_dataset_manifest,
    describe_file,
    sha256_file,
)


class DataManifestTests(unittest.TestCase):
    def test_manifest_is_sorted_and_uses_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.jsonl").write_text("b\n", encoding="utf-8")
            (root / "a.jsonl").write_text("a\n", encoding="utf-8")

            manifest = build_dataset_manifest(
                name="fixture",
                version="v1",
                data_root=root,
                files=[root / "b.jsonl", root / "a.jsonl"],
                metadata={"record_count": 2},
            )

        self.assertEqual(
            [item["path"] for item in manifest["files"]],
            ["a.jsonl", "b.jsonl"],
        )
        self.assertEqual(manifest["total_bytes"], 4)
        self.assertEqual(manifest["metadata"], {"record_count": 2})
        self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["files"]))

    def test_sha256_and_file_description_are_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.txt"
            path.write_text("sample\n", encoding="utf-8")

            digest = sha256_file(path, chunk_bytes=2)
            description = describe_file(root, path)

        expected = hashlib.sha256(b"sample\n").hexdigest()
        self.assertEqual(digest, expected)
        self.assertEqual(
            description,
            {"path": "sample.txt", "bytes": 7, "sha256": expected},
        )

    def test_rejects_missing_or_outside_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root.parent / "outside-manifest-fixture.txt"
            outside.write_text("outside", encoding="utf-8")
            try:
                with self.assertRaisesRegex(ValueError, "outside"):
                    describe_file(root, outside)
                with self.assertRaisesRegex(FileNotFoundError, "missing"):
                    describe_file(root, root / "missing.jsonl")
            finally:
                outside.unlink(missing_ok=True)

    def test_atomic_manifest_write_is_deterministic(self):
        manifest = {
            "schema_version": "dataset_manifest_v1",
            "name": "fixture",
            "version": "v1",
            "file_count": 0,
            "total_bytes": 0,
            "files": [],
            "metadata": {"z": 1, "a": 2},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "manifest.json"

            atomic_write_manifest(path, manifest)
            first = path.read_text(encoding="utf-8")
            atomic_write_manifest(path, manifest)
            second = path.read_text(encoding="utf-8")

        self.assertEqual(first, second)
        self.assertEqual(json.loads(first), manifest)
        self.assertTrue(first.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
