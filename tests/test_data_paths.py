import tempfile
import unittest
from pathlib import Path

from thought_action_retrieval.data.paths import (
    DataRootError,
    resolve_data_root,
    resolve_under_data_root,
)


class DataPathTests(unittest.TestCase):
    def test_resolves_explicit_root_before_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            explicit = base / "explicit"
            ignored = base / "ignored"
            explicit.mkdir()
            ignored.mkdir()

            result = resolve_data_root(
                explicit=explicit,
                environ={"TA_DATA_ROOT": str(ignored)},
            )

        self.assertEqual(result, explicit.resolve())

    def test_resolves_environment_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            result = resolve_data_root(environ={"TA_DATA_ROOT": str(root)})

        self.assertEqual(result, root.resolve())

    def test_rejects_missing_data_root(self):
        with self.assertRaisesRegex(DataRootError, "TA_DATA_ROOT"):
            resolve_data_root(environ={})

    def test_rejects_non_directory_data_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "data.txt"
            file_path.write_text("not a directory", encoding="utf-8")

            with self.assertRaisesRegex(DataRootError, "directory"):
                resolve_data_root(explicit=file_path)

    def test_rejects_path_escape_and_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(DataRootError, "outside"):
                resolve_under_data_root(root, "../secret.json")
            with self.assertRaisesRegex(DataRootError, "relative"):
                resolve_under_data_root(root, "/tmp/secret.json")

    def test_resolves_relative_path_below_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            result = resolve_under_data_root(root, "SWE/data.jsonl")

        self.assertEqual(result, (root / "SWE/data.jsonl").resolve())


if __name__ == "__main__":
    unittest.main()
