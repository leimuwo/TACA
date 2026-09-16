import subprocess
import tempfile
import unittest
from pathlib import Path

from thought_action_retrieval.repository_audit import audit_repository


def run_git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def initialize_git(root: Path) -> None:
    run_git(root, "init", "-q")


class RepositoryAuditTests(unittest.TestCase):
    def test_rejects_tracked_secret_large_file_and_machine_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initialize_git(root)
            (root / ".env").write_text("API_KEY=secret-value\n", encoding="utf-8")
            (root / "large.bin").write_bytes(b"x" * 33)
            machine_path = "/inspire" + "/hdd/private/data"
            (root / "run.py").write_text(
                f'DATA = "{machine_path}"\n', encoding="utf-8"
            )
            run_git(root, "add", "-f", ".")

            issues = audit_repository(root, max_file_bytes=32)

        self.assertEqual(
            {issue.code for issue in issues},
            {
                "forbidden_path",
                "file_too_large",
                "machine_absolute_path",
                "possible_secret",
            },
        )

    def test_allows_empty_secret_example_and_absolute_path_in_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initialize_git(root)
            (root / ".env.example").write_text(
                "INTENT_API_KEY=\n", encoding="utf-8"
            )
            documented_path = "/inspire" + "/hdd/project/data"
            (root / "README.md").write_text(
                f"Example: `{documented_path}`\n", encoding="utf-8"
            )
            run_git(root, "add", ".")

            issues = audit_repository(root, max_file_bytes=1024)

        self.assertEqual(issues, [])

    def test_rejects_forbidden_generated_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initialize_git(root)
            generated = root / "models" / "weights.txt"
            generated.parent.mkdir()
            generated.write_text("weights", encoding="utf-8")
            run_git(root, "add", ".")

            issues = audit_repository(root)

        self.assertEqual([issue.code for issue in issues], ["forbidden_path"])


if __name__ == "__main__":
    unittest.main()
