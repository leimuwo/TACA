import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = [ROOT / "README.md", ROOT / "data/README.md", ROOT / "docs/data-catalog.md"]
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


class DocumentationTests(unittest.TestCase):
    def test_readme_documents_core_commands_and_data_manifest(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("python -m unittest discover -s tests -v", readme)
        self.assertIn("scripts/audit_repository.py", readme)
        self.assertIn("scripts/select_swe_trajectories.py", readme)
        self.assertIn("scripts/extract_thought_intents.py", readme)
        self.assertIn("scripts/build_intent_action_dataset.py build", readme)
        self.assertIn("scripts/build_intent_action_dataset.py finalize", readme)
        self.assertIn('"candidate_id": "swe:CASE-0001:T1-I1"', readme)
        self.assertIn('"label": "direct_match"', readme)
        self.assertIn('"annotator": "reviewer-name"', readme)
        self.assertIn('"notes": "Action directly fulfills the Intent."', readme)
        self.assertIn("TA_DATA_ROOT", readme)
        self.assertIn(
            "data/manifests/swe-selected-500/dataset_manifest.json",
            readme,
        )

    def test_all_local_markdown_links_exist(self):
        for document in DOCUMENTS:
            text = document.read_text(encoding="utf-8")
            for raw_target in MARKDOWN_LINK_RE.findall(text):
                target = raw_target.split("#", 1)[0]
                if not target or target.startswith(("http://", "https://", "mailto:")):
                    continue
                resolved = (document.parent / target).resolve()
                self.assertTrue(
                    resolved.exists(),
                    f"broken local link in {document.relative_to(ROOT)}: {raw_target}",
                )


if __name__ == "__main__":
    unittest.main()
