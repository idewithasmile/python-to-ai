from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    "01-python-fundamentals",
    "02-calculus-and-linear-algebra",
    "03-probability-and-statistics",
    "04-classical-machine-learning",
    "05-deep-learning-foundations",
    "06-deep-learning-multimodal",
    "07-computer-vision",
    "08-advanced-architectures-nlp",
    "09-generative-ai",
    "10-advanced-deep-learning-and-llms",
]
ROOT_FILES = [
    "README.md",
    "LICENSE",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    "requirements.txt",
    "pyproject.toml",
    ".pre-commit-config.yaml",
]
MODULE_DIRECTORIES = ["notebooks", "exercises", "projects"]


class RepositoryScaffoldTest(unittest.TestCase):
    def test_root_files_exist(self) -> None:
        for name in ROOT_FILES:
            with self.subTest(name=name):
                self.assertTrue((ROOT / name).is_file())

    def test_module_directories_exist(self) -> None:
        for name in MODULES:
            with self.subTest(module=name):
                module = ROOT / name
                self.assertTrue(module.is_dir())
                self.assertTrue((module / "README.md").is_file())
                for directory in MODULE_DIRECTORIES:
                    self.assertTrue((module / directory).is_dir())

    def test_roadmap_contains_54_weeks(self) -> None:
        roadmap = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertEqual(roadmap.count("**Week "), 54)

    def test_roadmap_has_ten_modules_and_five_phases(self) -> None:
        roadmap = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertEqual(roadmap.count("**Module complete**"), 10)
        self.assertEqual(roadmap.count("#### Module"), 10)

        phases = [line for line in roadmap.splitlines() if line.startswith("### Phase")]
        self.assertEqual(len(phases), 5)
        windows = ["1\u201311", "12\u201322", "23\u201333", "34\u201344", "45\u201354"]
        for phase, window in zip(phases, windows, strict=True):
            self.assertIn(f"(Weeks {window})", phase)

    def test_module_readmes_contain_core_sections(self) -> None:
        for name in MODULES:
            with self.subTest(module=name):
                content = (ROOT / name / "README.md").read_text(encoding="utf-8")
                for heading in [
                    "## Overview",
                    "## Weekly Plan",
                    "## Workspace",
                    "## Module Checklist",
                ]:
                    self.assertIn(heading, content)

    def test_module_readmes_link_back_to_roadmap(self) -> None:
        for name in MODULES:
            with self.subTest(module=name):
                content = (ROOT / name / "README.md").read_text(encoding="utf-8")
                self.assertIn("../README.md", content)


if __name__ == "__main__":
    unittest.main()
