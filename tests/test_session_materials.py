from __future__ import annotations

import json
import re
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
SESSION_DIRECTORY = re.compile(r"^\d{4}-\d{2}-\d{2}-.+")
FORBIDDEN_TEXT = (
    "AI VIETNAM",
    "AIVIETNAM",
    "aivietnam",
    "google.colab",
    "/content/drive",
)
FORBIDDEN_PATTERNS = (re.compile(r"\bTA_[A-Za-z0-9]+"),)
RAW_SLIDE_SUFFIXES = {".pdf", ".pptx", ".ppt"}


def session_directories() -> list[Path]:
    directories = []
    for module in MODULES:
        module_directory = ROOT / module
        if not module_directory.is_dir():
            continue
        for child in module_directory.iterdir():
            if child.is_dir() and SESSION_DIRECTORY.match(child.name):
                directories.append(child)
    return sorted(directories)


class SessionMaterialsTest(unittest.TestCase):
    def test_every_session_has_a_study_guide(self) -> None:
        for session in session_directories():
            with self.subTest(session=session.name):
                readme = session / "README.md"
                self.assertTrue(readme.is_file())
                content = readme.read_text(encoding="utf-8")
                self.assertIn("## Overview", content)
                self.assertIn("## Learning outcomes", content)

    def test_every_session_has_valid_notebooks(self) -> None:
        for session in session_directories():
            with self.subTest(session=session.name):
                notebooks = sorted((session / "notebooks").glob("*.ipynb"))
                self.assertTrue(notebooks)
                for path in notebooks:
                    document = json.loads(path.read_text(encoding="utf-8"))
                    self.assertIn("cells", document)
                    self.assertTrue(document["cells"])

    def test_notebooks_are_debranded_and_portable(self) -> None:
        for session in session_directories():
            for path in sorted((session / "notebooks").glob("*.ipynb")):
                with self.subTest(notebook=path.name):
                    content = path.read_text(encoding="utf-8")
                    for term in FORBIDDEN_TEXT:
                        self.assertNotIn(term, content)
                    for pattern in FORBIDDEN_PATTERNS:
                        self.assertIsNone(pattern.search(content))

    def test_session_data_is_debranded(self) -> None:
        for session in session_directories():
            data_directory = session / "data"
            if not data_directory.is_dir():
                continue
            for path in data_directory.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in {".csv", ".md", ".txt"}:
                    continue
                with self.subTest(file=path.name):
                    content = path.read_text(encoding="utf-8")
                    for term in FORBIDDEN_TEXT:
                        self.assertNotIn(term, content)
                    for pattern in FORBIDDEN_PATTERNS:
                        self.assertIsNone(pattern.search(content))

    def test_sessions_do_not_contain_raw_slides(self) -> None:
        for session in session_directories():
            for path in session.rglob("*"):
                if path.is_file():
                    with self.subTest(file=path.name):
                        self.assertNotIn(path.suffix.lower(), RAW_SLIDE_SUFFIXES)

    def test_session_data_directories_are_documented(self) -> None:
        for session in session_directories():
            data_directory = session / "data"
            if data_directory.is_dir():
                with self.subTest(session=session.name):
                    self.assertTrue((data_directory / "README.md").is_file())


if __name__ == "__main__":
    unittest.main()
