from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_site import MODULES, build_site  # noqa: E402


class SeoMetadataTest(unittest.TestCase):
    def test_pyproject_has_seo_keywords_and_urls(self) -> None:
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = data["project"]
        self.assertGreaterEqual(len(project["keywords"]), 15)
        for url_key in ("Homepage", "Repository", "Documentation", "Issues"):
            self.assertIn(url_key, project["urls"])

    def test_citation_cff_is_valid_with_keywords(self) -> None:
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn("cff-version: 1.2.0", citation)
        self.assertIn("license: MIT", citation)
        self.assertIn("github.com/idewithasmile/python-to-ai", citation)
        keywords = re.findall(r"^\s*-\s+[a-z0-9-]+$", citation, flags=re.MULTILINE)
        self.assertGreaterEqual(len(keywords), 10)

    def test_social_preview_png_exists(self) -> None:
        preview = ROOT / "assets" / "social-preview.png"
        self.assertTrue(preview.is_file())
        self.assertEqual(preview.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_module_readmes_include_topics(self) -> None:
        for module in MODULES:
            with self.subTest(module=module):
                content = (ROOT / module / "README.md").read_text(encoding="utf-8")
                self.assertIn("**Topics:**", content)

    def test_build_site_generates_seo_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "site"
            pages = build_site(ROOT, out)
            self.assertGreaterEqual(len(pages), 13)
            self.assertTrue((out / "index.html").is_file())
            self.assertTrue((out / "robots.txt").is_file())
            self.assertTrue((out / "sitemap.xml").is_file())
            self.assertTrue((out / ".nojekyll").is_file())
            self.assertTrue((out / "assets" / "social-preview.png").is_file())
            verification = out / "googleee41da0f126d3fe3.html"
            self.assertTrue(verification.is_file())
            self.assertIn(
                "google-site-verification: googleee41da0f126d3fe3.html",
                verification.read_text(encoding="utf-8"),
            )

            robots = (out / "robots.txt").read_text(encoding="utf-8")
            self.assertIn(
                "Sitemap: https://idewithasmile.github.io/python-to-ai/sitemap.xml", robots
            )

            tree = ET.parse(out / "sitemap.xml")
            locs = [
                el.text
                for el in tree.getroot().iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
            ]
            self.assertEqual(len(locs), len(pages))

            for page in pages:
                with self.subTest(page=page.relative_to(out).as_posix()):
                    content = page.read_text(encoding="utf-8")
                    self.assertIn("<title>", content)
                    self.assertIn('<meta name="description"', content)
                    self.assertIn('<link rel="canonical"', content)
                    self.assertIn('property="og:title"', content)
                    self.assertIn('name="twitter:card"', content)
                    match = re.search(
                        r'<script type="application/ld\+json">(.*?)</script>',
                        content,
                        flags=re.DOTALL,
                    )
                    self.assertIsNotNone(match)
                    assert match is not None
                    parsed_ld = json.loads(match.group(1))
                    self.assertEqual(parsed_ld["@context"], "https://schema.org")


if __name__ == "__main__":
    unittest.main()
