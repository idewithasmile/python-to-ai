"""Build a static, SEO-optimized GitHub Pages site from the python-to-ai curriculum."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from xml.etree import ElementTree as ET

SITE_URL = "https://idewithasmile.github.io/python-to-ai"
REPO_URL = "https://github.com/idewithasmile/python-to-ai"
DEFAULT_DESCRIPTION = (
    "A free, open-source 54-week curriculum from Python fundamentals and math for ML "
    "to deep learning, computer vision, NLP, generative AI, and LLMs with PyTorch."
)
BASE_KEYWORDS = [
    "python",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "computer vision",
    "natural language processing",
    "generative ai",
    "large language models",
    "llm",
    "rag",
    "pytorch",
    "numpy",
    "data science",
    "ai curriculum",
    "python to ai",
    "jupyter notebooks",
]

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

SESSION_DIR_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)$")

FAQS = [
    (
        "What is python-to-ai?",
        "python-to-ai is a free, open-source, 54-week AI and machine learning curriculum "
        "organized into 10 progressive modules. It covers Python fundamentals, algorithms, "
        "calculus, linear algebra, probability, classical machine learning, deep learning "
        "with NumPy and PyTorch, computer vision, NLP, generative AI, and LLMs.",
    ),
    (
        "Who is this Python and AI roadmap designed for?",
        "It is built for self-learners, students, and software engineers who want a "
        "structured, project-driven path into artificial intelligence. Every topic is "
        "learned twice: first from first principles in Python/NumPy and then with modern "
        "production libraries.",
    ),
    (
        "Do I need a GPU to complete the curriculum?",
        "No. All required notebooks and exercises run on CPU with lightweight datasets, "
        "while optional GPU setup notes are documented in the environment setup guide.",
    ),
    (
        "Which libraries and tools are covered?",
        "Core technologies include Python 3.10+, NumPy, pandas, Matplotlib, Seaborn, "
        "scikit-learn, PyTorch 2.x, OpenCV, Hugging Face Transformers, PyTorch Geometric, "
        "Gymnasium, JupyterLab, and pytest.",
    ),
]

CSS_STYLES = """
:root {
  --bg: #07111f;
  --bg-elevated: #0d1b2e;
  --bg-card: #112238;
  --border: #1e3552;
  --text: #e2e8f0;
  --muted: #94a3b8;
  --accent: #38bdf8;
  --accent-2: #34d399;
  --accent-3: #818cf8;
  --code-bg: #08101c;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.65;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.container { max-width: 1120px; margin: 0 auto; padding: 0 1.25rem; }
.site-header {
  border-bottom: 1px solid var(--border);
  background: rgba(7, 17, 31, 0.92);
  backdrop-filter: blur(8px);
  position: sticky;
  top: 0;
  z-index: 20;
}
.nav {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.85rem 0;
}
.brand { font-weight: 700; font-size: 1.1rem; color: #f8fafc; letter-spacing: -0.02em; }
.nav-links { display: flex; flex-wrap: wrap; gap: 1.1rem; font-size: 0.95rem; }
.nav-links a { color: var(--muted); }
.nav-links a:hover { color: var(--text); }
.hero { padding: 2.5rem 0 1.75rem; }
.hero-banner {
  width: 100%;
  border-radius: 14px;
  border: 1px solid var(--border);
  display: block;
  margin-bottom: 1.5rem;
}
.hero h1 { font-size: clamp(1.85rem, 3.2vw, 2.75rem); margin: 0 0 0.75rem; line-height: 1.2; }
.lead { font-size: 1.12rem; color: #cbd5e1; max-width: 78ch; margin: 0 0 1.25rem; }
.cta-row { display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 1.25rem 0; }
.btn {
  display: inline-block;
  padding: 0.6rem 1.1rem;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.95rem;
  border: 1px solid var(--border);
  background: var(--bg-card);
  color: var(--text);
}
.btn-primary {
  background: linear-gradient(135deg, #0284c7, #2563eb);
  border-color: transparent;
  color: #ffffff;
}
.btn:hover { text-decoration: none; filter: brightness(1.1); }
.tag-list { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.75rem 0; padding: 0; list-style: none; }
.tag {
  display: inline-block;
  font-size: 0.82rem;
  padding: 0.22rem 0.65rem;
  border-radius: 999px;
  background: rgba(56, 189, 248, 0.12);
  border: 1px solid rgba(56, 189, 248, 0.35);
  color: #7dd3fc;
  cursor: pointer;
}
.tag.active, .tag:hover {
  background: rgba(52, 211, 153, 0.18);
  border-color: var(--accent-2);
  color: #6ee7b7;
}
.search-bar {
  width: 100%;
  max-width: 520px;
  padding: 0.65rem 0.95rem;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: var(--bg-elevated);
  color: var(--text);
  font-size: 0.98rem;
  margin: 0.75rem 0 1rem;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1rem;
  margin: 1rem 0 2rem;
}
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.1rem 1.2rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.card-meta { font-size: 0.84rem; color: var(--muted); }
.card h3 { margin: 0; font-size: 1.08rem; line-height: 1.35; }
.card p { margin: 0; color: #cbd5e1; font-size: 0.93rem; }
.section-title { font-size: 1.55rem; margin: 2.25rem 0 0.75rem; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }
.breadcrumb { font-size: 0.9rem; color: var(--muted); margin: 1.25rem 0 0.5rem; }
.prose { max-width: 860px; padding-bottom: 2.5rem; }
.prose h1 { font-size: clamp(1.75rem, 3vw, 2.35rem); line-height: 1.2; margin-top: 0.5rem; }
.prose h2 { margin-top: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 0.35rem; }
.prose h3 { margin-top: 1.5rem; color: #f1f5f9; }
.prose pre {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem;
  overflow-x: auto;
  font-size: 0.9rem;
}
.prose code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: rgba(148, 163, 184, 0.14);
  padding: 0.12rem 0.35rem;
  border-radius: 5px;
  font-size: 0.9em;
}
.prose pre code { background: transparent; padding: 0; }
.prose blockquote {
  margin: 1rem 0;
  padding: 0.6rem 1rem;
  border-left: 3px solid var(--accent);
  background: rgba(56, 189, 248, 0.08);
  border-radius: 0 8px 8px 0;
}
.prose table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
  font-size: 0.93rem;
  display: block;
  overflow-x: auto;
}
.prose th, .prose td {
  border: 1px solid var(--border);
  padding: 0.55rem 0.75rem;
  text-align: left;
}
.prose th { background: var(--bg-elevated); }
.faq-item {
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem 1.15rem;
  margin-bottom: 0.75rem;
}
.faq-item h3 { margin: 0 0 0.4rem; font-size: 1.05rem; }
.faq-item p { margin: 0; color: #cbd5e1; }
.site-footer {
  border-top: 1px solid var(--border);
  padding: 1.75rem 0 2.5rem;
  color: var(--muted);
  font-size: 0.9rem;
  margin-top: 2rem;
}
.hidden { display: none !important; }
""".strip()

SEARCH_SCRIPT = """
<script>
(function () {
  var searchInput = document.getElementById("curriculum-search");
  var tagButtons = document.querySelectorAll("[data-filter-tag]");
  var cards = document.querySelectorAll("[data-searchable]");
  var activeTag = "all";

  function applyFilters() {
    var q = searchInput ? searchInput.value.trim().toLowerCase() : "";
    cards.forEach(function (card) {
      var text = (card.getAttribute("data-searchable") || "").toLowerCase();
      var tags = (card.getAttribute("data-tags") || "").toLowerCase();
      var matchesQuery = !q || text.indexOf(q) !== -1;
      var matchesTag = activeTag === "all" || tags.split(",").indexOf(activeTag) !== -1;
      card.classList.toggle("hidden", !(matchesQuery && matchesTag));
    });
  }

  if (searchInput) {
    searchInput.addEventListener("input", applyFilters);
  }
  tagButtons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      activeTag = btn.getAttribute("data-filter-tag");
      tagButtons.forEach(function (b) { b.classList.toggle("active", b === btn); });
      applyFilters();
    });
  });
})();
</script>
""".strip()


@dataclass(frozen=True)
class SessionInfo:
    module_slug: str
    slug: str
    schedule_date: str
    title: str
    overview: str
    topics: list[str]
    notebooks: list[str]
    readme_path: Path


@dataclass(frozen=True)
class ModuleInfo:
    slug: str
    number: str
    title: str
    weeks_line: str
    overview: str
    topics: list[str]
    readme_path: Path
    sessions: list[SessionInfo]


def slugify_heading(text: str) -> str:
    clean = re.sub(r"`([^`]+)`", r"\1", text)
    clean = re.sub(r"[^a-zA-Z0-9\s-]", "", clean).strip().lower()
    return re.sub(r"[\s-]+", "-", clean) or "section"


def strip_markdown(text: str) -> str:
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_description(text: str, limit: int = 158) -> str:
    cleaned = strip_markdown(text)
    if len(cleaned) <= limit:
        return cleaned
    clipped = cleaned[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:")
    return f"{clipped}…"


def extract_title(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return strip_markdown(line[2:])
    return fallback


def extract_section(markdown: str, heading: str) -> str:
    lines = markdown.splitlines()
    target = heading.lower()
    capturing = False
    collected: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if capturing:
                break
            if line[3:].strip().lower() == target:
                capturing = True
            continue
        if capturing:
            if line.startswith("---"):
                continue
            collected.append(line)
    return "\n".join(collected).strip()


def extract_overview_paragraph(markdown: str, fallback: str) -> str:
    section = extract_section(markdown, "Overview")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]
    for para in paragraphs:
        if not para.startswith(("|", "-", "*", "#")):
            return truncate_description(para)
    return truncate_description(fallback)


def extract_topics(markdown: str) -> list[str]:
    for line in markdown.splitlines():
        if line.startswith("**Topics:**"):
            tags = re.findall(r"`([^`]+)`", line)
            if tags:
                return tags
    return []


def extract_weeks_line(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("**Weeks "):
            return strip_markdown(line)
    return ""


def rewrite_link(
    url: str,
    *,
    source_rel: Path,
    page_depth: int,
    module_slugs: set[str],
    session_lookup: dict[tuple[str, str], SessionInfo],
) -> str:
    if url.startswith(("http://", "https://", "mailto:", "#")):
        return url

    prefix = "../" * page_depth
    path_part, _, anchor = url.partition("#")
    anchor_suffix = f"#{anchor}" if anchor else ""

    resolved = (source_rel.parent / path_part).as_posix()
    # Normalize posix path segments manually against repo root
    parts: list[str] = []
    for seg in resolved.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
        else:
            parts.append(seg)
    norm = "/".join(parts)

    if norm in ("", "README.md"):
        return f"{prefix}index.html{anchor_suffix}"
    if norm == "docs/setup.md":
        return f"{prefix}guides/setup/index.html{anchor_suffix}"
    if norm == "docs/study-guide.md":
        return f"{prefix}guides/study-guide/index.html{anchor_suffix}"

    if len(parts) == 1 and parts[0] in module_slugs:
        return f"{prefix}modules/{parts[0]}/index.html{anchor_suffix}"
    if len(parts) == 2 and parts[0] in module_slugs and parts[1] == "README.md":
        return f"{prefix}modules/{parts[0]}/index.html{anchor_suffix}"

    if (
        len(parts) >= 2
        and (parts[0], parts[1]) in session_lookup
        and (len(parts) == 2 or (len(parts) == 3 and parts[2] == "README.md"))
    ):
        return f"{prefix}sessions/{parts[0]}/{parts[1]}/index.html{anchor_suffix}"

    if norm.startswith("assets/") and norm in {"assets/banner.svg", "assets/social-preview.png"}:
        return f"{prefix}{norm}"

    return f"{REPO_URL}/blob/main/{norm}{anchor_suffix}"


def render_inline(
    text: str,
    *,
    source_rel: Path,
    page_depth: int,
    module_slugs: set[str],
    session_lookup: dict[tuple[str, str], SessionInfo],
) -> str:
    code_placeholders: list[str] = []

    def _hold_code(match: re.Match[str]) -> str:
        idx = len(code_placeholders)
        code_placeholders.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"\x00CODE{idx}\x00"

    text = re.sub(r"`([^`]+)`", _hold_code, text)

    link_placeholders: list[str] = []

    def _hold_link(match: re.Match[str]) -> str:
        label = html.escape(match.group(1))
        raw_url = match.group(2).strip()
        target = rewrite_link(
            raw_url,
            source_rel=source_rel,
            page_depth=page_depth,
            module_slugs=module_slugs,
            session_lookup=session_lookup,
        )
        idx = len(link_placeholders)
        link_placeholders.append(f'<a href="{html.escape(target, quote=True)}">{label}</a>')
        return f"\x00LINK{idx}\x00"

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _hold_link, text)
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)

    for idx, replacement in enumerate(link_placeholders):
        escaped = escaped.replace(f"\x00LINK{idx}\x00", replacement)
    for idx, replacement in enumerate(code_placeholders):
        escaped = escaped.replace(f"\x00CODE{idx}\x00", replacement)
    return escaped


def render_markdown(
    markdown: str,
    *,
    source_rel: Path,
    page_depth: int,
    module_slugs: set[str],
    session_lookup: dict[tuple[str, str], SessionInfo],
) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    i = 0

    def inline(chunk: str) -> str:
        return render_inline(
            chunk,
            source_rel=source_rel,
            page_depth=page_depth,
            module_slugs=module_slugs,
            session_lookup=session_lookup,
        )

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            cls = f' class="language-{html.escape(lang)}"' if lang else ""
            out.append(f"<pre><code{cls}>{html.escape(chr(10).join(code_lines))}</code></pre>")
            continue

        if stripped.startswith("$$"):
            math_lines = [line]
            if not (len(stripped) > 2 and stripped.endswith("$$")):
                i += 1
                while i < len(lines):
                    math_lines.append(lines[i])
                    if lines[i].strip().endswith("$$"):
                        break
                    i += 1
            i += 1
            out.append(f'<div class="math-block">{html.escape(chr(10).join(math_lines))}</div>')
            continue

        if stripped == "---":
            out.append("<hr>")
            i += 1
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            hid = slugify_heading(text)
            out.append(f'<h{level} id="{hid}">{inline(text)}</h{level}>')
            i += 1
            continue

        if stripped.startswith(">"):
            quote_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append(f"<blockquote><p>{inline(' '.join(quote_lines))}</p></blockquote>")
            continue

        if (
            "|" in stripped
            and i + 1 < len(lines)
            and re.match(r"^\|?[\s:|-]+\|[\s:|-]*$", lines[i + 1].strip())
        ):
            header_cells = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and "|" in lines[i].strip() and lines[i].strip():
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            ths = "".join(f"<th>{inline(c)}</th>" for c in header_cells)
            trs = "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>" for row in rows
            )
            out.append(f"<table><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>")
            continue

        if re.match(r"^[-*]\s+", stripped):
            items: list[str] = []
            while i < len(lines) and re.match(r"^[-*]\s+", lines[i].strip()):
                item_text = re.sub(r"^[-*]\s+", "", lines[i].strip())
                if item_text.startswith("[x] "):
                    item_text = f"&#9745; {item_text[4:]}"
                elif item_text.startswith("[ ] "):
                    item_text = f"&#9744; {item_text[4:]}"
                items.append(f"<li>{inline(item_text)}</li>")
                i += 1
            out.append(f"<ul>{''.join(items)}</ul>")
            continue

        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                item_text = re.sub(r"^\d+\.\s+", "", lines[i].strip())
                items.append(f"<li>{inline(item_text)}</li>")
                i += 1
            out.append(f"<ol>{''.join(items)}</ol>")
            continue

        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (
                not nxt
                or nxt.startswith(("#", "```", "$$", ">", "---"))
                or re.match(r"^[-*]\s+", nxt)
                or re.match(r"^\d+\.\s+", nxt)
            ):
                break
            para_lines.append(nxt)
            i += 1
        out.append(f"<p>{inline(' '.join(para_lines))}</p>")

    return "\n".join(out)


def discover_curriculum(root: Path) -> tuple[list[ModuleInfo], list[SessionInfo]]:
    modules: list[ModuleInfo] = []
    all_sessions: list[SessionInfo] = []

    for module_slug in MODULES:
        module_dir = root / module_slug
        readme_path = module_dir / "README.md"
        if not readme_path.is_file():
            continue
        md = readme_path.read_text(encoding="utf-8")
        raw_title = extract_title(md, module_slug)
        clean_title = re.sub(r"^Module\s+\d+:\s*", "", raw_title)
        number = module_slug.split("-", 1)[0]
        topics = extract_topics(md)
        overview = extract_overview_paragraph(md, clean_title)
        weeks_line = extract_weeks_line(md)

        sessions: list[SessionInfo] = []
        for child in sorted(module_dir.iterdir()):
            match = SESSION_DIR_RE.match(child.name)
            if not (child.is_dir() and match):
                continue
            s_readme = child / "README.md"
            if not s_readme.is_file():
                continue
            s_md = s_readme.read_text(encoding="utf-8")
            s_title = extract_title(s_md, child.name)
            s_overview = extract_overview_paragraph(s_md, s_title)
            notebooks = sorted(p.name for p in (child / "notebooks").glob("*.ipynb"))
            session_info = SessionInfo(
                module_slug=module_slug,
                slug=child.name,
                schedule_date=match.group(1),
                title=s_title,
                overview=s_overview,
                topics=topics,
                notebooks=notebooks,
                readme_path=s_readme,
            )
            sessions.append(session_info)
            all_sessions.append(session_info)

        modules.append(
            ModuleInfo(
                slug=module_slug,
                number=number,
                title=clean_title,
                weeks_line=weeks_line,
                overview=overview,
                topics=topics,
                readme_path=readme_path,
                sessions=sessions,
            )
        )

    all_sessions.sort(key=lambda s: s.schedule_date, reverse=True)
    return modules, all_sessions


def render_page_shell(
    *,
    title: str,
    description: str,
    keywords: list[str],
    canonical_url: str,
    og_type: str,
    page_depth: int,
    json_ld: dict[str, object],
    body_html: str,
    include_math: bool = False,
    extra_scripts: str = "",
) -> str:
    prefix = "../" * page_depth
    image_url = f"{SITE_URL}/assets/social-preview.png"
    kw_str = ", ".join(dict.fromkeys(keywords))
    ld_json = json.dumps(json_ld, ensure_ascii=False, separators=(",", ":"))
    math_head = ""
    if include_math:
        math_head = (
            '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" crossorigin="anonymous">\n'
            '<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" crossorigin="anonymous"></script>\n'
            '<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" crossorigin="anonymous" '
            "onload=\"renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}]});\"></script>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <meta name="description" content="{html.escape(description, quote=True)}">
  <meta name="keywords" content="{html.escape(kw_str, quote=True)}">
  <meta name="author" content="python-to-ai contributors">
  <meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1">
  <link rel="canonical" href="{html.escape(canonical_url, quote=True)}">
  <meta property="og:type" content="{html.escape(og_type, quote=True)}">
  <meta property="og:site_name" content="python-to-ai">
  <meta property="og:locale" content="en_US">
  <meta property="og:title" content="{html.escape(title, quote=True)}">
  <meta property="og:description" content="{html.escape(description, quote=True)}">
  <meta property="og:url" content="{html.escape(canonical_url, quote=True)}">
  <meta property="og:image" content="{html.escape(image_url, quote=True)}">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="630">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="{html.escape(title, quote=True)}">
  <meta name="twitter:description" content="{html.escape(description, quote=True)}">
  <meta name="twitter:image" content="{html.escape(image_url, quote=True)}">
  <script type="application/ld+json">{ld_json}</script>
  {math_head}
  <style>{CSS_STYLES}</style>
</head>
<body>
  <header class="site-header">
    <div class="container nav">
      <a class="brand" href="{prefix}index.html">python-to-ai</a>
      <nav class="nav-links" aria-label="Primary navigation">
        <a href="{prefix}index.html#modules">Modules</a>
        <a href="{prefix}index.html#sessions">Sessions</a>
        <a href="{prefix}guides/setup/index.html">Setup</a>
        <a href="{prefix}guides/study-guide/index.html">Study Guide</a>
        <a href="{REPO_URL}" rel="noopener">GitHub</a>
      </nav>
    </div>
  </header>
  <main class="container">
    {body_html}
  </main>
  <footer class="site-footer">
    <div class="container">
      <p><strong>python-to-ai</strong> &middot; Open-source 54-week Python, Machine Learning, Deep Learning, Computer Vision, NLP, Generative AI &amp; LLM curriculum (MIT License).</p>
      <p><a href="{REPO_URL}">GitHub Repository</a> &middot; <a href="{prefix}sitemap.xml">Sitemap</a> &middot; <a href="{prefix}robots.txt">robots.txt</a></p>
    </div>
  </footer>
  {extra_scripts}
</body>
</html>
"""


def build_home_page(modules: list[ModuleInfo], sessions: list[SessionInfo]) -> str:
    all_tags: list[str] = []
    for mod in modules:
        for t in mod.topics:
            if t not in all_tags:
                all_tags.append(t)

    featured_tags = all_tags[:18]
    tag_buttons = ['<button type="button" class="tag active" data-filter-tag="all">all</button>']
    for tag in featured_tags:
        tag_buttons.append(
            f'<button type="button" class="tag" data-filter-tag="{html.escape(tag, quote=True)}">'
            f"#{html.escape(tag)}</button>"
        )

    module_cards: list[str] = []
    for mod in modules:
        tag_str = ",".join(mod.topics)
        search_blob = f"module {mod.number} {mod.title} {mod.overview} {' '.join(mod.topics)}"
        pills = "".join(f'<span class="tag">#{html.escape(t)}</span>' for t in mod.topics[:5])
        module_cards.append(
            f'<article class="card" data-searchable="{html.escape(search_blob, quote=True)}" '
            f'data-tags="{html.escape(tag_str, quote=True)}">'
            f'<div class="card-meta">Module {html.escape(mod.number)} &middot; {len(mod.sessions)} published sessions</div>'
            f'<h3><a href="modules/{mod.slug}/index.html">{html.escape(mod.title)}</a></h3>'
            f"<p>{html.escape(mod.overview)}</p>"
            f'<div class="tag-list">{pills}</div>'
            f"</article>"
        )

    session_cards: list[str] = []
    for sess in sessions:
        tag_str = ",".join(sess.topics)
        search_blob = f"{sess.schedule_date} {sess.title} {sess.overview} {sess.module_slug} {' '.join(sess.topics)}"
        session_cards.append(
            f'<article class="card" data-searchable="{html.escape(search_blob, quote=True)}" '
            f'data-tags="{html.escape(tag_str, quote=True)}">'
            f'<div class="card-meta">{html.escape(sess.schedule_date)} &middot; {len(sess.notebooks)} notebooks</div>'
            f'<h3><a href="sessions/{sess.module_slug}/{sess.slug}/index.html">{html.escape(sess.title)}</a></h3>'
            f"<p>{html.escape(sess.overview)}</p>"
            f"</article>"
        )

    faq_html = "".join(
        f'<div class="faq-item"><h3>{html.escape(q)}</h3><p>{html.escape(a)}</p></div>'
        for q, a in FAQS
    )

    body_html = f"""
    <section class="hero">
      <img class="hero-banner" src="assets/banner.svg" alt="python-to-ai: 54-week Python to AI curriculum banner">
      <h1>python-to-ai: 54-Week Python, Machine Learning &amp; AI Curriculum</h1>
      <p class="lead">{html.escape(DEFAULT_DESCRIPTION)}</p>
      <div class="cta-row">
        <a class="btn btn-primary" href="{REPO_URL}">Explore on GitHub</a>
        <a class="btn" href="guides/setup/index.html">Environment Setup</a>
        <a class="btn" href="guides/study-guide/index.html">Study Guide</a>
      </div>
      <input id="curriculum-search" class="search-bar" type="search" placeholder="Search modules, sessions, PyTorch, YOLO, gradient descent, RAG..." aria-label="Search curriculum">
      <div class="tag-list" aria-label="Filter by topic tag">{"".join(tag_buttons)}</div>
    </section>

    <section id="modules">
      <h2 class="section-title">10-Module Curriculum Roadmap (Weeks 1&ndash;54)</h2>
      <div class="grid">{"".join(module_cards)}</div>
    </section>

    <section id="sessions">
      <h2 class="section-title">Published Study Guides &amp; Notebooks ({len(sessions)})</h2>
      <div class="grid">{"".join(session_cards)}</div>
    </section>

    <section id="faq">
      <h2 class="section-title">Frequently Asked Questions</h2>
      {faq_html}
    </section>
    """

    json_ld = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "WebSite",
                "name": "python-to-ai",
                "url": f"{SITE_URL}/",
                "description": DEFAULT_DESCRIPTION,
            },
            {
                "@type": "Course",
                "name": "python-to-ai: From Python Fundamentals to Modern AI Systems",
                "description": DEFAULT_DESCRIPTION,
                "url": f"{SITE_URL}/",
                "inLanguage": "en",
                "isAccessibleForFree": True,
                "license": "https://opensource.org/licenses/MIT",
                "provider": {
                    "@type": "Organization",
                    "name": "python-to-ai",
                    "url": REPO_URL,
                },
                "teaches": BASE_KEYWORDS,
                "hasPart": [
                    {
                        "@type": "Course",
                        "name": f"Module {mod.number}: {mod.title}",
                        "description": mod.overview,
                        "url": f"{SITE_URL}/modules/{mod.slug}/",
                    }
                    for mod in modules
                ],
            },
            {
                "@type": "FAQPage",
                "mainEntity": [
                    {
                        "@type": "Question",
                        "name": q,
                        "acceptedAnswer": {"@type": "Answer", "text": a},
                    }
                    for q, a in FAQS
                ],
            },
        ],
    }

    return render_page_shell(
        title="python-to-ai — 54-Week Python, Machine Learning & AI Curriculum",
        description=DEFAULT_DESCRIPTION,
        keywords=BASE_KEYWORDS + all_tags,
        canonical_url=f"{SITE_URL}/",
        og_type="website",
        page_depth=0,
        json_ld=json_ld,
        body_html=body_html,
        extra_scripts=SEARCH_SCRIPT,
    )


def build_module_page(
    mod: ModuleInfo,
    *,
    root: Path,
    module_slugs: set[str],
    session_lookup: dict[tuple[str, str], SessionInfo],
) -> str:
    md = mod.readme_path.read_text(encoding="utf-8")
    source_rel = mod.readme_path.relative_to(root)
    rendered = render_markdown(
        md,
        source_rel=source_rel,
        page_depth=2,
        module_slugs=module_slugs,
        session_lookup=session_lookup,
    )
    canonical = f"{SITE_URL}/modules/{mod.slug}/"
    title = f"Module {mod.number}: {mod.title} | python-to-ai"
    tags_html = "".join(f'<li class="tag">#{html.escape(t)}</li>' for t in mod.topics)

    body_html = f"""
    <nav class="breadcrumb" aria-label="Breadcrumb">
      <a href="../../index.html">Home</a> / Module {html.escape(mod.number)}
    </nav>
    <ul class="tag-list">{tags_html}</ul>
    <article class="prose">
      {rendered}
    </article>
    """

    json_ld = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "BreadcrumbList",
                "itemListElement": [
                    {"@type": "ListItem", "position": 1, "name": "Home", "item": f"{SITE_URL}/"},
                    {
                        "@type": "ListItem",
                        "position": 2,
                        "name": f"Module {mod.number}: {mod.title}",
                        "item": canonical,
                    },
                ],
            },
            {
                "@type": "Course",
                "name": f"Module {mod.number}: {mod.title}",
                "description": mod.overview,
                "url": canonical,
                "isAccessibleForFree": True,
                "inLanguage": "en",
                "keywords": mod.topics,
                "provider": {"@type": "Organization", "name": "python-to-ai", "url": REPO_URL},
            },
        ],
    }

    return render_page_shell(
        title=title,
        description=mod.overview,
        keywords=BASE_KEYWORDS + mod.topics,
        canonical_url=canonical,
        og_type="article",
        page_depth=2,
        json_ld=json_ld,
        body_html=body_html,
    )


def build_session_page(
    sess: SessionInfo,
    mod: ModuleInfo,
    *,
    root: Path,
    module_slugs: set[str],
    session_lookup: dict[tuple[str, str], SessionInfo],
) -> str:
    md = sess.readme_path.read_text(encoding="utf-8")
    source_rel = sess.readme_path.relative_to(root)
    rendered = render_markdown(
        md,
        source_rel=source_rel,
        page_depth=3,
        module_slugs=module_slugs,
        session_lookup=session_lookup,
    )
    canonical = f"{SITE_URL}/sessions/{sess.module_slug}/{sess.slug}/"
    title = f"{sess.title} — Module {mod.number} | python-to-ai"
    nb_links = "".join(
        f'<li><a href="{REPO_URL}/blob/main/{sess.module_slug}/{sess.slug}/notebooks/{html.escape(nb)}" '
        f'rel="noopener">{html.escape(nb)}</a></li>'
        for nb in sess.notebooks
    )
    nb_box = (
        f"<section><h2>Executable Notebooks</h2><ul>{nb_links}</ul></section>" if nb_links else ""
    )

    body_html = f"""
    <nav class="breadcrumb" aria-label="Breadcrumb">
      <a href="../../../index.html">Home</a> /
      <a href="../../../modules/{mod.slug}/index.html">Module {html.escape(mod.number)}</a> /
      {html.escape(sess.title)}
    </nav>
    <article class="prose">
      {nb_box}
      {rendered}
    </article>
    """

    json_ld = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "BreadcrumbList",
                "itemListElement": [
                    {"@type": "ListItem", "position": 1, "name": "Home", "item": f"{SITE_URL}/"},
                    {
                        "@type": "ListItem",
                        "position": 2,
                        "name": f"Module {mod.number}: {mod.title}",
                        "item": f"{SITE_URL}/modules/{mod.slug}/",
                    },
                    {"@type": "ListItem", "position": 3, "name": sess.title, "item": canonical},
                ],
            },
            {
                "@type": "TechArticle",
                "headline": sess.title,
                "description": sess.overview,
                "datePublished": sess.schedule_date,
                "inLanguage": "en",
                "url": canonical,
                "keywords": sess.topics,
                "author": {"@type": "Organization", "name": "python-to-ai contributors"},
                "isPartOf": {"@type": "Course", "name": f"Module {mod.number}: {mod.title}"},
            },
        ],
    }

    return render_page_shell(
        title=title,
        description=sess.overview,
        keywords=BASE_KEYWORDS + sess.topics + [sess.title.lower()],
        canonical_url=canonical,
        og_type="article",
        page_depth=3,
        json_ld=json_ld,
        body_html=body_html,
        include_math=True,
    )


def build_guide_page(
    *,
    root: Path,
    source_path: Path,
    slug: str,
    title: str,
    description: str,
    module_slugs: set[str],
    session_lookup: dict[tuple[str, str], SessionInfo],
) -> str:
    md = source_path.read_text(encoding="utf-8")
    rendered = render_markdown(
        md,
        source_rel=source_path.relative_to(root),
        page_depth=2,
        module_slugs=module_slugs,
        session_lookup=session_lookup,
    )
    canonical = f"{SITE_URL}/guides/{slug}/"
    body_html = f"""
    <nav class="breadcrumb" aria-label="Breadcrumb">
      <a href="../../index.html">Home</a> / {html.escape(title)}
    </nav>
    <article class="prose">
      {rendered}
    </article>
    """
    json_ld = {
        "@context": "https://schema.org",
        "@type": "TechArticle",
        "headline": title,
        "description": description,
        "url": canonical,
        "inLanguage": "en",
    }
    return render_page_shell(
        title=f"{title} | python-to-ai",
        description=description,
        keywords=BASE_KEYWORDS,
        canonical_url=canonical,
        og_type="article",
        page_depth=2,
        json_ld=json_ld,
        body_html=body_html,
    )


def write_sitemap(output_dir: Path, urls: list[tuple[str, str, str, str]]) -> None:
    urlset = ET.Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")
    for loc, lastmod, changefreq, priority in urls:
        url_el = ET.SubElement(urlset, "url")
        ET.SubElement(url_el, "loc").text = loc
        ET.SubElement(url_el, "lastmod").text = lastmod
        ET.SubElement(url_el, "changefreq").text = changefreq
        ET.SubElement(url_el, "priority").text = priority
    tree = ET.ElementTree(urlset)
    ET.indent(tree, space="  ")
    tree.write(output_dir / "sitemap.xml", encoding="utf-8", xml_declaration=True)


def build_site(root: Path, output_dir: Path) -> list[Path]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modules, sessions = discover_curriculum(root)
    module_map = {m.slug: m for m in modules}
    module_slugs = set(module_map)
    session_lookup = {(s.module_slug, s.slug): s for s in sessions}

    assets_out = output_dir / "assets"
    assets_out.mkdir(parents=True, exist_ok=True)
    for asset_name in ("banner.svg", "social-preview.png"):
        src = root / "assets" / asset_name
        if src.is_file():
            shutil.copy2(src, assets_out / asset_name)

    for search_dir in (root, root / "docs"):
        if not search_dir.is_dir():
            continue
        for verification_file in sorted(search_dir.glob("google*.html")):
            shutil.copy2(verification_file, output_dir / verification_file.name)

    generated: list[Path] = []
    today_iso = date.today().isoformat()
    sitemap_entries: list[tuple[str, str, str, str]] = [
        (f"{SITE_URL}/", today_iso, "weekly", "1.0"),
    ]

    index_file = output_dir / "index.html"
    index_file.write_text(build_home_page(modules, sessions), encoding="utf-8")
    generated.append(index_file)

    for mod in modules:
        mod_dir = output_dir / "modules" / mod.slug
        mod_dir.mkdir(parents=True, exist_ok=True)
        mod_file = mod_dir / "index.html"
        mod_file.write_text(
            build_module_page(
                mod, root=root, module_slugs=module_slugs, session_lookup=session_lookup
            ),
            encoding="utf-8",
        )
        generated.append(mod_file)
        sitemap_entries.append((f"{SITE_URL}/modules/{mod.slug}/", today_iso, "weekly", "0.8"))

    for sess in sessions:
        mod = module_map[sess.module_slug]
        sess_dir = output_dir / "sessions" / sess.module_slug / sess.slug
        sess_dir.mkdir(parents=True, exist_ok=True)
        sess_file = sess_dir / "index.html"
        sess_file.write_text(
            build_session_page(
                sess, mod, root=root, module_slugs=module_slugs, session_lookup=session_lookup
            ),
            encoding="utf-8",
        )
        generated.append(sess_file)
        sitemap_entries.append(
            (
                f"{SITE_URL}/sessions/{sess.module_slug}/{sess.slug}/",
                sess.schedule_date,
                "monthly",
                "0.7",
            )
        )

    guides = [
        (
            root / "docs" / "setup.md",
            "setup",
            "Environment Setup",
            "Step-by-step Python, virtual environment, JupyterLab, and PyTorch GPU setup guide for python-to-ai.",
        ),
        (
            root / "docs" / "study-guide.md",
            "study-guide",
            "Study Guide",
            "Weekly study rhythm, notebook standards, project rubric, and experiment checklist for python-to-ai.",
        ),
    ]
    for source_path, slug, title, description in guides:
        if not source_path.is_file():
            continue
        guide_dir = output_dir / "guides" / slug
        guide_dir.mkdir(parents=True, exist_ok=True)
        guide_file = guide_dir / "index.html"
        guide_file.write_text(
            build_guide_page(
                root=root,
                source_path=source_path,
                slug=slug,
                title=title,
                description=description,
                module_slugs=module_slugs,
                session_lookup=session_lookup,
            ),
            encoding="utf-8",
        )
        generated.append(guide_file)
        sitemap_entries.append((f"{SITE_URL}/guides/{slug}/", today_iso, "monthly", "0.6"))

    write_sitemap(output_dir, sitemap_entries)
    (output_dir / "robots.txt").write_text(
        f"User-agent: *\nAllow: /\n\nSitemap: {SITE_URL}/sitemap.xml\n",
        encoding="utf-8",
    )
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SEO-ready GitHub Pages site.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("site"),
        help="Output directory for the generated static site (default: site).",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = args.output if args.output.is_absolute() else repo_root / args.output
    pages = build_site(repo_root, out_dir)
    print(f"Built {len(pages)} HTML pages into {out_dir}")


if __name__ == "__main__":
    main()
