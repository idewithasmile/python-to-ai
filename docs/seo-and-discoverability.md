# SEO and Discoverability Guide

This document records the search-engine and GitHub discoverability setup for `python-to-ai`.

## 1. Repository Metadata

- **Package & citation metadata:** [`pyproject.toml`](../pyproject.toml) (`keywords`, `classifiers`, `[project.urls]`) and [`CITATION.cff`](../CITATION.cff).
- **Social preview image:** [`assets/social-preview.png`](../assets/social-preview.png) (`1200x630` PNG for Open Graph, Twitter Cards, and GitHub Settings).
- **Topic tags:** [`README.md`](../README.md) and each module `README.md` include curated `**Topics:**` tags.

## 2. Recommended GitHub Topics (Up to 20)

Apply these topics in **GitHub Repository Settings -> About -> Topics** (or via GitHub CLI):

```text
python, artificial-intelligence, machine-learning, deep-learning, computer-vision, nlp, generative-ai, llm, rag, pytorch, numpy, data-science, ai-curriculum, roadmap, study-guide, jupyter-notebook, linear-algebra, neural-networks, transformers, ai-agents
```

Command-line equivalent using `gh`:

```powershell
gh repo edit idewithasmile/python-to-ai `
  --homepage "https://idewithasmile.github.io/python-to-ai/" `
  --description "54-week open-source AI curriculum: Python, math for ML, PyTorch, computer vision, NLP, generative AI, LLMs & RAG." `
  --add-topic "python,artificial-intelligence,machine-learning,deep-learning,computer-vision,nlp,generative-ai,llm,rag,pytorch,numpy,data-science,ai-curriculum,roadmap,study-guide,jupyter-notebook,linear-algebra,neural-networks,transformers,ai-agents"
```

## 3. GitHub Pages and Google Search

The workflow [`.github/workflows/pages.yml`](../.github/workflows/pages.yml) builds [`scripts/build_site.py`](../scripts/build_site.py) into static HTML with:

- Unique `<title>`, `<meta name="description">`, `<meta name="keywords">`, and `<link rel="canonical">` per page.
- Open Graph (`og:*`) and Twitter Card (`twitter:*`) metadata.
- JSON-LD `schema.org` structured data (`Course`, `WebSite`, `FAQPage`, `TechArticle`, and `BreadcrumbList`).
- Automatically generated `sitemap.xml` and `robots.txt`.

After pushing to `main`:

1. Enable **Settings -> Pages -> Source: GitHub Actions**.
2. Upload [`assets/social-preview.png`](../assets/social-preview.png) under **Settings -> General -> Social preview**.
3. Verify `https://idewithasmile.github.io/python-to-ai/sitemap.xml` in **Google Search Console** so Google indexes the landing page, all 10 modules, and all published study guides.
