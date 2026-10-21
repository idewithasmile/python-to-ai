# Contributing

Thank you for improving `python-to-ai`. The repository is a learning curriculum, so clarity, reproducibility, and honest evaluation matter more than clever code.

## Ways to Contribute

- Fix a broken link, typo, or unclear explanation.
- Add a focused exercise, test, or worked example.
- Modernize deprecated code or package usage.
- Improve a project brief, rubric, or dataset card.
- Report a notebook that does not run from a clean environment.

## Before You Start

1. Search existing issues and pull requests.
2. Open an issue for substantial changes so the scope can be discussed.
3. Keep each pull request focused on one learning problem.
4. Use English for code, comments, docstrings, notebooks, commit messages, and documentation.

## Content Standards

- Teach one idea at a time.
- State assumptions and limitations explicitly.
- Include a baseline and a meaningful evaluation metric.
- Prefer reproducible commands over prose descriptions of commands.
- Link to public, stable references when possible.
- Do not include private course material, instructor names, cohort dates, private links, enrollment details, or branded classroom content.
- Do not include personal data, secrets, credentials, or unlicensed datasets.

## Code Standards

- Target Python 3.10 or newer.
- Use type hints for public functions and small, testable components.
- Prefer the standard library unless a dependency is already justified.
- Add a seed for any stochastic process.
- Keep notebooks deterministic and free of hidden state.
- Avoid committing generated outputs unless they are essential for learning.

Run the available checks:

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

## Notebook Standards

- Start with a title, objective, and expected artifact.
- Record imports, seeds, paths, and versions.
- Use short markdown explanations between code cells.
- Restart the kernel and run all cells before submitting.
- Clear large outputs, credentials, and absolute personal paths.
- Include at least one exercise that is not solved in the lesson.

## Data and Model Policy

- Never commit datasets larger than 50 MB, model weights, or generated media.
- Put local data in `data/`; its contents are ignored by Git.
- Document the dataset license, source, version, and split strategy.
- Use download scripts or dataset cards instead of committed copies.
- Keep model checkpoints and experiment artifacts in ignored directories.

## Commit Messages

Use a short imperative summary with a type and optional scope:

```text
feat(m03): add bootstrap confidence interval lab
fix(m04): prevent target leakage in preprocessing example
docs(m08): clarify tokenization exercise
test(m01): cover empty input edge case
chore: update Ruff configuration
```

## Pull Request Checklist

- [ ] The change is focused and explained.
- [ ] Notebooks run from a clean kernel.
- [ ] Tests and lint checks pass.
- [ ] Local links resolve.
- [ ] No large files, secrets, or personal data are included.
- [ ] The roadmap is updated if progress or dates changed.
- [ ] The contribution is written in English.

## Review Process

A maintainer will review the change for correctness, clarity, reproducibility, and fit with the curriculum. Reviews may request smaller examples, clearer assumptions, additional tests, or stronger evaluation. Negative results and honest limitations are welcome when they are documented clearly.

## Questions

Use the learning-question issue template for focused questions that include a reproducible example. For general study advice, consult [`docs/study-guide.md`](docs/study-guide.md).
