# Module 01: Python Fundamentals & Algorithmic Thinking

[Back to the master roadmap](../README.md) · [Environment setup](../docs/setup.md)

**Weeks 1–4 · November 2–29, 2026 · Status: In progress**

## Overview

This module builds the programming foundation for the rest of the curriculum. The emphasis is on writing correct, readable, tested Python rather than memorizing syntax. You will move from control flow and data structures to object-oriented design, recursion, algorithmic problem solving, tabular data manipulation, and a small but complete application.

The module deliberately combines practical Python with algorithmic thinking. Big-O, recursion, two pointers, and dynamic programming are introduced where they solve real problems, not as isolated theory.

## Learning Outcomes

By the end of this module, you should be able to:

- Write idiomatic Python using functions, comprehensions, generators, classes, dataclasses, and type hints.
- Choose appropriate built-in data structures and reason about their time and space complexity.
- Read and write files, handle exceptions, and structure code into reusable modules.
- Use pandas, requests, and an HTML parser to build a small data pipeline.
- Apply recursion, two pointers, and dynamic programming to well-scoped problems.
- Write unit tests, debug systematically, and document a reproducible command-line project.

## Weekly Plan

| Week | Dates | Focus | Output |
| :---: | :--- | :--- | :--- |
| 01 | Nov 2–8, 2026 | Environment setup, syntax, control flow, functions, and Big-O | Verified environment and algorithm exercises |
| 02 | Nov 9–15, 2026 | Lists, dictionaries, sets, tuples, comprehensions, iterators, files, and exceptions | Data-structure lab |
| 03 | Nov 16–22, 2026 | OOP, dataclasses, typing, recursion, two pointers, testing, and debugging | Tested OOP mini-library |
| 04 | Nov 23–29, 2026 | pandas, APIs, web parsing, dynamic programming, and CLI design | Python Data Toolkit milestone |

## Published Sessions

| Date | Topic | Materials |
| :--- | :--- | :--- |
| 2026-10-21 | Basic Python (Part 1) | [Study guide and notebooks](2026-10-21-basic-python-part-1/) |
| 2026-10-22 | Basic Python (Part 2) | [Study guide and notebooks](2026-10-22-basic-python-part-2/) |
| 2026-10-23 | Big-O (Time Analysis) | [Study guide and notebooks](2026-10-23-big-o-time-analysis/) |
| 2026-10-24 | Basic Python Practice Lab | [Study guide and notebooks](2026-10-24-basic-python-practice-lab/) |
| 2026-10-25 | Data Structure | [Study guide and notebooks](2026-10-25-data-structure/) |
| 2026-10-26 | Data Visualization | [Study guide and notebooks](2026-10-26-data-visualization/) |

## Milestone Project: Python Data Toolkit

Build a command-line application that ingests data from a CSV file and a public JSON API, cleans and merges the data, produces a small summary report, and exports the result.

### Requirements

- A reproducible environment file and a short setup section.
- Typed, documented functions with clear module boundaries.
- Input validation and useful error messages.
- Data cleaning and transformation with pandas.
- A command-line interface built with `argparse`.
- At least six unit tests covering normal and edge cases.
- A README with usage examples and a sample output.
- No credentials, personal data, or large files committed.

### Stretch Goals

- Add caching for API requests.
- Add structured logging and a configurable output format.
- Package the project with `pyproject.toml`.
- Add property-based tests with Hypothesis.
- Profile the slowest stage and improve it.

## Workspace

```text
01-python-fundamentals/
├── 2026-10-21-basic-python-part-1/
│   ├── assets/
│   ├── notebooks/
│   └── README.md
├── 2026-10-22-basic-python-part-2/
│   ├── data/
│   ├── notebooks/
│   └── README.md
├── 2026-10-23-big-o-time-analysis/
│   ├── notebooks/
│   └── README.md
├── 2026-10-24-basic-python-practice-lab/
│   ├── notebooks/
│   └── README.md
├── 2026-10-25-data-structure/
│   ├── data/
│   ├── notebooks/
│   └── README.md
├── 2026-10-26-data-visualization/
│   ├── data/
│   ├── notebooks/
│   └── README.md
├── notebooks/
│   ├── 01-python-basics.ipynb
│   ├── 02-data-structures.ipynb
│   ├── 03-oop-and-recursion.ipynb
│   └── 04-pandas-and-apis.ipynb
├── exercises/
└── projects/
    └── python-data-toolkit/
```

## Tools

- Python 3.10+
- pandas, requests, BeautifulSoup4
- pytest, Hypothesis, Ruff
- JupyterLab
- Git

## Deep Dives

- Recursive backtracking and search spaces.
- Dynamic programming patterns: memoization and tabulation.
- Clean code, naming, and small functions.
- Profiling CPU and memory usage.
- Reading and writing data safely.

## Assessment

| Criterion | Weight | What good looks like |
| :--- | :---: | :--- |
| Problem and data | 15% | Inputs are well-defined and the pipeline handles real-world imperfections. |
| Correctness | 30% | Functions are tested; edge cases are handled; the CLI behaves predictably. |
| Code quality | 20% | Readable names, small modules, type hints, and no duplicated logic. |
| Reproducibility | 15% | Setup, commands, and sample data are documented and repeatable. |
| Communication | 20% | README explains the problem, usage, design, and limitations clearly. |

## References

- Python documentation: <https://docs.python.org/3/>
- pandas documentation: <https://pandas.pydata.org/docs/>
- pytest documentation: <https://docs.pytest.org/>
- *Python Crash Course* by Eric Matthes.
- *Fluent Python* by Luciano Ramalho.
- *Introduction to Algorithms* by Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.

## Module Checklist

- [ ] Environment verified with `python scripts/verify_environment.py`.
- [ ] All four weekly notebooks run from a clean kernel.
- [ ] At least one independent exercise per week is complete.
- [ ] The milestone project passes its tests and runs from a documented command.
- [ ] The project README includes a dataset and API attribution.
- [ ] The module reflection is written and the master roadmap is updated.
