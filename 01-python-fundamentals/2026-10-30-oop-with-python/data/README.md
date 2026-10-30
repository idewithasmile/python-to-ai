# Object-Oriented Modeling Datasets: Catalog & Organization Roster

This directory contains structured datasets used in object-oriented programming labs for class modeling, object aggregation, composite hierarchies, and custom collection sorting.

## Overview of Datasets

| File | Format | Records | Description | Primary Use Case |
| :--- | :---: | :---: | :--- | :--- |
| `books_catalog.json` | JSON | 8 items | Catalog of technical and scientific publications with metadata | Object instantiation, attribute encapsulation, custom comparator sorting (`__lt__`) |
| `organization_roster.json` | JSON | 2 departments, 3 teams, 5 employee assignments | Hierarchical corporate roster with nested date-of-birth and compensation objects | Demonstrating Aggregation (shared employee references) vs Composition (company-owned departments and teams) |

---

## 1. `books_catalog.json`

### Purpose and Provenance

`books_catalog.json` provides a clean, self-contained record set representing published computer science and artificial intelligence literature. It serves as benchmark data for:
1. Converting raw dictionary payloads into typed, validated `Book` class instances.
2. Encapsulating state with `@property` getters and price validation setters.
3. Sorting collections of `Book` instances by multiple dimensions (price, rating, publication year, page count).

### Schema Definition

| Field | Type | Nullable | Description |
| :--- | :---: | :---: | :--- |
| `isbn` | `string` | No | International Standard Book Number (unique identifier) |
| `title` | `string` | No | Full publication title |
| `author` | `string` | No | Primary author(s) or editor(s) |
| `genre` | `string` | No | Academic or technical classification |
| `price` | `float` | No | Retail price in USD ($> 0.0$) |
| `publication_year` | `integer` | No | Calendar year of release |
| `pages` | `integer` | No | Total page count |
| `rating` | `float` | No | Review score on a $1.0 - 5.0$ scale |

### Loading Example (Standard Library `json`)

```python
import json
from pathlib import Path

catalog_path = Path("data/books_catalog.json")
with open(catalog_path, mode="r", encoding="utf-8") as f:
    books_data = json.load(f)

print(f"Loaded {len(books_data)} book records.")
```

---

## 2. `organization_roster.json`

### Purpose and Provenance

`organization_roster.json` models real-world enterprise entity relationships to illustrate the critical architectural distinction between **Aggregation** and **Composition**:
- **Composition (Part-Of):** The `Company` owns its `departments`, and each department owns its `teams`. If a department is dissolved, its teams cease to exist within the corporate structure.
- **Aggregation (Has-A):** A `Team` references `employees`. Employees maintain independent lifecycles; they can transition between teams or be assigned concurrently across multiple cross-functional teams (e.g., Sarah Connor in both Core Algorithms and Computer Vision).
- **Composite Value Object:** Each employee holds an aggregated `Date` (`birth_date`) and a composed `Salary` structure (`base_monthly`, `annual_bonus`).

### Schema Definition

```text
Company
└── departments: list[Department]
    └── teams: list[Team]
        └── members: list[Employee]
            ├── birth_date: Date (day, month, year)
            └── salary: Salary (base_monthly, annual_bonus)
```

| Field Path | Type | Nullable | Description |
| :--- | :---: | :---: | :--- |
| `company_name` | `string` | No | Corporate legal entity name |
| `headquarters` | `string` | No | Geographic headquarters location |
| `founded_year` | `integer` | No | Incorporation year |
| `departments[].department_name` | `string` | No | Division identifier |
| `departments[].teams[].team_id` | `string` | No | Unique alphanumeric team identifier |
| `departments[].teams[].team_name` | `string` | No | Team operational title |
| `departments[].teams[].mission` | `string` | No | High-level engineering or research objective |
| `departments[].teams[].members[].employee_id` | `string` | No | Unique personnel badge code |
| `departments[].teams[].members[].name` | `string` | No | Employee full name |
| `departments[].teams[].members[].role` | `string` | No | Job title |
| `departments[].teams[].members[].email` | `string` | No | Corporate contact address |
| `departments[].teams[].members[].birth_date.day` | `integer` | No | Calendar day of birth ($1 - 31$) |
| `departments[].teams[].members[].birth_date.month` | `integer` | No | Calendar month of birth ($1 - 12$) |
| `departments[].teams[].members[].birth_date.year` | `integer` | No | Four-digit calendar year |
| `departments[].teams[].members[].salary.base_monthly` | `float` | No | Monthly salary payment ($> 0.0$) |
| `departments[].teams[].members[].salary.annual_bonus` | `float` | No | Performance bonus per annum |

### Loading Example (Standard Library `json`)

```python
import json
from pathlib import Path

roster_path = Path("data/organization_roster.json")
with open(roster_path, mode="r", encoding="utf-8") as f:
    roster_data = json.load(f)

company_name = roster_data["company_name"]
total_teams = sum(len(d["teams"]) for d in roster_data["departments"])
print(f"Company: {company_name} | Total Teams: {total_teams}")
```
