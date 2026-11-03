# OOP Practice Lab Dataset Catalog

This directory contains synthetic datasets used for testing, validation, and benchmarking throughout the practice lab notebooks.

## Datasets Overview

| Filename | Format | Description | Target Notebook |
| :--- | :--- | :--- | :--- |
| `ward_residents.json` | JSON | Municipal citizen census records with polymorphic occupations | Notebook 02 |
| `device_catalog.csv` | CSV | Hardware devices with manufacturer association metadata | Notebook 01 |
| `buffer_workload.json` | JSON | Computational pipeline task queue stream for buffer benchmarks | Notebooks 03, 04, 05 |

---

## 1. Ward Residents Census (`ward_residents.json`)

Structured JSON records representing citizens residing within an administrative ward division. Used to demonstrate class inheritance hierarchies, polymorphism, and aggregate querying.

### Schema Definition

- `role` (string): Occupational role (`student`, `teacher`, `doctor`). Determines subclass instantiation.
- `name` (string): Full name of the resident.
- `yob` (integer): Year of birth (Gregorian calendar year, four digits).
- `grade` (string, optional): Academic class or grade level (present if `role == "student"`).
- `subject` (string, optional): Teaching subject specialization (present if `role == "teacher"`).
- `specialist` (string, optional): Medical specialty field (present if `role == "doctor"`).

---

## 2. Hardware Device Catalog (`device_catalog.csv`)

Tabular record of specialized computing devices and their manufacturing origins. Used to demonstrate object aggregation, reference encapsulation, and catalog serialization.

### Schema Definition

| Column | Type | Description |
| :--- | :--- | :--- |
| `device_name` | String | Hardware product commercial identifier |
| `price_usd` | Float | Unit price in US dollars |
| `manufacturer_id` | Integer | Unique identifier of the manufacturing entity |
| `manufacturer_country` | String | Country of origin of the manufacturer |

---

## 3. Buffer Workload Telemetry (`buffer_workload.json`)

JSON payload representing an execution workload of tensor operators in deep learning workflows. Used to evaluate bounded stack and circular queue data structures under burst traffic and memory limits.

### Schema Definition

- `task_id` (string): Unique identifier of the computational task.
- `priority` (integer): Priority rank where 1 indicates high priority.
- `operation` (string): Neural network primitive or tensor operator name.
- `tensor_shape` (list[int]): Dimensions of the input tensor array.
- `latency_ms` (float): Execution latency in milliseconds.
