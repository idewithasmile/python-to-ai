# Session Data: Sorting Benchmarks, Subsequence Sequences, and Dynamic Programming Assets

This directory provides standardized, portable test datasets and evaluation fixtures for the study guide and executable notebooks in [Dynamic Programming and Divide-and-Conquer](../README.md). All files are completely self-contained and run locally in Python without network access or third-party dependencies.

---

## Files and Schemas

### 1. `sorting_benchmarks.json`

Structured benchmark sequences designed to evaluate divide-and-conquer sorting implementations (Merge Sort and Quick Sort), testing best-case, average-case, and pathological worst-case inputs.

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `array` | `list[integer]` | Input integer array to be sorted | `[7, 2, 5, 30, 16, 8, 4, 17]` |
| `description` | `string` | Pedagogical role and structural characteristics of the case | `"Primary sequence analyzed step-by-step in lecture slides"` |

#### Benchmark Cases Included:
- `lecture_example`: The primary 8-element sequence (`[7, 2, 5, 30, 16, 8, 4, 17]`) evaluated across the lecture slides.
- `already_sorted`: Pre-sorted increasing array (`[1, 3, 5, 7, 9, 11, 13, 15]`); demonstrates $O(n^2)$ degeneration for fixed-pivot Quick Sort vs $O(n \log n)$ stability for Merge Sort.
- `reverse_sorted`: Strictly descending array (`[20, 18, 15, 12, 10, 8, 5, 2]`); tests symmetric inversion handling.
- `duplicates_heavy`: Array with recurring integers (`[4, 2, 4, 3, 2, 4, 1, 3, 2, 1, 4]`); tests equal-key partitioning stability.
- `single_element`: Terminal base-case array (`[42]`) of length 1.
- `empty_array`: Base-case array (`[]`) of length 0.
- `random_integers`: 20-element pseudorandom integer array for general verification.

---

### 2. `lis_sequences.json`

Test sequences for the Longest Increasing Subsequence (LIS) problem, paired with their mathematically exact maximal lengths and sample reconstructed subsequences.

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `nums` | `list[integer]` | Input integer sequence | `[3, 10, 2, 1, 20]` |
| `expected_length` | `integer` | Mathematically verified maximum length of any strictly increasing subsequence | `3` |
| `example_subsequence` | `list[integer]` | Valid candidate subsequence achieving `expected_length` | `[3, 10, 20]` |
| `description` | `string` | Structural rationale of the test sequence | `"Primary LIS walkthrough from lecture slides"` |

#### Benchmark Cases Included:
- `lecture_example`: The primary sequence from the slides (`[3, 10, 2, 1, 20]`), yielding optimal LIS length 3 (`[3, 10, 20]`).
- `standard_increasing`: Common textbook sequence (`[10, 22, 9, 33, 21, 50, 41, 60, 80]`) yielding length 6 (`[10, 22, 33, 50, 60, 80]`).
- `strictly_decreasing`: Descending sequence (`[9, 8, 7, 6, 5, 4, 3, 2, 1]`) verifying minimal length 1.
- `all_identical`: Array with identical elements (`[7, 7, 7, 7, 7]`) verifying strict inequality ($a_i < a_j$).
- `alternating_wave`: Zig-zag array (`[1, 5, 2, 6, 3, 7, 4, 8]`) with branching candidate paths, yielding length 5.
- `stock_price_trend`: Simulated financial closing prices testing momentum recovery.

---

### 3. `knapsack_items.csv`

Tabular dataset representing feature sets in a production machine learning deployment pipeline. Used for the 0/1 Knapsack optimization problem, where the objective is to maximize predictive utility under a bounded memory/latency budget.

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `item_id` | `integer` | Unique identifier of candidate feature | `1` |
| `feature_name` | `string` | Descriptive feature identifier | `"user_demographics"` |
| `memory_cost_kb` | `integer` | Memory / latency cost weight ($w_i$) in kilobytes | `12` |
| `utility_score` | `integer` | Information gain / feature importance score ($v_i$) | `35` |

- **Format:** Comma-separated values (UTF-8 encoding)
- **Pedagogical Target:** Solved using dynamic programming with 2D tabulation ($O(n \cdot W)$) and optimized 1D memory compression.

---

### 4. `edit_distance_pairs.json`

Evaluation pairs for Levenshtein minimum edit distance, modeling real-world text discrepancies in OCR transcription, automated speech recognition (ASR), and search query autocorrection.

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `id` | `string` | Unique benchmark identifier | `"pair_01"` |
| `str1` | `string` | Source character string | `"algorithm"` |
| `str2` | `string` | Target character string | `"algor1thm"` |
| `expected_distance` | `integer` | Minimum operations (insert, delete, substitute) required | `1` |
| `description` | `string` | Error modality modeled | `"Single character substitution (digit 1 for letter i)"` |

---

## Usage Example

```python
import json
from pathlib import Path

data_dir = Path(__file__).resolve().parent

# Load LIS test suite
with open(data_dir / "lis_sequences.json", encoding="utf-8") as f:
    lis_benchmarks = json.load(f)

for case_name, case in lis_benchmarks.items():
    print(f"Case: {case_name} -> Expected LIS length: {case['expected_length']}")
```
