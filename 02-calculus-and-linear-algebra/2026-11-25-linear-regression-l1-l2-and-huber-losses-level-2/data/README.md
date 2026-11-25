# Data Directory

This directory contains three datasets used in the Linear Regression (L1, L2, and Huber
Losses — Level 2) session.

---

## `data.csv` — Toy Housing Dataset

A minimal synthetic dataset for single-feature linear regression experiments.

| Column | Index | Type | Unit | Description |
|---|---|---|---|---|
| `area` | 0 | `float` | ×100 m² | Floor area of the property |
| `price` | 1 | `float` | Tael (金兩) | Sale price of the property |

**Shape:** 4 rows × 2 columns (no header row).

**Sample rows:**

```
6.7, 9.1
4.6, 5.9
3.5, 4.6
5.5, 6.7
```

**Notes:** This dataset is intentionally tiny — it is meant for hand-tracing gradient updates
and visualising the fitted line, not for benchmarking.

---

## `data_outlier.csv` — Toy Housing Dataset with Outlier

Identical to `data.csv` with one additional row that acts as an outlier (high price relative
to its area), making it ideal for demonstrating the difference in robustness between MSE and
MAE / Huber loss.

| Column | Index | Type | Unit | Description |
|---|---|---|---|---|
| `area` | 0 | `float` | ×100 m² | Floor area of the property |
| `price` | 1 | `float` | Tael | Sale price of the property |

**Shape:** 5 rows × 2 columns (no header row).

**Sample rows:**

```
6.7, 8.6
4.6, 5.9
3.5, 4.6
5.5, 6.7
4.0, 7.2   ← outlier (high price for a small area)
```

---

## `advertising.csv` — Advertising Budget and Sales

A classic marketing dataset often used in introductory regression courses. Each row represents
one market, recording the advertising budget (in thousands of dollars) across three channels
and the resulting sales volume.

| Column | Index | Type | Unit | Description |
|---|---|---|---|---|
| `TV` | 0 | `float` | \$000 | TV advertising budget |
| `Radio` | 1 | `float` | \$000 | Radio advertising budget |
| `Newspaper` | 2 | `float` | \$000 | Newspaper advertising budget |
| `Sales` | 3 | `float` | 000 units | Product sales volume |

**Shape:** 200 rows × 4 columns (1 header row).

**Value ranges:**

| Column | Min | Max | Mean |
|---|---|---|---|
| TV | 0.7 | 296.4 | 147.0 |
| Radio | 0.0 | 49.6 | 23.3 |
| Newspaper | 0.3 | 114.0 | 30.6 |
| Sales | 1.6 | 27.0 | 14.0 |

**Source:** This dataset is widely used in academia and appears in *An Introduction to
Statistical Learning* (James et al., 2021). It is in the public domain and contains no
personally identifiable information.

**Usage notes:**
- The three advertising channels are used as input features; `Sales` is the target.
- Apply mean–range feature scaling to all three input columns before training. Store
  `mean_v`, `max_v`, and `min_v` from the training statistics for use at inference time.
- The `Sales` column is **not** scaled in the provided notebooks.
