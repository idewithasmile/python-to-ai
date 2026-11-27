# Data Directory

This directory contains the dataset used in the **Linear Regression Practice Lab** (`2026-11-27-linear-regression-practice-lab`).

---

## `advertising.csv` — Advertising Budget and Sales

A classic multivariate regression dataset recording advertising expenditure (in thousands of US dollars) across three media channels alongside the resulting product sales volume (in thousands of units) across 200 regional markets.

### Schema

| Column | Index | Type | Unit | Description |
| :--- | :---: | :---: | :--- | :--- |
| `TV` | 0 | `float` | \$000 | Television advertising budget |
| `Radio` | 1 | `float` | \$000 | Radio advertising budget |
| `Newspaper` | 2 | `float` | \$000 | Newspaper advertising budget |
| `Sales` | 3 | `float` | 000 units | Product sales volume (regression target) |

**Shape:** 200 data rows × 4 columns (1 header row: `TV,Radio,Newspaper,Sales`).

### Summary Statistics

| Column | Min | Max | Mean | Std (Sample) |
| :--- | :---: | :---: | :---: | :---: |
| `TV` | 0.7 | 296.4 | 147.04 | 85.85 |
| `Radio` | 0.0 | 49.6 | 23.26 | 14.85 |
| `Newspaper` | 0.3 | 114.0 | 30.55 | 21.78 |
| `Sales` | 1.6 | 27.0 | 15.13 | 5.28 |

### Sample Rows

```csv
TV,Radio,Newspaper,Sales
230.1,37.8,69.2,22.1
44.5,39.3,45.1,10.4
17.2,45.9,69.3,12
151.5,41.3,58.5,16.5
180.8,10.8,58.4,17.9
```

### Usage Notes

- **Problem 1–3 layout (column-major):** `X = [tv_data, radio_data, newspaper_data]` of shape $3 \times 200$, and `y = sales_data` of length $200$.
- **Problem 4–5 layout (row-major with bias):** Each sample is formatted as `[1.0, tv, radio, newspaper]`, yielding a $200 \times 4$ feature representation where $x_0 = 1$ absorbs the bias parameter $b$.
- **Feature scaling:** Because `TV` spans $[0.7, 296.4]$ while `Radio` spans $[0.0, 49.6]$, apply Min-Max scaling (or standardization) to the input features using training-set statistics only, and retain `(max_vals, min_vals)` to transform test queries consistently at inference time. The target `Sales` remains unscaled.

### Source

Public benchmark dataset popularized in *An Introduction to Statistical Learning* (James, Witten, Hastie, and Tibshirani, 2021). Contains no personally identifiable information.
