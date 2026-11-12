# Session Data: Tabular Regression Sample for Hyperparameter Optimization

This directory contains a lightweight, self-contained subset of tabular regression data used by the notebooks in [Hyper-parameter optimization](../README.md).

> **Repository size policy:** The full dataset files (`train.csv` at ~93.7 MB and `test.csv` at ~58.9 MB from the Kaggle Tabular Playground Series — Feb 2021 benchmark) exceed the repository's 50 MB file limit and are intentionally omitted from Git tracking. `tabular_regression_sample.csv` preserves the exact schema and statistical structure in a portable 500-row subset (~155 KB).

## Files and Schemas

### `tabular_regression_sample.csv`

Comma-separated tabular dataset (UTF-8 encoding, 500 records + header row) combining 10 categorical features, 14 continuous features, an integer identifier, and a continuous regression target.

| Column(s) | Type | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `id` | `integer` | Unique row ID | Primary sample identifier (set as index before training) |
| `cat0`–`cat9` | `string` | Single uppercase letters (`A`–`O`) | 10 anonymized categorical predictors encoded to integer/category codes |
| `cont0`–`cont13` | `float` | Approximately `[-0.1, 1.1]` | 14 anonymized continuous predictors |
| `target` | `float` | Continuous real target | Regression response variable evaluated via RMSE |

- **Format:** CSV with header row.
- **Target usage:** Cross-validated hyperparameter optimization (Bayesian Optimization with Gaussian Process surrogate and LightGBM/Gradient Boosting benchmarks).
