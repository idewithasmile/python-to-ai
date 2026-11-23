# Toy House Prices Datasets (Single-Variable Linear Regression)

This directory contains lightweight, single-feature benchmark datasets used to study the
fundamentals of univariate linear regression, computational graphs, and loss function
robustness.

## Files

| File | Rows | Columns | Description |
|---|---|---|---|
| `house_prices_toy.csv` | 8 | 2 | Nominal dataset representing house area versus sales price. |
| `house_prices_outlier.csv` | 9 | 2 | Nominal dataset with a single severe outlier injected at index 8 (`area=6.0, price=25.0`). |

---

## Schema

| Column | Type | Unit | Description |
|---|---|---|---|
| `area` | `float64` | Arbitrary units (e.g., $100\,\text{m}^2$) | Input feature representing the floor area of the property. |
| `price` | `float64` | Arbitrary units (e.g., $\$100{,}000$ or billion VND) | Target scalar continuous label representing the property valuation. |

---

## Benchmark Values from Lecture Syllabus

The first 4 rows match the exact numerical examples from the curriculum:

| Index | Area ($x$) | Price ($y$) | Usage |
|---|---|---|---|
| 0 | 6.7 | 9.1 | Primary one-sample stochastic gradient descent (SGD) trace. |
| 1 | 4.6 | 5.9 | Multi-sample batch gradient descent evaluations. |
| 2 | 3.5 | 4.6 | Multi-sample batch gradient descent evaluations. |
| 3 | 5.5 | 6.7 | Multi-sample batch gradient descent evaluations. |

The nominal relationship exhibits high linearity approximately governed by:

$$\text{price} \approx 1.25 \times \text{area} + 0.30$$

The corrupted dataset `house_prices_outlier.csv` includes an anomalous observation at `area = 6.0` with `price = 25.0` (more than triple the expected price), serving as a benchmark to compare the robustness of **Mean Squared Error (MSE / L2)**, **Mean Absolute Error (MAE / L1)**, and **Huber loss**.
