# Session Data

This directory contains the small, redistributable datasets used by the notebooks in [Data Structure](../README.md). Every file is deliberately lightweight so that all examples and exercises execute offline and remain easy to inspect.

## Files and Schemas

### `feedback.txt`

A text file containing 25 records of customer feedback categories from an e-commerce survey. Each line contains a single category string.

| Field | Type | Description |
| :--- | :--- | :--- |
| `category` | string | Feedback topic: `product quality`, `delivery time`, `price`, `quantity`, or `customer service`. |

- **Total records:** 25
- **Format:** Plain text, newline-delimited (`\n`)
- **Missing values:** None
- **Target usage:** Categorical aggregation, frequency histograms with `collections.Counter`, and Pandas bar chart visualization.

### `sales_records.txt`

A text file containing 10 numeric entries representing daily transaction totals (in USD) across consecutive trading sessions.

| Field | Type | Description |
| :--- | :--- | :--- |
| `sales_amount` | float | Daily revenue total formatted with two decimal places. |

- **Total records:** 10
- **Format:** Plain text, newline-delimited (`\n`)
- **Missing values:** None
- **Target usage:** Sequential file reading, defensive floating-point conversion, cumulative prefix sum accumulation, and continuous range querying.
