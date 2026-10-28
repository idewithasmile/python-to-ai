# Session Data: Combinatorial Optimization, Search Graphs, and Empirical Benchmarks

This directory contains lightweight, portable datasets and test assets used by the notebooks and study guide in [Brute-force Exhaustive](../README.md). Every file is self-contained so that all examples, algorithms, and exercises run locally without external cloud dependencies.

## Files and Schemas

### `tsp_distances_4.json`

A structured JSON document defining a symmetric 4-node Travelling Salesman Problem graph, directly corresponding to the introductory problem studied in the lecture materials.

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `name` | `string` | Benchmark identifier | `"4-City Symmetric TSP Benchmark"` |
| `num_cities` | `integer` | Total number of vertices in the graph | `4` |
| `cities` | `list[string]` | Labels of the graph vertices | `["A", "B", "C", "D"]` |
| `distance_matrix` | `list[list[integer]]` | Symmetric $4 \times 4$ cost matrix where $D_{ij}$ is the weight of edge $(i, j)$ | `[[0, 2, 8, 5], ...]` |
| `optimal_tours` | `list[list[string]]` | Hamiltonian cycles achieving minimum total weight | `[["A", "B", "C", "D", "A"], ...]` |
| `optimal_cost` | `integer` | Minimum path weight of a complete tour | `17` |

- **Format:** JSON object (UTF-8 encoding)
- **Target usage:** Validating permutation generation and verifying brute-force TSP search logic on a small state space ($3! = 6$ tours).

### `tsp_cities_10.json`

A 10-city 2D Euclidean coordinate map and precomputed pairwise distance matrix designed for profiling combinatorial scaling.

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `name` | `string` | Benchmark identifier | `"10-City 2D Euclidean TSP Benchmark"` |
| `num_cities` | `integer` | Total number of vertices | `10` |
| `cities` | `list[string]` | City labels from A through J | `["A", "B", ..., "J"]` |
| `coordinates` | `dict[string, dict]` | Planar $(x, y)$ coordinates for each city | `{"A": {"x": 0.0, "y": 0.0}}` |
| `distance_matrix` | `list[list[float]]` | Symmetric $10 \times 10$ Euclidean distance matrix | `[[0.0, 5.1, ...], ...]` |

- **Format:** JSON object (UTF-8 encoding)
- **Target usage:** Demonstrating the factorial time complexity barrier ($9! = 362{,}880$ candidate tours) and timing brute-force exhaustive search versus partial subsets.

### `knapsack_items.csv`

A tabular item inventory for the 0/1 Knapsack Problem, an NP-complete combinatorial decision problem solvable by exhaustive subset evaluation.

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `item_id` | `integer` | Unique identifier for each item | `1` |
| `name` | `string` | Descriptive item name | `"laptop"` |
| `weight` | `integer` | Resource weight consumed by selecting this item | `3` |
| `value` | `integer` | Utility score or value gained | `10` |

- **Format:** Comma-separated values (UTF-8 encoding)
- **Number of rows:** 8 items (header + 8 records)
- **Target usage:** Generating all $2^8 = 256$ candidate item subsets (power set), verifying capacity constraints ($W \le 8$), and identifying the optimal value subset.

### `feature_selection_data.csv`

A synthetic tabular regression dataset containing 100 samples with 5 explanatory variables and a continuous response variable.

| Column | Type | Role | Description |
| :--- | :--- | :--- | :--- |
| `x1` | `float` | Informative feature | True coefficient $\beta_1 = 3.0$ |
| `x2` | `float` | Informative feature | True coefficient $\beta_2 = -2.0$ |
| `x3` | `float` | Noise feature | Uncorrelated Gaussian noise ($\beta_3 = 0.0$) |
| `x4` | `float` | Informative feature | True coefficient $\beta_4 = 1.5$ |
| `x5` | `float` | Noise feature | Uncorrelated Gaussian noise ($\beta_5 = 0.0$) |
| `target` | `float` | Response variable | $y = 3x_1 - 2x_2 + 1.5x_4 + \epsilon$ where $\epsilon \sim \mathcal{N}(0, 0.1^2)$ |

- **Format:** Comma-separated values (UTF-8 encoding)
- **Dimensions:** 100 observations $\times$ 6 columns
- **Target usage:** Performing Best Subset Selection by exhaustively fitting linear models to all $2^5 - 1 = 31$ non-empty feature combinations and selecting the model that optimizes adjusted $R^2$ / AIC.
