# Session Data: 2D Matrix and Grid Benchmarks

This directory contains lightweight reference fixtures used by the study guide and notebooks in [List for 2D data](../README.md).

## Files and Schemas

### `matrix_benchmarks.json`

A UTF-8 JSON object containing small 2D integer matrices and ground-truth results for verifying pure-Python matrix and sliding-window algorithms.

| Key | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `square_matrix` | `list[list[int]]` | $3 \times 3$ | Standard square matrix for diagonal and transpose checks |
| `expected_main_diagonal_sum` | `int` | scalar | Trace $\sum_i A_{i,i} = 15$ |
| `expected_anti_diagonal_sum` | `int` | scalar | Anti-diagonal sum $\sum_i A_{i, n-1-i} = 15$ |
| `unsorted_rows_matrix` | `list[list[int]]` | $3 \times 3$ | Matrix whose rows are sorted by row sum (`12`, `13`, `13`) |
| `expected_sorted_by_row_sum` | `list[list[int]]` | $3 \times 3$ | Expected stable row-sum-sorted output |
| `matrix_a` | `list[list[int]]` | $2 \times 3$ | Left operand for matrix multiplication |
| `matrix_b` | `list[list[int]]` | $3 \times 2$ | Right operand for matrix multiplication |
| `expected_product_ab` | `list[list[int]]` | $2 \times 2$ | Exact product $A \times B$ |
| `image_patch_5x5` | `list[list[int]]` | $5 \times 5$ | Synthetic grayscale intensity patch for 2D convolution / prefix sums |
| `box_kernel_3x3` | `list[list[int]]` | $3 \times 3$ | Unnormalized $3 \times 3$ averaging filter kernel |
