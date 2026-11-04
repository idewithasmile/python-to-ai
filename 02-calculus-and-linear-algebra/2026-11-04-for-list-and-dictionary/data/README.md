# Session Data: Numerical Convergence Benchmarks, ML Hyperparameters, and Activation Inputs

This directory contains lightweight, portable datasets and benchmark fixtures used by the notebooks and study guide in [For, List, and Dictionary](../README.md). Every file is self-contained so that all examples, algorithms, and exercises run locally without external network dependencies.

## Files and Schemas

### `numerical_series_benchmarks.json`

A structured JSON document containing reference parameters, analytical formulas, and ground truth estimates for classical numerical iteration algorithms.

| Section | Key | Type | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `pi_series` | `method` | `string` | Approximation algorithm name | `"gregory_leibniz"` |
| `pi_series` | `formula` | `string` | Mathematical series representation | `"4 * sum((-1)**(i+1) / (2*i - 1) for i in 1..n)"` |
| `pi_series` | `iterations` | `integer` | Number of summation terms | `1000` |
| `pi_series` | `expected_estimate` | `float` | Computed estimate after $N$ iterations | `3.140592653839794` |
| `pi_series` | `true_pi` | `float` | High-precision true mathematical value | `3.141592653589793` |
| `pi_series` | `absolute_error` | `float` | Absolute deviation $|	ext{estimate} - 	ext{true}|$ | `0.0009999997500000845` |
| `euler_number` | `method` | `string` | Approximation formula type (`limit` or `taylor_series`) | `"taylor_series"` |
| `euler_number` | `formula` | `string` | Mathematical formula | `"sum(1 / factorial(k) for k in 0..n)"` |
| `euler_number` | `estimate` | `float` | Computed approximation | `2.7182818011463845` |
| `newton_square_root` | `number` | `float` | Input radicand $N$ | `9.0` |
| `newton_square_root` | `initial_guess` | `float` | Initial starting approximation $x_0 = N/2$ | `4.5` |
| `newton_square_root` | `expected_root` | `float` | Converged square root $\sqrt{N}$ | `3.0` |
| `newton_square_root` | `iteration_trace` | `list[float]` | Sequence of intermediate iterates $x_1, x_2, \dots$ | `[3.25, 3.009615, 3.000015, 3.0]` |

- **Format:** JSON object (UTF-8 encoding).
- **Target usage:** Validating numerical iteration loops, comparing polynomial vs super-exponential convergence rates, and testing error threshold conditions.

---

### `ml_hyperparameters.json`

A hierarchical machine learning configuration schema used for demonstrating dictionary mutations, deep versus shallow copy semantics, and hyperparameter grid generation.

| Section | Field | Type | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `experiment_id` | - | `string` | Unique experiment identifier | `"exp_vision_mlp_001"` |
| `model` | `architecture` | `string` | Neural network model class | `"MultilayerPerceptron"` |
| `model` | `input_dim` | `integer` | Feature dimension of input tensors | `784` |
| `model` | `hidden_dims` | `list[integer]` | Dimensionality of hidden layers | `[256, 128, 64]` |
| `model` | `activation` | `string` | Hidden activation function | `"relu"` |
| `training` | `learning_rate` | `float` | Optimization step size | `0.001` |
| `training` | `optimizer` | `string` | First-order optimization algorithm | `"Adam"` |
| `training` | `batch_size` | `integer` | Mini-batch sample size | `64` |
| `tuning_grid` | - | `dict[string, list]` | Hyperparameter search space for grid search | `{"learning_rate": [0.01, 0.001]}` |
| `metrics_log` | - | `dict[string, dict]` | Epoch-level validation loss and accuracy records | `{"epoch_1": {"train_loss": 0.842}}` |

- **Format:** JSON object (UTF-8 encoding).
- **Target usage:** Benchmarking dictionary manipulation, defensive copying (`copy.deepcopy`), nested dictionary parsing, and parameter updates.

---

### `activation_inputs.csv`

Tabular test vectors spanning negative, zero, and positive real domains to evaluate element-wise activation function implementations.

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `sample_id` | `string` | Unique record identifier | `"s01"` |
| `input_value` | `float` | Raw real input scalar $x$ | `-5.0` |
| `expected_relu` | `float` | Rectified Linear Unit output $\max(0, x)$ | `0.0` |
| `expected_sigmoid` | `float` | Logistic sigmoid output $\frac{1}{1 + e^{-x}}$ | `0.00669285` |
| `expected_leaky_relu` | `float` | Leaky ReLU output with slope $\alpha = 0.01$ | `-0.05` |

- **Format:** Comma-separated values (UTF-8 encoding).
- **Target usage:** Unit testing list comprehension activation mappings and evaluating numerical floating-point stability across negative saturation regimes.
