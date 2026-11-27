# Linear Regression Practice Lab: Cost Geometry, SGD vs. Full-Batch GD, Vectorized Features, and Feature Scaling

[Back to Module 02](../README.md) · [Back to the master roadmap](../../README.md)

**Session date:** 2026-11-27 · **Module:** 02 — Mathematics for Machine Learning (Calculus & Linear Algebra)  
**Format:** Textbook-grade study guide, convex cost geometry, computational graph hand-traces, five end-to-end coding problems on `advertising.csv`, feature scaling theory and benchmarks, and three interactive notebooks

---

## Overview

Linear regression turns geometric intuition into a concrete optimization pipeline: choose a linear hypothesis family, measure residual errors with a differentiable loss surface, compute parameter sensitivities via partial derivatives, and iteratively update weights using gradient descent. While Level 1 and Level 2 introduced single-sample updates and alternative losses ($L_1$, $L_2$, Huber), turning those equations into reliable software requires mastering four engineering transitions:

1. **Univariate & Convex Cost Geometry:** Understanding why the squared-error objective $J(\theta_0, \theta_1)$ forms a convex paraboloid with no non-global local minima, how contour maps relate to hypothesis lines, and how the learning rate $\alpha$ governs convergence versus divergence.
2. **Stochastic vs. Full-Batch Training Regimes:** Contrasting noisy per-sample Stochastic Gradient Descent (SGD) against smooth $N$-sample full-batch gradient descent for both Mean Squared Error (MSE) and Mean Absolute Error (MAE) on multi-feature tabular data.
3. **Unified Bias-Augmented Representation:** Folding the intercept $b$ into the weight vector $\mathbf{w} = [b, w_1, w_2, w_3]^\top$ by prepending $x_0 = 1$ to every sample $\mathbf{x}_i = [1, x_{i,1}, x_{i,2}, x_{i,3}]^\top$, simplifying prediction, gradient computation, and parameter updates into uniform list comprehensions or vector operations.
4. **Feature Scaling & Conditioning:** Diagnosing why mismatched feature magnitudes (`TV` up to $296.4$ vs. `Radio` up to $49.6$) stretch loss contours into ill-conditioned ellipses, and applying Maximum Absolute Scaling, Min-Max Scaling, Mean Normalization, and Z-Score Standardization to accelerate and stabilize gradient descent.

This study guide synthesizes the complete exercise curriculum into formal derivations, verified hand calculations, bug-checked reference implementations, and three executable Jupyter notebooks.

---

## Learning outcomes

By the end of this session, you should be able to:

- Formulate the univariate linear hypothesis $h_\theta(x) = \theta_0 + \theta_1 x$ and its squared-error cost surface $J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$, and explain why linear regression admits a unique global minimum.
- Trace a single forward and backward pass through a linear regression computational graph by hand (verifying $x = 6.7, y = 9.1, w = -0.34, b = 0.04, \eta = 0.01 \implies w' = 1.179292, b' = 0.26676$, reducing loss from $128.55$ to $0.868$).
- Implement `get_column` and `prepare_data` to parse CSV datasets into clean column-major and row-major feature structures without external high-level data loaders.
- Build and compare one-sample SGD (`implement_linear_regression`) and $N$-sample full-batch gradient descent (`implement_linear_regression_nsamples`) under both MSE and MAE objectives, explaining why batch updates yield smooth monotonic curves while SGD oscillates across iterations.
- Refactor separate scalar parameter updates into a unified bias-augmented feature vector `[1, x1, x2, x3]` with `predict`, `compute_gradient_w`, and `update_weight`.
- Implement and compare four feature scaling techniques—**Max-Abs Scaling**, **Min-Max Scaling**, **Mean Normalization**, and **Z-Score Standardization**—and preserve training statistics `(min, max)` so unseen test samples are transformed consistently at inference time.

---

## Session Map

| Notebook | Focus | Key Topics & Implementations |
| :--- | :--- | :--- |
| [01 · Cost Geometry & SGD Basics](notebooks/01_univariate_cost_geometry_and_sgd_basics.ipynb) | Univariate hypothesis, loss surfaces, and gradient descent mechanics | 1D/2D cost plots $J(\theta_1)$ and $J(\theta_0, \theta_1)$, contour maps, learning rate convergence vs. overshoot, worked computational graph trace ($x=6.7, y=9.1$), conceptual diagnostic quizzes |
| [02 · Advertising SGD, Batch & Vectorized GD](notebooks/02_advertising_sgd_batch_and_vectorized_gd.ipynb) | Problems 1–4 on `advertising.csv` | `prepare_data` & `get_column`, 1-sample SGD (MSE & MAE), $N$-sample full-batch GD (MSE & MAE), unified `[1, x1, x2, x3]` bias-augmented implementation, sales inference on `(19.2, 35.9, 51.3)` |
| [03 · Feature Scaling & Convergence](notebooks/03_feature_scaling_methods_and_convergence.ipynb) | Problem 5 & feature scaling taxonomy | `min_max_scaling` in `prepare_data`, inference-time rescaling, before/after MSE convergence comparison, Max-Abs (per-column & global), Min-Max, Mean Normalization, Z-Score Standardization, contour conditioning |

---

## 1. Univariate Linear Regression & Cost Function Geometry

### 1.1 Hypothesis and Squared-Error Objective

Given $m$ training pairs $\{(x^{(i)}, y^{(i)})\}_{i=1}^{m}$ (such as house floor area $x$ in $\mathrm{ft}^2$ and sale price $y$ in thousands of dollars), univariate linear regression fits an affine function:

$$h_\theta(x) = \theta_0 + \theta_1 x$$

where $\theta_0$ is the intercept (bias $b$) and $\theta_1$ is the slope (weight $w$). To measure how closely the hypothesis matches the training targets, we minimize the half-mean squared error cost function:

$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} \bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr)^2$$

The factor $\frac{1}{2}$ cancels the exponent $2$ during differentiation, while $\frac{1}{m}$ makes the gradient scale-invariant to the training set size $m$.

### 1.2 Why $J(\theta_0, \theta_1)$ Has No Non-Global Local Minima

Fixing $\theta_0 = 0$ on the identity points $(1, 1), (2, 2), (3, 3)$ yields a 1D parabola:

$$J(\theta_1) = \frac{1}{2 \cdot 3} \bigl[(\theta_1 - 1)^2 + (2\theta_1 - 2)^2 + (3\theta_1 - 3)^2\bigr] = \frac{14}{6} (\theta_1 - 1)^2$$

whose unique minimum is $J(1) = 0$. In general, for any $m$ samples, $J(\theta_0, \theta_1)$ is a sum of squares of affine functions of $(\theta_0, \theta_1)$—a **convex quadratic bowl**. Consequently:

1. Every local minimum of $J(\theta_0, \theta_1)$ is guaranteed to be a **global minimum** (there are no spurious local traps).
2. With a sufficiently small fixed learning rate $\alpha > 0$, gradient descent converges to the global optimum without needing to decay $\alpha$ over time (though decaying $\alpha$ is common in stochastic or non-convex settings).
3. If $\alpha$ is too small, convergence is slow; if $\alpha$ is too large, updates overshoot the valley and $J(\theta)$ increases or oscillates wildly.

### 1.3 Gradient Descent Update Equations

Taking partial derivatives of $J(\theta_0, \theta_1)$ with respect to $\theta_0$ and $\theta_1$:

$$\frac{\partial J(\theta_0, \theta_1)}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^{m} \bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr)$$

$$\frac{\partial J(\theta_0, \theta_1)}{\partial \theta_1} = \frac{1}{m} \sum_{i=1}^{m} \bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr) x^{(i)}$$

Both parameters must be updated **simultaneously** (computing both right-hand sides before overwriting $\theta_0$ or $\theta_1$):

$$\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} \bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr), \qquad \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^{m} \bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr) x^{(i)}$$

---

## 2. Computational Graph & Hand-Traced SGD Update

When training on one sample $(x, y)$ at a time with unaveraged squared loss $L(\hat{y}, y) = (\hat{y} - y)^2$ and hypothesis $\hat{y} = w x + b$:

```text
  x ──┐
      (*) ── t = w*x ──┐
  w ──┘                (+) ── y_hat = w*x + b ──┐
                   b ──┘                        (-) ── r = y_hat - y ── (sq) ── L = r^2
                                            y ──┘
```

Applying the chain rule yields the single-sample gradients:

$$\frac{\partial L}{\partial w} = 2 x (\hat{y} - y), \qquad \frac{\partial L}{\partial b} = 2 (\hat{y} - y)$$

### Worked Trace (Toy House Price Sample)

Given $x = 6.7$, $y = 9.1$, initial parameters $w = -0.34$, $b = 0.04$, and learning rate $\eta = 0.01$:

1. **Forward pass:**
   $$\hat{y} = (-0.34)(6.7) + 0.04 = -2.278 + 0.04 = -2.238$$
   $$L = (-2.238 - 9.1)^2 = (-11.338)^2 = 128.550244 \approx 128.55$$

2. **Backward pass:**
   $$\frac{\partial L}{\partial w} = 2(6.7)(-11.338) = -151.9292, \qquad \frac{\partial L}{\partial b} = 2(-11.338) = -22.676$$

3. **Parameter update ($\eta = 0.01$):**
   $$w_{\text{new}} = -0.34 - 0.01(-151.9292) = 1.179292$$
   $$b_{\text{new}} = 0.04 - 0.01(-22.676) = 0.26676$$

4. **Post-update forward evaluation on the same sample:**
   $$\hat{y}_{\text{new}} = (1.179292)(6.7) + 0.26676 = 8.1680164$$
   $$L_{\text{new}} = (8.1680164 - 9.1)^2 = (-0.9319836)^2 \approx 0.8686$$

One gradient step drops the squared error on this sample from $128.55$ to $0.8686$.

---

## 3. Multivariate Linear Regression & Training Regimes

Extending the model to $d = 3$ features (`TV`, `Radio`, `Newspaper`) predicting `Sales`:

$$\hat{y} = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$$

Using the bias-augmented feature convention $x_0 = 1$, define $\mathbf{x} = [1, x_1, x_2, x_3]^\top$ and $\boldsymbol{\theta} = [b, w_1, w_2, w_3]^\top$:

$$\hat{y} = \boldsymbol{\theta}^\top \mathbf{x} = \sum_{j=0}^{3} \theta_j x_j$$

### 3.1 Loss Functions and Partial Derivatives

| Loss | Sample Loss $\ell(\hat{y}, y)$ | Dataset Objective $\mathcal{L}$ | Gradient w.r.t. $\theta_j$ (per sample) |
| :--- | :--- | :--- | :--- |
| **MSE ($L_2$)** | $(\hat{y} - y)^2$ | $\frac{1}{N}\sum_{i=1}^{N}(\hat{y}^{(i)} - y^{(i)})^2$ | $2 x_j (\hat{y} - y)$ |
| **MAE ($L_1$)** | $|\hat{y} - y|$ | $\frac{1}{N}\sum_{i=1}^{N}|\hat{y}^{(i)} - y^{(i)}|$ | $x_j \, \mathrm{sign}(\hat{y} - y)$ (or $0$ at $\hat{y} = y$) |

Note on the exercise slides: In Problem 3, the slide snippet computing MAE loss in `implement_linear_regression_nsamples` calls `compute_gradient_wi(xi, y, y_hat)` (the MSE gradient $2 x_i (\hat{y}-y)$) and has a variable-name typo (`dl_dw1 / N` instead of `dw1_total / N` in the post-loop update). In Notebook 02, we explicitly fix the accumulator bug (`dw1_total / N`, `dw2_total / N`, `dw3_total / N`, `db_total / N`) and provide the exact MAE subgradient `compute_gradient_mae_wi(xi, y, y_hat) = xi * sign(y_hat - y)` alongside the slide-matching variant so learners can verify both!

### 3.2 Comparing One-Sample SGD, Mini-Batch GD, and Full-Batch GD

| Regime | Batch Size | Updates per Epoch | Loss Curve Behavior | Typical Use Case |
| :--- | :---: | :---: | :--- | :--- |
| **One-Sample SGD** | $m = 1$ | $N$ | High-variance per-iteration spikes; converges toward a noisy ball around optimum | Online learning, very large datasets |
| **Mini-Batch GD** | $1 < m < N$ | $\lceil N / m \rceil$ | Moderate variance; balances hardware vectorization and escape from flat regions | Standard deep learning & large tabular training |
| **Full-Batch GD** | $m = N$ | $1$ | Smooth, deterministic, monotonic decrease (for valid $\eta$) | Small-to-medium convex problems (e.g., `advertising.csv` with $N=200$) |

---

## 4. Coding Exercises Overview (Problems 1–5)

All five problems operate on `data/advertising.csv` ($N = 200$ rows, 3 input channels + `Sales` target):

1. **Problem 1 — Data Loading (`prepare_data` & `get_column`):**
   Extract each column via list comprehension `[row[index] for row in data]` after `np.genfromtxt(..., delimiter=",", skip_header=1).tolist()`, returning column-major `X = [tv_data, radio_data, newspaper_data]` and `y = sales_data`.
2. **Problem 2 — One-Sample Linear Regression (SGD):**
   Using the fixed benchmark initialization:
   ```python
   w1, w2, w3, b = (0.016992259082509283, 0.0070783670518262355, -0.002307860847821344, 0)
   ```
   implement `predict`, `compute_loss_mse`, `compute_loss_mae`, `compute_gradient_wi`, `compute_gradient_b`, `update_weight_wi`, `update_weight_b`, and the per-sample loop over `epoch_max=50` epochs at `lr=1e-5`. Evaluate inference at `(TV=19.2, Radio=35.9, Newspaper=51.3)`.
3. **Problem 3 — Full-Batch $N$-Sample Linear Regression:**
   Accumulate `loss_total`, `dw1_total`, `dw2_total`, `dw3_total`, `db_total` across all $N=200$ samples before applying a single averaged update `w_i = w_i - lr * (dw_i_total / N)` per epoch (`epoch_max=1000`). Plot smooth MSE and MAE epoch curves.
4. **Problem 4 — Bias-Augmented Feature List (`[1, x1, x2, x3]`):**
   Restructure `prepare_data` so `X = [[1, x1, x2, x3] for x1, x2, x3 in zip(tv_data, radio_data, newspaper_data)]` and unify weights into a 4-element list `[b, w1, w2, w3]`. Replace scalar functions with `sum(f * w for ...)` and `[2 * xi * (y_hat - y) for xi in x_features]`. (Fixes the slide typo `y = sales_data[i]` inside the training loop to `y = y_output[i]`.)
5. **Problem 5 — Min-Max Feature Scaling:**
   Implement `min_max_scaling(data1, data2, data3)` returning both scaled columns and the six extreme statistics `(max_1, max_2, max_3, min_1, min_2, min_3)` so inference queries are scaled with identical training bounds before calling `predict`.

---

## 5. Feature Scaling Theory & Four Standard Methods

### 5.1 Scaling vs. Normalization & Contour Conditioning

In the `advertising.csv` dataset, `TV` reaches $296.4$ while `Radio` stays below $49.6$. Without scaling, gradients with respect to $w_1$ (`TV`) are roughly $6\times$ larger in magnitude than gradients with respect to $w_2$ (`Radio`), stretching the MSE cost contours into narrow ellipses:

```text
  Unscaled features (elongated contours)        Scaled features (isotropic contours)
      b ^                                           b ^
        |       /  /   /                                |       .---.
        |     /  / ^ /                                  |     /   ^   \
        |   /  /  /  /   (zig-zag path)                 |    |    |    |  (direct path)
        |  /  / ^  /                                    |     \   |   /
        +───────────────> w                             +───────────────> w
```

- **Feature Scaling** rescales the **range** of a feature without changing the shape of its distribution (e.g., Min-Max scaling, Max-Abs scaling).
- **Normalization / Standardization** shifts and scales values relative to location and spread (e.g., Z-score standardization centers data at zero mean and unit variance).

### 5.2 Summary of the Four Scaling Formulas

Given a feature vector $x$ with minimum $x_{\min}$, maximum $x_{\max}$, mean $\mu = \mathrm{mean}(x)$, and standard deviation $\sigma$:

| Method | Formula | Output Range / Properties | Sensitive to Outliers? |
| :--- | :--- | :--- | :---: |
| **Max-Abs Scaling** | $x' = \frac{x}{\max(\|x\|)}$ | $[-1, 1]$ (preserves sparsity / zero entries) | Yes |
| **Min-Max Scaling** | $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | $[0, 1]$ (bounded interval) | Yes |
| **Mean Normalization** | $x' = \frac{x - \mu}{x_{\max} - x_{\min}}$ | Approximately $[-0.5, 0.5]$, zero mean | Yes |
| **Z-Score Standardization** | $x' = \frac{x - \mu}{\sigma}$ | Unbounded, $\mu' = 0$, $\sigma' = 1$ | Moderate (less than Min-Max) |

### 5.3 Benchmark DataFrame Verification (4×4 Example)

On the 4-row reference table (`Feature 1 = [180000, 360000, 230000, 60000]`, `Feature 2 = [110, 905, 230, 450]`, `Feature 3 = [18.9, 23.4, 14.0, 13.5]`, `Feature 4 = [1400, 1800, 1300, 1500]`):

- **Per-column Max-Abs Scaling** normalizes each column independently so each column's peak equals $1.0$.
- **Global Max-Abs Scaling** divides the entire table by the global maximum ($360000$), which preserves cross-column magnitude ratios but leaves `Feature 2–4` near zero ($\le 0.005$)—demonstrating why scaling **per feature column** is essential before linear regression.
- **Min-Max Scaling** maps every column's minimum to $0.0$ and maximum to $1.0$.
- **Z-Score Standardization** centers every column around $0$ with unit sample standard deviation (`pandas.Series.std()` uses `ddof=1`).

### 5.4 Conceptual Quiz Checkpoints

1. **House Age Mean Normalization:** Houses have ages between $30$ and $50$ years with mean $38$. Since $x_{\max} - x_{\min} = 50 - 30 = 20$, mean normalization gives:
   $$x_i = \frac{\text{age} - 38}{20}$$
2. **Learning Rate Diagnosis:** Running gradient descent for 15 iterations with $\alpha = 0.3$ causes $J(\theta)$ to decrease rapidly and then level off—indicating $\alpha = 0.3$ is an **effective learning rate** that has already reached near-convergence.

---

## 6. Verification Checklist

- [x] **Data loading (`Problem 1`):** `prepare_data("data/advertising.csv")` yields $3 \times 200$ features and $200$ sales targets.
- [x] **Toy computational graph trace:** Hand calculation ($w' = 1.179292$, $b' = 0.26676$, $L_{\text{new}} \approx 0.8686$) verified in Python.
- [x] **One-sample SGD (`Problem 2`):** Both MSE and MAE loops execute cleanly with deterministic initialization and produce expected predictions.
- [x] **Full-batch GD (`Problem 3`):** Accumulator bug (`dw_total / N`) corrected; smooth monotonic convergence verified over $1000$ epochs for both MSE and MAE.
- [x] **Bias-augmented feature list (`Problem 4`):** Unified `[1, x1, x2, x3]` formulation matches explicit `w1, w2, w3, b` SGD trajectories to machine precision.
- [x] **Feature scaling (`Problem 5` & Section 5):** All four scalers verified against the 4×4 benchmark DataFrame, and Min-Max scaling improves conditioning on `advertising.csv`.

---

## References

1. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.
2. Ng, A. (2012). *Machine Learning* (Univariate & Multivariate Linear Regression, Gradient Descent, Feature Scaling). Stanford / Coursera Lecture Notes.
3. Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). *Mathematics for Machine Learning*. Cambridge University Press.
4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
