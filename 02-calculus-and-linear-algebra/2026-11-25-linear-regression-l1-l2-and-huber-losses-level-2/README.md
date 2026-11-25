# Linear Regression — L1, L2, and Huber Losses (Level 2)

> **Module 02 · Calculus and Linear Algebra** · Session `230616` · Scheduled `2026-11-25`

---

## Overview

This session extends the Level 1 introduction to linear regression by working through three
distinct loss functions—**Mean Squared Error (MSE / L2 loss)**, **Mean Absolute Error (MAE /
L1 loss)**, and the **Huber loss**—from both a mathematical and an implementation perspective.
You will implement gradient-descent training loops from scratch, observe how each loss behaves
in the presence of outliers, scale multi-feature inputs, and apply everything to a realistic
multi-variable dataset (the classic Advertising dataset).

The session is split across five progressively richer notebooks:

| Notebook | Key idea |
|---|---|
| `01_one_sample_mse.ipynb` | SGD with MSE on a single-feature toy dataset |
| `02_one_sample_mae.ipynb` | SGD with MAE; non-differentiable point and sign-gradient |
| `03_n_sample_mse_batch.ipynb` | Full-batch gradient descent with MSE; average gradient |
| `04_n_sample_mae_batch.ipynb` | Full-batch gradient descent with MAE |
| `05_outlier_mse_batch.ipynb` | Effect of outliers on MSE; motivates robust losses |
| `06_advertising_stochastic.ipynb` | Multi-feature SGD with feature scaling on Advertising data |
| `07_advertising_batch.ipynb` | Multi-feature batch GD with feature scaling on Advertising data |

---

## Learning outcomes

After completing this session you will be able to:

1. **Derive** the gradient of MSE, MAE, and Huber loss with respect to weight `w` and bias `b`
   for a single-variable linear model.
2. **Implement** stochastic gradient descent (SGD) and full-batch gradient descent loops
   entirely in plain Python and NumPy.
3. **Explain** why MAE is more robust to outliers than MSE, and when Huber loss provides a
   practical middle ground.
4. **Apply** mean–range feature scaling to multi-feature inputs and correctly rescale test
   points at inference time.
5. **Extend** a single-feature model to multiple features with separate weights per feature.
6. **Read and interpret** loss curves to diagnose convergence, oscillation, and overshoot.

---

## Mathematical Background

### 1. The Linear Model

For a single input feature $x$ the model predicts

$$\hat{y} = w \cdot x + b$$

where $w$ (weight) and $b$ (bias) are the learnable parameters.

For $d$ features the model is

$$\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_d x_d + b = \mathbf{w}^\top \mathbf{x} + b$$

---

### 2. Loss Functions

#### 2.1 Mean Squared Error (MSE / L2 Loss)

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

**Properties:**
- Smooth and continuously differentiable everywhere.
- Heavily penalises large errors (quadratic growth).
- Sensitive to outliers: a single extreme value can dominate the total loss and pull the fitted
  line far from the majority of points.

**Gradients** for the single-feature case:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{2}{N} \sum_{i=1}^{N} x_i (\hat{y}_i - y_i)$$

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{2}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)$$

For stochastic (one sample at a time) updates drop the $1/N$ average and use the per-sample
gradient:

$$\frac{\partial \ell}{\partial w} = 2 x (\hat{y} - y), \qquad \frac{\partial \ell}{\partial b} = 2 (\hat{y} - y)$$

---

#### 2.2 Mean Absolute Error (MAE / L1 Loss)

$$\mathcal{L}_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^{N} |\hat{y}_i - y_i|$$

**Properties:**
- Grows linearly with error magnitude — much more robust to outliers than MSE.
- Not differentiable at $\hat{y}_i = y_i$ (where the residual is exactly zero).
- In practice the derivative is taken as the **sign** of the residual:

$$\frac{\partial |\hat{y} - y|}{\partial w} = x \cdot \text{sign}(\hat{y} - y),
\qquad \frac{\partial |\hat{y} - y|}{\partial b} = \text{sign}(\hat{y} - y)$$

where $\text{sign}(r) = r / |r|$ for $r \neq 0$.

**Batch gradients:**

$$\frac{\partial \mathcal{L}_{\text{MAE}}}{\partial w} = \frac{1}{N} \sum_{i=1}^{N} x_i \cdot \text{sign}(\hat{y}_i - y_i)$$

$$\frac{\partial \mathcal{L}_{\text{MAE}}}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} \text{sign}(\hat{y}_i - y_i)$$

---

#### 2.3 Huber Loss

The Huber loss blends MSE and MAE using a threshold $\delta > 0$:

$$\mathcal{L}_{\delta}(r) = \begin{cases}
  \dfrac{1}{2} r^2 & \text{if } |r| \leq \delta \\[6pt]
  \delta \bigl(|r| - \dfrac{\delta}{2}\bigr) & \text{if } |r| > \delta
\end{cases}$$

where $r = \hat{y} - y$.

**Properties:**
- Behaves like MSE for small residuals (smooth near zero, easier to optimise).
- Behaves like MAE for large residuals (robust to outliers).
- Continuously differentiable everywhere (unlike MAE).
- $\delta$ is a hyperparameter; a common choice is $\delta = 1.0$.

**Gradient:**

$$\frac{\partial \mathcal{L}_\delta}{\partial \hat{y}} = \begin{cases}
  r & \text{if } |r| \leq \delta \\
  \delta \cdot \text{sign}(r) & \text{if } |r| > \delta
\end{cases}$$

---

### 3. Gradient Descent Variants

| Variant | Gradient computed over | Update frequency | Noise level |
|---|---|---|---|
| **Stochastic (SGD)** | 1 sample | Every sample | High — noisy trajectory |
| **Full-batch GD** | All $N$ samples | Once per epoch | Low — stable but slow per step |
| **Mini-batch SGD** | $B$ samples ($B \ll N$) | $N/B$ times per epoch | Medium — practical sweet spot |

The update rule for all variants is

$$w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}, \qquad b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}$$

where $\eta$ is the **learning rate** (step size).

---

### 4. Feature Scaling

When features have different units or magnitudes the loss surface becomes an elongated ellipsoid,
making gradient descent slow and sensitive to learning-rate choice.

**Mean–range scaling** (used in the notebooks):

$$x_{\text{scaled}} = \frac{x - \bar{x}}{x_{\max} - x_{\min}}$$

where $\bar{x}$, $x_{\max}$, and $x_{\min}$ are computed over **training features only** and
saved so that the same transformation can be applied at inference time.

> [!IMPORTANT]
> Save `mean_v`, `max_v`, and `min_v` from training. At inference, apply the same scaling
> to new inputs **before** calling `predict`. Never recompute the statistics on the test point.

---

### 5. Multi-Feature Linear Regression

With three features (TV, Radio, Newspaper budgets) the model is

$$\hat{y} = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$$

The MSE gradient for each weight $w_i$ is

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial w_i} = \frac{2}{N} \sum_{j=1}^{N} x_{ij} (\hat{y}_j - y_j)$$

The bias gradient is unchanged:

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial b} = \frac{2}{N} \sum_{j=1}^{N} (\hat{y}_j - y_j)$$

---

## Datasets

See [`data/README.md`](data/README.md) for full schema documentation.

| File | Description |
|---|---|
| `data/data.csv` | Toy housing dataset — 4 samples, 2 columns (area, price) |
| `data/data_outlier.csv` | Same housing dataset with one outlier row added |
| `data/advertising.csv` | Advertising spend vs. sales — 200 samples, 4 columns |

---

## Notebooks

All notebooks are self-contained and run end-to-end with only `numpy` and `matplotlib`.
Data is loaded from relative paths under `../data/`.

| File | Description |
|---|---|
| [`notebooks/01_one_sample_mse.ipynb`](notebooks/01_one_sample_mse.ipynb) | SGD with MSE, single feature |
| [`notebooks/02_one_sample_mae.ipynb`](notebooks/02_one_sample_mae.ipynb) | SGD with MAE, single feature |
| [`notebooks/03_n_sample_mse_batch.ipynb`](notebooks/03_n_sample_mse_batch.ipynb) | Batch GD with MSE |
| [`notebooks/04_n_sample_mae_batch.ipynb`](notebooks/04_n_sample_mae_batch.ipynb) | Batch GD with MAE |
| [`notebooks/05_outlier_mse_batch.ipynb`](notebooks/05_outlier_mse_batch.ipynb) | Outlier effect on MSE |
| [`notebooks/06_advertising_stochastic.ipynb`](notebooks/06_advertising_stochastic.ipynb) | Multi-feature SGD + scaling |
| [`notebooks/07_advertising_batch.ipynb`](notebooks/07_advertising_batch.ipynb) | Multi-feature batch GD + scaling |

---

## Exercises

1. **Huber from scratch** — implement a `huber_loss(y_hat, y, delta)` and `huber_gradient(y_hat, y, x, delta)` and train the same toy housing model. Plot loss curves side-by-side for MSE, MAE, and Huber.

2. **Outlier robustness** — load `data_outlier.csv` and train with MSE, MAE, and Huber. Compare the final fitted lines visually. Which loss is least affected by the outlier?

3. **Learning-rate sensitivity** — fix the Advertising SGD notebook, vary `lr` in `{0.1, 0.01, 0.001}`, and plot all three loss curves on a single figure. What is the optimal learning rate?

4. **Per-feature scaling** — replace the joint scaling in `06_advertising_stochastic.ipynb` with per-feature min–max normalisation. Does convergence speed change?

5. **Mini-batch GD** — implement a mini-batch gradient descent loop with `batch_size=16` for the Advertising dataset. Compare convergence to the full-batch version.

---

## Key Takeaways

- **MSE** is the natural choice when the data is clean and Gaussian noise is assumed. Its
  quadratic form is easy to differentiate and converges smoothly.
- **MAE** is preferred when the target distribution has heavy tails or outliers. The constant
  gradient magnitude means it never overcorrects.
- **Huber loss** is the pragmatic choice: smooth near zero (like MSE) and linear for large
  residuals (like MAE). Tune $\delta$ on a held-out validation set.
- **Feature scaling** is mandatory for gradient-based optimisation on multi-feature problems.
  Always scale using training-set statistics and store them for inference.
- **Stochastic vs. batch** GD: SGD converges faster in wall-clock time on large datasets but
  with higher variance; batch GD is more stable but slower per epoch.

---

## References

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Ch. 3 — Linear Regression.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Ch. 5 — Machine Learning Basics. <https://www.deeplearningbook.org>
3. Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73–101.
4. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer. Ch. 3 — Linear Regression. <https://www.statlearning.com>
