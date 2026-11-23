# Linear Regression — L1, L2, and Huber Losses (Level 1)

> **Module 02 · Calculus and Linear Algebra** · Session `230614` · Scheduled `2026-11-23`

[Back to Module 02](../README.md) · [Back to Master Roadmap](../../README.md) · [Proceed to Level 2 (Session 230616)](../2026-11-25-linear-regression-l1-l2-and-huber-losses-level-2/)

---

## Overview

Linear regression is the foundational cornerstone of supervised machine learning and mathematical
optimization. At its core, the algorithm models an affine relationship between input observations
and real-valued scalar targets. While seemingly elementary, a rigorous treatment of univariate linear
regression bridges multiple mathematical domains: multivariate differential calculus, chain-rule
computational graphs, loss function geometry, and numerical optimization.

This Level 1 study guide deconstructs linear regression from first principles without relying on
black-box estimators. We examine:
1. **The Univariate Hypothesis:** Expressing the continuous mapping $\hat{y} = w x + b$ through elementary arithmetic nodes.
2. **Computational Graph Architecture:** Tracking scalar forward signals and analytic backward gradients across forward propagation and backpropagation paths.
3. **Loss Function Spectrum ($L_1$, $L_2$, Huber):** Contrasting Mean Squared Error (MSE), Mean Absolute Error (MAE), and smooth Huber loss under nominal and outlier-corrupted distributions.
4. **Exact Numerical Traces:** Verifying manual calculations against the reference house price dataset step by step.
5. **Optimization Geometry:** Exploring the quadratic loss surface $\mathcal{L}(w, b)$, learning rate stability bounds, and stopping criteria.

---

## Learning outcomes

After completing this session, you will be able to:

1. **Deconstruct** the univariate linear model $\hat{y} = wx + b$ into elementary computational graph nodes for forward evaluation and analytical backpropagation.
2. **Derive** exact partial derivatives for Mean Squared Error ($L_2$), Mean Absolute Error ($L_1$), and Huber loss with respect to weight $w$ and bias $b$.
3. **Execute** manual numerical traces of single-sample Stochastic Gradient Descent (SGD) and verify parameter updates against analytical benchmarks.
4. **Contrast** the mathematical properties, sub-gradient behaviors, and outlier sensitivities of $L_1$, $L_2$, and Huber loss formulations.
5. **Verify** analytical gradient implementations using centered finite-difference numerical approximations.
6. **Analyze** the 2D loss surface geometry, contour topology, and optimization trajectories under underfitting, optimal, and oscillatory learning rates.

---

## 1. Mathematical Formulation

### 1.1 The Univariate Hypothesis

For a single scalar feature $x \in \mathbb{R}$, the linear regression hypothesis $\hat{y}: \mathbb{R} \to \mathbb{R}$ is defined as:

$$\hat{y} = w \cdot x + b$$

where:
- $w \in \mathbb{R}$ is the **weight** (slope parameter), governing the rate of change of the prediction with respect to the input feature.
- $b \in \mathbb{R}$ is the **bias** (intercept parameter), translating the hypothesis line independently of the input magnitude.

### 1.2 Computational Graph Representation

To compute analytical gradients systematically, we represent the hypothesis and loss as a directed acyclic graph (DAG) composed of elementary arithmetic operations:

```
    [ x ] -----\
               (*) Node 1: t = w * x
    [ w ] -----/                     \
                                      (+) Node 2: y_hat = t + b
                           [ b ] -----/                         \
                                                                 (-) Node 3: r = y_hat - y
                                                      [ y ] -----/                         \
                                                                                            (sq) Node 4: L = r^2
```

The forward propagation relations are:
$$\begin{aligned}
t &= w \cdot x && \text{(Node 1: Scalar Multiplication)} \\
\hat{y} &= t + b && \text{(Node 2: Affine Addition)} \\
r &= \hat{y} - y && \text{(Node 3: Prediction Residual)} \\
L &= r^2 = (\hat{y} - y)^2 && \text{(Node 4: Squared Loss Function)}
\end{aligned}$$

---

## 2. Backward Differentiation via the Chain Rule

To minimize the scalar loss $L$ using gradient descent, we compute the sensitivities $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial b}$ by propagating adjoint signals backward through the graph.

### 2.1 Local Node Derivatives

From elementary differential calculus:
$$\frac{\partial L}{\partial r} = 2 r = 2(\hat{y} - y)$$
$$\frac{\partial r}{\partial \hat{y}} = 1, \qquad \frac{\partial r}{\partial y} = -1$$
$$\frac{\partial \hat{y}}{\partial t} = 1, \qquad \frac{\partial \hat{y}}{\partial b} = 1$$
$$\frac{\partial t}{\partial w} = x, \qquad \frac{\partial t}{\partial x} = w$$

### 2.2 Analytical Parameter Gradients

Applying the multivariate chain rule:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial r} \cdot \frac{\partial r}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial t} \cdot \frac{\partial t}{\partial w} = [2(\hat{y} - y)] \cdot [1] \cdot [1] \cdot [x] = 2 x (\hat{y} - y)$$

$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial r} \cdot \frac{\partial r}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial b} = [2(\hat{y} - y)] \cdot [1] \cdot [1] = 2 (\hat{y} - y)$$

Notice the intuitive structure:
- Both gradients share the common residual scale factor $2(\hat{y} - y)$.
- The weight gradient is amplified by the feature magnitude $x$.
- The bias gradient acts uniformly across the residual regardless of $x$.

---

## 3. Step-by-Step Numerical Walkthrough

We step through the exact single-sample SGD iteration detailed in the syllabus:

### Initial Conditions
- **Training sample:** $x = 6.7$ (area in $100\,\text{m}^2$), $y = 9.1$ (price in valuation units)
- **Parameters:** $w^{(0)} = -0.34$, $b^{(0)} = 0.04$
- **Learning rate:** $\eta = 0.01$

### Step 0: Initial Forward Evaluation
1. **Multiplication:** $t^{(0)} = w^{(0)} \cdot x = -0.34 \times 6.7 = -2.278$
2. **Prediction:** $\hat{y}^{(0)} = t^{(0)} + b^{(0)} = -2.278 + 0.04 = -2.238$
3. **Residual:** $r^{(0)} = \hat{y}^{(0)} - y = -2.238 - 9.1 = -11.338$
4. **Squared Loss:** $L^{(0)} = (r^{(0)})^2 = (-11.338)^2 = 128.549284 \approx 128.55$

### Step 0: Backward Evaluation & Parameter Updates
1. **Weight gradient:**
   $$\frac{\partial L}{\partial w} = 2 \times 6.7 \times (-11.338) = -151.9292$$
2. **Bias gradient:**
   $$\frac{\partial L}{\partial b} = 2 \times (-11.338) = -22.6760$$
3. **Weight update:**
   $$w^{(1)} = w^{(0)} - \eta \frac{\partial L}{\partial w} = -0.34 - 0.01 \times (-151.9292) = -0.34 + 1.519292 = 1.179292$$
4. **Bias update:**
   $$b^{(1)} = b^{(0)} - \eta \frac{\partial L}{\partial b} = 0.04 - 0.01 \times (-22.6760) = 0.04 + 0.22676 = 0.26676$$

### Step 1: Forward Evaluation After Single Update
1. **Multiplication:** $t^{(1)} = w^{(1)} \cdot x = 1.179292 \times 6.7 \approx 7.901256$
2. **Prediction:** $\hat{y}^{(1)} = 7.901256 + 0.26676 = 8.168016 \approx 8.168$
3. **Residual:** $r^{(1)} = 8.168016 - 9.1 = -0.931984$
4. **New Loss:** $L^{(1)} = (-0.931984)^2 \approx 0.8686 \approx 0.868$

> **Key takeaway:** In a single gradient descent step with $\eta=0.01$, the loss drops dramatically from $128.55$ to $0.868$ (a $99.3\%$ error reduction).

---

## 4. The Loss Function Spectrum: L1, L2, and Huber Loss

The choice of loss function $\ell(\hat{y}, y)$ reflects underlying probabilistic assumptions about the noise distribution in the data (Gaussian vs. Laplace).

| Property | Mean Squared Error (L2 / MSE) | Mean Absolute Error (L1 / MAE) | Huber Loss (Smooth L1) |
|---|---|---|---|
| **Formula $\ell(r)$** | $r^2$ (or $\frac{1}{2}r^2$) | $|r|$ | $\begin{cases} \frac{1}{2}r^2 & |r| \le \delta \\ \delta(|r| - \frac{1}{2}\delta) & |r| > \delta \end{cases}$ |
| **Derivative $\frac{\partial \ell}{\partial r}$** | $2r$ | $\text{sign}(r)$ for $r \ne 0$ | $\begin{cases} r & |r| \le \delta \\ \delta \cdot \text{sign}(r) & |r| > \delta \end{cases}$ |
| **Weight Gradient $\frac{\partial \ell}{\partial w}$** | $2 x r$ | $x \cdot \text{sign}(r)$ | $x \cdot \frac{\partial \ell}{\partial r}$ |
| **Differentiability** | $\mathcal{C}^\infty$ (smooth everywhere) | Non-differentiable at $r = 0$ | $\mathcal{C}^1$ (continuous 1st derivative) |
| **Outlier Resilience** | Very poor ($r^2$ amplifies large errors) | Strong (linear penalty) | Excellent (hybrid quadratic/linear) |
| **Optimization Behavior** | Fast convergence near optimum | May oscillate around minimum | Smooth convergence near zero residual |
| **Probabilistic Prior** | Normal / Gaussian noise: $p(y|x) \sim \mathcal{N}$ | Laplacian noise: $p(y|x) \sim \text{Laplace}$ | Contaminated normal / heavy-tailed |

---

## 5. Loss Surface Topology & Closed-Form Solutions

For an arbitrary dataset $\{(x_i, y_i)\}_{i=1}^N$, the empirical Mean Squared Error is:

$$\mathcal{L}(w, b) = \frac{1}{N} \sum_{i=1}^N (w x_i + b - y_i)^2$$

### 5.1 Convexity and the Hessian Matrix

The Hessian matrix $\mathbf{H}$ of $\mathcal{L}$ with respect to $\mathbf{\theta} = [w, b]^\top$ is:

$$\mathbf{H} = \begin{bmatrix}
\frac{\partial^2 \mathcal{L}}{\partial w^2} & \frac{\partial^2 \mathcal{L}}{\partial w \partial b} \\
\frac{\partial^2 \mathcal{L}}{\partial b \partial w} & \frac{\partial^2 \mathcal{L}}{\partial b^2}
\end{bmatrix} = \begin{bmatrix}
\frac{2}{N} \sum_{i=1}^N x_i^2 & \frac{2}{N} \sum_{i=1}^N x_i \\
\frac{2}{N} \sum_{i=1}^N x_i & 2
\end{bmatrix}$$

Because $\det(\mathbf{H}) = 4 \left( \frac{1}{N} \sum x_i^2 - \bar{x}^2 \right) = 4 \text{Var}(x) > 0$ (for non-degenerate datasets with at least two distinct feature values), $\mathbf{H}$ is strictly positive definite.
Therefore, the loss surface $\mathcal{L}(w, b)$ is a **strictly convex elliptic paraboloid** possessing a unique global minimum.

### 5.2 Closed-Form Ordinary Least Squares (OLS)

Setting $\nabla \mathcal{L} = \mathbf{0}$ yields the analytical global minimizer:

$$w^* = \frac{\sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^N (x_i - \bar{x})^2} = \frac{\text{Cov}(x, y)}{\text{Var}(x)}$$
$$b^* = \bar{y} - w^* \bar{x}$$

where $\bar{x} = \frac{1}{N} \sum x_i$ and $\bar{y} = \frac{1}{N} \sum y_i$.

### 5.3 Learning Rate Stability

Gradient descent iterations converge if and only if:

$$0 < \eta < \frac{2}{\lambda_{\max}(\mathbf{H})}$$

where $\lambda_{\max}(\mathbf{H})$ is the largest eigenvalue of the Hessian matrix. If $\eta > \frac{2}{\lambda_{\max}}$, parameter updates oscillate with increasing amplitude and diverge toward $\pm \infty$.

---

## 6. Directory Structure & Session Inventory

```
2026-11-23-linear-regression-l1-l2-and-huber-losses-level-1/
├── README.md                                          <- This comprehensive study guide
├── data/
│   ├── README.md                                      <- Schema documentation and metadata
│   ├── house_prices_toy.csv                           <- 8 nominal benchmark samples (x=area, y=price)
│   └── house_prices_outlier.csv                       <- Nominal dataset with injected severe outlier
└── notebooks/
    ├── 01_computational_graph_and_single_sample_sgd.ipynb  <- Graph decomposition & exact trace verification
    ├── 02_loss_functions_l1_l2_and_huber_derivation.ipynb   <- Gradient checks & outlier sensitivity experiments
    └── 03_loss_surface_and_gradient_descent_dynamics.ipynb <- 2D contour mapping, OLS, and trajectory analysis
```

---

## 7. References and Further Reading

1. **Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020).** *Mathematics for Machine Learning*. Cambridge University Press. Chapter 9: Linear Regression.
2. **Boyd, S., & Vandenberghe, L. (2004).** *Convex Optimization*. Cambridge University Press. Chapter 4: Convex Optimization Problems.
3. **Strang, G. (2016).** *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press. Section 4.3: Least Squares Approximations.
4. **Huber, P. J. (1964).** Robust Estimation of a Location Parameter. *The Annals of Mathematical Statistics*, 35(1), 73–101.
