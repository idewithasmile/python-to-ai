# Hyperparameter Optimization: Search Spaces, Classical Search, Bayesian Optimization, and Deep Learning Tuning

[Back to Module 02](../README.md) · [Back to the master roadmap](../../README.md)

**Session date:** 2026-11-12 · **Module:** 02 — Mathematics for Machine Learning (Calculus & Linear Algebra)  
**Format:** Textbook-grade study guide, model-family hyperparameter reference, Bayesian optimization derivations, and five practical notebooks

---

## Overview

Training a machine learning model involves two distinct levels of optimization: an **inner loop** that updates internal model parameters (such as regression weights or neural network kernels) from training data, and an **outer loop** that selects **hyperparameters**—external configuration choices such as regularization strength, kernel bandwidth, tree depth, learning rate, and optimizer momentum—that govern how the inner loop behaves and generalizes.

Because validation loss as a function of hyperparameters has no closed-form gradient and each evaluation requires training a full cross-validation run, choosing hyperparameters systematically is critical to both model accuracy and compute efficiency. This session surveys the full Automated Machine Learning (AutoML) tuning spectrum:

1. **Foundations of Hyperparameter Optimization:** Formalizing the four pillars of an AutoML tuning pipeline (model, search space, cross-validation scheme, and objective metric) and contrasting learnable parameters against external hyperparameters.
2. **Hyperparameter Mechanics Across Classical Model Families:** Analyzing the bias–variance effect and effective search ranges for regularized linear models, Support Vector Machines (SVMs), and Random Forests.
3. **Uninformed Search Strategies:** Exhaustive **Grid Search**, **Random Search** with log-uniform priors, and multi-fidelity **Successive Halving** (`HalvingRandomSearchCV`).
4. **Informed Sequential Search (Bayesian Optimization):** Surrogate modeling with Gaussian Processes (GP) and Tree-structured Parzen Estimators (TPE), balancing exploration vs. exploitation via acquisition functions, and early-stopping callbacks.
5. **Deep Learning Optimizer & Architecture Tuning:** Structuring PyTorch training/validation loops, implementing Adam from scratch, and understanding learning rate, $(\beta_1, \beta_2)$, $\epsilon$, and weight decay.

---

## Learning outcomes

By the end of this session, you should be able to:

- Distinguish internal **model parameters** (learned automatically during `fit` / backpropagation) from external **hyperparameters** (specified prior to training to control capacity, regularization, and optimization).
- Specify the four core components of a hyperparameter optimization pipeline: estimator, bounded search space (linear vs. log-uniform), $K$-fold cross-validation, and scoring objective.
- Select appropriate search ranges and interpret regularization trade-offs for Linear Models (`C`, `alpha`, `l1_ratio`), SVMs (`C`, `kernel`, `degree`, `gamma`, `nu`, `epsilon`), and Random Forests (`n_estimators`, `max_features`, `min_samples_leaf`, `bootstrap`).
- Implement `GridSearchCV`, `RandomizedSearchCV`, and `HalvingRandomSearchCV` in `scikit-learn`, explaining why Random Search outperforms Grid Search when only a subset of hyperparameters strongly affects validation performance.
- Explain how **Successive Halving** allocates a small initial budget (e.g., `n_samples`) to many candidates and iteratively promotes top-performing configurations to larger resource budgets.
- Formulate **Bayesian Optimization** mathematically using a surrogate posterior $p(f \mid \mathcal{D})$ (Gaussian Process or TPE) and an **acquisition function** (such as Expected Improvement) to balance exploration and exploitation.
- Implement a Gaussian Process + Expected Improvement Bayesian search loop on tabular regression data and configure `BayesSearchCV` spaces (`Real`, `Integer`, `Categorical`) with stopping callbacks.
- Build a clean PyTorch CNN training/validation loop and implement the Adam update rule with bias correction, learning rate, $\beta_1, \beta_2$, $\epsilon$, and weight decay.

---

## 1. Parameters vs. Hyperparameters & Tuning Pipeline Architecture

An automated hyperparameter tuning system requires four coupled components:

```mermaid
flowchart LR
    SS["Search Space (Priors & Bounds)"] --> Alg["Search Strategy (Grid / Random / Halving / Bayesian)"]
    Alg -->|"Candidate config theta"| CV["K-Fold Cross-Validation"]
    Model["Estimator / Architecture"] --> CV
    CV --> Metric["Evaluation Metric & Objective"]
    Metric -->|"Validation score"| Alg
```

1. **Model with tunable hyperparameters.**
2. **Search space** defining discrete sets or continuous bounds (uniform or log-uniform) for each hyperparameter.
3. **Cross-validation mechanism** (e.g., 5-fold `KFold`) to estimate generalization performance reliably without leaking the test set.
4. **Evaluation metric and objective function** (e.g., accuracy for balanced classification, negative RMSE for regression).

### Parameters vs. Hyperparameters

| Dimension | Model Parameters | Hyperparameters |
| :--- | :--- | :--- |
| **Role** | Internal weights defining the learned mapping $f_\theta(x)$ | External settings controlling training dynamics and model capacity |
| **Usage phase** | Required directly during inference (`predict`) | Required during model construction and training (`fit`) |
| **Update mechanism** | Updated automatically by optimizer / solver during training | Set before training begins (by practitioner or AutoML search) |
| **Examples** | Linear coefficients $\mathbf{w}$, bias $b$, CNN filter weights | Regularization `C`, learning rate $\eta$, tree `max_depth`, SVM `gamma` |

---

## 2. Key Hyperparameters by Model Family

### 2.1 Regularized Linear Models (Ridge, Lasso, ElasticNet, Logistic Regression)

- **`C` (Logistic Regression / LinearSVC):** Inverse regularization strength ($C = 1/\lambda$). **Smaller `C` imposes stronger regularization**, shrinking weights toward zero.
- **`alpha` (Ridge, Lasso, ElasticNet):** Direct regularization penalty $\alpha > 0$. Larger `alpha` increases shrinkage; in Lasso ($\ell_1$), higher `alpha` can require more coordinate-descent iterations.
- **`l1_ratio` (ElasticNet only):** Mixing weight $\rho \in [0, 1]$ between $\ell_1$ and $\ell_2$ penalties; commonly searched over `[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]`.

### 2.2 Support Vector Machines (SVM)

- **`C`:** Soft-margin misclassification penalty. Smaller `C` widens the margin and ignores more noisy points (higher bias, lower variance); commonly searched over `np.logspace(-3, 3, 7)` (or `loguniform(1e-3, 1e3)`).
- **`kernel`:** Maps inputs into higher-dimensional feature spaces; standard choices are `['linear', 'poly', 'rbf', 'sigmoid']`, with `'rbf'` as the most versatile nonlinear default.
- **`degree`:** Polynomial kernel degree when `kernel='poly'`; effective values typically lie in `[2, 5]`.
- **`gamma`:** Kernel coefficient for `'rbf'`, `'poly'`, and `'sigmoid'`. High `gamma` narrows each point's influence radius (risking overfitting); low `gamma` produces smoother decision boundaries. Typical range: `np.logspace(-3, 3, 7)`.
- **`nu` (`NuSVC`, `NuSVR`):** Upper bound on margin errors / lower bound on support vectors, in $(0, 1]$.
- **`epsilon` (`SVR`):** Width of the $\epsilon$-insensitive tube where residuals incur zero penalty; typically searched over `np.logspace(-4, 2, 7)`.
- **`penalty`, `loss`, `dual` (`LinearSVC`):** Valid combinations include `('l1', 'squared_hinge', False)`, `('l2', 'hinge', True)` (equivalent to `SVC(kernel='linear')`), `('l2', 'squared_hinge', True)`, and `('l2', 'squared_hinge', False)`.

### 2.3 Random Forests

- **`n_estimators`:** Number of trees in the ensemble. More trees reduce ensemble variance monotonically at linear compute cost.
- **`max_features`:** Number/fraction of candidate features evaluated at each split; lower values decorrelate trees and accelerate training.
- **`min_samples_leaf`:** Minimum samples required in a leaf node, indirectly limiting tree depth; higher values reduce variance and increase bias.
- **`bootstrap`:** `True` or `False` indicating whether bootstrap resampling is used when constructing each tree.

---

## 3. Uninformed Search: Grid, Random, and Successive Halving

### 3.1 Grid Search

Grid Search evaluates the full Cartesian product of discrete candidate lists. For $D$ hyperparameters with $|S_d|$ choices each, total fits equal $K \prod_{d=1}^D |S_d|$ under $K$-fold CV. While exhaustive within the grid, it suffers severely from the **curse of dimensionality**: adding an unimportant hyperparameter multiplies runtime without increasing coverage of important dimensions.

### 3.2 Random Search

Random Search samples configurations independently from prior distributions (e.g., `scipy.stats.loguniform` for scale parameters like `C` and `gamma`). Because validation loss often depends strongly on only a few hyperparameters (low *effective dimensionality*), $N$ random draws explore $N$ distinct values along every axis rather than repeating the same grid projections—making Random Search substantially more efficient in high-dimensional spaces.

### 3.3 Successive Halving (`HalvingGridSearchCV` / `HalvingRandomSearchCV`)

Both Grid and Random Search are *uninformed* and waste full compute evaluating clearly weak candidates. **Successive Halving** introduces multi-fidelity resource allocation:
1. Evaluate a large pool of candidates using a small resource budget (e.g., a small subsample `n_samples` or few boosting iterations).
2. Keep only the top fraction ($1 / \text{factor}$, e.g. top $50\%$ or $1/3$) of configurations.
3. Multiply the resource budget for survivors and repeat until a single best configuration remains at `max_resources`.

---

## 4. Informed Search: Bayesian Optimization

When each evaluation of the true objective $f(\mathbf{x})$ is expensive (e.g., 5-fold CV on hundreds of thousands of rows), **Bayesian Optimization** replaces blind sampling with sequential model-based optimization:

1. **Surrogate Model (Proxy Function):** Given observed evaluations $\mathcal{D}_t = \{(\mathbf{x}_i, y_i)\}_{i=1}^t$, fit a probabilistic surrogate predicting both mean $\mu(\mathbf{x})$ and epistemic uncertainty $\sigma(\mathbf{x})$ over untested configurations:
   - **Gaussian Process (GP):** Effective for smooth, continuous, low-to-moderate dimensional spaces.
   - **Random Forest / Extra Trees:** Handles discrete and conditional spaces cleanly.
   - **Tree-structured Parzen Estimators (TPE):** Models $p(\mathbf{x} \mid y < y^*)$ vs. $p(\mathbf{x} \mid y \ge y^*)$ using kernel density estimators, scaling well to complex search spaces.
2. **Acquisition Function (Exploration vs. Exploitation):** Choose the next candidate $\mathbf{x}_{t+1} = \arg\max_{\mathbf{x}} \alpha(\mathbf{x}; \mathcal{D}_t)$. A standard choice is **Expected Improvement (EI)**:

$$\text{EI}(\mathbf{x}) = \bigl(\mu(\mathbf{x}) - f^+ - \xi\bigr)\,\Phi(Z) + \sigma(\mathbf{x})\,\phi(Z), \qquad Z = \frac{\mu(\mathbf{x}) - f^+ - \xi}{\sigma(\mathbf{x})}$$

where $f^+$ is the best score observed so far, $\Phi$ and $\phi$ are the standard normal CDF and PDF, and $\xi \ge 0$ tunes exploration. High $\mu(\mathbf{x})$ drives **exploitation**; high $\sigma(\mathbf{x})$ drives **exploration**.

3. **Log-Uniform Priors & Early Stopping:** Scale parameters spanning orders of magnitude (e.g., `learning_rate` in $[10^{-2}, 1]$, `reg_alpha` / `reg_lambda` in $[10^{-9}, 10^2]$) should always be sampled in log space (`'log-uniform'`), paired with convergence stoppers (`DeltaYStopper`, `DeadlineStopper`).

---

## 5. Deep Learning Hyperparameters & Optimizers

In deep learning, optimization hyperparameters—learning rate $\eta$, momentum coefficients $(\beta_1, \beta_2)$, numerical stability constant $\epsilon$, and weight decay $\lambda$—directly govern convergence speed and generalization. For example, the **Adam** update rule tracks biased first and second moment estimates of the gradient $g_t$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \qquad \theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Clean training loops keep `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()` separated per mini-batch and evaluate validation loss and accuracy inside `torch.no_grad()` under `model.eval()`.

---

## Notebooks

| Notebook | Focus |
| :--- | :--- |
| [`01_grid_search.ipynb`](notebooks/01_grid_search.ipynb) | Exhaustive `GridSearchCV` over SVM `C`, `kernel`, and `gamma` grids |
| [`02_random_search.ipynb`](notebooks/02_random_search.ipynb) | `RandomizedSearchCV` with continuous `loguniform` priors |
| [`03_halving_search.ipynb`](notebooks/03_halving_search.ipynb) | Multi-fidelity `HalvingRandomSearchCV` with `n_samples` resource scaling |
| [`04_bayesian_optimization.ipynb`](notebooks/04_bayesian_optimization.ipynb) | Gaussian Process + Expected Improvement surrogate search & `BayesSearchCV` tabular tuning |
| [`05_deep_learning_optimizers.ipynb`](notebooks/05_deep_learning_optimizers.ipynb) | PyTorch CNN training loop, from-scratch Adam reference, and optimizer hyperparameter comparison |

---

## References

- Bergstra, J., & Bengio, Y. (2012), *Random Search for Hyper-Parameter Optimization*, JMLR 13:281–305.
- Snoek, J., Larochelle, H., & Adams, R. P. (2012), *Practical Bayesian Optimization of Machine Learning Algorithms*, NeurIPS.
- Jamieson, K., & Talwalkar, A. (2016), *Non-stochastic Best Arm Identification and Hyperparameter Optimization* (Successive Halving), AISTATS.
- scikit-learn User Guide, *Tuning the hyper-parameters of an estimator*: <https://scikit-learn.org/stable/modules/grid_search.html>
