# Module 02: Mathematics for Machine Learning (Calculus & Linear Algebra)

[Back to the master roadmap](../README.md) · [Environment setup](../docs/setup.md)

**Weeks 5–11 · November 30, 2026–January 17, 2027 · Status: Planned**

## Overview

This module develops the mathematical language used throughout machine learning. You will work with vectors, matrices, norms, derivatives, gradients, and matrix calculus, then implement the same ideas in NumPy. The goal is not abstract proof fluency alone; it is the ability to read a model description, derive its objective, implement it numerically, and diagnose why training behaves the way it does.

The module also connects linear algebra to practical operations such as cosine similarity, retrieval, image interpolation, upsampling, edge detection, and low-rank approximation.

## Learning Outcomes

By the end of this module, you should be able to:

- Perform vector and matrix operations with NumPy and explain their shapes and semantics.
- Compute derivatives and gradients, apply the chain rule, and implement gradient descent from scratch.
- Derive the objective and update rule for linear regression with MSE, MAE, and Huber loss.
- Vectorize regression and compare closed-form and iterative solutions.
- Use eigenvalues, eigenvectors, and singular value decomposition for compression and retrieval.
- Interpret interpolation, upsampling, and convolution-like image operators as linear operations.
- Communicate derivations, numerical checks, and limitations in a reproducible report.

## Weekly Plan

| Week | Dates | Focus | Output |
| :---: | :--- | :--- | :--- |
| 05 | Nov 30–Dec 6, 2026 | NumPy arrays, broadcasting, vectorization, and numerical stability | NumPy warm-up |
| 06 | Dec 7–13, 2026 | Derivatives, gradients, chain rule, and gradient descent | Optimization notebook |
| 07 | Dec 14–20, 2026 | Linear regression, MSE, MAE, Huber loss, and closed-form solutions | Loss-function experiments |
| 08 | Dec 21–27, 2026 | Matrix calculus and vectorized regression | Vectorized implementation |
| 09 | Dec 28, 2026–Jan 3, 2027 | Consolidation, eigenvalues, SVD, and PCA foundations | Linear-algebra drills and concept map |
| 10 | Jan 4–10, 2027 | Cosine similarity, retrieval, interpolation, upsampling, and edge operators | Retrieval and image-operator lab |
| 11 | Jan 11–17, 2027 | Integration of regression, SVD, and image operators | Math-for-ML Toolkit milestone |

## Milestone Project: Math-for-ML Toolkit

Build a small, tested toolkit that demonstrates the mathematical foundations of machine learning in code.

### Requirements

- Implement MSE, MAE, and Huber loss with their gradients.
- Implement batch gradient descent with feature standardization and convergence checks.
- Compare closed-form linear regression with gradient descent on at least two datasets or synthetic regimes.
- Implement a numerical gradient check for at least one function.
- Use SVD for image compression or embedding-based retrieval.
- Demonstrate interpolation, upsampling, or edge detection as a linear operator.
- Produce a report with derivations, plots, numerical checks, and limitations.
- Keep all experiments reproducible with fixed seeds and recorded versions.

### Stretch Goals

- Add ridge and lasso regression with derived gradients.
- Implement PCA from the covariance matrix and compare with scikit-learn.
- Explore condition numbers and numerical instability.
- Visualize singular values and reconstruction error.
- Build a small image-retrieval demo with cosine similarity.

## Workspace

```text
02-calculus-and-linear-algebra/
├── notebooks/
│   ├── 05-numpy-and-vectorization.ipynb
│   ├── 06-gradients-and-optimization.ipynb
│   ├── 07-linear-regression-and-losses.ipynb
│   ├── 08-matrix-calculus.ipynb
│   ├── 09-svd-and-pca.ipynb
│   └── 10-similarity-and-image-operators.ipynb
├── exercises/
└── projects/
    └── math-for-ml-toolkit/
```

## Tools

- NumPy
- Matplotlib
- SymPy
- SciPy
- JupyterLab
- pytest

## Deep Dives

- Interpolation and image resampling.
- Convolution as a linear operator.
- Eigenfaces and low-rank approximation.
- PCA and whitening.
- Automatic differentiation and finite-difference checks.

## Assessment

| Criterion | Weight | What good looks like |
| :--- | :---: | :--- |
| Mathematical correctness | 30% | Derivations are correct and consistent with the implementation. |
| Numerical validation | 20% | Gradient checks, shape checks, and convergence diagnostics are included. |
| Implementation quality | 20% | Vectorized, readable code with tests for core functions. |
| Experimentation | 15% | Closed-form and iterative methods are compared fairly. |
| Communication | 15% | The report explains intuition, results, and limitations clearly. |

## References

- *Mathematics for Machine Learning* by Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong.
- *Introduction to Linear Algebra* by Gilbert Strang.
- *Dive into Deep Learning*: linear algebra and calculus chapters: <https://d2l.ai/>
- *Essence of Linear Algebra* and *Essence of Calculus* video series by 3Blue1Brown.
- NumPy documentation: <https://numpy.org/doc/stable/>
- SymPy documentation: <https://docs.sympy.org/>

## Module Checklist

- [ ] All weekly notebooks run from a clean kernel.
- [ ] At least one derivation is written independently, without copying the lesson.
- [ ] Numerical gradient checks pass for the implemented objectives.
- [ ] The milestone report includes both closed-form and iterative solutions.
- [ ] SVD or PCA is explained in terms of reconstruction or retrieval.
- [ ] The module reflection is written and the master roadmap is updated.
