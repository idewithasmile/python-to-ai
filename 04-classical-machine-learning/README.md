# Module 04: Classical Machine Learning & Feature Engineering

[Back to the master roadmap](../README.md) · [Environment setup](../docs/setup.md)

**Weeks 18–22 · March 1–April 4, 2027 · Status: Planned**

## Overview

This module covers the classical machine learning toolkit that remains essential for tabular data, baselines, and model interpretation. You will learn the full workflow: problem framing, leakage-free splits, feature pipelines, model selection, evaluation, error analysis, and documentation.

The module progresses from K-nearest neighbors and decision trees to ensembles, gradient boosting, support vector machines, and imbalanced learning. Every model is compared against a simple baseline, and every result is examined for failure modes rather than reduced to a single leaderboard score.

## Learning Outcomes

By the end of this module, you should be able to:

- Frame a supervised learning problem and choose metrics that match the decision context.
- Prevent data leakage with pipelines, cross-validation, and disciplined feature engineering.
- Explain and implement KNN, decision trees, random forests, boosting, and SVMs.
- Tune hyperparameters without overfitting the validation set.
- Handle imbalanced data with sampling, class weights, and appropriate metrics.
- Interpret model behavior with feature importance, partial dependence, and error analysis.
- Deliver a reproducible tabular machine learning project with a model card.

## Weekly Plan

| Week | Dates | Focus | Output |
| :---: | :--- | :--- | :--- |
| 18 | Mar 1–7, 2027 | ML workflow, leakage, metrics, KNN, pipelines, and cross-validation | Baseline pipeline |
| 19 | Mar 8–14, 2027 | Decision trees for classification and regression, pruning, and feature engineering | Tree-model lab |
| 20 | Mar 15–21, 2027 | Bagging, random forests, boosting, gradient boosting, and hyperparameter search | Ensemble tuning experiment |
| 21 | Mar 22–28, 2027 | SVMs, kernels, imbalanced learning, calibration, and interpretability | SVM and imbalance lab |
| 22 | Mar 29–Apr 4, 2027 | Integration with held-out evaluation and model documentation | Tabular ML Challenge milestone |

## Milestone Project: Tabular ML Challenge

Build an end-to-end tabular machine learning system for a public dataset. Treat the project as if another engineer must reproduce and audit it.

### Requirements

- A dataset card with source, license, target definition, and split strategy.
- A documented baseline, such as a majority class or simple logistic regression.
- A preprocessing pipeline using `Pipeline` and `ColumnTransformer`.
- At least three model families compared with the same evaluation protocol.
- Cross-validation and a final held-out test set that is touched only once.
- Hyperparameter tuning with a recorded search space.
- Metrics appropriate to the problem, including per-class or subgroup results where relevant.
- Error analysis with concrete examples and a confusion matrix.
- A model card covering intended use, limitations, bias, and ethical considerations.
- A reproducible training command and an experiment log.

### Stretch Goals

- Add stacking or a calibrated ensemble.
- Use SHAP or partial dependence for interpretation.
- Compare probability calibration before and after rescaling.
- Add Optuna for Bayesian hyperparameter search.
- Track experiments with MLflow.
- Test robustness to missing values or distribution shift.

## Workspace

```text
04-classical-machine-learning/
├── notebooks/
│   ├── 18-ml-workflow-and-knn.ipynb
│   ├── 19-decision-trees.ipynb
│   ├── 20-ensembles-and-tuning.ipynb
│   └── 21-svm-and-imbalance.ipynb
├── exercises/
└── projects/
    └── tabular-ml-challenge/
```

## Tools

- scikit-learn
- XGBoost, LightGBM, CatBoost
- Optuna
- SHAP
- imbalanced-learn
- pandas, NumPy, Matplotlib, Seaborn
- MLflow

## Deep Dives

- Imbalanced classification and cost-sensitive learning.
- Probability calibration.
- Interpretability and model risk.
- Feature selection and leakage traps.
- Research design for a fair model comparison.

## Assessment

| Criterion | Weight | What good looks like |
| :--- | :---: | :--- |
| Problem and data | 20% | The target, metric, and split match the real decision problem. |
| Modeling | 25% | Models are appropriate, tuned fairly, and compared to a baseline. |
| Evaluation | 25% | No leakage, honest held-out results, and subgroup/error analysis. |
| Reproducibility | 15% | Pipeline, seeds, search space, and commands are documented. |
| Communication | 15% | The model card and report explain limitations and intended use. |

## References

- *Hands-On Machine Learning with Scikit-Learn and PyTorch* by Aurélien Géron.
- *The Elements of Statistical Learning* by Trevor Hastie, Robert Tibshirani, and Jerome Friedman.
- *Pattern Recognition and Machine Learning* by Christopher M. Bishop.
- scikit-learn user guide: <https://scikit-learn.org/stable/user_guide.html>
- XGBoost documentation: <https://xgboost.readthedocs.io/>
- Optuna documentation: <https://optuna.readthedocs.io/>

## Module Checklist

- [ ] All weekly notebooks run from a clean kernel.
- [ ] The preprocessing pipeline is fit only on training folds.
- [ ] At least three model families are compared with the same protocol.
- [ ] The held-out test set is used once, after model selection.
- [ ] Error analysis includes concrete failing examples.
- [ ] The model card documents limitations, bias, and intended use.
- [ ] The module reflection is written and the master roadmap is updated.
