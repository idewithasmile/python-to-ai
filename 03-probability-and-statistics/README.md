# Module 03: Probability & Mathematical Statistics for Data Science

[Back to the master roadmap](../README.md) · [Environment setup](../docs/setup.md)

**Weeks 12–17 · January 18–February 28, 2027 · Status: Planned**

## Overview

This module builds the probabilistic and statistical reasoning needed to design experiments, quantify uncertainty, and evaluate models honestly. You will move from probability axioms and Bayes' theorem to distributions, estimation, hypothesis testing, resampling, and exploratory data analysis, then connect the theory to a Naive Bayes classifier and a data-driven report.

The emphasis is on assumptions and interpretation. A number without uncertainty, a test without a design, or a correlation without a causal caveat is treated as incomplete work.

## Learning Outcomes

By the end of this module, you should be able to:

- Apply probability rules, conditional probability, and Bayes' theorem to concrete problems.
- Work with common distributions and explain their parameters, assumptions, and failure modes.
- Compute and interpret expectation, variance, covariance, and correlation.
- Estimate parameters, construct confidence intervals, and perform hypothesis tests.
- Use bootstrap, permutation tests, and Monte Carlo simulation to quantify uncertainty.
- Build an exploratory data analysis and a Naive Bayes baseline with a leakage-free split.
- Communicate statistical findings with appropriate caveats about causality and generalization.

## Weekly Plan

| Week | Dates | Focus | Output |
| :---: | :--- | :--- | :--- |
| 12 | Jan 18–24, 2027 | Probability axioms, conditional probability, Bayes' theorem, and distributions | Probability lab |
| 13 | Jan 25–31, 2027 | Random variables, expectation, variance, covariance, and sampling | Distribution and sampling notebook |
| 14 | Feb 1–7, 2027 | Estimation, confidence intervals, hypothesis testing, and A/B testing | Inference report |
| 15 | Feb 8–14, 2027 | Naive Bayes, generative versus discriminative models, and text baselines | Naive Bayes classifier |
| 16 | Feb 15–21, 2027 | Monte Carlo, bootstrap, genetic algorithms, and exploratory data analysis | Simulation and EDA notebook |
| 17 | Feb 22–28, 2027 | Integration of inference, visualization, and baselines | Statistical Analysis Report milestone |

## Milestone Project: Statistical Analysis Report

Choose a public dataset with a clear question and produce a rigorous statistical analysis. The report should be understandable to a technical reader who has not seen the dataset.

### Requirements

- A documented dataset card with source, license, and limitations.
- Exploratory analysis with well-chosen visualizations, not chart clutter.
- Descriptive statistics with appropriate measures of center, spread, and association.
- At least one hypothesis test with assumptions checked, or a justified alternative.
- Confidence intervals or bootstrap uncertainty for the main estimate.
- A Naive Bayes baseline with a leakage-free train, validation, and test split.
- A discussion of causality, sampling bias, and external validity.
- A reproducible notebook and a short written summary.

### Stretch Goals

- Add a power analysis for an A/B test.
- Compare bootstrap and asymptotic confidence intervals.
- Calibrate the Naive Bayes probabilities and discuss the result.
- Implement a genetic algorithm for a small optimization problem.
- Add a permutation test for a custom statistic.

## Workspace

```text
03-probability-and-statistics/
├── notebooks/
│   ├── 12-probability-and-bayes.ipynb
│   ├── 13-distributions-and-sampling.ipynb
│   ├── 14-inference-and-hypothesis-testing.ipynb
│   ├── 15-naive-bayes.ipynb
│   └── 16-simulation-and-eda.ipynb
├── exercises/
└── projects/
    └── statistical-analysis-report/
```

## Tools

- NumPy, pandas, Matplotlib, Seaborn
- SciPy statistics
- statsmodels
- scikit-learn
- JupyterLab

## Deep Dives

- Information theory and entropy.
- Statistical language models.
- Experimental design and multiple comparisons.
- Calibration and proper scoring rules.
- Genetic algorithms as stochastic optimization.

## Assessment

| Criterion | Weight | What good looks like |
| :--- | :---: | :--- |
| Problem and data | 20% | The question is precise and the dataset is appropriate and documented. |
| Statistical reasoning | 30% | Assumptions, uncertainty, and causal limits are handled correctly. |
| Baseline and evaluation | 20% | The Naive Bayes baseline uses a fair split and relevant metrics. |
| Reproducibility | 15% | Analysis can be rerun from a clean environment with fixed seeds. |
| Communication | 15% | Figures and prose support the conclusion without overclaiming. |

## References

- *Introduction to Probability* by Joseph K. Blitzstein and Jessica Hwang.
- *All of Statistics* by Larry Wasserman.
- *Think Stats* by Allen B. Downey.
- *Dive into Deep Learning*: probability and statistics chapters: <https://d2l.ai/>
- SciPy statistics documentation: <https://docs.scipy.org/doc/scipy/reference/stats.html>
- statsmodels documentation: <https://www.statsmodels.org/stable/index.html>

## Module Checklist

- [ ] All weekly notebooks run from a clean kernel.
- [ ] At least one probability problem is solved by hand and checked by simulation.
- [ ] The report states assumptions before interpreting a test.
- [ ] Confidence intervals or bootstrap uncertainty are reported.
- [ ] The Naive Bayes baseline uses a leakage-free split.
- [ ] The module reflection is written and the master roadmap is updated.
