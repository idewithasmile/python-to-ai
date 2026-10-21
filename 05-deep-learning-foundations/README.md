# Module 05: Foundations of Deep Learning (NumPy & PyTorch)

[Back to the master roadmap](../README.md) · [Environment setup](../docs/setup.md)

**Weeks 23–27 · April 5–May 9, 2027 · Status: Planned**

## Overview

This module makes the transition from classical models to neural networks. You will implement logistic regression, softmax regression, and a multilayer perceptron first with NumPy and manual gradients, then with PyTorch tensors, autograd, datasets, and training loops.

The goal is to remove the magic from deep learning. By the end, you should be able to explain what a loss function, gradient, optimizer, initialization scheme, and normalization layer are doing, and debug a training run when the loss does not move.

## Learning Outcomes

By the end of this module, you should be able to:

- Derive and implement binary and multiclass logistic regression with manual gradients.
- Use PyTorch tensors, autograd, `Dataset`, and `DataLoader` correctly.
- Build and train an MLP for tabular, text, or image data.
- Compare activation functions, initialization schemes, and normalization strategies.
- Explain SGD, momentum, Adam, and AdamW, including their trade-offs.
- Diagnose vanishing gradients, exploding gradients, overfitting, and unstable learning rates.
- Track experiments with seeds, configuration, metrics, and training curves.

## Weekly Plan

| Week | Dates | Focus | Output |
| :---: | :--- | :--- | :--- |
| 23 | Apr 5–11, 2027 | Logistic regression, cross-entropy, tensors, and automatic differentiation | Autograd exercises |
| 24 | Apr 12–18, 2027 | Softmax regression, multiclass evaluation, datasets, and training loops | PyTorch classification pipeline |
| 25 | Apr 19–25, 2027 | MLPs, activations, initialization, normalization, and regularization | MLP experiment suite |
| 26 | Apr 26–May 2, 2027 | SGD, momentum, AdamW, learning-rate schedules, and debugging | Optimizer benchmark |
| 27 | May 3–9, 2027 | Integration of a NumPy implementation and a PyTorch counterpart | Neural Network Laboratory milestone |

## Milestone Project: Neural Network Laboratory

Implement the same small neural network twice: once in NumPy with manual gradients and once in PyTorch with autograd. Compare the implementations under controlled conditions.

### Requirements

- A NumPy implementation of logistic regression, softmax regression, or a small MLP with manual backpropagation.
- A PyTorch implementation of the same architecture and objective.
- A numerical gradient check for the NumPy implementation.
- A controlled comparison of at least three activation functions or initialization schemes.
- A controlled comparison of at least three optimizers or learning-rate schedules.
- Training and validation curves with fixed seeds.
- A clear analysis of vanishing gradients, overfitting, or optimization instability.
- A reproducible training command and a short technical report.

### Stretch Goals

- Add mini-batch training and gradient accumulation.
- Add mixed precision and compare speed and stability.
- Implement a learning-rate finder.
- Use `torch.compile` and measure the effect.
- Add TensorBoard or Weights & Biases tracking.
- Train on a small text-classification or time-series dataset.

## Workspace

```text
05-deep-learning-foundations/
├── notebooks/
│   ├── 23-logistic-regression-and-autograd.ipynb
│   ├── 24-softmax-and-training-loops.ipynb
│   ├── 25-mlps-and-initialization.ipynb
│   └── 26-optimizers-and-debugging.ipynb
├── exercises/
└── projects/
    └── neural-network-laboratory/
```

## Tools

- NumPy
- PyTorch 2.x
- torchvision
- PyTorch Lightning (optional)
- TensorBoard
- Weights & Biases (optional)
- scikit-learn for metrics
- JupyterLab

## Deep Dives

- Vanishing and exploding gradients.
- Initialization theory and variance propagation.
- Optimization algorithms and schedules.
- Mixed precision and memory profiling.
- Experiment tracking and reproducibility.

## Assessment

| Criterion | Weight | What good looks like |
| :--- | :---: | :--- |
| Mathematical correctness | 25% | Manual gradients are correct and verified numerically. |
| Implementation | 25% | Training loops, data handling, and evaluation are clean and tested. |
| Experimentation | 20% | Comparisons are controlled and training curves are interpreted. |
| Reproducibility | 15% | Seeds, versions, configurations, and commands are recorded. |
| Communication | 15% | The report explains the mechanism behind observed training behavior. |

## References

- *Dive into Deep Learning*: <https://d2l.ai/>
- *Deep Learning* by Ian Goodfellow, Yoshua Bengio, and Aaron Courville.
- PyTorch tutorials: <https://pytorch.org/tutorials/>
- PyTorch documentation: <https://pytorch.org/docs/stable/index.html>
- *Neural Networks: Zero to Hero* video series by Andrej Karpathy.
- scikit-learn metrics documentation: <https://scikit-learn.org/stable/modules/model_evaluation.html>

## Module Checklist

- [ ] All weekly notebooks run from a clean kernel.
- [ ] The NumPy implementation passes a numerical gradient check.
- [ ] The PyTorch implementation matches the NumPy objective on a small example.
- [ ] At least three optimizers or schedules are compared under controlled conditions.
- [ ] Training curves and failure cases are documented.
- [ ] Seeds and package versions are recorded.
- [ ] The module reflection is written and the master roadmap is updated.
