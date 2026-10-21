# Module 06: Deep Learning for Multi-Modal Data (CNNs, RNNs & LSTMs)

[Back to the master roadmap](../README.md) · [Environment setup](../docs/setup.md)

**Weeks 28–33 · May 10–June 20, 2027 · Status: Planned**

## Overview

This module extends neural networks to structured data: images, sequences, text, and time series. You will build convolutional networks from scratch, study training and generalization, use pretrained vision models, implement recurrent models, and fine-tune transformer encoders.

The final project combines at least two data modalities or model families under one evaluation protocol. The point is not to stack architectures, but to understand which inductive bias fits which data and how to measure the trade-offs.

## Learning Outcomes

By the end of this module, you should be able to:

- Explain convolution arithmetic, pooling, receptive fields, and parameter sharing.
- Train and regularize CNNs with augmentation, normalization, and early stopping.
- Fine-tune modern pretrained vision models with `timm` and PyTorch.
- Implement RNN, LSTM, and GRU models for sequences and time series.
- Explain attention and the transformer encoder at the level of tensors and shapes.
- Build a reproducible multi-modal or multi-family classification pipeline.
- Analyze compute cost, errors, and the effect of transfer learning.

## Weekly Plan

| Week | Dates | Focus | Output |
| :---: | :--- | :--- | :--- |
| 28 | May 10–16, 2027 | Convolution arithmetic, pooling, receptive fields, and CNNs from scratch | CNN image classifier |
| 29 | May 17–23, 2027 | Augmentation, batch normalization, regularization, and generalization | Regularization study |
| 30 | May 24–30, 2027 | Transfer learning with ResNet, EfficientNet, ConvNeXt, and `timm` | Fine-tuning benchmark |
| 31 | May 31–Jun 6, 2027 | RNNs, LSTMs, GRUs, sequence data, and time-series forecasting | Sequence-model lab |
| 32 | Jun 7–13, 2027 | Attention, transformers, and fine-tuning for text classification | Transformer fine-tuning lab |
| 33 | Jun 14–20, 2027 | Integration of image, text, and sequence models | Multi-Modal Classification Capstone milestone |

## Milestone Project: Multi-Modal Classification Capstone

Build a classification system that uses at least two data modalities or two fundamentally different model families. For example, combine product images with descriptions, or compare a CNN on images with a transformer on text under a shared evaluation protocol.

### Requirements

- A documented dataset with license, modality descriptions, and split strategy.
- A simple baseline before any deep model.
- A CNN trained from scratch or fine-tuned from a pretrained backbone.
- A sequence model using an RNN, LSTM, GRU, or transformer encoder.
- A shared metric protocol and leakage-free splits.
- At least one controlled comparison: scratch versus pretrained, or one modality versus fusion.
- Training curves, confusion matrices, and concrete error examples.
- Compute and runtime reporting.
- A reproducible training command and a model card.
- A short discussion of whether the additional modality or model complexity is justified.

### Stretch Goals

- Add late fusion or cross-attention between modalities.
- Add Grad-CAM or attention visualization.
- Compare parameter count, latency, and accuracy.
- Deploy the model behind a small Gradio interface.
- Test robustness to corrupted or missing inputs.
- Explore parameter-efficient fine-tuning for the transformer.

## Workspace

```text
06-deep-learning-multimodal/
├── notebooks/
│   ├── 28-cnns-from-scratch.ipynb
│   ├── 29-cnn-training-and-generalization.ipynb
│   ├── 30-transfer-learning.ipynb
│   ├── 31-rnns-and-lstms.ipynb
│   └── 32-attention-and-transformers.ipynb
├── exercises/
└── projects/
    └── multimodal-classification-capstone/
```

## Tools

- PyTorch 2.x and torchvision
- `timm`
- Hugging Face Transformers
- torchaudio (optional for audio extensions)
- Albumentations
- Matplotlib, scikit-learn
- TensorBoard or Weights & Biases

## Deep Dives

- Attention internals and positional information.
- Sequence-to-sequence models.
- Transfer learning and domain shift.
- Multimodal fusion strategies.
- Model interpretability for images and text.

## Assessment

| Criterion | Weight | What good looks like |
| :--- | :---: | :--- |
| Data and problem | 20% | Modalities, labels, splits, and license are clearly documented. |
| Modeling | 25% | Architectures match the data and are compared fairly. |
| Evaluation | 25% | Shared metrics, error analysis, and compute trade-offs are reported. |
| Reproducibility | 15% | Training is reproducible with recorded seeds and configuration. |
| Communication | 15% | The report explains when the added complexity is worth it. |

## References

- *Dive into Deep Learning*: <https://d2l.ai/>
- *Attention Is All You Need* by Ashish Vaswani et al.
- *Deep Residual Learning for Image Recognition* by Kaiming He et al.
- *Long Short-Term Memory* by Sepp Hochreiter and Jürgen Schmidhuber.
- Hugging Face NLP course: <https://huggingface.co/learn/nlp-course>
- `timm` documentation: <https://huggingface.co/docs/timm>
- PyTorch vision tutorials: <https://pytorch.org/tutorials/>

## Module Checklist

- [ ] All weekly notebooks run from a clean kernel.
- [ ] A CNN is trained from scratch and compared with a pretrained backbone.
- [ ] At least one sequence model is trained and evaluated.
- [ ] The transformer lab explains tensor shapes and attention masks.
- [ ] The capstone reports compute cost and error analysis.
- [ ] The model card documents data, limitations, and intended use.
- [ ] The module reflection is written and the master roadmap is updated.
