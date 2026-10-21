# Module 09: Generative AI, GANs & Diffusion Models

[Back to the master roadmap](../README.md) · [Environment setup](../docs/setup.md)

**Weeks 45–50 · September 6–October 17, 2027 · Status: Planned**

## Overview

This module introduces generative modeling for images and language. You will study autoencoders and variational autoencoders, generative adversarial networks, diffusion models, latent diffusion, controllable image generation, and parameter-efficient LLM fine-tuning.

Generative systems are evaluated differently from classifiers. You will use both quantitative metrics and structured human review, and you will document consent, licensing, bias, and misuse risks. The milestone can focus on image generation or language-model adaptation, provided the evaluation and documentation are rigorous.

## Learning Outcomes

By the end of this module, you should be able to:

- Explain the objectives of VAEs, GANs, and diffusion models and compare their trade-offs.
- Train or fine-tune a generative image model on a small, licensed dataset.
- Implement and interpret a diffusion sampling process and its key hyperparameters.
- Use latent diffusion and conditioning mechanisms for controllable generation.
- Apply parameter-efficient fine-tuning to a language model.
- Evaluate generative outputs with appropriate automatic metrics and human review.
- Build a demo that clearly states provenance, limitations, and responsible-use constraints.

## Weekly Plan

| Week | Dates | Focus | Output |
| :---: | :--- | :--- | :--- |
| 45 | Sep 6–12, 2027 | Generative modeling foundations, VAEs, and neural style transfer | VAE and style-transfer lab |
| 46 | Sep 13–19, 2027 | GANs, DCGAN, training stability, Pix2Pix, and CycleGAN | Conditional GAN lab |
| 47 | Sep 20–26, 2027 | DDPM, denoising score matching, samplers, and image inpainting | Diffusion-model lab |
| 48 | Sep 27–Oct 3, 2027 | Latent diffusion, Stable Diffusion, ControlNet, and generative evaluation | Controllable generation lab |
| 49 | Oct 4–10, 2027 | LoRA, QLoRA, instruction tuning, DPO, XAI, and video diffusion | LLM adaptation lab |
| 50 | Oct 11–17, 2027 | Integration of a generative AI application and demo | Generative AI Studio milestone |

## Milestone Project: Generative AI Studio

Choose one track:

- **Image track:** train or fine-tune a diffusion model for generation, inpainting, colorization, or controllable editing.
- **Language track:** fine-tune a small language model with LoRA or QLoRA for a well-scoped task.

### Requirements

- A dataset card with license, consent, provenance, and preprocessing.
- A clear task definition and intended use.
- A baseline, such as a pretrained model without adaptation or a simpler generative method.
- A reproducible training or fine-tuning configuration.
- Quantitative evaluation appropriate to the track, such as FID, CLIPScore, perplexity, or task-specific accuracy.
- Structured qualitative review with a rubric and multiple examples.
- A discussion of bias, representation, copyright, consent, and misuse.
- A small demo with visible provenance and usage limitations.
- A model card and a short technical report.
- No generated media that misrepresents real people or events.

### Stretch Goals

- Add ControlNet or another conditioning mechanism.
- Compare schedulers or samplers at a fixed compute budget.
- Add DPO after supervised fine-tuning.
- Explore video diffusion with a small model.
- Apply explainability methods to generated outputs.
- Measure inference latency and memory across quantization settings.

## Workspace

```text
09-generative-ai/
├── notebooks/
│   ├── 45-vaes-and-style-transfer.ipynb
│   ├── 46-gans-and-image-translation.ipynb
│   ├── 47-diffusion-models.ipynb
│   ├── 48-latent-diffusion-and-control.ipynb
│   └── 49-llm-adaptation.ipynb
├── exercises/
└── projects/
    └── generative-ai-studio/
```

## Tools

- PyTorch 2.x
- Hugging Face Diffusers
- Hugging Face Transformers
- `peft`, `trl`, `accelerate`
- `safetensors`
- `bitsandbytes` (optional, CUDA)
- Gradio
- Matplotlib, NumPy, scikit-image

## Deep Dives

- Neural style transfer.
- Video diffusion and temporal consistency.
- Explainable AI for generative models.
- Preference optimization and alignment.
- Evaluation of generated media.

## Assessment

| Criterion | Weight | What good looks like |
| :--- | :---: | :--- |
| Problem and data | 20% | The task, data provenance, license, and consent are explicit. |
| Generative modeling | 25% | The method is appropriate and compared to a meaningful baseline. |
| Evaluation | 25% | Automatic metrics and structured human review are both included. |
| Reproducibility | 15% | Training, sampling, and evaluation are reproducible. |
| Ethics and communication | 15% | The report addresses bias, copyright, misuse, and limitations. |

## References

- *Generative Adversarial Nets* by Ian Goodfellow et al.
- *Auto-Encoding Variational Bayes* by Diederik P. Kingma and Max Welling.
- *Denoising Diffusion Probabilistic Models* by Jonathan Ho, Ajay Jain, and Pieter Abbeel.
- *High-Resolution Image Synthesis with Latent Diffusion Models* by Robin Rombach et al.
- *Deep Generative Modeling* by Jakub M. Tomczak.
- Hugging Face diffusion course: <https://huggingface.co/learn/diffusion-course>
- Diffusers documentation: <https://huggingface.co/docs/diffusers>

## Module Checklist

- [ ] All weekly notebooks run from a clean kernel.
- [ ] The dataset license and provenance are documented.
- [ ] At least one baseline is reported.
- [ ] Evaluation combines automatic metrics with structured human review.
- [ ] The demo states provenance and usage limitations.
- [ ] The model card addresses bias, consent, copyright, and misuse.
- [ ] The module reflection is written and the master roadmap is updated.
