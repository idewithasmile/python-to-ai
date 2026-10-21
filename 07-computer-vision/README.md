# Module 07: Computer Vision & Deep Learning Applications

[Back to the master roadmap](../README.md) · [Environment setup](../docs/setup.md)

**Weeks 34–39 · June 21–August 1, 2027 · Status: Planned**

## Overview

This module applies deep learning to real computer vision tasks. You will work on image restoration, segmentation, object detection, tracking, video understanding, and medical imaging, with an emphasis on evaluation and failure analysis.

The module treats computer vision as a full system problem: dataset construction, augmentation, architecture choice, training, post-processing, latency, and deployment. Modern pretrained models and current detection families are used, and obsolete APIs are replaced rather than preserved.

## Learning Outcomes

By the end of this module, you should be able to:

- Build image restoration pipelines for denoising, super-resolution, and colorization.
- Train and evaluate semantic segmentation models with IoU and Dice metrics.
- Explain anchors, IoU, non-maximum suppression, and modern detection heads.
- Fine-tune a modern detector and analyze false positives, false negatives, and localization errors.
- Implement tracking-by-detection and evaluate identity consistency.
- Adapt a vision pipeline to a medical or other high-stakes domain with appropriate caveats.
- Package inference for reproducible evaluation and a simple demo.

## Weekly Plan

| Week | Dates | Focus | Output |
| :---: | :--- | :--- | :--- |
| 34 | Jun 21–27, 2027 | Denoising, super-resolution, colorization, autoencoders, and U-Net foundations | Image-restoration lab |
| 35 | Jun 28–Jul 4, 2027 | Semantic segmentation, IoU, Dice, and medical-image workflows | Segmentation lab |
| 36 | Jul 5–11, 2027 | Detection foundations: anchors, IoU, NMS, and the YOLO family | Detection foundations lab |
| 37 | Jul 12–18, 2027 | Modern detectors, fine-tuning, evaluation, and inference optimization | Detector fine-tuning experiment |
| 38 | Jul 19–25, 2027 | Object tracking, tracking-by-detection, and video understanding | Tracking and video lab |
| 39 | Jul 26–Aug 1, 2027 | Integration of a complete vision application | Vision Application milestone |

## Milestone Project: Vision Application

Choose one primary task: image restoration, semantic segmentation, object detection, tracking, or medical image analysis. Build a complete application around it, not just a training script.

### Requirements

- A dataset card with license, annotation quality, and split strategy.
- A baseline, such as a classical method or a small pretrained model.
- A modern deep model appropriate to the task.
- Data augmentation and preprocessing documented as part of the pipeline.
- Task-appropriate metrics: PSNR/SSIM, IoU/Dice, mAP, or tracking metrics.
- Qualitative results with both successes and failures.
- Latency and hardware reporting.
- A simple inference demo or command-line entry point.
- A model card covering intended use, limitations, bias, and safety.
- No patient-identifiable or sensitive data committed to the repository.

### Stretch Goals

- Export the model to ONNX and compare latency.
- Add a real-time webcam or video demo.
- Compare two detector families or two segmentation backbones.
- Add test-time augmentation and measure the effect.
- Explore self-supervised pretraining or knowledge distillation.
- Deploy behind a small local web interface.

## Workspace

```text
07-computer-vision/
├── notebooks/
│   ├── 34-image-restoration.ipynb
│   ├── 35-segmentation.ipynb
│   ├── 36-detection-foundations.ipynb
│   ├── 37-modern-detectors.ipynb
│   └── 38-tracking-and-video.ipynb
├── exercises/
└── projects/
    └── vision-application/
```

## Tools

- PyTorch 2.x and torchvision
- OpenCV
- Albumentations
- scikit-image
- `timm`
- Ultralytics YOLO family
- `torchmetrics`
- ONNX and ONNX Runtime
- Gradio or Streamlit

## Deep Dives

- Self-supervised and semi-supervised learning.
- Knowledge distillation.
- Imbalanced vision datasets.
- Medical imaging evaluation and domain shift.
- Efficient inference and model compression.

## Assessment

| Criterion | Weight | What good looks like |
| :--- | :---: | :--- |
| Data and problem | 20% | Task, annotations, license, and splits are appropriate and documented. |
| Modeling | 25% | The model and training strategy fit the task and beat a baseline. |
| Evaluation | 25% | Task metrics, error analysis, and qualitative examples are thorough. |
| System quality | 15% | Inference is reproducible, measured, and usable. |
| Communication and ethics | 15% | The model card addresses limitations, bias, privacy, and intended use. |

## References

- *Computer Vision: Algorithms and Applications* by Richard Szeliski.
- *U-Net: Convolutional Networks for Biomedical Image Segmentation* by Olaf Ronneberger, Philipp Fischer, and Thomas Brox.
- *You Only Look Once* object detection papers by Joseph Redmon et al. and later authors.
- *End-to-End Object Detection with Transformers* by Nicolas Carion et al.
- *Deep Learning for Vision Systems* by Mohamed Elgendy.
- OpenCV documentation: <https://docs.opencv.org/>
- `torchvision` models documentation: <https://pytorch.org/vision/stable/models.html>

## Module Checklist

- [ ] All weekly notebooks run from a clean kernel.
- [ ] The dataset license and annotation limitations are documented.
- [ ] At least one task-appropriate baseline is reported.
- [ ] Evaluation includes qualitative failures, not only aggregate metrics.
- [ ] Latency and hardware are measured for inference.
- [ ] The model card addresses privacy and high-stakes use.
- [ ] The module reflection is written and the master roadmap is updated.
