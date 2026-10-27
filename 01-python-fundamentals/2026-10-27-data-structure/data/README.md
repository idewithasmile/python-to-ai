# Session Data: Compound Structures, Text Processing, and Vision Assets

This directory contains lightweight, portable datasets and test assets used by the notebooks and study guide in [Data Structure](../README.md). Every file is self-contained so that all examples and exercises run locally without external cloud dependencies.

## Files and Schemas

### `data.json`

A structured JSON document containing machine learning training configuration parameters and evaluation metrics.

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `learning_rate` | `float` | Optimization step size for gradient descent | `0.1` |
| `optimizer` | `string` | First-order gradient-based optimization algorithm | `"Adam"` |
| `metric` | `string` | Primary evaluation criterion | `"Accuracy"` |

- **Format:** JSON object (UTF-8 encoding)
- **Target usage:** Parsing and deserializing JSON key-value configurations into Python dictionaries, validating dictionary keys defensively, modifying parameters, and serializing configuration back to disk.

### `data.txt`

A plain-text excerpt summarizing Python's core language design philosophy and characteristics.

- **Content:** An overview of Python's high-level interpreted nature, dynamic typing, automatic memory management (garbage collection), and multi-paradigm programming support (structured, object-oriented, functional).
- **Format:** UTF-8 encoded plain text
- **Word count:** 42 words across 4 sentences
- **Target usage:**
  - String cleaning, lowercasing, and punctuation stripping.
  - Set-based unique vocabulary extraction ($O(1)$ membership deduplication).
  - Dictionary-based term frequency analysis (Bag-of-Words and histogram construction).

### `binary_cat.png`

A small binary (two-tone / thresholded) raster image used for discrete pixel frequency analysis and 2D histogram computation.

| Property | Value |
| :--- | :--- |
| **Dimensions** | 70 × 70 pixels |
| **Color Space** | Grayscale / Binary (intensity values in $\{0, 255\}$) |
| **File Format** | Portable Network Graphics (PNG) |
| **Target usage:** Demonstrating dictionary-based 2D matrix frequency counting where pixel intensity serves as the dictionary key and occurrence count serves as the mapped value, followed by discrete bar chart plotting.

### `cat_nms.png`

A photographic test image containing candidate bounding boxes for object localization and Non-Maximum Suppression (NMS).

| Property | Value |
| :--- | :--- |
| **Dimensions** | 300 × 300 pixels |
| **Color Space** | RGB / BGR (8-bit per channel) |
| **File Format** | Portable Network Graphics (PNG) |
| **Target usage:** Visualizing overlapping object candidate bounding boxes $(x_1, y_1, x_2, y_2, \text{confidence})$, computing pairwise Intersection over Union (IoU) ratios, and applying greedy Non-Maximum Suppression to filter redundant detections.
