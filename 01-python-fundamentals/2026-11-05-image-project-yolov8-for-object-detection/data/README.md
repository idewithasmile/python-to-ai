# Dataset Documentation: YOLOv8 Object Detection Benchmark

This directory provides the dataset artifacts, training telemetry logs, model configuration profiles, and evaluation ground truth benchmarks used across the YOLOv8 Object Detection session.

---

## Directory Topologies & File Manifest

```text
data/
├── README.md                          # Data dictionary, schema definitions, and validation checks
├── human_detection_mini/              # Self-contained YOLO-format benchmark dataset
│   ├── data.yaml                      # Ultralytics dataset specification file
│   ├── images/
│   │   ├── train/                     # Training split images (5 samples, 640x640 JPEG)
│   │   └── val/                       # Validation split images (3 samples, 640x640 JPEG)
│   └── labels/
│       ├── train/                     # YOLO-format normalized bounding box annotations (.txt)
│       └── val/                       # YOLO-format normalized bounding box annotations (.txt)
├── detection_benchmark.json           # Ground-truth annotations, prediction sets, and pairwise IoU values
├── model_configs.json                 # Architectural specs across YOLOv8 scales (Nano to XLarge)
└── training_telemetry.csv             # 20-epoch training and validation loss/metric trajectories
```

---

## 1. YOLO Dataset Specification: `human_detection_mini/`

The `human_detection_mini` dataset adheres strictly to the canonical Ultralytics YOLOv8 directory structure and coordinate normalization format.

### 1.1 Dataset Configuration File (`data.yaml`)

```yaml
path: ./human_detection_mini
train: images/train
val: images/val
test: ~

nc: 1
names:
  0: human
```

- **`path`**: Root dataset directory (evaluated relative to execution environment or Ultralytics global settings).
- **`train`**: Relative path to directory containing training images.
- **`val`**: Relative path to directory containing validation images.
- **`test`**: Optional test split (set to null `~` when not in use).
- **`nc`**: Integer number of distinct object classes (for this benchmark, $nc = 1$).
- **`names`**: Dictionary mapping numeric zero-indexed class IDs to human-readable string identifiers (`0: human`).

### 1.2 Coordinate System & Text Label Format

Each image in `images/` corresponds to an identically named `.txt` file in `labels/` (e.g., `images/train/human_train_01.jpg` $\leftrightarrow$ `labels/train/human_train_01.txt`).

Each line within an annotation file represents one detected bounding box encoded as 5 space-separated floating-point values:

$$\text{Format:} \quad \langle \text{class\_id} \rangle \quad \langle x_{\text{center}} \rangle \quad \langle y_{\text{center}} \rangle \quad \langle \text{width} \rangle \quad \langle \text{height} \rangle$$

All coordinates are normalized relative to the image dimensions $W \times H$:

$$x_{\text{center}} = \frac{x_{\min} + x_{\max}}{2 \cdot W}, \quad y_{\text{center}} = \frac{y_{\min} + y_{\max}}{2 \cdot H}, \quad w = \frac{x_{\max} - x_{\min}}{W}, \quad h = \frac{y_{\max} - y_{\min}}{H}$$

All coordinate values satisfy $0 \le x_{\text{center}}, y_{\text{center}}, w, h \le 1$.

#### Sample Annotation (`human_val_01.txt`)
```text
0 0.281250 0.585938 0.187500 0.484375
0 0.648438 0.562500 0.171875 0.500000
```

---

## 2. Evaluation Benchmark: `detection_benchmark.json`

Structured benchmark records pairing ground truth annotations with candidate model predictions across the validation set for IoU calculation, confidence threshold tuning, and Precision-Recall/mAP evaluation.

### Schema Definition
```json
{
  "evaluation_split": "string (e.g., 'val')",
  "iou_thresholds": "list of floats (e.g., [0.50, 0.55, ..., 0.95])",
  "samples": [
    {
      "image_name": "string (filename in images/val/)",
      "dimensions": {"width": "int", "height": "int", "channels": "int"},
      "ground_truths": [
        {
          "id": "int (unique object identifier)",
          "class_id": "int (0-indexed)",
          "class_name": "string",
          "bbox_voc": "[xmin, ymin, xmax, ymax] (absolute pixel coordinates)",
          "bbox_yolo": "[xc, yc, w, h] (normalized coordinates in [0, 1])"
        }
      ],
      "predictions": [
        {
          "id": "int (unique prediction identifier)",
          "class_id": "int (0-indexed)",
          "class_name": "string",
          "confidence": "float in range [0.0, 1.0]",
          "bbox_voc": "[xmin, ymin, xmax, ymax] (absolute pixel coordinates)",
          "iou_with_gt": "float (pairwise overlap with ground truth)"
        }
      ]
    }
  ]
}
```

---

## 3. Training Telemetry: `training_telemetry.csv`

Comprehensive per-epoch metrics captured across a 20-epoch training run of YOLOv8s on the benchmark dataset.

### Column Dictionary
| Column Name | Data Type | Physical Meaning & Optimization Target |
| :--- | :--- | :--- |
| `epoch` | Integer | Zero/one-indexed training iteration ($1 \dots 20$) |
| `train/box_loss` | Float | Complete IoU (CIoU) bounding box regression loss on training batch |
| `train/cls_loss` | Float | Binary Cross-Entropy (BCE) classification loss on training batch |
| `train/dfl_loss` | Float | Distribution Focal Loss (DFL) regression loss on fine boundary bins |
| `metrics/precision(B)` | Float | Global bounding box precision at confidence threshold $\tau = 0.50$ |
| `metrics/recall(B)` | Float | Global bounding box recall at confidence threshold $\tau = 0.50$ |
| `metrics/mAP50(B)` | Float | Mean Average Precision computed at IoU threshold $\alpha = 0.50$ |
| `metrics/mAP50-95(B)`| Float | Mean Average Precision averaged over $\alpha \in [0.50:0.05:0.95]$ |
| `val/box_loss` | Float | CIoU bounding box regression loss evaluated on validation set |
| `val/cls_loss` | Float | BCE classification loss evaluated on validation set |
| `val/dfl_loss` | Float | DFL regression loss evaluated on validation set |

---

## 4. Model Architectures & Scaling Profiles: `model_configs.json`

Captures structural specifications, parameter complexities, and latency benchmarks across the five primary YOLOv8 model scales:

| Model Scale | Identifier | Parameters ($M$) | Compute ($\text{GFLOPs}$) | COCO $\text{mAP}_{50-95}$ | CPU Latency ($\text{ms}$) | A100 GPU Latency ($\text{ms}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Nano | `YOLOv8n` | 3.2 | 8.7 | 37.3 | 80.4 | 0.99 |
| Small | `YOLOv8s` | 11.2 | 28.6 | 44.9 | 128.4 | 1.20 |
| Medium | `YOLOv8m` | 25.9 | 78.9 | 50.2 | 234.7 | 1.83 |
| Large | `YOLOv8l` | 43.7 | 165.2 | 52.9 | 375.2 | 2.39 |
| XLarge | `YOLOv8x` | 68.2 | 257.8 | 53.9 | 479.1 | 3.53 |

All latency benchmarks reflect float32 inference on standard $640 \times 640 \times 3$ resolution tensors.
