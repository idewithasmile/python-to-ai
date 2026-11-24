# Session Data: Binary Image Classification Fixture (`sample_dataset`)

This directory provides a self-contained, offline image classification dataset (`sample_dataset/`) used by all nine notebooks in [Image Augmentation](../README.md).

---

## Directory Topology

```text
data/
├── README.md                    # Dataset specification and schema documentation
└── sample_dataset/              # Self-contained RGB JPEG benchmark (28 images, ~175 KB)
    ├── train/
    │   ├── dogs/                # 10 training images (dog_00.jpg .. dog_09.jpg)
    │   └── cats/                # 10 training images (cat_00.jpg .. cat_09.jpg)
    └── test/
        ├── dogs/                # 4 evaluation images (dog_00.jpg .. dog_03.jpg)
        └── cats/                # 4 evaluation images (cat_00.jpg .. cat_03.jpg)
```

---

## Schema and Class Encoding

| Field | Specification |
| :--- | :--- |
| **Task** | Binary image classification (`dogs` vs. `cats`) |
| **Directory-to-Label Mapping** | `{"dogs": 0, "cats": 1}` |
| **File Format** | 8-bit RGB JPEG (`.jpg`) |
| **Native Dimensions** | `128 × 128 × 3` pixels (`uint8`, values in `[0, 255]`) |
| **Training Split Size** | 20 images (10 `dogs`, 10 `cats`) |
| **Test Split Size** | 8 images (4 `dogs`, 4 `cats`) |
| **Model Input Convention** | Resized to `224 × 224` (`C × H × W`, normalized `float32`) |

### Visual Structure

The fixture images are procedurally generated with fixed seed `20261124` so that all augmentation experiments are deterministic and require zero external downloads:

- **Class `0` (`dogs`):** Cool background tint (`RGB ≈ (210, 225, 240)`), rounded rectangular face geometry, floppy rectangular ears, dark circular eyes, and snout ellipse with subtle additive pixel noise ($\pm 12$ intensity levels).
- **Class `1` (`cats`):** Warm background tint (`RGB ≈ (240, 225, 210)`), circular face geometry, triangular pointed ears, green eyes, triangular nose, and horizontal whiskers with additive pixel noise.

---

## Using an External Dataset

Any image folder following the standard `train/<class_name>/*.jpg` and `test/<class_name>/*.jpg` hierarchy can be dropped into `data/` and loaded via `DATA_DIR` in the notebooks without modifying model or augmentation code.
