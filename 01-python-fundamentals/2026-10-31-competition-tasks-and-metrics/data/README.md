# Session Data: Competition Tasks, Evaluation Metrics, and Image Pipelines

This directory contains lightweight, portable datasets and test assets used by the notebooks and study guide in [Competition tasks and metrics](../README.md). Every file is self-contained so that all examples, algorithms, and exercises run locally without external cloud dependencies or remote downloads.

## Files and Schemas

### `regression_house_prices.csv`

A continuous response tabular dataset representing residential real estate valuation. Contains actual property sale prices alongside benchmark predictions from an ordinary least squares baseline model and a regularized Ridge regression model.

| Column | Type | Role | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `house_id` | `string` | Identifier | Unique dwelling code | `"H001"` |
| `area_sqft` | `integer` | Feature | Usable interior living area in square feet | `1725` |
| `bedrooms` | `integer` | Feature | Number of dedicated bedrooms | `3` |
| `bathrooms` | `float` | Feature | Number of bathrooms (including half-baths) | `2.8` |
| `location_score` | `float` | Feature | Municipal walkability and transit score $[1.0, 10.0]$ | `7.45` |
| `actual_price_k` | `float` | Ground Truth | Actual transaction price in thousands of USD | `478.32` |
| `pred_linear_k` | `float` | Prediction | Baseline Linear Regression prediction | `485.10` |
| `pred_ridge_k` | `float` | Prediction | Regularized Ridge Regression prediction | `476.90` |

- **Format:** Comma-separated values (UTF-8 encoding).
- **Number of records:** 50 observations.
- **Target usage:** Computing and comparing regression evaluation metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and the Coefficient of Determination ($R^2$).

---

### `classification_eval_data.csv`

A multi-class classification evaluation dataset containing predicted probability distributions and hard class assignments across three animal categories (`cat`, `dog`, `bird`).

| Column | Type | Role | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `sample_id` | `string` | Identifier | Unique observation identifier | `"S001"` |
| `true_label` | `string` | Ground Truth | Human-annotated true category label | `"cat"` |
| `true_class_id` | `integer` | Ground Truth | Zero-indexed integer label ($0=\text{cat}, 1=\text{dog}, 2=\text{bird}$) | `0` |
| `prob_cat` | `float` | Probability | Softmax prediction probability for class 0 | `0.8540` |
| `prob_dog` | `float` | Probability | Softmax prediction probability for class 1 | `0.1120` |
| `prob_bird` | `float` | Probability | Softmax prediction probability for class 2 | `0.0340` |
| `pred_label` | `string` | Prediction | Hard argmax predicted category label | `"cat"` |
| `pred_class_id` | `integer` | Prediction | Hard argmax predicted class index | `0` |

- **Format:** Comma-separated values (UTF-8 encoding).
- **Number of records:** 60 observations.
- **Target usage:** Constructing confusion matrices, computing binary and multi-class metrics (Accuracy, Precision, Recall, Specificity, F1-Score), evaluating macro/micro/weighted averages, and calculating categorical Log Loss (Cross-Entropy).

---

### `detection_bounding_boxes.json`

A structured JSON document containing ground-truth bounding boxes and candidate detector outputs with confidence scores across 5 reference images.

| Field | Type | Description |
| :--- | :--- | :--- |
| `description` | `string` | High-level summary of the dataset scope and coordinate convention. |
| `coordinate_format` | `string` | Bounding box specification: Pascal VOC `[xmin, ymin, xmax, ymax]`. |
| `images` | `list[dict]` | Array of image records containing dimensional metadata and annotations. |
| `images[].image_id` | `string` | Reference identifier corresponding to an image canvas. |
| `images[].image_dimensions` | `list[int]` | Canvas width and height in pixels $[W, H]$. |
| `images[].ground_truth` | `list[dict]` | Verified ground-truth boxes with `category` and `box` coordinates. |
| `images[].predictions` | `list[dict]` | Candidate detector outputs with `category`, `box`, and `confidence` score. |

- **Format:** Standard JSON (UTF-8 encoding).
- **Target usage:** Vectorized Intersection over Union (IoU) calculation, coordinate format transformations, threshold matching (IoU $\ge 0.50$), False Positive / True Positive classification, and Average Precision (AP) calculation.

---

### `sample_manifest.csv`

A file-level manifest describing an image collection organized for supervised training, validation, and testing.

| Column | Type | Role | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `image_id` | `string` | Identifier | Unique image token | `"img_c1"` |
| `filename` | `string` | File Path | Relative path to image within the data directory | `"sample_images/cat_01.png"` |
| `class_name` | `string` | Label | High-level semantic category | `"cat"` |
| `label_id` | `integer` | Target | Numerical encoding for model consumption | `0` |
| `split` | `string` | Partition | Designated partition: `train`, `val`, or `test` | `"train"` |

- **Format:** Comma-separated values (UTF-8 encoding).
- **Number of records:** 6 image entries.
- **Target usage:** Demonstrating reproducible dataset splitting, stratified partitioning, manifest indexing, and batch loading pipelines.

---

### `sample_images/`

A local directory containing synthetic sample images for testing computer vision pipelines without downloading multi-gigabyte remote archives.

- `cat_01.png`, `cat_02.png`, `cat_03.png`: Synthetic images containing distinct warm-palette features and labeled shapes representing class 0.
- `dog_01.png`, `dog_02.png`, `dog_03.png`: Synthetic images containing distinct cool-palette features and labeled shapes representing class 1.
- **Dimensions:** $100 \times 100$ pixels, 3-channel RGB/BGR.
- **Target usage:** OpenCV reading (`cv2.imread`), color-space conversion (`cv2.COLOR_BGR2RGB`), resizing, normalization, and matplotlib grid plotting.
