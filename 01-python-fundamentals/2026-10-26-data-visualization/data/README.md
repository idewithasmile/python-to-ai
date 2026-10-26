# Session Data: Computer Vision EDA and High-Performance Tabular Datasets

This directory contains lightweight, portable datasets designed for the notebooks and study guide of [Data Visualization](../README.md). All files are completely self-contained and run offline without external dependencies or authentication.

## Files and Schemas

### `bounding_boxes.csv`

A tabular dataset of object detection annotations representing labeled objects across images of diverse spatial resolutions.

| Field | Type | Description |
| :--- | :--- | :--- |
| `image_id` | string | Filename identifier of the source image (e.g., `img_0001.jpg`). |
| `image_width` | integer | Total image horizontal pixel resolution (e.g., 1920, 1280, 640). |
| `image_height` | integer | Total image vertical pixel resolution (e.g., 1080, 720, 480). |
| `class_name` | string | Target semantic category: `person`, `car`, `bicycle`, `traffic_light`, `dog`, `bus`. |
| `x_min` | integer | Horizontal pixel coordinate of the top-left bounding box corner. |
| `y_min` | integer | Vertical pixel coordinate of the top-left bounding box corner. |
| `box_width` | integer | Bounding box width in pixels. |
| `box_height` | integer | Bounding box height in pixels. |
| `area` | integer | Total bounding box pixel area ($W \times H$). |
| `aspect_ratio` | float | Bounding box aspect ratio ($W / H$) rounded to 3 decimal places. |
| `scale_category` | string | Scale tier under COCO metrics: `small` ($< 32^2$), `medium` ($[32^2, 96^2]$), `large` ($> 96^2$). |

- **Format:** Comma-Separated Values (CSV) with header.
- **Target usage:** Aspect ratio histograms, spatial density analysis, COCO scale distribution, class imbalance modeling, and anchor box clustering.

---

### `tracking_sequences.csv`

A temporal dataset containing bounding box trajectories from multi-frame visual tracking benchmark sequences.

| Field | Type | Description |
| :--- | :--- | :--- |
| `sequence_id` | string | Unique video sequence name (e.g., `seq_pedestrian_crossing`, `seq_vehicle_turn`). |
| `frame_index` | integer | Monotonically increasing sequential frame index (1 to $N$). |
| `timestamp_ms` | integer | Frame capture timestamp relative to sequence start in milliseconds (~30 FPS). |
| `track_id` | integer | Unique identifier of the tracked target instance. |
| `x` | float | Top-left horizontal coordinate of the target bounding box. |
| `y` | float | Top-left vertical coordinate of the target bounding box. |
| `width` | float | Bounding box width in pixels. |
| `height` | float | Bounding box height in pixels. |
| `area` | float | Bounding box pixel area at current timestamp. |
| `confidence` | float | Detector or tracker prediction confidence score ($[0.0, 1.0]$). |
| `occlusion` | integer | Binary occlusion indicator (0 = visible, 1 = partially/fully occluded). |

- **Format:** Comma-Separated Values (CSV) with header.
- **Target usage:** Visual Object Tracking (VOT) exploratory data analysis, trajectory continuity verification, drift detection, and frame-rate stride downsampling.

---

### `food_products.csv`

A high-performance benchmark dataset detailing nutritional profiles of consumer products, used for Polars and Pandas comparative evaluation.

| Field | Type | Description |
| :--- | :--- | :--- |
| `product_id` | string | Unique alphanumeric product SKU identifier. |
| `product_name` | string | Descriptive commercial product title. |
| `category` | string | High-level food category (e.g., `Beverages`, `Dairy`, `Bakery`). |
| `brand` | string | Brand name of the manufacturer. |
| `energy_kcal` | integer | Caloric energy per 100g/ml in kilocalories. |
| `fat_g` | float | Total lipids/fat in grams per 100g. |
| `carbohydrates_g` | float | Total carbohydrates in grams per 100g. |
| `protein_g` | float | Total protein in grams per 100g. |
| `nutrition_grade` | string | Letter rating indicating nutritional balance (`A`, `B`, `C`, `D`, `E`). |
| `created_date` | string | ISO-8601 registration date (`YYYY-MM-DD`). |

- **Format:** Comma-Separated Values (CSV) with header.
- **Target usage:** Eager and lazy query execution, predicate filtering, multi-column group-by aggregations, datetime parsing, and benchmark comparisons.

---

### `categories_metadata.csv`

A relational lookup table for category metadata, enabling join operations with product tables.

| Field | Type | Description |
| :--- | :--- | :--- |
| `category` | string | Primary key representing the product category. |
| `department` | string | Corporate retail department handling this category. |
| `tax_rate` | float | Applicable value-added tax rate ($[0.0, 1.0]$). |
| `storage_type` | string | Supply chain storage condition requirement (e.g., `Ambient`, `Chilled`, `Dry Storage`). |

- **Format:** Comma-Separated Values (CSV) with header.
- **Target usage:** Relational table joins (`inner`, `left`, `right`), denormalization, and multi-table analytics in Polars.
