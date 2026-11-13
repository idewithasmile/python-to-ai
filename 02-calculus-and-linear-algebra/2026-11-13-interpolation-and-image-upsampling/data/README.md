# Session Data: Sample Image and Time-Series Imputation Benchmarks

This directory contains portable image and 1D sequence fixtures used by the study guide and notebooks in [Interpolation and Image upsampling](../README.md).

## Files and Schemas

### `nature.jpg`

- **Format:** JPEG image (`uint8`), resolution `500 × 1200` pixels (Height = 500, Width = 1200, 3 RGB/BGR channels; ~219 KB).
- **Target usage:** Reading and round-tripping 2D grayscale lists, brightness clipping, synthetic upsampling visualization, and comparing Nearest Neighbor, separable 1D linear, and 2D bilinear image resizing (`1000 × 2400`).

---

### `time_series_missing.json`

UTF-8 JSON object containing 1D time-series test sequences with missing `null` (`None` in Python) gaps, including leading, interior, and trailing missing segments.

| Key | Type | Length | Description |
| :--- | :--- | :---: | :--- |
| `boundary_missing_series` | `list[float \| null]` | 9 | Series `[None, 2.2, 2.0, 2.2, None, 2.7, None, 4.9, None]` testing leading, interior, and trailing gaps |
| `expected_linear_imputed` | `list[float]` | 9 | Exact expected output `[2.2, 2.2, 2.0, 2.2, 2.45, 2.7, 3.8, 4.9, 4.9]` under linear interpolation + boundary clamping |
| `trailing_missing_series` | `list[float \| null]` | 8 | Series `[2.0, 2.2, None, 2.7, 4.9, 6.2, None, None]` with consecutive trailing `None` values |
