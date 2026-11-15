# Session Data: Image Resizing Benchmark Fixture

This directory contains the local image fixture used by the study guide and notebooks in [Interpolation Practice Lab](../README.md).

## Files and Schemas

### `tree.jpg`

- **Format:** JPEG image (`uint8`), resolution `400 × 400` pixels (Height = 400, Width = 400; ~43 KB).
- **Color space:** Single-subject grayscale frost-covered tree photograph (stored as 3-channel JPEG; loaded in grayscale via `cv2.imread(..., cv2.IMREAD_GRAYSCALE)` to yield shape `(400, 400)` or in BGR/RGB via `cv2.IMREAD_COLOR` to yield shape `(400, 400, 3)`).
- **Target usage:**
  - Converting a `400 × 400` grayscale `numpy.ndarray` to a pure-Python 2D list (`list[list[int]]`) for Nearest-Neighbor upsampling to `1200 × 1200` (`3×` scale).
  - Diagnosing the integer-coordinate zeroing bug in naive bilinear interpolation (`bl_resize_normalcase` vs. `bl_resize_special`) when upsampling to `800 × 800` (`2×` scale).
  - Extending bilinear resizing to 3-channel color images (`bl_resize_color`) and benchmarking against `cv2.resize(..., interpolation=cv2.INTER_LINEAR)`.
