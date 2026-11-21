# Session Data: Image Gradient and Discrete Differentiation Fixtures

This directory contains the image fixture used by the study guide and interactive notebooks in [Derivative and its Applications Practice Lab](../README.md).

## Files and Schemas

### `cat.jpg`

- **Format:** JPEG image (`uint8`).
- **Dimensions:** `400 × 400` pixels (Height $H = 400$, Width $W = 400$, 3 color channels BGR/RGB; ~40 KB).
- **Grayscale representation:** When loaded in grayscale mode (`cv2.imread(..., cv2.IMREAD_GRAYSCALE)` or converted via luminance weighting $0.299R + 0.587G + 0.114B$), it forms a discrete 2D matrix $I \in [0, 255]^{400 \times 400}$ of unsigned 8-bit integers.
- **Statistical characteristics:**
  - Dynamic range: $\min(I) = 0$, $\max(I) = 255$.
  - Spatial composition: Contains a kitten subject with high-contrast structural edges (fur outlines, whiskers, ear contours, eye boundaries) and a dark background, making it an ideal test fixture for discrete spatial differentiation and edge magnitude extraction.
- **Target usage:**
  - **Pure-Python 2D list differentiation:** Converting grayscale `numpy.ndarray` to nested lists (`list[list[int]]`) and computing horizontal differences `computeXDerivative(image)`, vertical differences `computeYDerivative(image)`, and combined gradient magnitude `computeMagnitude(image)` using explicit nested loops.
  - **Kernel-based convolution:** Convolving image patches with 2D difference kernels ($x$-kernel `[[0, 0, 0], [-1, 0, 1], [0, 0, 0]]` and $y$-kernel `[[0, -1, 0], [0, 0, 0], [0, 1, 0]]`) via `computeGradientUsingConv(image, kernel)`.
  - **Sobel operator filtering:** Convolving with separable $3 \times 3$ Sobel kernels $K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$ and $K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$, verifying numerical results against manual hand calculations and OpenCV's `cv2.Sobel`.
