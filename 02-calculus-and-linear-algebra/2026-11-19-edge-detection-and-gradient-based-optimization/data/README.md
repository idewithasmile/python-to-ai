# Session Data: Edge Detection Benchmark Image and Discrete Derivative Fixtures

This directory contains the image fixture used by the study guide and interactive notebooks in [Edge Detection and Gradient-based Optimization](../README.md).

## Files and Schemas

### `nature.jpg`

- **Format:** JPEG image (`uint8`).
- **Dimensions:** `500 × 1200` pixels (Height $H = 500$, Width $W = 1200$, 3 color channels BGR/RGB; ~98.5 KB).
- **Grayscale representation:** When loaded in grayscale mode (`cv2.imread(..., 0)` or converted via $0.299R + 0.587G + 0.114B$), it forms a single 2D matrix $I \in [0, 255]^{500 \times 1200}$ of unsigned 8-bit integers.
- **Statistical characteristics:**
  - Dynamic range: $\min(I) = 0$, $\max(I) = 255$.
  - Spatial composition: Contains distinct horizontal horizons, vertical tree trunks, foliage textures, and sky gradients, providing rich multidirectional gradient features.
- **Target usage:**
  - **1D horizontal differentiation:** Treating each row $i$ as a discrete function $f(j) = I[i, j]$ and computing single-sided forward differences $\Delta_x I = I[i, j+1] - I[i, j]$ and symmetric central differences $\delta_x I = \frac{I[i, j+1] - I[i, j-1]}{2}$.
  - **1D vertical differentiation:** Treating each column $j$ as a discrete function $g(i) = I[i, j]$ and computing vertical differences $\Delta_y I = I[i+1, j] - I[i, j]$ and central differences $\delta_y I = \frac{I[i+1, j] - I[i-1, j]}{2}$.
  - **Signed gradient visualization:** Mapping negative and positive derivative values back to valid 8-bit display ranges through absolute magnitude $|d|$ and midpoint level-shifting $d + 128$.
  - **2D Sobel convolution:** Convolving $I$ with horizontal kernel $K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$ and vertical kernel $K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$ to compute continuous gradient vectors $\nabla I = [G_x, G_y]^T$, gradient magnitudes $G = \sqrt{G_x^2 + G_y^2}$, and gradient orientations $\theta = \arctan2(G_y, G_x)$.
