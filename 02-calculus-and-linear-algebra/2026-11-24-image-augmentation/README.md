# Image Augmentation: Geometric, Photometric, Occlusion, Mixing, and Consistency Methods

> **Module 02 · Calculus and Linear Algebra** · Session `230615` · Scheduled `2026-11-24`

[Back to Module 02](../README.md) · [Back to Master Roadmap](../../README.md)

---

## Overview

Deep neural networks trained on finite visual datasets routinely memorize training-set artifacts—background textures, illumination biases, or exact object positions—rather than learning invariant semantic representations. **Image augmentation** expands the effective support of the empirical data distribution $p_{\text{data}}(x, y)$ by applying label-preserving (or smoothly label-interpolating) transformations $\mathcal{T}$ during training and inference.

Mathematically, data augmentation approximates **vicinal risk minimization (VRM)** rather than pure **empirical risk minimization (ERM)**. Instead of evaluating loss solely at observed Dirac point masses $\delta(x - x_i, y - y_i)$, we integrate over a neighborhood distribution $v(\tilde{x}, \tilde{y} \mid x_i, y_i)$ induced by geometric warps, photometric perturbations, structured occlusions, and convex sample mixing.

This study guide provides a rigorous, textbook-grade treatment of the complete augmentation taxonomy:

1. **Classical Geometric Transformations:** Flipping, scaling, translation, and cropping formulated via homogeneous affine coordinate matrices.
2. **Photometric & Intensity Transformations:** Brightness shifts, linear/gamma/sigmoid contrast curves, global Histogram Equalization, and CLAHE.
3. **Spatial Filtering & Blurring Operators:** Linear and edge-preserving convolutions (Average, Gaussian, Median, Bilateral, Motion, and Mean-Shift blur).
4. **Occlusion & Regional Deletion:** Cutout, Random Erasing, Hide-and-Seek, and GridMask for robustness against partial occlusion.
5. **Region-Mixing & Label-Smoothing:** MixUp, CutMix, Attentive CutMix, RICAP, and SaliencyMix.
6. **Inference & Semi-Supervised Consistency:** Test-Time Augmentation (TTA) and $\Pi$-Model consistency regularization.

---

## Learning outcomes

After completing this session, you will be able to:

1. **Formulate** flipping, scaling, and translation as $3 \times 3$ homogeneous affine transformation matrices and compute bilinear resampling coordinates.
2. **Derive** intensity remapping equations for linear contrast, gamma correction, sigmoid contrast, histogram equalization (CDF mapping), and CLAHE.
3. **Compare** linear smoothing filters (Gaussian, Average, Motion) against nonlinear edge-preserving filters (Median, Bilateral, Mean-Shift) in terms of kernel weights and noise suppression.
4. **Implement** structured occlusion strategies (Cutout, Random Erasing, Hide-and-Seek, GridMask) and analyze the deletion-to-retention ratio $(r, d)$.
5. **Derive** mixed-label targets and Beta-distributed mixing coefficients $\lambda \sim \text{Beta}(\alpha, \alpha)$ for MixUp, CutMix, RICAP, Attentive CutMix, and SaliencyMix.
6. **Apply** Test-Time Augmentation (TTA) probability averaging and $\Pi$-Model consistency loss ($\mathcal{L}_{\text{CE}} + \alpha \mathcal{L}_{\text{MSE}}$) in clean, portable PyTorch pipelines.

---

## 1. Classical Geometric Transformations

Geometric augmentations transform the spatial coordinates $(x, y)$ of an image $I \in \mathbb{R}^{H \times W \times C}$ without altering pixel intensities (aside from resampling interpolation). Using homogeneous coordinates $\mathbf{p} = [x, y, 1]^T$, every 2D affine transformation takes the unified matrix form:

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \mathbf{M} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} & t_x \\ a_{21} & a_{22} & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

In practice, frameworks compute the **inverse mapping** $\mathbf{p} = \mathbf{M}^{-1}\mathbf{p}'$ for every integer destination pixel $(x', y')$ and evaluate $I(\mathbf{p})$ via nearest-neighbor or bilinear interpolation to prevent holes in the output grid.

### 1.1 Flipping (Reflection)

Flipping reflects an image across a central spatial axis, teaching models orientation invariance where reflection preserves semantic class (e.g., natural objects, animals—though not directional text or traffic arrows).

- **Horizontal Flip** (reflection across the vertical centerline $x = (W - 1)/2$):
  $$x' = (W - 1) - x, \qquad y' = y, \qquad \mathbf{M}_{\text{hflip}} = \begin{bmatrix} -1 & 0 & W - 1 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$
- **Vertical Flip** (reflection across the horizontal centerline $y = (H - 1)/2$):
  $$x' = x, \qquad y' = (H - 1) - y, \qquad \mathbf{M}_{\text{vflip}} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & H - 1 \\ 0 & 0 & 1 \end{bmatrix}$$

### 1.2 Scaling (Resizing)

Scaling stretches or shrinks the image by independent or isotropic scale factors $(s_x, s_y)$:

$$\mathbf{M}_{\text{scale}} = \begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{bmatrix}, \qquad x' = s_x x, \quad y' = s_y y$$

When $s_x = s_y > 1$, the image is zoomed in (upsampled); when $s_x = s_y < 1$, it is zoomed out (downsampled) and padded or cropped back to the network's fixed input resolution.

### 1.3 Translation (Shifting)

Translation shifts all pixels horizontally by $t_x$ and vertically by $t_y$:

$$\mathbf{M}_{\text{trans}} = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}, \qquad x' = x + t_x, \quad y' = y + t_y$$

Boundary pixels vacated by the shift are filled using a border policy such as constant zero-padding (`BORDER_CONSTANT`), edge replication (`BORDER_REPLICATE`), or mirror reflection (`BORDER_REFLECT_101`).

### 1.4 Cropping

Cropping extracts a rectangular sub-region $[y_0 : y_0 + h_c, \, x_0 : x_0 + w_c]$ and optionally resizes it to the target model input shape $(H_{\text{in}}, W_{\text{in}})$:

- **Center Crop:** Deterministic evaluation crop centered at $(W/2, H/2)$.
- **Random Resized Crop:** Samples a crop area fraction $s \sim \mathcal{U}(s_{\min}, s_{\max})$ and aspect ratio $r \sim \mathcal{U}(r_{\min}, r_{\max})$, forcing scale and position invariance while preserving object context.

---

## 2. Photometric & Intensity Transformations

Photometric transformations apply point-wise or neighborhood intensity mappings $I'(x, y) = f(I(x, y))$ on normalized intensities $u = I(x, y) / 255 \in [0, 1]$ (or raw `[0, 255]` values) to simulate varying lighting, exposure, and sensor response.

### 2.1 Brightness Adjustment

Brightness adjustment adds a constant offset $\beta$ across channels, clipped to the valid range:

$$I'(x, y) = \operatorname{clip}\!\big(I(x, y) + \beta, \; 0, \; 255\big)$$

Positive $\beta > 0$ brightens the image; negative $\beta < 0$ darkens it.

### 2.2 Contrast Transformations

Contrast controls the separation between dark and bright intensity levels. Five widely used operators are:

| Method | Mathematical Formulation ($u \in [0, 1]$) | Behavioral Profile |
| :--- | :--- | :--- |
| **Linear Contrast** | $u' = \operatorname{clip}\!\big(\alpha (u - 0.5) + 0.5, \; 0, \; 1\big)$ | Uniformly scales deviations around midpoint $0.5$; $\alpha > 1$ boosts contrast, $\alpha < 1$ compresses it. |
| **Gamma Correction** | $u' = u^{\gamma}, \quad \gamma > 0$ | Nonlinear power-law curve; $\gamma < 1$ expands shadow detail, $\gamma > 1$ darkens midtones and highlights. |
| **Sigmoid Contrast** | $u' = \dfrac{1}{1 + \exp\!\big(-g(u - c)\big)}$ | S-shaped curve with gain $g$ and cutoff $c$; boosts midtone contrast smoothly while compressing extremes. |
| **Histogram Equalization (HE)** | $u' = \operatorname{CDF}_I(u) = \sum_{k=0}^{\lfloor 255 u \rfloor} p_I(k)$ | Global monotonic mapping that flattens the intensity histogram across $[0, 255]$. |
| **CLAHE** | Local HE inside $M \times N$ tiles with clip limit $\tau$ + bilinear tile blending | Enhances local contrast while preventing noise over-amplification in uniform regions. |

#### Histogram Equalization Derivation

Let $p_I(k) = \frac{n_k}{HW}$ be the empirical probability of grayscale level $k \in \{0, \dots, 255\}$. Treating intensity as a continuous random variable $U$ with PDF $p_U(u)$, the cumulative distribution function $T(u) = \int_0^u p_U(t)\,dt$ produces a transformed variable $V = T(U)$ whose density is uniform:

$$p_V(v) = p_U(u) \left|\frac{du}{dv}\right| = p_U(u) \cdot \frac{1}{p_U(u)} = 1 \quad \text{for } v \in [0, 1]$$

In discrete images, **Contrast Limited Adaptive Histogram Equalization (CLAHE)** refines global HE by:
1. Partitioning the image into non-overlapping contextual tiles (e.g., $8 \times 8$).
2. Clipping each tile histogram at threshold $\tau$ and redistributing excess counts uniformly across all bins before computing the local CDF.
3. Bilinearly interpolating between neighboring tile mappings to eliminate block boundary seams.

---

## 3. Spatial Filtering & Blurring Operators

Blurring reduces high-frequency spatial detail, helping classifiers avoid overfitting to sensor noise or sharp texture artifacts.

| Operator | Formulation / Kernel Mechanism | Preserves Edges? | Typical Use Case |
| :--- | :--- | :---: | :--- |
| **Average (Box) Blur** | Uniform kernel $K_{i,j} = \frac{1}{k^2}$ over $k \times k$ window | No | Fast isotropic smoothing |
| **Gaussian Blur** | $G(x, y) = \frac{1}{2\pi\sigma^2}\exp\!\left(-\frac{x^2+y^2}{2\sigma^2}\right)$ | No | Natural optical defocus, noise suppression |
| **Median Blur** | $I'(x, y) = \operatorname{median}_{(u, v) \in \Omega} I(x+u, y+v)$ | Mostly | Removing salt-and-pepper impulse noise |
| **Bilateral Filter** | $W \propto \exp\!\left(-\frac{\|\mathbf{p}-\mathbf{q}\|^2}{2\sigma_s^2} - \frac{(I(\mathbf{p})-I(\mathbf{q}))^2}{2\sigma_r^2}\right)$ | **Yes** | Smoothing flat regions while keeping edges sharp |
| **Motion Blur** | Directional 1D line kernel oriented at angle $\theta$ and length $k$ | No | Simulating camera shake or moving subjects |
| **Mean-Shift Blur** | Iterative mode seeking in joint 5D space $(x, y, R, G, B)$ | **Yes** | Color quantization and region smoothing |

Notice the mathematical elegance of the **Bilateral Filter**: unlike Gaussian blur, which weights neighbors strictly by spatial distance $\|\mathbf{p} - \mathbf{q}\|$, the bilateral kernel multiplies the spatial Gaussian by a range Gaussian over intensity differences $|I(\mathbf{p}) - I(\mathbf{q})|$. Across a sharp edge, $|I(\mathbf{p}) - I(\mathbf{q})|$ is large, driving the cross-edge weight toward zero and preventing edge bleeding.

---

## 4. Occlusion & Regional Deletion Methods

Occlusion augmentations deliberately zero out or replace image regions during training so that convolutional and transformer backbones cannot rely on a single discriminative part (e.g., only a dog's nose or a bird's beak).

### 4.1 Cutout

**Cutout** (DeVries & Taylor, 2017) masks a fixed-size square patch of side $L \times L$ centered at a uniformly sampled coordinate $(c_x, c_y)$ with a constant value (typically $0$ or dataset mean):

$$M(x, y) = \begin{cases} 0 & \text{if } |x - c_x| \le \frac{L}{2} \text{ and } |y - c_y| \le \frac{L}{2} \\ 1 & \text{otherwise} \end{cases}, \qquad \tilde{I} = I \odot M$$

### 4.2 Random Erasing

**Random Erasing** (Zhong et al., 2020) generalizes Cutout by applying occlusion stochastically with probability $p_e$, sampling both:
- Area ratio $s_e \sim \mathcal{U}(s_l, s_h)$ so erased area $S_e = s_e \cdot (H \cdot W)$,
- Aspect ratio $r_e \sim \mathcal{U}(r_1, r_2)$, yielding $H_e = \sqrt{S_e r_e}$ and $W_e = \sqrt{S_e / r_e}$.

The erased rectangle is filled with either a constant value, dataset mean, or per-pixel uniform random noise $\mathcal{U}(0, 255)$.

### 4.3 Hide-and-Seek (HaS)

**Hide-and-Seek** (Singh & Lee, 2017) divides the image into a uniform $S \times S$ grid of patches and independently hides each grid cell with probability $p_{\text{hide}}$ (setting its pixels to the dataset channel mean). Because multiple scattered cells can be erased simultaneously, the network must learn distributed object representations across all parts.

### 4.4 GridMask

Random single-patch deletion risks either erasing the entire foreground object or missing it completely. **GridMask** (Chen et al., 2020) resolves this trade-off with a structured periodic grid of square masks parameterized by $(r, d)$:
- $d$: spatial period (distance between adjacent masked units),
- $r \in (0, 1)$: ratio of kept edge length, so each deleted square has side $k = (1 - r) d$ (or equivalently the keep ratio is $k_{\text{keep}} = 1 - (1 - r)^2$ depending on convention; specifically in the original paper, $r = 1 - \frac{\text{deleted\_side}}{d}$ controls the exact balance between deletion and retention).

As illustrated across parameter pairs such as $(r, d) = (120, 0.1)$, $(100, 0.4)$, and $(50, 0.3)$ (pixel period and keep ratio), periodic spacing guarantees that neither whole-object erasure nor zero-occlusion occurs systematically.

---

## 5. Region-Mixing & Label-Smoothing Augmentations

Rather than preserving hard one-hot targets, **mixing augmentations** synthesize convex combinations of input pixels (or patches) and assign proportional soft labels.

### 5.1 MixUp

**MixUp** (Zhang et al., 2018) linearly interpolates both inputs and one-hot target vectors for a randomly paired sample $(x_i, y_i)$ and $(x_j, y_j)$:

$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1 - \lambda) y_j$$

where the mixing coefficient $\lambda \sim \text{Beta}(\alpha, \alpha)$ with hyperparameter $\alpha > 0$. Equivalently, the training objective blends the two cross-entropy losses:

$$\mathcal{L}_{\text{MixUp}} = \lambda \, \mathcal{L}_{\text{CE}}(f_\theta(\tilde{x}), y_i) + (1 - \lambda) \, \mathcal{L}_{\text{CE}}(f_\theta(\tilde{x}), y_j)$$

For example, mixing a cat image ($y_i = [1.0, 0.0]$) and a dog image ($y_j = [0.0, 1.0]$) with $\lambda = 0.7$ produces an overlaid blended image labeled $[0.7, 0.3]$. MixUp encourages linear behavior between training examples and improves model calibration.

### 5.2 CutMix

Because MixUp blends pixels globally, it can create unnatural double-exposure images. **CutMix** (Yun et al., 2019) replaces a rectangular region in image $x_A$ with the corresponding patch from image $x_B$:

$$\tilde{x} = \mathbf{M} \odot x_A + (\mathbf{1} - \mathbf{M}) \odot x_B$$

where $\mathbf{M} \in \{0, 1\}^{H \times W}$ is a binary mask with zeroed bounding box $(r_x, r_y, r_w, r_h)$ sampled from:

$$r_x \sim \mathcal{U}(0, W), \quad r_y \sim \mathcal{U}(0, H), \qquad r_w = W\sqrt{1 - \lambda}, \quad r_h = H\sqrt{1 - \lambda}, \qquad \lambda \sim \text{Beta}(\beta, \beta)$$

To keep labels strictly proportional to the visible pixel area after clipping the box to image boundaries $[0, W] \times [0, H]$, $\lambda$ is recomputed from the clipped box area:

$$\lambda_{\text{adj}} = 1 - \frac{(x_2 - x_1)(y_2 - y_1)}{W \cdot H}, \qquad \tilde{y} = \lambda_{\text{adj}} y_A + (1 - \lambda_{\text{adj}}) y_B$$

### 5.3 Attentive CutMix

Random box placement in CutMix may copy background sky or grass while assigning foreground class weight to the target. **Attentive CutMix** (Walawalkar et al., 2020) uses a pretrained feature extractor (e.g., ResNet-50 prior to global pooling, producing a $7 \times 7$ spatial feature grid on $224 \times 224$ inputs) to select the **top-$k$ most activated patches** (typically $k = 6$ of the $7 \times 7$ grid cells, each $32 \times 32$ pixels) from the source image and paste only those high-attention patches at the corresponding grid positions in the target image.

### 5.4 RICAP (Random Image Cropping and Patching)

**RICAP** (Takahashi et al., 2019) crops four random patches from four distinct images $(I_0, I_1, I_2, I_3)$ in a mini-batch and tiles them into a $2 \times 2$ composite image of size $I_x \times I_y$:

1. Sample a split boundary point $(w, h)$ via Beta distributions:
   $$w = \operatorname{round}\!\big(I_x \cdot \text{Beta}(\beta, \beta)\big), \qquad h = \operatorname{round}\!\big(I_y \cdot \text{Beta}(\beta, \beta)\big)$$
   which defines the four quadrant dimensions:
   $$(w_k, h_k) \in \{(w, h),\; (I_x - w, h),\; (w, I_y - h),\; (I_x - w, I_y - h)\}$$
2. Randomly sample crop origins $(x_k, y_k)$ inside each of the four source images and stitch the four crops into $\tilde{I}$.
3. Compute the area-weighted loss across all four source labels $c_k$ with weights $W_k = \frac{w_k h_k}{I_x I_y}$ (note $\sum_{k=0}^3 W_k = 1$):
   $$\mathcal{L}_{\text{RICAP}} = \sum_{k=0}^{3} W_k \, \mathcal{L}_{\text{CE}}\big(f_\theta(\tilde{I}), c_k\big)$$

Three boundary placement regimes exist:
- **anywhere-RICAP:** $(w, h)$ can fall anywhere in $[0, I_x] \times [0, I_y]$.
- **center-RICAP:** $(w, h)$ is restricted to a central window $[t I_x, (1-t)I_x] \times [t I_y, (1-t)I_y]$.
- **corner-RICAP:** $(w, h)$ is restricted near corners (e.g., $[0, u I_x]$ or $[(1-u)I_x, I_x]$), which achieves the strongest empirical accuracy because at least one quadrant displays a large, unobstructed object view for the network to learn from.

### 5.5 SaliencyMix

**SaliencyMix** (Uddin et al., 2021) bridges CutMix and visual attention without requiring an auxiliary pretrained network:

1. Compute a **saliency map** $S(x_B)$ of the source image $x_B$ (e.g., OpenCV's Fine-Grained Static Saliency or spectral/gradient energy map).
2. Locate the **peak saliency coordinate** $(x^*, y^*) = \arg\max_{(x,y)} S(x_B)[x, y]$.
3. Center the CutMix bounding box of size $(r_w, r_h)$ around $(x^*, y^*)$ and paste that salient patch onto the target image $x_A$, adjusting $\lambda_{\text{adj}}$ by the clipped area ratio.

---

## 6. Inference & Consistency Regularization

### 6.1 Test-Time Augmentation (TTA)

Data augmentation need not stop at training. **Test-Time Augmentation (TTA)** generates $K$ augmented views $\{\mathcal{T}_1(x), \dots, \mathcal{T}_K(x)\}$ of a single test image (for example, original + horizontal flip + slight crop/shift) and averages the model's predicted class probabilities:

$$\hat{\mathbf{p}}_{\text{TTA}}(x) = \frac{1}{K} \sum_{k=1}^{K} \operatorname{softmax}\!\big(f_\theta(\mathcal{T}_k(x))\big), \qquad \hat{y} = \arg\max_c \, [\hat{\mathbf{p}}_{\text{TTA}}(x)]_c$$

Because random perturbations across views cancel out variance in predictions (an implicit single-model ensemble), TTA typically boosts classification accuracy and reduces calibration error at the cost of $K\times$ inference latency.

### 6.2 $\Pi$-Model Consistency Regularization

The **$\Pi$-Model** (Laine & Aila, 2017) uses data augmentation as a **consistency regularizer** (originally for semi-supervised learning, and also effective as a supervised regularizer). Given input $x$, we evaluate the network twice under independent stochastic perturbations or noise ($\tilde{x} = x + \epsilon$ or two random augmentations):

$$\mathbf{z} = f_\theta(x), \qquad \tilde{\mathbf{z}} = f_\theta(\tilde{x})$$

and penalize prediction disagreement between the two views using Mean Squared Error alongside the supervised Cross-Entropy loss:

$$\mathcal{L}_{\Pi} = \mathcal{L}_{\text{CE}}(\mathbf{z}, y) + \alpha(t) \, \|\mathbf{z} - \tilde{\mathbf{z}}\|_2^2$$

where $\alpha(t)$ controls the consistency weight (often ramped up during early epochs so the model stabilizes before enforcing invariance).

---

## 7. Comparison & Practical Selection Guide

| Technique | Operates On | Label Modification | Preserves Global Structure? | Key Hyperparameters | Best Used When |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Flip / Scale / Shift** | Single image | No | Yes | `p`, `shift_limit`, `scale_limit`, `rotate_limit` | Baseline pipeline for almost all vision tasks |
| **Brightness / Contrast / CLAHE** | Single image | No | Yes | `gamma`, `clip_limit`, `tile_grid_size` | Illumination changes, medical or outdoor scenes |
| **Blur (Gaussian / Motion / Bilateral)** | Single image | No | Yes | `sigma`, `kernel_size` | Camera shake, sensor noise robustness |
| **Cutout / Random Erasing / GridMask** | Single image | No | Partial | `area_ratio`, `(r, d)` | Occluded objects, fine-grained recognition |
| **MixUp** | Image pair | Soft convex ($\lambda$) | Blended | $\alpha \in [0.2, 1.0]$ | Improving calibration and decision boundaries |
| **CutMix** | Image pair | Soft area ratio ($\lambda_{\text{adj}}$) | Patch-wise | $\beta = 1.0$, $p_{\text{cutmix}}$ | General CNN/ViT image classification |
| **Attentive CutMix / SaliencyMix** | Image pair | Soft area ratio | Salient patches | `top_k`, $\beta$ | Foreground objects in cluttered backgrounds |
| **RICAP** | 4 images | Soft 4-way ($W_k$) | Quadrant-wise | $\beta = 0.3$, boundary mode | Small-to-medium classification datasets |
| **TTA** | Inference views | None (averages $\mathbf{p}$) | Yes | Number of views $K$ | Squeezing extra accuracy at test time |
| **$\Pi$-Model** | Perturbed pair | Unsupervised MSE term | Yes | Consistency weight $\alpha$ | Noisy inputs or limited labeled data |

---

## 8. Notebooks & Practice Lab Index

All nine notebooks are self-contained, de-branded, and wired to the local fixture in [`data/sample_dataset/`](data/README.md):

| Notebook | Technique Covered | Key Implementation Focus |
| :--- | :--- | :--- |
| [`01_albumentations_pipeline.ipynb`](notebooks/01_albumentations_pipeline.ipynb) | Albumentations + K-Fold ResNet-18 | Composing flip, compression, affine `ShiftScaleRotate`/`Affine`, and normalization in a PyTorch `Dataset` |
| [`02_custom_augmentation_transforms.ipynb`](notebooks/02_custom_augmentation_transforms.ipynb) | Multi-stage OpenCV / `imgaug`-style Pipeline | Implementing crop/pad, affine warps, blur, additive Gaussian noise, and contrast transforms in portable Python |
| [`03_cutmix_augmentation.ipynb`](notebooks/03_cutmix_augmentation.ipynb) | CutMix | Sampling Beta bounding boxes (`rand_bbox`), pixel-ratio $\lambda$ adjustment, and mixed Cross-Entropy loss |
| [`04_mixup_augmentation.ipynb`](notebooks/04_mixup_augmentation.ipynb) | MixUp | Linear input & target interpolation (`mixup_data`, `mixup_criterion`) with Beta($\alpha, \alpha$) weights |
| [`05_ricap_augmentation.ipynb`](notebooks/05_ricap_augmentation.ipynb) | RICAP | Four-image quadrant cropping, boundary sampling, and area-weighted 4-term Cross-Entropy loss |
| [`06_saliencymix_augmentation.ipynb`](notebooks/06_saliencymix_augmentation.ipynb) | SaliencyMix | Extracting peak saliency regions (`saliency_bbox`) with OpenCV/gradient fallback and pasting salient patches |
| [`07_attentive_cutmix.ipynb`](notebooks/07_attentive_cutmix.ipynb) | Attentive CutMix | Using ResNet-50 feature maps to locate and transfer top-$k$ attentive $32 \times 32$ patches |
| [`08_test_time_augmentation.ipynb`](notebooks/08_test_time_augmentation.ipynb) | Test-Time Augmentation (TTA) | Training K-Fold checkpoints and comparing single-pass vs. multi-view probability-averaged inference |
| [`09_pi_model_consistency_regularization.ipynb`](notebooks/09_pi_model_consistency_regularization.ipynb) | $\Pi$-Model Consistency | Combining supervised Cross-Entropy with perturbation consistency MSE loss ($\mathcal{L}_{\text{CE}} + \alpha \mathcal{L}_{\text{MSE}}$) |

---

## References

1. Shorten, C., & Khoshgoftaar, T. M. (2019). *A survey on Image Data Augmentation for Deep Learning*. Journal of Big Data, 6(1), 60.
2. DeVries, T., & Taylor, G. W. (2017). *Improved Regularization of Convolutional Neural Networks with Cutout*. arXiv:1708.04552.
3. Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2020). *Random Erasing Data Augmentation*. AAAI.
4. Singh, K. K., & Lee, Y. J. (2017). *Hide-and-Seek: Forcing a Network to be Meticulous for Weakly-Supervised Object and Action Localization*. ICCV.
5. Chen, P., Liu, S., Zhao, H., & Jia, J. (2020). *GridMask Data Augmentation*. arXiv:2001.04086.
6. Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). *mixup: Beyond Empirical Risk Minimization*. ICLR.
7. Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y. (2019). *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*. ICCV.
8. Walawalkar, D., Shen, Z., Liu, Z., & Savvides, M. (2020). *Attentive CutMix: An Enhanced Data Augmentation Approach for Deep Learning Based Image Classification*. ICASSP.
9. Takahashi, R., Matsubara, T., & Uehara, K. (2019). *Data Augmentation Using Random Image Cropping and Patching for Deep CNNs (RICAP)*. IEEE TCSVT.
10. Uddin, A. F. M. S., Monira, M. S., Shin, W., Chung, T., & Bae, S.-H. (2021). *SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Visual Recognition*. ICLR.
11. Laine, S., & Aila, T. (2017). *Temporal Ensembling for Semi-Supervised Learning ($\Pi$-Model)*. ICLR.
12. Buslaev, A., et al. (2020). *Albumentations: Fast and Flexible Image Augmentations*. Information, 11(2), 125.
