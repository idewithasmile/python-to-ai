# Edge Detection and Gradient-based Optimization: Activation Derivatives, Discrete Image Calculus, Sobel Filters, and Momentum Dynamics

[Back to Module 02](../README.md) · [Back to the master roadmap](../../README.md)

**Session date:** 2026-11-19 · **Module:** 02 — Mathematics for Machine Learning (Calculus & Linear Algebra)  
**Format:** Textbook-grade study guide, complete analytical derivations, Taylor error analysis, Sobel convolution mechanics, discrete vs. continuous optimization, Polyak momentum dynamics, and five interactive notebooks

---

## Overview

Calculus provides the universal mathematical framework that connects physical signal processing to deep learning parameter optimization. At the center of this framework lies the concept of the **derivative**—the instantaneous rate of change of a mathematical function. While basic calculus treats differentiation primarily as an analytical exercise on continuous scalar functions $f: \mathbb{R} \to \mathbb{R}$, modern artificial intelligence applies calculus across two foundational domains:

1. **Digital Signal & Computer Vision (Discrete Spatial Differentiation):**  
   In digital images, visual features such as edges, object boundaries, contours, and surface transitions correspond to rapid spatial fluctuations in pixel intensity. Because a digital sensor samples light intensity on a discrete 2D grid $I(x, y)$, differentiation cannot evaluate an infinitesimal limit $\Delta x \to 0$. Instead, spatial derivatives must be approximated through **finite differences** (single-sided and symmetric central differences). Combining orthogonal spatial differentiation with Gaussian-like triangular smoothing yields the classical **Sobel operator**, which constructs a continuous gradient vector field $\nabla I = [G_x, G_y]^T$ capturing edge magnitude and directional orientation.

2. **Neural Network Construction & Parameter Optimization (Continuous Directional Differentiation):**  
   In deep neural architectures, non-linear activation functions such as **Sigmoid**, **Hyperbolic Tangent (Tanh)**, and **Swish** inject the non-linear expressive capacity required to break linear subspace collapse. Training these architectures relies on backpropagating loss gradients via the chain rule. When optimizing objective loss surfaces $\mathcal{L}(\theta)$, the gradient vector $\nabla_\theta \mathcal{L}$ points in the direction of steepest ascent; moving in the negative gradient direction $-\nabla_\theta \mathcal{L}$ guarantees instantaneous descent. However, naive gradient descent struggles in the presence of pathological curvature, ill-conditioned ravines, flat plateaus, and saddle points. Introducing physical inertia via **Polyak's Momentum** dampens high-frequency transverse oscillations and accelerates parameter convergence along consistent ravines.

This guide provides a rigorous, textbook-grade mathematical treatment unifying activation derivatives, discrete image operators, finite difference Taylor expansions, discrete coordinate stepping, and continuous momentum-accelerated gradient descent.

---

## Learning outcomes

By the end of this session, you should be able to:

- Derive the exact analytical derivative of the **Sigmoid** function $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ from first principles using the chain and quotient rules, and explain why $\sigma'(x) \le 0.25$ induces the vanishing gradient problem in deep networks.
- Prove the derivative identities for **Hyperbolic Tangent** $\tanh'(x) = 1 - \tanh^2(x)$ and **Swish** $\text{swish}'(x) = \text{swish}(x) + \sigma(x)(1 - \text{swish}(x))$, analyzing their zero-centeredness and non-monotonicity.
- Contrast single-sided forward differences $\mathcal{O}(h)$ with symmetric central differences $\mathcal{O}(h^2)$ using Taylor series expansions, identifying the theoretical error cancellation mechanisms.
- Formulate 2D digital images as discrete scalar fields and compute row-wise and column-wise discrete differences, resolving the signed gradient representation problem via absolute value mapping $|d|$ and midpoint level-shifting $d + 128$.
- Derive the **Sobel operator** kernels $K_x$ and $K_y$ as outer products of a 1D discrete central difference filter and an orthogonal 1D triangular smoothing filter, proving rank-1 separability.
- Compute the 2D spatial gradient vector $\nabla I$, the gradient magnitude $G = \sqrt{G_x^2 + G_y^2}$, and the gradient orientation $\theta = \arctan2(G_y, G_x)$ across digital image patches from scratch.
- Implement discrete coordinate-wise **sign-based stepping** algorithms ($x_t = x_{t-1} - \text{sign}(\nabla f)$) for 1D and 2D convex paraboloids ($f(x) = x^2, f(x, y) = x^2 + y^2$), proving exact step convergence on integer lattices.
- Formulate continuous **Gradient Descent** $\theta_t = \theta_{t-1} - \eta \nabla f(\theta_{t-1})$ and derive the theoretical upper bound on learning rate $\eta < \frac{2}{L}$ under $L$-Lipschitz gradient continuity.
- Analyze the geometry of non-convex loss surfaces, distinguishing local minima, global minima, plateaus, and saddle points via the eigenvalues of the Hessian matrix $\mathbf{H}$.
- Formulate **Polyak's Heavy Ball Momentum** $v_t = \gamma v_{t-1} + \eta \nabla f(\theta_{t-1})$, proving how velocity unrolls into an exponentially weighted moving average of past gradients that accelerates convergence by a factor of $\frac{1}{1 - \gamma}$ along consistent valleys while damping orthogonal oscillations.

---

## Session Map

| Notebook | Focus | Key Topics & Implementations |
| :--- | :--- | :--- |
| [01 · Activation Functions & Derivatives](notebooks/01_activation_functions_and_derivatives.ipynb) | Analytical derivations, numerical finite differences, and gradient saturation | Sigmoid, Tanh, and Swish; step-by-step chain rule proofs; forward vs. central difference autograd; `np.gradient`; vanishing gradient visualization |
| [02 · Discrete 1D Image Derivatives](notebooks/02_discrete_image_derivatives_1d.ipynb) | 1D discrete differences on 2D image matrices | Row/column 1D slices, forward vs. central difference, signed value visualization ($|d|$ vs. level shift $d+128$), pure Python loops vs. vectorized NumPy slicing |
| [03 · Sobel Operators & Edge Detection](notebooks/03_sobel_operators_and_edge_detection.ipynb) | 2D spatial gradients, kernel separability, and edge maps | $K_x$ and $K_y$ outer-product decomposition, scratch 2D convolution, slide 22 hand-calculation check ($a = -211$), gradient magnitude, angle $\theta$, OpenCV comparison |
| [04 · Gradient-based Optimization](notebooks/04_gradient_based_optimization_and_learning_rates.ipynb) | Discrete coordinate descent and continuous Gradient Descent | Sign-based discrete optimization ($x_0 = 70 \to 0$ in 70 steps; 2D $(3,4) \to (0,0)$ in 4 steps), continuous GD, learning rate tuning ($\eta$ small vs. optimal vs. divergent) |
| [05 · Momentum & Optimization Landscapes](notebooks/05_momentum_and_optimization_landscapes.ipynb) | Non-convex surfaces, ill-conditioned ravines, and momentum | Polyak heavy ball formulation, effective step $\frac{\eta}{1-\gamma}$, anisotropic quadratic bowl ($\kappa = 20$), saddle point escape ($f = x^4 - 2x^2 + y^2$), loss curves |

---

## 1. Activation Functions & Analytical Derivatives

### 1.1 The Role of Activation Functions
In a feedforward neural layer, an affine linear transformation maps input $x \in \mathbb{R}^d$ to pre-activation $z = Wx + b$. Composing linear transformations without non-linearities collapses an arbitrary deep network into a single linear map:

$$W_L(W_{L-1}(\dots(W_1 x + b_1)\dots) + b_{L-1}) + b_L = \widetilde{W} x + \widetilde{b}$$

To approximate complex non-linear manifolds, an element-wise **activation function** $\phi: \mathbb{R} \to \mathbb{R}$ is applied: $a = \phi(z)$. During backpropagation, the chain rule scales incoming downstream error signals by the local derivative $\phi'(z)$.

---

### 1.2 The Sigmoid Function

The standard logistic Sigmoid function maps real numbers into probabilities $\sigma(x) \in (0, 1)$:

$$\sigma(x) = \frac{1}{1 + e^{-x}} = (1 + e^{-x})^{-1}$$

#### Step-by-Step Analytical Derivation
Applying the power rule and the chain rule:

$$\sigma'(x) = \frac{d}{dx}\left[(1 + e^{-x})^{-1}\right] = -1 \cdot (1 + e^{-x})^{-2} \cdot \frac{d}{dx}(1 + e^{-x})$$

Evaluating the inner derivative $\frac{d}{dx}(1 + e^{-x}) = -e^{-x}$:

$$\sigma'(x) = -(1 + e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1 + e^{-x})^2}$$

To express this strictly in terms of $\sigma(x)$, rewrite the numerator by adding and subtracting $1$:

$$\frac{e^{-x}}{(1 + e^{-x})^2} = \frac{(1 + e^{-x}) - 1}{(1 + e^{-x})^2} = \frac{1 + e^{-x}}{(1 + e^{-x})^2} - \frac{1}{(1 + e^{-x})^2} = \frac{1}{1 + e^{-x}} - \left(\frac{1}{1 + e^{-x}}\right)^2$$

Factoring out $\frac{1}{1 + e^{-x}}$:

$$\sigma'(x) = \frac{1}{1 + e^{-x}} \left(1 - \frac{1}{1 + e^{-x}}\right) = \mathbf{\sigma(x)(1 - \sigma(x))}$$

#### Mathematical Properties & Saturation Analysis
- **Symmetry:** $\sigma(-x) = 1 - \sigma(x)$.
- **Peak Derivative:** At $x = 0$, $\sigma(0) = 0.5$, yielding $\sigma'(0) = 0.5(1 - 0.5) = \mathbf{0.25}$.
- **Vanishing Gradient:** Because $\sigma'(x) \le 0.25$ for all $x$, backpropagating through $L$ consecutive Sigmoid layers scales gradients by at most $(0.25)^L$. For $L = 5$, $(0.25)^5 \approx 0.00097$. For $|x| > 4$, $\sigma'(x) < 0.0177$, rendering early layers essentially unlearnable.

```
       sigma(x) and sigma'(x)
  1.0 +                      .-'""'-.               --- sigma(x)
      |                    .'        '.             ... sigma'(x)
  0.5 + - - - - - - - - - / - - - - - -\ - - - - -
      |             . ' ' :            : ' ' .
  0.25+ - - - - - -' . . .:. . . . . . .:. . . '- - - (Peak derivative = 0.25)
      |            :      :            :      :
  0.0 +------------+------+------------+------+----> x
                  -4      0            4
```

---

### 1.3 The Hyperbolic Tangent (Tanh) Function

The Tanh function is a zero-centered sigmoid scaled to $(-1, 1)$:

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{2}{1 + e^{-2x}} - 1 = 1 - \frac{2}{e^{2x} + 1}$$

#### Derivation 1: Quotient Rule
Let $u(x) = e^x - e^{-x}$ (with $u'(x) = e^x + e^{-x}$) and $v(x) = e^x + e^{-x}$ (with $v'(x) = e^x - e^{-x}$):

$$\tanh'(x) = \frac{u'v - uv'}{v^2} = \frac{(e^x + e^{-x})(e^x + e^{-x}) - (e^x - e^{-x})(e^x - e^{-x})}{(e^x + e^{-x})^2}$$
$$\tanh'(x) = \frac{(e^x + e^{-x})^2 - (e^x - e^{-x})^2}{(e^x + e^{-x})^2} = 1 - \left(\frac{e^x - e^{-x}}{e^x + e^{-x}}\right)^2 = \mathbf{1 - \tanh^2(x)}$$

#### Derivation 2: Scaled Exponential Form
Using $\tanh(x) = 2(1 + e^{-2x})^{-1} - 1$:

$$\tanh'(x) = 2 \cdot (-1)(1 + e^{-2x})^{-2} \cdot (-2e^{-2x}) = \frac{4e^{-2x}}{(1 + e^{-2x})^2}$$
$$\tanh'(x) = \frac{4(e^{-2x} + 1 - 1)}{(1 + e^{-2x})^2} = \frac{4}{1 + e^{-2x}} - \frac{4}{(1 + e^{-2x})^2} = 1 - \left(\frac{2}{1 + e^{-2x}} - 1\right)^2 = \mathbf{1 - \tanh^2(x)}$$

#### Properties & Comparison with Sigmoid
- **Zero-Centered:** $\tanh(0) = 0$, meaning activations have zero mean, avoiding directional zigzagging in downstream weight updates.
- **Peak Derivative:** At $x = 0$, $\tanh'(0) = 1 - 0^2 = \mathbf{1.0}$ (four times larger than Sigmoid).
- **Saturation:** For $|x| > 3$, $\tanh(x) \approx \pm 1$ and $\tanh'(x) \to 0$, retaining the saturation vulnerability of squashing functions.

---

### 1.4 The Swish Activation Function

Proposed by Ramachandran, Zoph, and Le (2017), **Swish** is a self-gated activation function:

$$\text{swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

#### Analytical Derivation
Using the product rule $(uv)' = u'v + uv'$ where $u(x) = x$ ($u'(x) = 1$) and $v(x) = \sigma(x)$ ($v'(x) = \sigma(x)(1 - \sigma(x))$):

$$\text{swish}'(x) = 1 \cdot \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \sigma(x) + x\sigma(x) - x\sigma(x)^2$$

Substituting $x\sigma(x) = \text{swish}(x)$:

$$\text{swish}'(x) = \sigma(x) + \text{swish}(x) - \text{swish}(x)\sigma(x) = \text{swish}(x) + \sigma(x)(1 - \text{swish}(x))$$

Alternatively, factoring $\sigma(x)$:

$$\text{swish}'(x) = \sigma(x)[1 + x(1 - \sigma(x))]$$

#### Geometric Superiority
1. **Smooth Non-Monotonicity:** Swish exhibits a shallow dip into negative values for $x \in (-1.28, 0)$, with a global minimum at $x \approx -1.278$ where $\text{swish}(x) \approx -0.278$.
2. **Gradient Flow for Negative Inputs:** Unlike ReLU which zeroes out all negative inputs ($f'(x) = 0$ for $x < 0$), Swish preserves a small negative gradient, preventing dead neurons.
3. **Upper Unboundedness:** As $x \to \infty$, $\text{swish}(x) \to x$ and $\text{swish}'(x) \to 1$, completely eliminating saturation for large positive activations.

---

### 1.5 Taxonomy of Neural Activation Functions

| Function Name | Mathematical Form $f(x)$ | Analytical Derivative $f'(x)$ | Output Range | Peak Gradient |
| :--- | :--- | :--- | :---: | :---: |
| **Sigmoid** | $\frac{1}{1 + e^{-x}}$ | $\sigma(x)(1 - \sigma(x))$ | $(0, 1)$ | $0.25$ (at $x=0$) |
| **Tanh** | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2(x)$ | $(-1, 1)$ | $1.00$ (at $x=0$) |
| **ReLU** | $\max(0, x)$ | $\begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}$ | $[0, \infty)$ | $1.00$ ($x > 0$) |
| **Leaky ReLU** | $\max(\alpha x, x), \; \alpha \approx 0.01$ | $\begin{cases} 1 & x > 0 \\ \alpha & x \le 0 \end{cases}$ | $(-\infty, \infty)$ | $1.00$ ($x > 0$) |
| **ELU** | $\begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \le 0 \end{cases}$ | $\begin{cases} 1 & x > 0 \\ f(x) + \alpha & x \le 0 \end{cases}$ | $(-\alpha, \infty)$ | $1.00$ ($x > 0$) |
| **Swish** | $x \cdot \sigma(x)$ | $\text{swish}(x) + \sigma(x)(1 - \text{swish}(x))$ | $[-0.278, \infty)$ | $\approx 1.099$ |
| **Softplus** | $\ln(1 + e^x)$ | $\frac{1}{1 + e^{-x}} = \sigma(x)$ | $(0, \infty)$ | $1.00$ ($x \to \infty$) |

---

### 1.6 Finite Difference Truncation Error Analysis

Numerical differentiation checks analytical implementations against finite step approximations.

#### Forward Difference $\mathcal{O}(h)$
Taylor expanding $f(x + h)$:

$$f(x + h) = f(x) + h f'(x) + \frac{h^2}{2} f''(x) + \mathcal{O}(h^3)$$
$$\frac{f(x + h) - f(x)}{h} = f'(x) + \frac{h}{2} f''(x) + \mathcal{O}(h^2) = f'(x) + \mathcal{O}(h)$$

The leading truncation error is first-order in step size $h$.

#### Central Difference $\mathcal{O}(h^2)$
Taylor expanding both $f(x + h/2)$ and $f(x - h/2)$:

$$f(x + h/2) = f(x) + \frac{h}{2} f'(x) + \frac{h^2}{8} f''(x) + \frac{h^3}{48} f'''(x) + \mathcal{O}(h^4)$$
$$f(x - h/2) = f(x) - \frac{h}{2} f'(x) + \frac{h^2}{8} f''(x) - \frac{h^3}{48} f'''(x) + \mathcal{O}(h^4)$$

Subtracting the two series:

$$f(x + h/2) - f(x - h/2) = h f'(x) + \frac{h^3}{24} f'''(x) + \mathcal{O}(h^5)$$
$$\frac{f(x + h/2) - f(x - h/2)}{h} = f'(x) + \frac{h^2}{24} f'''(x) + \mathcal{O}(h^4) = f'(x) + \mathbf{\mathcal{O}(h^2)}$$

The quadratic error term ensures that reducing $h$ by $10\times$ reduces truncation error by $100\times$.

---

## 2. Discrete Calculus in Digital Image Processing & Edge Detection

### 2.1 The Image as a 2D Discrete Scalar Field
A digital grayscale image $I$ is a matrix of discrete spatial samples:

$$I \in \{0, 1, \dots, 255\}^{H \times W}$$

where $i \in \{0, \dots, H-1\}$ denotes row (vertical axis, $y$) and $j \in \{0, \dots, W-1\}$ denotes column (horizontal axis, $x$).

An **edge** represents a discontinuity or sharp transition in pixel intensity. Mathematically, edges correspond to local extrema in the spatial derivative.

```
Intensity Profile I(x)                First Derivative dI/dx
        ^                                     ^
255 +   |        .-----------         +255 +  |           /\  <--- Peak at edge!
        |       /                             |          /  \
        |      /                              |         /    \
  0 +---+-----'-------------> x          0 +--+--------'------'--------> x
```

---

### 2.2 1D Discrete Differences on Pixel Lattices

Treating each row as a discrete 1D function $f(j) = I[i, j]$:
- **Forward difference ($\Delta x = 1$):**
  $$d_x[i, j] = I[i, j+1] - I[i, j]$$
- **Central difference ($\Delta x = 2$):**
  $$\delta_x[i, j] = \frac{I[i, j+1] - I[i, j-1]}{2}$$

#### The Visualization Mapping Problem
Because pixel values are bounded within $[0, 255]$, differences $d \in [-255, +255]$ contain negative numbers that cannot be directly rendered in standard 8-bit displays.

Two canonical transformations solve this:
1. **Magnitude Mapping (Rectified):**
   $$I_{\text{mag}}[i, j] = |d_x[i, j]|$$
   Transitions are rendered as bright contours regardless of direction (dark-to-bright or bright-to-dark).
2. **Midpoint Level-Shifting (Biased):**
   $$I_{\text{shift}}[i, j] = \text{clip}(d_x[i, j] + 127.5, 0, 255)$$
   Zero rate of change (flat regions) maps to neutral gray ($128$). Positive gradients (dark-to-bright transitions) appear light ($>128$), while negative gradients (bright-to-dark transitions) appear dark ($<128$).

---

### 2.3 The 2D Spatial Gradient Vector

In continuous 2D space, the gradient of an image scalar field $I(x, y)$ is a 2D vector field:

$$\nabla I(x, y) = \begin{bmatrix} \frac{\partial I}{\partial x} \\ \frac{\partial I}{\partial y} \end{bmatrix} = \begin{bmatrix} G_x \\ G_y \end{bmatrix}$$

From this vector field, two invariant geometric quantities are computed at every pixel:

1. **Gradient Magnitude (Edge Strength):**
   $$G = \|\nabla I\|_2 = \sqrt{G_x^2 + G_y^2} \quad \text{or } L_1 \text{ approximation: } G \approx |G_x| + |G_y|$$
2. **Gradient Orientation (Direction of Steepest Intensity Ascent):**
   $$\theta = \arctan2(G_y, G_x) \in [-\pi, \pi]$$

The edge boundary tangent is strictly orthogonal to the gradient orientation: $\theta_{\text{edge}} = \theta \pm \frac{\pi}{2}$.

---

### 2.4 The Sobel Operator

Pure 1D differences amplify high-frequency image noise. The **Sobel operator** suppresses noise by combining numerical differentiation with smoothing.

#### Rank-1 Separable Decomposition
The $3 \times 3$ horizontal Sobel kernel $K_x$ and vertical kernel $K_y$ decompose into outer products:

$$K_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix} \ast \begin{bmatrix} -1 & 0 & +1 \end{bmatrix}$$
$$K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix} = \begin{bmatrix} -1 \\ 0 \\ +1 \end{bmatrix} \ast \begin{bmatrix} 1 & 2 & 1 \end{bmatrix}$$

- In $K_x$, the vector $[-1, 0, +1]$ calculates a horizontal central difference across columns, while the orthogonal vector $[1, 2, 1]^T$ computes a weighted triangular moving average across rows.
- In $K_y$, $[-1, 0, +1]^T$ calculates a vertical central difference across rows, while $[1, 2, 1]$ smooths across columns.

#### 2D Discrete Convolution Formula
For an image $I$ and $3 \times 3$ kernel $K$:

$$G[i, j] = (I * K)[i, j] = \sum_{u=-1}^1 \sum_{v=-1}^1 K[u+1, v+1] \cdot I[i+u, j+v]$$

#### Hand Calculation Walkthrough (Lecture Slide 22)
Consider a local $3 \times 3$ image patch:

$$\mathbf{P} = \begin{bmatrix} 230 & p_{01} & 147 \\ 237 & p_{11} & 188 \\ 210 & p_{21} & 180 \end{bmatrix}$$

Convolving with horizontal Sobel kernel $K_x$:

$$a = \sum_{u, v} K_x[u, v] \cdot \mathbf{P}[u, v]$$
$$a = (-1)(230) + (1)(147) + (-2)(237) + (2)(188) + (-1)(210) + (1)(180)$$
$$a = -230 + 147 - 474 + 376 - 210 + 180 = \mathbf{-211}$$

The absolute response is:
$$|a| = |-211| = \mathbf{211}$$

This confirms the exact hand derivation presented in lecture slide 22.

---

## 3. Mathematical Foundations of Gradient-based Optimization

### 3.1 The Gradient as an Optimization Cue
Let $f: \mathbb{R}^d \to \mathbb{R}$ be a continuously differentiable objective function. The first-order directional derivative along a unit vector $\mathbf{u}$ ($\|\mathbf{u}\| = 1$) is:

$$D_{\mathbf{u}} f(\mathbf{x}) = \nabla f(\mathbf{x})^T \mathbf{u} = \|\nabla f(\mathbf{x})\| \cos \phi$$

where $\phi$ is the angle between $\nabla f(\mathbf{x})$ and $\mathbf{u}$.
- To maximize the increase, set $\cos \phi = +1 \implies \mathbf{u} = \frac{\nabla f(\mathbf{x})}{\|\nabla f(\mathbf{x})\|}$ (**Steepest Ascent**).
- To maximize the decrease, set $\cos \phi = -1 \implies \mathbf{u} = -\frac{\nabla f(\mathbf{x})}{\|\nabla f(\mathbf{x})\|}$ (**Steepest Descent**).

---

### 3.2 Discrete Sign-based Coordinate Optimization

When optimizing over integer lattices ($\mathbf{x} \in \mathbb{Z}^d$), step magnitudes cannot be infinitesimally small. Taking discrete unit steps opposite to the sign of the partial derivatives guarantees descent:

$$x_{i, t} = x_{i, t-1} - \text{sign}\left(\frac{\partial f}{\partial x_i}\right) \cdot \Delta s$$

#### 1D Integer Paraboloid: $f(x) = x^2$ (Slides 33–35)
With $x_0 = 70.0$, $f'(x) = 2x$, and $\text{step} = 1.0$:
- Iteration 1: $f'(70.0) = +140.0 \implies \text{sign} = +1 \implies x_1 = 70 - 1 = 69.0$
- Iteration 2: $f'(69.0) = +138.0 \implies \text{sign} = +1 \implies x_2 = 69 - 1 = 68.0$
- $\dots$
- Iteration 64: $f'(7.0) = +14.0 \implies x_{64} = 6.0$
- Iteration 70: $f'(1.0) = +2.0 \implies x_{70} = 0.0$
- At $x = 0.0$: $f'(0.0) = 0.0 \implies \text{sign} = 0 \implies x_{71} = 0.0$ (Stationary convergence)

#### 2D Integer Paraboloid: $f(x, y) = x^2 + y^2$ (Slide 37)
Starting at $(x_0, y_0) = (3.0, 4.0)$:
- Step 0: $\nabla f = (2(3), 2(4)) = (6.0, 8.0) \implies \Delta = (-1, -1) \implies (x_1, y_1) = (2.0, 3.0)$
- Step 1: $\nabla f = (4.0, 6.0) \implies \Delta = (-1, -1) \implies (x_2, y_2) = (1.0, 2.0)$
- Step 2: $\nabla f = (2.0, 4.0) \implies \Delta = (-1, -1) \implies (x_3, y_3) = (0.0, 1.0)$
- Step 3: $\nabla f = (0.0, 2.0) \implies \Delta = (0, -1) \implies (x_4, y_4) = \mathbf{(0.0, 0.0)}$

The discrete coordinate algorithm reaches the global minimum at the origin in exactly $4$ steps.

---

### 3.3 Continuous Gradient Descent Dynamics & Learning Rate Bounds

In continuous parameter spaces $\theta \in \mathbb{R}^d$, Gradient Descent scales updates by learning rate $\eta > 0$:

$$\theta_t = \theta_{t-1} - \eta \nabla f(\theta_{t-1})$$

#### Lipschitz Gradient Continuity and the Convergence Bound
Assume $\nabla f$ is $L$-Lipschitz continuous:

$$\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \le L \|\mathbf{x} - \mathbf{y}\| \quad \forall \mathbf{x}, \mathbf{y}$$

By the Descent Lemma:

$$f(\theta_t) \le f(\theta_{t-1}) + \nabla f(\theta_{t-1})^T(\theta_t - \theta_{t-1}) + \frac{L}{2} \|\theta_t - \theta_{t-1}\|^2$$

Substituting $\theta_t - \theta_{t-1} = -\eta \nabla f(\theta_{t-1})$:

$$f(\theta_t) \le f(\theta_{t-1}) - \eta \|\nabla f(\theta_{t-1})\|^2 + \frac{L \eta^2}{2} \|\nabla f(\theta_{t-1})\|^2 = f(\theta_{t-1}) - \eta \left(1 - \frac{L\eta}{2}\right) \|\nabla f(\theta_{t-1})\|^2$$

For strict monotonic descent ($f(\theta_t) < f(\theta_{t-1})$ whenever $\nabla f \neq 0$):

$$1 - \frac{L\eta}{2} > 0 \implies \mathbf{\eta < \frac{2}{L}}$$

- **Optimal Damping:** Maximizing progress $\eta(1 - L\eta/2)$ yields $\eta^* = \frac{1}{L}$.
- **Under-stepping ($\eta \ll \frac{1}{L}$):** Guarantees monotonic descent but requires an excessive number of iterations.
- **Over-stepping ($\frac{1}{L} < \eta < \frac{2}{L}$):** Induces damped oscillatory overshoots across the optimum.
- **Divergence ($\eta > \frac{2}{L}$):** Updates overshoot the valley walls with exponentially expanding amplitudes, driving loss to infinity.

```
       Learning Rate Dynamics on f(x) = x^2 (L = 2)
  Small (eta = 0.05)         Optimal (eta = 0.40)          Divergent (eta = 1.05)
      |                          |                             \      /
     / \                        / \                           / \    / \
    /   \                      /   \                         /   \  /   \
   *-->*->*->*                *--->*->*                     *------>*    \
  (Monotonic crawl)          (Rapid convergence)            \            *
                                                             (Exploding oscillation)
```

---

## 4. Optimization Landscapes, Saddle Points, and Momentum

### 4.1 Non-Convex Curvature and Stationary Points

For a twice-differentiable function $f(\mathbf{x})$, stationary points satisfy $\nabla f(\mathbf{x}^*) = \mathbf{0}$. The local curvature is determined by the eigenvalues $\{\lambda_1, \dots, \lambda_d\}$ of the symmetric **Hessian matrix** $\mathbf{H} = \nabla^2 f(\mathbf{x}^*)$:

1. **Local Minimum:** All eigenvalues are strictly positive ($\lambda_i > 0, \forall i \iff \mathbf{H} \succ 0$). The surface curves upward in all directions.
2. **Local Maximum:** All eigenvalues are strictly negative ($\lambda_i < 0, \forall i \iff \mathbf{H} \prec 0$). The surface curves downward in all directions.
3. **Saddle Point:** The Hessian is indefinite, possessing both positive and negative eigenvalues ($\exists i, j$ such that $\lambda_i > 0$ and $\lambda_j < 0$). In deep networks, high-dimensional saddle points vastly outnumber local minima.
4. **Ill-Conditioned Ravine:** All $\lambda_i > 0$, but the **condition number** $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} \gg 1$. The surface forms an elongated steep-walled canyon with a very gentle slope along the valley floor.

---

### 4.2 Polyak's Heavy Ball Momentum Formulation

To overcome ill-conditioned ravines and saddle points, Boris Polyak (1964) introduced the **Heavy Ball method**, modeling a particle with mass $m$ experiencing friction:

$$v_t = \gamma v_{t-1} + \eta \nabla f(\theta_{t-1})$$
$$\theta_t = \theta_{t-1} - v_t$$

where $\gamma \in [0, 1)$ is the momentum decay factor (typically $0.9$), and $v_t$ is the accumulated velocity.

#### Velocity as an Exponentially Weighted Moving Average
Unrolling the recurrence from $v_0 = 0$:

$$v_1 = \eta \nabla f(\theta_0)$$
$$v_2 = \gamma \eta \nabla f(\theta_0) + \eta \nabla f(\theta_1)$$
$$v_t = \eta \sum_{\tau=0}^{t-1} \gamma^\tau \nabla f(\theta_{t-1-\tau})$$

The effective memory horizon is approximately $\frac{1}{1 - \gamma}$ steps (e.g., $10$ steps when $\gamma = 0.9$).

#### Mechanism of Action
1. **Valley Acceleration:** Along directions where gradients consistently point in the same direction (along the valley floor), terms accumulate constructively:
   $$v_{\parallel} \to \frac{\eta}{1 - \gamma} \nabla f_{\parallel}$$
   The effective step size is amplified by a factor of $\frac{1}{1 - \gamma} = 10\times$.
2. **Oscillation Damping:** Along directions where gradients alternate signs (bouncing between the steep walls of the ravine), consecutive gradient terms cancel out:
   $$v_{\perp} \to 0$$
   This suppresses transverse oscillations.
3. **Escaping Saddle Points:** Near a saddle point where $\nabla f \approx 0$, vanilla gradient descent crawls with negligible speed. Momentum maintains non-zero velocity accumulated from previous steps, coasting past flat saddle regions.

```
       Navigating an Ill-Conditioned Ravine (f = 0.05x^2 + y^2)
  Vanilla GD:                               Momentum GD:
   \      /\      /\      /                  \
    \    /  \    /  \    /                    \___________
     \  /    \  /    \  /                                 \__________
      \/      \/      \/                                             ======> Minimum
  (Violent cross-wall zigzagging)           (Smooth acceleration along valley floor)
```

---

## 5. Summary & Practitioner's Cheat Sheet

### 5.1 Activation Functions Comparison

| Criteria | Sigmoid $\sigma(x)$ | Tanh $\tanh(x)$ | Swish $x \cdot \sigma(x)$ |
| :--- | :--- | :--- | :--- |
| **Output Bounds** | $(0, 1)$ | $(-1, 1)$ | $[-0.278, \infty)$ |
| **Zero-Centered** | No ($\mathbb{E}[\sigma] > 0$) | Yes | Approximately near origin |
| **Max Derivative** | $0.25$ (at $x=0$) | $1.00$ (at $x=0$) | $\approx 1.099$ (at $x \approx 0.67$) |
| **Saturation Behavior** | Saturates at both $\pm \infty$ | Saturates at both $\pm \infty$ | Saturates only as $x \to -\infty$; linear as $x \to \infty$ |
| **Primary Use Cases** | Binary output classification | Shallow hidden layers, RNN cells | Deep CNNs, Transformers, EfficientNet |

### 5.2 Edge Operators & Discrete Derivatives

| Operator / Difference | Kernel / Stencil | Truncation Error | Primary Advantage |
| :--- | :--- | :---: | :--- |
| **Forward Difference** | $[ -1, 1 ]$ | $\mathcal{O}(h)$ | Simplest discrete implementation |
| **Central Difference** | $[ -0.5, 0, 0.5 ]$ | $\mathcal{O}(h^2)$ | Symmetric; cancels first-order Taylor error |
| **Sobel Horizontal $K_x$** | $\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$ | $\mathcal{O}(h^2)$ | Combines central diff with smoothing; noise-robust |
| **Sobel Vertical $K_y$** | $\begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$ | $\mathcal{O}(h^2)$ | Combines vertical diff with horizontal smoothing |

### 5.3 Optimization Algorithms

| Algorithm | Update Rule | Memory Overhead | Curvature Robustness |
| :--- | :--- | :---: | :--- |
| **Sign-based Stepping** | $x_t = x_{t-1} - \text{sign}(\nabla f) \cdot \Delta s$ | None | Robust on discrete integer lattices |
| **Standard Gradient Descent** | $\theta_t = \theta_{t-1} - \eta \nabla f(\theta_{t-1})$ | None | Poor on ravines ($\kappa \gg 1$); stalls at saddles |
| **Polyak Momentum** | $v_t = \gamma v_{t-1} + \eta \nabla f; \; \theta_t = \theta_{t-1} - v_t$ | $1 \times$ params ($v$) | Accelerates along valleys; dampens transverse oscillations |

---

## 6. References & Further Reading

1. **Ramachandran, P., Zoph, B., & Le, Q. V. (2017).** *Searching for Activation Functions.* arXiv preprint arXiv:1710.05941.
2. **Polyak, B. T. (1964).** *Some methods of speeding up the convergence of iteration methods.* USSR Computational Mathematics and Mathematical Physics, 4(5), 1-17.
3. **Sobel, I., & Feldman, G. (1968).** *A $3 \times 3$ Isotropic Gradient Operator for Image Processing.* Stanford Artificial Intelligence Project.
4. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning.* MIT Press. Chapter 4 (Numerical Computation) and Chapter 8 (Optimization for Training Deep Models).
5. **Nocedal, J., & Wright, S. J. (2006).** *Numerical Optimization.* Springer Science & Business Media.
6. **Nesterov, Y. (1983).** *A method for solving the convex programming problem with convergence rate $O(1/k^2)$.* Soviet Mathematics Doklady, 27(2), 372-376.
