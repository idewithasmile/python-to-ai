# Derivative and its Applications Practice Lab: Numerical Differentiation, Activation Derivatives, Gradient Optimization, and Image Convolution

[Back to Module 02](../README.md) · [Back to the master roadmap](../../README.md)

**Session date:** 2026-11-21 · **Module:** 02 — Mathematics for Machine Learning (Calculus & Linear Algebra)  
**Format:** Textbook-grade study guide, complete analytical derivations, numerical Taylor analysis, optimization hand-trace verification, Sobel convolution mechanics, and three interactive notebooks

---

## Overview

Calculus provides the mathematical language of continuous change and optimization that underpins all modern artificial intelligence. While basic calculus treats differentiation primarily as an analytical procedure on continuous scalar functions $f: \mathbb{R} \to \mathbb{R}$, machine learning applies differential calculus across three distinct computational regimes:

1. **Analytical & Algorithmic Derivatives:** In deep neural networks, non-linear activation functions (such as Sigmoid, Exponential Linear Units, Softplus, and Softsign) break affine subspace collapse. Training these architectures requires computing exact gradients through backpropagation to adjust millions of parameters.
2. **Numerical Calculus & Finite Differences:** In digital computing, analytical limits cannot evaluate infinitesimal intervals $\Delta x \to 0$. Instead, approximations must rely on finite difference schemes (forward, backward, and symmetric central differences), navigating the delicate trade-off between truncation error and catastrophic floating-point cancellation.
3. **Discrete Spatial Calculus & Computer Vision:** Digital images sample physical continuous scenes onto discrete 2D pixel grids $I(x, y)$. Features such as edges, object contours, and textures represent spatial fluctuations in pixel intensity, approximated via discrete difference operators and 2D convolution filters such as the Sobel operator.
4. **Gradient-Based Optimization:** Minimizing objective loss surfaces $\mathcal{L}(\theta)$ requires stepping parameters in the direction opposite to the gradient. Understanding convergence rates, learning rate thresholds, and momentum dynamics is essential for designing robust optimization loops.

This study guide provides a rigorous, textbook-grade synthesis of these four pillars, combining formal mathematical derivations with first-principles Python implementations and verified numerical experiments.

---

## Learning outcomes

By the end of this session, you should be able to:

- Formulate the difference quotient limit definition of the derivative and interpret the geometric rotation of secant lines into tangent lines.
- Classify finite difference schemes—**forward difference** $\mathcal{O}(h)$, **backward difference** $\mathcal{O}(h)$, and **symmetric central difference** $\mathcal{O}(h^2)$—and prove the second-order truncation accuracy of central differences using Taylor series expansions.
- Analyze the floating-point trade-off between truncation error $\mathcal{O}(h^2)$ and round-off cancellation $\mathcal{O}(\epsilon_{\text{mach}} / h)$, identifying the optimal step size $h \approx 10^{-5}$ to $10^{-8}$ for IEEE 754 double precision.
- Derive the analytical derivatives of six core activation functions from first principles: **Sigmoid** ($\sigma'(x) = \sigma(x)(1 - \sigma(x))$), **Exponential** ($e^x$), **PReLU** ($\alpha$ for $x \le 0$, $1$ for $x > 0$), **ELU** ($\alpha e^x$ for $x \le 0$, $1$ for $x > 0$), **Softplus** ($\text{softplus}'(x) = \sigma(x)$), and **Softsign** ($\frac{1}{(1 + |x|)^2}$).
- Construct an automated, relative-error **numerical gradient checker** comparing analytical gradients against central differences with tolerance $\delta < 10^{-5}$.
- Derive and verify the step-by-step numerical trajectory of Gradient Descent on $f(x) = x^2$ with $x_0 = 70.0$ and $\eta = 0.1$, proving the geometric contraction $x_t = (1 - 2\eta)^t x_0$ that yields $x_1 = 56.0$, $x_2 = 44.8$, $x_3 = 35.84$, and $x_4 = 28.672$.
- Categorize optimization dynamics under varying learning rates $\eta$: under-stepping ($0 < \eta < 0.5$), exact single-step convergence ($\eta = 0.5$), damped oscillation ($0.5 < \eta < 1.0$), limit-cycle oscillation ($\eta = 1.0$), and divergent explosion ($\eta > 1.0$).
- Formulate **Polyak's Heavy-Ball Momentum** ($v_t = \gamma v_{t-1} + \eta \nabla f$, $x_t = x_{t-1} - v_t$), proving how velocity buffers accelerate progress along flat valleys by $\frac{1}{1 - \gamma}$ while suppressing transverse oscillations on anisotropic ravines.
- Represent digital images as discrete 2D scalar fields ($p \in [0, 255]$) and implement horizontal differences $G_x$, vertical differences $G_y$, and gradient magnitude $G = \sqrt{G_x^2 + G_y^2}$ using pure-Python nested 2D lists without external dependencies.
- Prove that sliding spatial difference operators over an image grid is mathematically identical to 2D convolution with discrete kernels $K_x = \begin{bmatrix} 0 & 0 & 0 \\ -1 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}$ and $K_y = \begin{bmatrix} 0 & -1 & 0 \\ 0 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}$.
- Derive the $3 \times 3$ **Sobel operator** kernels as the outer product of a 1D discrete central difference filter and an orthogonal 1D triangular smoothing filter ($K_x = [1, 2, 1]^\top [-1, 0, 1]$), and verify manual hand-calculations on image patches ($a = -211 \implies |a| = 211$) against OpenCV's `cv2.Sobel`.

---

## Session Map

| Notebook | Focus | Key Topics & Implementations |
| :--- | :--- | :--- |
| [01 · Numerical Derivatives & Activations](notebooks/01_numerical_derivatives_and_activation_functions.ipynb) | Finite differences, Taylor error analysis, and 6 activation functions | Difference quotient limit, forward vs. backward vs. central differences, Taylor expansion error proof, Sigmoid, Exponential, PReLU, ELU, Softplus, Softsign, gradient checking |
| [02 · Gradient Optimization & Momentum](notebooks/02_gradient_based_optimization_and_momentum.ipynb) | Steepest descent geometry, learning rate analysis, and momentum | Slide hand-trace verification ($x_0 = 70.0 \to 56.0 \to 44.8 \to 35.84 \to 28.672$), geometric contraction ratio, learning rate regimes ($\eta \in [0.05, 1.05]$), Polyak momentum on anisotropic ravines |
| [03 · Image Gradients & Convolution](notebooks/03_image_gradients_finite_differences_and_convolution.ipynb) | Discrete 2D image differentiation, convolution, and Sobel operators | Pure-Python 2D list gradients without convolution, 2D discrete convolution implementation, equivalence proof, Sobel kernels, hand calculation verification ($a = -211$), OpenCV parity on `cat.jpg` |

---

## 1. Mathematical Foundations: Limits, Tangent Lines, and Derivatives

### 1.1 The Physical Intuition of Limits

Consider observing a continuous dynamical physical process—such as a soccer match broadcast where the video transmission briefly drops out at timestamp $t = 2.00$ minutes. Although the frame at exactly $t = 2.00$ is missing, one can estimate the position of the ball with high precision by inspecting adjacent frames:

- At $t = 1.999$, the ball is at position $y = 2.999$ meters.
- At $t = 2.001$, the ball is at position $y = 3.001$ meters.

As observation intervals narrow from both the past ($t \to 2^-$) and the future ($t \to 2^+$), both directional sequences converge toward $y = 3.00$ meters.

```text
Time t:        1.98  ...  1.999  -->  [ 2.00 ]  <--  2.001  ...  2.02
Position y:    2.94  ...  2.999  -->  [ ???  ]  <--  3.001  ...  3.06
                                       |
                               Predicted: ~3.00 m
```

Mathematically, the **limit** of a function $f(x)$ as $x$ approaches $c$ is the value $L$ that $f(x)$ approaches arbitrarily closely, provided $x$ is sufficiently close to $c$ (with $x \neq c$):

$$\lim_{x \to c} f(x) = L \iff \forall \epsilon > 0, \; \exists \delta > 0 \; \text{s.t.} \; 0 < |x - c| < \delta \implies |f(x) - L| < \epsilon$$

A two-sided limit exists if and only if both directional one-sided limits exist and coincide:

$$\lim_{x \to c} f(x) = L \iff \lim_{x \to c^-} f(x) = \lim_{x \to c^+} f(x) = L$$

If the left and right limits disagree—such as in piecewise step functions—the limit does not exist:

$$f(x) = \begin{cases} -x + 5 & \text{if } x \le 2 \\ x - 1 & \text{if } x > 2 \end{cases} \implies \lim_{x \to 2^-} f(x) = 3, \quad \lim_{x \to 2^+} f(x) = 1 \implies \lim_{x \to 2} f(x) \text{ does not exist.}$$

### 1.2 From Secant Lines to the Tangent Line

On a straight line $y = mx + b$, the rate of change is constant everywhere:

$$m = \frac{\Delta y}{\Delta x} = \frac{y_2 - y_1}{x_2 - x_1}$$

On a non-linear curve $y = f(x)$, the rate of change varies continuously from point to point. Choosing two distinct points $A(x, f(x))$ and $B(x + \Delta x, f(x + \Delta x))$ defines a **secant line** with slope:

$$m_{\text{secant}} = \frac{\Delta y}{\Delta x} = \frac{f(x + \Delta x) - f(x)}{\Delta x}$$

As $\Delta x$ decreases toward zero, point $B$ travels along the curve toward point $A$. The secant line pivots around $A$, rotating until it coincides with the **tangent line** at $A$:

$$\lim_{\Delta x \to 0} m_{\text{secant}} = m_{\text{tangent}} = f'(x) = \frac{df}{dx}$$

```text
 y ^                         . B (x+Δx, f(x+Δx))
   |                       .´ /
   |                     .´  /  Secant line (slope = Δy / Δx)
   |                   .´   /
   |             A   .´    /
   |           (x, f(x))  /
   |            *────────/──── Tangent line (instantaneous slope f'(x))
   |           .´
   +─────────────────────────────────────> x
              x         x+Δx
```

### 1.3 Notational Conventions in Calculus

Different scientific disciplines employ distinct notations for the derivative:

| Notation | Creator | Expression | Common Domain |
| :--- | :--- | :--- | :--- |
| **Leibniz** | Gottfried Wilhelm Leibniz | $\frac{df}{dx}, \frac{d}{dx}f(x), \frac{d^2 f}{dx^2}$ | Differential equations, chain rule, physics |
| **Lagrange** | Joseph-Louis Lagrange | $f'(x), f''(x), f^{(n)}(x)$ | Pure mathematics, analysis, optimization |
| **Euler** | Leonhard Euler | $D_x f, D^2 f$ | Operator theory, functional analysis |
| **Newton** | Sir Isaac Newton | $\dot{x}, \ddot{x}$ | Classical mechanics (derivatives with respect to time $t$) |

In machine learning, Leibniz and Lagrange notations predominate: Lagrange's prime notation $f'(x)$ is preferred for 1D scalar activations, while Leibniz's partial fraction notation $\frac{\partial \mathcal{L}}{\partial W}$ is standard for multi-parameter network weights.

---

## 2. Numerical Differentiation & Taylor Truncation Analysis

### 2.1 The Three Core Finite Difference Schemes

In digital computation, evaluating derivatives numerically requires selecting a non-zero step size $h$:

1. **Forward Difference (One-Sided):**
   $$D_+ f(x) = \frac{f(x + h) - f(x)}{h}$$

2. **Backward Difference (One-Sided):**
   $$D_- f(x) = \frac{f(x) - f(x - h)}{h}$$

3. **Central Difference (Symmetric):**
   $$D_0 f(x) = \frac{f(x + h) - f(x - h)}{2h}$$

### 2.2 Proof of Error Order via Taylor Series Expansions

To determine the approximation error of each scheme, expand $f(x + h)$ and $f(x - h)$ as Taylor series around $x$:

$$f(x + h) = f(x) + h f'(x) + \frac{h^2}{2!} f''(x) + \frac{h^3}{3!} f'''(x) + \mathcal{O}(h^4)$$

$$f(x - h) = f(x) - h f'(x) + \frac{h^2}{2!} f''(x) - \frac{h^3}{3!} f'''(x) + \mathcal{O}(h^4)$$

#### Forward Difference Error:
Subtracting $f(x)$ from the expansion of $f(x + h)$:

$$f(x + h) - f(x) = h f'(x) + \frac{h^2}{2} f''(x) + \mathcal{O}(h^3)$$

Dividing by $h$:

$$\frac{f(x + h) - f(x)}{h} = f'(x) + \underbrace{\frac{h}{2} f''(x) + \mathcal{O}(h^2)}_{\text{Truncation error } \mathcal{O}(h)}$$

The error decays linearly with $h$ (first-order accuracy).

#### Central Difference Error:
Subtracting the expansion of $f(x - h)$ from $f(x + h)$:

$$f(x + h) - f(x - h) = 2h f'(x) + \frac{2h^3}{6} f'''(x) + \mathcal{O}(h^5)$$

Notice that all even-powered terms ($\frac{h^2}{2} f''(x)$, $\frac{h^4}{24} f^{(4)}(x)$) cancel out identically due to anti-symmetry! Dividing both sides by $2h$:

$$\frac{f(x + h) - f(x - h)}{2h} = f'(x) + \underbrace{\frac{h^2}{6} f'''(x) + \mathcal{O}(h^4)}_{\text{Truncation error } \mathcal{O}(h^2)}$$

The error decays quadratically with $h$ (second-order accuracy). For $h = 10^{-4}$, the forward difference error is $\approx 10^{-4}$, while the central difference error is $\approx 10^{-8}$—a four order-of-magnitude precision improvement.

### 2.3 The Floating-Point Dilemma: Truncation vs. Catastrophic Cancellation

While analytical theory suggests choosing $h \to 0$, digital hardware evaluates floating-point numbers with finite precision (IEEE 754 64-bit float has machine epsilon $\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$). When $h$ becomes excessively small:

1. **Truncation Error:** Decreases as $h \to 0$ ($\sim \mathcal{O}(h^2)$).
2. **Round-Off Error (Catastrophic Cancellation):** Increases as $h \to 0$ ($\sim \frac{\epsilon_{\text{mach}}}{h}$), because subtracting two nearly identical numbers $f(x + h) - f(x - h)$ loses most significant digits.

Total numerical error follows a U-shaped curve:

$$\text{Total Error}(h) \approx C_1 h^2 + C_2 \frac{\epsilon_{\text{mach}}}{h}$$

Setting the derivative with respect to $h$ to zero:

$$2 C_1 h - C_2 \frac{\epsilon_{\text{mach}}}{h^2} = 0 \implies h^* \approx \sqrt[3]{\epsilon_{\text{mach}}} \approx (10^{-16})^{1/3} \approx 10^{-5} \text{ to } 10^{-6}$$

For double precision, $h = 10^{-5}$ represents the optimal sweet spot for numerical gradient checking.

---

## 3. Core Activation Functions & Analytical Derivatives

### 3.1 Why Non-Linear Activation Functions Are Necessary

Consider an $L$-layer multi-layer perceptron where each layer computes an affine linear transformation $z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$. If no non-linear activation is applied ($a^{[l]} = z^{[l]}$):

$$a^{[L]} = W^{[L]} (W^{[L-1]} (\dots (W^{[1]} x + b^{[1]}) \dots) + b^{[L-1]}) + b^{[L]}$$

Because the product of linear transformation matrices is itself another linear transformation matrix:

$$\widetilde{W} = W^{[L]} W^{[L-1]} \dots W^{[1]}, \qquad \widetilde{b} = W^{[L]} \dots b^{[1]} + \dots + b^{[L]}$$

The entire deep architecture collapses into a single-layer linear model $\hat{y} = \widetilde{W} x + \widetilde{b}$, incapable of solving non-linear problems such as XOR or modeling high-dimensional image manifolds. Element-wise non-linear activations $\phi(z)$ shatter linear subspace collapse.

---

### 3.2 Analytical Derivations of Six Core Activations

#### 1. Sigmoid Function
$$\sigma(x) = \frac{1}{1 + e^{-x}} = (1 + e^{-x})^{-1}$$

Differentiating via the chain and power rules:
$$\sigma'(x) = -(1 + e^{-x})^{-2} \cdot \frac{d}{dx}(1 + e^{-x}) = -(1 + e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1 + e^{-x})^2}$$

Decomposing the fraction into a product:
$$\sigma'(x) = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} = \frac{1}{1 + e^{-x}} \cdot \left(\frac{1 + e^{-x} - 1}{1 + e^{-x}}\right) = \sigma(x) (1 - \sigma(x))$$

- **Maximum derivative:** $\sigma'(0) = 0.5 \times (1 - 0.5) = 0.25$.
- **Vanishing gradient hazard:** For $|x| > 4$, $\sigma'(x) \to 0$. Multiplying many sub-0.25 factors across layers causes gradients to vanish exponentially during backpropagation.

#### 2. Exponential Function
$$f(x) = e^x, \qquad f'(x) = e^x$$

- **Characteristics:** Strictly positive, unbounded as $x \to \infty$, rapidly decays to $0$ as $x \to -\infty$.

#### 3. Parametric ReLU (PReLU)
$$\text{PReLU}(x) = \begin{cases} \alpha x & \text{if } x < 0 \\ x & \text{if } x \ge 0 \end{cases}$$

Differentiating each piecewise interval:
$$\text{PReLU}'(x) = \begin{cases} \alpha & \text{if } x \le 0 \\ 1 & \text{if } x > 0 \end{cases}$$

- With $\alpha = 0.1$, negative activations retain a non-zero gradient ($0.1$), eliminating the "dying ReLU" failure mode where neurons become permanently inactive.

#### 4. Exponential Linear Unit (ELU)
$$\text{ELU}(x) = \begin{cases} \alpha(e^x - 1) & \text{if } x \le 0 \\ x & \text{if } x > 0 \end{cases}$$

Differentiating:
$$\text{ELU}'(x) = \begin{cases} \alpha e^x & \text{if } x \le 0 \\ 1 & \text{if } x > 0 \end{cases}$$

- For $x \le 0$, the derivative smoothly scales with $e^x$, bringing the mean activation closer to zero while preserving gradient flow.

#### 5. Softplus Function
$$\text{softplus}(x) = \log(1 + e^x)$$

Differentiating using the chain rule:
$$\text{softplus}'(x) = \frac{1}{1 + e^x} \cdot \frac{d}{dx}(1 + e^x) = \frac{e^x}{1 + e^x} = \frac{1}{1 + e^{-x}} = \sigma(x)$$

- **Mathematical beauty:** The exact derivative of the smooth Softplus curve is the Sigmoid activation! Softplus serves as a smooth, differentiable surrogate for standard ReLU ($\max(0, x)$).

#### 6. Softsign Function
$$\text{softsign}(x) = \frac{x}{1 + |x|}$$

Derivation:
- For $x > 0$: $f(x) = \frac{x}{1 + x} \implies f'(x) = \frac{(1+x)(1) - x(1)}{(1+x)^2} = \frac{1}{(1+x)^2}$
- For $x < 0$: $f(x) = \frac{x}{1 - x} \implies f'(x) = \frac{(1-x)(1) - x(-1)}{(1-x)^2} = \frac{1-x+x}{(1-x)^2} = \frac{1}{(1-x)^2}$

Unifying both branches:
$$\text{softsign}'(x) = \frac{1}{(1 + |x|)^2}$$

- Compared to Tanh, Softsign approaches its horizontal asymptotes ($\pm 1$) polynomially rather than exponentially, mitigating gradient saturation in deeper regimes.

---

### 3.3 Comparative Summary Table of Activation Functions

| Activation | Analytical Formula $f(x)$ | Analytical Derivative $f'(x)$ | Output Range | Derivative Max | Saturation Property |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Sigmoid** | $\frac{1}{1 + e^{-x}}$ | $f(x)(1 - f(x))$ | $(0, 1)$ | $0.25$ at $x=0$ | Saturates as $x \to \pm\infty$ |
| **Exponential** | $e^x$ | $e^x$ | $(0, \infty)$ | Unbounded | Explodes as $x \to \infty$ |
| **PReLU ($\alpha=0.1$)** | $\max(x, \alpha x)$ | $1$ if $x > 0$ else $\alpha$ | $(-\infty, \infty)$ | $1.0$ for $x > 0$ | No saturation; constant slope |
| **ELU ($\alpha=0.1$)** | $x$ if $x > 0$ else $\alpha(e^x - 1)$ | $1$ if $x > 0$ else $\alpha e^x$ | $(-\alpha, \infty)$ | $1.0$ for $x > 0$ | Smooth negative saturation |
| **Softplus** | $\log(1 + e^x)$ | $\sigma(x)$ | $(0, \infty)$ | $1.0$ as $x \to \infty$ | Smooth non-saturating positive |
| **Softsign** | $\frac{x}{1 + |x|}$ | $\frac{1}{(1 + |x|)^2}$ | $(-1, 1)$ | $1.0$ at $x=0$ | Polynomial tail saturation |

---

## 4. Gradient-Based Optimization & Convergence Dynamics

### 4.1 Optimization Geometry

Consider a differentiable scalar objective function $f(x)$. The derivative $f'(x)$ determines the slope of the tangent line:
- If $f'(x) > 0$: The function increases as $x$ increases. To decrease $f(x)$, we must move to the left by subtracting a positive quantity.
- If $f'(x) < 0$: The function decreases as $x$ increases. To decrease $f(x)$, we must move to the right by subtracting a negative quantity (adding).

Thus, moving opposite to the gradient guarantees descent for sufficiently small step sizes:

$$x_{t+1} = x_t - \eta f'(x_t)$$

```text
       f(x) ^               .
            |              / \  Slope f'(x) > 0  --> Move LEFT
            |             /   \
            |    Slope < 0     \
            |    --> Move RIGHT \
            |        \          /
            |         \________/  <-- Minimum f'(x*) = 0
            +───────────────────────────> x
```

---

### 4.2 Recreating the Lecture Slide Trace: Minimizing $f(x) = x^2$

Let $f(x) = x^2$ on $-100 \le x \le 100$.
- Analytical derivative: $f'(x) = 2x$.
- Initial parameter: $x_0 = 70.0$.
- Learning rate: $\eta = 0.1$.

Let us evaluate the step-by-step recurrence:
- **Step 0:**
  $$f'(x_0) = 2(70.0) = 140.0$$
  $$x_1 = x_0 - \eta f'(x_0) = 70.0 - 0.1 \times 140.0 = 70.0 - 14.0 = \mathbf{56.0}$$
  
- **Step 1:**
  $$f'(x_1) = 2(56.0) = 112.0$$
  $$x_2 = x_1 - \eta f'(x_1) = 56.0 - 0.1 \times 112.0 = 56.0 - 11.2 = \mathbf{44.8}$$
  
- **Step 2:**
  $$f'(x_2) = 2(44.8) = 89.6$$
  $$x_3 = x_2 - \eta f'(x_2) = 44.8 - 0.1 \times 89.6 = 44.8 - 8.96 = \mathbf{35.84}$$
  
- **Step 3:**
  $$f'(x_3) = 2(35.84) = 71.68$$
  $$x_4 = x_3 - \eta f'(x_3) = 35.84 - 0.1 \times 71.68 = 35.84 - 7.168 = \mathbf{28.672}$$

These values reproduce the exact numerical outputs in the lecture slides:

| Iteration $t$ | Parameter $x_t$ | Derivative $f'(x_t) = 2x_t$ | Update Step $\Delta x = \eta f'$ | Objective Value $f(x_t) = x_t^2$ |
| :---: | :---: | :---: | :---: | :---: |
| **0** | $70.0000$ | $140.0000$ | $14.0000$ | $4900.000$ |
| **1** | $56.0000$ | $112.0000$ | $11.2000$ | $3136.000$ |
| **2** | $44.8000$ | $89.6000$ | $8.9600$ | $2007.040$ |
| **3** | $35.8400$ | $71.6800$ | $7.1680$ | $1284.506$ |
| **4** | $28.6720$ | $57.3440$ | $5.7344$ | $822.084$ |

---

### 4.3 General Closed-Form Solution and Learning Rate Regimes

For any quadratic function $f(x) = a x^2$ ($a > 0$), substituting $f'(x) = 2ax$ into the update rule gives:

$$x_{t+1} = x_t - \eta(2a x_t) = (1 - 2a\eta) x_t$$

Unrolling this geometric recurrence across $t$ steps:

$$x_t = (1 - 2a\eta)^t x_0$$

The convergence behavior is entirely governed by the contraction multiplier $\rho = 1 - 2a\eta$:

```text
       Contraction factor ρ = 1 - 2aη
       <─────── Divergent ────────><───── Oscillation ─────><──── Monotonic ────><── Slow ──>
      -inf                        -1                        0                    1         +inf
                                   |                        |                    |
                               η = 1/a                   η = 1/(2a)            η = 0
```

1. **Under-Stepping ($0 < \eta < \frac{1}{2a}$, for $a=1$: $\eta < 0.5$):**
   $0 < \rho < 1$. Monotonic geometric decay toward $0$. As $\eta \to 0$, convergence slows.
2. **Exact Dead-Center Convergence ($\eta = \frac{1}{2a}$, for $a=1$: $\eta = 0.5$):**
   $\rho = 1 - 2(1)(0.5) = 0$. $x_1 = 0 \cdot x_0 = 0$. Reaches the exact minimum in a **single step**!
3. **Damped Oscillation ($\frac{1}{2a} < \eta < \frac{1}{a}$, for $a=1$: $0.5 < \eta < 1.0$):**
   $-1 < \rho < 0$. Alternates signs across the origin ($x_t$ flips positive/negative) while converging in absolute magnitude toward $0$.
4. **Limit Cycle Oscillation ($\eta = \frac{1}{a}$, for $a=1$: $\eta = 1.0$):**
   $\rho = -1$. $x_t = (-1)^t x_0$. Parameter bounces endlessly between $+70.0$ and $-70.0$ without ever converging.
5. **Divergent Explosion ($\eta > \frac{1}{a}$, for $a=1$: $\eta > 1.0$):**
   $|\rho| > 1$. Sequence diverges exponentially to $\pm \infty$.

---

### 4.4 Momentum Dynamics: Polyak's Heavy Ball

When optimizing multi-dimensional functions where the loss surface curves much more steeply along some directions than others (anisotropic quadratic bowls or ravines), standard Gradient Descent suffers from severe transverse oscillations.

**Polyak's Heavy-Ball Momentum** models a physical particle of mass moving through a viscous fluid:

$$v_t = \gamma v_{t-1} + \eta \nabla f(x_{t-1})$$
$$x_t = x_{t-1} - v_t$$

where $v_t$ is the velocity vector and $\gamma \in [0, 1)$ is the momentum decay factor (typically $0.9$).

#### Analytical Acceleration:
Unrolling the velocity recurrence:

$$v_t = \eta \sum_{k=0}^{t-1} \gamma^k \nabla f(x_{t-1-k})$$

The velocity is an exponentially weighted moving average of past gradients:
- Along directions where gradients maintain a consistent sign (the valley floor), velocities accumulate constructively, increasing the effective step size by a factor of:
  $$\text{Effective Step Multiplier} = \sum_{k=0}^{\infty} \gamma^k = \frac{1}{1 - \gamma}$$
  For $\gamma = 0.9$, convergence along the valley accelerates by $10\times$!
- Along directions where gradients oscillate and flip signs (the steep walls), successive terms cancel out destructively, dampening high-frequency oscillations.

---

## 5. Image Gradients, Discrete Spatial Calculus & 2D Convolution

### 5.1 Digital Images as Discrete 2D Grids

A digital image samples light intensity on a discrete 2D grid:
- **Grayscale Image:** $I \in [0, 255]^{H \times W}$, where $H$ denotes height (rows) and $W$ denotes width (columns). Each pixel $p = I(i, j)$ is an 8-bit unsigned integer scalar.
- **Color Image:** $I \in [0, 255]^{H \times W \times 3}$, where each pixel is a 3-element vector $[R, G, B]^\top$.

```text
      (0,0) ─────── Column j (Width W) ───────> (0, W-1)
        |
        |      Pixel I[i, j]
        |      Neighborhood:
        |           (i-1, j)
     Row i    (i, j-1)  (i, j)  (i, j+1)
        |           (i+1, j)
        v
      (H-1, 0) ───────────────────────────────> (H-1, W-1)
```

### 5.2 Discrete Finite Differences on Images (Without Convolution)

Because pixel coordinates $(i, j)$ exist only on an integer lattice, spatial derivatives cannot evaluate limits. Instead, finite difference approximations are applied along orthogonal grid directions with unit pixel spacing:

- **Horizontal Derivative ($X$-Direction / Column Difference):**
  $$G_x(i, j) = I(i, j+1) - I(i, j-1)$$

- **Vertical Derivative ($Y$-Direction / Row Difference):**
  $$G_y(i, j) = I(i+1, j) - I(i-1, j)$$

- **Gradient Magnitude:**
  $$G(i, j) = \sqrt{G_x(i, j)^2 + G_y(i, j)^2}$$

- **Gradient Orientation:**
  $$\theta(i, j) = \arctan2(G_y(i, j), G_x(i, j))$$

In pure Python, these spatial differences are computed via nested loops across the interior pixels $i \in [1, H-2]$ and $j \in [1, W-2]$:

```python
def compute_x_derivative(image):
    h, w = len(image), len(image[0])
    x_derivative = [[0.0] * w for _ in range(h)]
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            x_derivative[i][j] = float(image[i][j + 1] - image[i][j - 1])
    return x_derivative
```

---

### 5.3 Equivalence to 2D Discrete Convolution

Direct spatial differences can be formulated as a **2D discrete convolution** (or cross-correlation) with small difference kernels. For an image $I$ and kernel $K \in \mathbb{R}^{(2k+1) \times (2k+1)}$:

$$(I * K)(i, j) = \sum_{u=-k}^k \sum_{v=-k}^k K(u, v) I(i + u, j + v)$$

Defining the central difference kernels:

$$K_x = \begin{bmatrix} 0 & 0 & 0 \\ -1 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}, \qquad K_y = \begin{bmatrix} 0 & -1 & 0 \\ 0 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}$$

Applying $K_x$ to a $3 \times 3$ window around $(i, j)$:

$$(I * K_x)(i, j) = (-1) \cdot I(i, j-1) + (0) \cdot I(i, j) + (1) \cdot I(i, j+1) = I(i, j+1) - I(i, j-1) \equiv G_x(i, j)$$

Direct neighborhood indexing and 2D convolution produce mathematically identical outputs!

---

### 5.4 The Sobel Operator

Simple difference kernels are highly susceptible to sensor noise because high-frequency noise creates large artificial pixel differences. The **Sobel operator** overcomes this by combining discrete differentiation along one axis with triangular Gaussian-like smoothing along the orthogonal axis:

$$K_x = \underbrace{\begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix}}_{\text{Smoothing in } y} \underbrace{\begin{bmatrix} -1 & 0 & 1 \end{bmatrix}}_{\text{Difference in } x} = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$$

$$K_y = \underbrace{\begin{bmatrix} -1 \\ 0 \\ 1 \end{bmatrix}}_{\text{Difference in } y} \underbrace{\begin{bmatrix} 1 & 2 & 1 \end{bmatrix}}_{\text{Smoothing in } x} = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

Because each Sobel kernel is an outer product of two 1D vectors ($K = u v^\top$), the filter is **rank-1 separable**. Convolving with two 1D filters requires only $3 + 3 = 6$ multiplications per pixel instead of $3 \times 3 = 9$ multiplications, offering significant computational savings on large images.

---

### 5.5 Verification of Lecture Slide Hand Calculation (Slide 69)

Consider the sample image patch presented in the lecture slides:

$$\text{Patch} = \begin{bmatrix} 230 & \cdot & 147 \\ 237 & \cdot & 188 \\ 210 & \cdot & 180 \end{bmatrix}$$

Convolving with the Sobel horizontal kernel $K_x$:

$$a = (-1 \times 230) + (1 \times 147) + (-2 \times 237) + (2 \times 188) + (-1 \times 210) + (1 \times 180)$$
$$a = -230 + 147 - 474 + 376 - 210 + 180$$
$$a = -211$$

The absolute gradient response is:

$$f(a) = |a| = |-211| = \mathbf{211}$$

This confirms the exact hand-calculation mechanics presented in the lecture curriculum.

---

## 6. Implementation Summary & Verification Checklist

To ensure full numerical and code quality compliance, this practice lab implements and validates:

- [x] **Finite difference schemes:** Verified that central difference achieves $\mathcal{O}(h^2)$ second-order accuracy.
- [x] **6 Activation functions:** Complete analytical derivatives for Sigmoid, Exponential, PReLU, ELU, Softplus, and Softsign.
- [x] **Gradient checking:** Automated relative error checks with $\delta < 10^{-5}$ across all activations.
- [x] **Optimization trace verification:** Verified the exact lecture slide trace ($x_0 = 70.0 \to 56.0 \to 44.8 \to 35.84 \to 28.672$) on $f(x) = x^2$.
- [x] **Learning rate dynamics:** Demonstrated stability conditions $\eta < \frac{1}{a}$, single-step convergence at $\eta = \frac{1}{2a}$, and divergence for $\eta > \frac{1}{a}$.
- [x] **Momentum acceleration:** Verified that Polyak momentum dampens transverse oscillations on ill-conditioned ravines.
- [x] **Discrete image calculus:** Pure-Python 2D list implementations of $G_x$, $G_y$, and gradient magnitude $G = \sqrt{G_x^2 + G_y^2}$.
- [x] **Convolution equivalence:** Proved equivalence between direct discrete differencing and 2D convolution.
- [x] **Sobel operator mechanics:** Verified slide 69 manual calculation ($a = -211 \implies |a| = 211$) and benchmarked parity against OpenCV `cv2.Sobel`.

---

## References

1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). *Deep Learning*. Nature, 521(7553), 436–444.
2. Polyak, B. T. (1964). *Some methods of speeding up the convergence of iteration methods*. USSR Computational Mathematics and Mathematical Physics, 4(5), 1–17.
3. Sobel, I., & Feldman, G. (1968). *A $3 \times 3$ Isotropic Gradient Operator for Image Processing*. Stanford Artificial Intelligence Project.
4. Burden, R. L., & Faires, J. D. (2011). *Numerical Analysis* (9th ed.). Brooks/Cole, Cengage Learning.
5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
