# Basic Python: Applied Exercises and Algorithmic Practice Lab

[Back to Module 01](../README.md) · [Back to the master roadmap](../../README.md)

**Session date:** 2026-10-24 · **Module:** 01 — Python Fundamentals & Algorithmic Thinking  
**Format:** Comprehensive study guide, mathematical foundations, formal specifications, and executable practice notebooks

## Overview

Writing reliable code for computational science and artificial intelligence requires more than syntactic literacy. It demands defensive input validation, numerical awareness, algorithmic discipline, and a deep appreciation for the trade-offs between exact analytical formulations and computational approximations.

This practice lab bridges fundamental Python syntax and practical software engineering for machine learning. Across seven core domains, we transition from elementary console I/O mechanics to non-linear model activations, classification performance metrics, regression loss functions, combinatorial algorithms, Taylor series polynomial expansions, and low-level arithmetic manipulation.

Key areas explored in this session:

1. **Standard stream I/O and modern formatting:** Deconstructing `sys.stdout` and `sys.stderr`, buffer flushing mechanics (`flush=True`), custom delimiters (`sep`), line terminators (`end`), and formatted string literals (f-strings, PEP 498).
2. **Geometric solid modeling and numerical constants:** Parameterizing 3D Euclidean volumes (cubes, rectangular prisms, cylinders, spheres), analyzing floating-point truncation error ($\pi \approx 3.14$ vs. IEEE 754 `math.pi`), and defensive parameter assertions.
3. **Classification model evaluation:** Constructing the $2 \times 2$ Confusion Matrix, formulating Precision and Recall, proving mathematically why the harmonic mean ($F_1$-score) is required over the arithmetic mean in skewed distributions, and enforcing strict type and boundary validations.
4. **Neural network activation functions:** Examining why non-linearities are mathematically required in deep learning architectures, analyzing the mathematical properties and gradients of Binary Step, Sigmoid, and Exponential Linear Unit (ELU) activations, and implementing dynamic dispatchers with defensive numerical parsing.
5. **Regression loss functions and Monte Carlo error analysis:** Formulating Absolute Error (AE), Squared Error (SE), Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE); simulating stochastic prediction pipelines; and comparing L1 vs. L2 robustness to outliers.
6. **Combinatorics, sequences, and recurrence relations:** Deriving Pascal's triangle through multiplicative step ratios ($\binom{n}{r+1} = \binom{n}{r} \frac{n-r}{r+1}$) to avoid full factorial evaluation; and implementing the Fibonacci recurrence in $O(n)$ time and $O(1)$ space.
7. **Taylor series expansions and transcendental approximations:** Deriving Maclaurin polynomial series for trigonometric ($\sin, \cos$) and hyperbolic ($\sinh, \cosh$) functions, analyzing truncation error, and comparing against standard library implementations.
8. **Integer arithmetic and positional number theory:** Reversing decimal digits using pure modulo and integer division arithmetic without string allocation, analyzing time complexity ($O(\log_{10} n)$) and invariants.

## Learning outcomes

By the end of this session, you should be able to:

- Master `print()` parameters (`*objects`, `sep`, `end`, `file`, `flush`) to build structured console logs and real-time terminal output.
- Apply modern f-string format specifications for floating-point precision, integer padding, scientific notation, and inline debug expressions (`{var=}`).
- Implement parameterized functions for geometric solid volumes with input assertions and evaluate constant truncation errors.
- Construct a $2 \times 2$ Confusion Matrix and compute Precision, Recall, and the $F_1$-score.
- Explain mathematically why the harmonic mean prevents metric distortion under class imbalance.
- Implement strict type checking (`type(x) is int`) and boundary verification ($x > 0$) with graceful error reporting.
- Formulate the mathematical equations, ranges, and derivatives of Binary Step, Sigmoid, and ELU activations.
- Build safe numerical string validators (`is_number`) that detect float castability without unexpected crashes.
- Compute AE, SE, MAE, MSE, and RMSE, and explain the differing sensitivities of L1 and L2 losses to anomalous outliers.
- Generate synthetic data using Monte Carlo uniform sampling and execute step-by-step error tracking.
- Derive the ratio formula for binomial coefficients in Pascal's triangle and generate arbitrary rows in $O(n)$ arithmetic operations.
- Implement iterative linear recurrence algorithms with $O(1)$ auxiliary memory.
- Approximate transcendental functions using Taylor series expansions and quantify numerical convergence vs. expansion order $N$.
- Reverse positive integer digits using pure base-10 arithmetic operations ($n \pmod{10}$ and $n // 10$).

## Session map

| Notebook | Focus | Key Topics |
| :--- | :--- | :--- |
| [01 — Print & Formatting](notebooks/01-print-and-formatting.ipynb) | Stream I/O & Modern Formatting | `print()` signature, `sep`, `end`, `file`, `flush`, buffer flushing, PEP 498 f-string specifiers |
| [02 — Geometric Volumes](notebooks/02-geometric-volume-functions.ipynb) | Solid Geometry Modeling | Cube, rectangular prism, cylinder, sphere, $\pi$ precision analysis, dataclass encapsulation |
| [03 — Classification F1-Score](notebooks/03-classification-f1-score.ipynb) | Evaluation Metrics | Confusion matrix, Precision, Recall, harmonic mean $F_1$, strict integer type & range verification |
| [04 — Activation Functions](notebooks/04-activation-functions.ipynb) | Neural Non-Linearities | Binary Step, Sigmoid, ELU ($\alpha=0.01$), `is_number()` validator, dynamic dispatcher, Matplotlib plots |
| [05 — Regression Losses](notebooks/05-regression-losses-and-simulation.ipynb) | Error Metrics & Simulation | AE, SE, MAE, MSE, RMSE, Monte Carlo uniform sampling, outlier sensitivity analysis |
| [06 — Combinatorics & Series Lab](notebooks/06-combinatorics-and-series-lab.ipynb) | Algorithmic Problem Solving | Pascal's triangle via ratio updates, $O(1)$-space Fibonacci, Taylor polynomial approximations ($\sin, \cos, \sinh, \cosh$), arithmetic digit reversal |

---

## 1. Python Stream I/O and Formatting Mechanics

### 1.1 The Anatomy of `print()`

Python's built-in `print()` is the primary interface to the standard output stream. While frequently used with simple arguments, its full signature offers precise control:

```python
print(*objects, sep=" ", end="\n", file=sys.stdout, flush=False)
```

1. **`*objects` (Arbitrary Argument Tuple):**
   Any number of positional arguments can be passed. Python automatically converts each argument to a string by invoking its `__str__()` method. If an object does not define `__str__()`, Python falls back to its `__repr__()`.

2. **`sep` (Object Separator):**
   A string specifying what character sequence is inserted between consecutive objects. By default, `sep=' '` (a single whitespace). Setting `sep='/'` is useful for constructing file paths or date representations, while `sep=''` concatenates items directly.

3. **`end` (Stream Terminator):**
   A string appended to the stream after the final object has been written. Defaults to `'\\n'` (standard newline). Setting `end=' '` allows subsequent `print()` invocations to continue on the same terminal line, which is useful for loop progress indicators.

4. **`file` (Output Stream):**
   An object implementing a `.write(str)` method. The default is `sys.stdout`. By reassigning `file=sys.stderr`, error messages and warnings can be directed to the standard error stream, ensuring they remain visible even if `stdout` is redirected to a log file.

5. **`flush` (Buffer Flush Control):**
   Operating systems buffer I/O operations to minimize costly system calls. Under line buffering, characters accumulate in memory until a newline character (`'\\n'`) is emitted. Setting `flush=True` bypasses buffering and forces immediate transmission to the kernel device driver.

```text
+---------------------------------------------------------------+
|                       print(*objects)                         |
+---------------------------------------------------------------+
                               |
         +---------------------+---------------------+
         |                                           |
         v                                           v
+------------------+                       +-------------------+
| Interleave 'sep' |                       |  Append 'end'     |
| between objects  |                       |  after last item  |
+------------------+                       +-------------------+
                               |
                               v
               +-------------------------------+
               | Write to destination 'file'   |
               | (default: sys.stdout)         |
               +-------------------------------+
                               |
                               v
                     Is 'flush == True'?
                     /                 \
                   YES                  NO
                   /                     \
       +--------------------+    +--------------------+
       | Force immediate    |    | Buffer in memory   |
       | OS write flush     |    | until next newline |
       +--------------------+    +--------------------+
```

### 1.2 Formatted String Literals (f-strings)

Introduced in PEP 498 (Python 3.6), f-strings provide concise syntax and execution speed superior to legacy `%` formatting and `str.format()`.

```python
# Format specification syntax: {expression:format_spec}
value = 123.456789
print(f"Standard float:      {value}")
print(f"Rounded (4 decimals): {value:.4f}")
print(f"Scientific notation:  {value:.2e}")
print(f"Padded (width 10):    {value:10.2f}")
print(f"Percentage:           {0.8765:.2%}")
```

Key format specifiers for machine learning and scientific computing:
- `:.Nf`: Fixed-point real number rounded to $N$ decimal places.
- `:0Nd`: Zero-padded integer filled to minimum width $N$ (e.g., `{epoch:03d}`).
- `:.Ne`: Exponential scientific notation with $N$ decimal digits in the mantissa.
- `:.N%`: Multiplies the value by 100, formats as float with $N$ decimals, and appends a `%` sign.
- `{x=}`: Self-documenting debugging specifier that prints the expression text followed by its evaluated value.

---

## 2. Geometric Solid Modeling and Numerical Precision

### 2.1 Euclidean Volume Formulations

Solid geometry problems appear regularly in computer graphics, 3D perception, and physical simulations. The four fundamental regular solids are:

1. **Cube:**
   A regular hexahedron whose three spatial dimensions are identical ($s = l = w = h$):
   $$V_{\text{cube}}(s) = s^3$$

2. **Rectangular Prism (Cuboid):**
   A six-faced polyhedron bounded by rectangular faces with orthogonal side lengths $l$, $w$, and $h$:
   $$V_{\text{prism}}(l, w, h) = l \cdot w \cdot h$$

3. **Right Circular Cylinder:**
   A ruled surface bounded by two parallel circular bases of radius $r$ separated by height $h$:
   $$V_{\text{cylinder}}(r, h) = \pi r^2 h$$

4. **Sphere:**
   The locus of all points in 3D Euclidean space equidistant from a central point by radius $r$:
   $$V_{\text{sphere}}(r) = \frac{4}{3} \pi r^3$$

### 2.2 Numerical Constant Precision and Error Propagation

When computing non-linear spatial measures involving transcendental numbers like $\pi$, the choice of constant directly affects numerical accuracy:
- Crude approximation: $\pi_0 = 3.14$
- Standard 64-bit IEEE 754 float: `math.pi` $= 3.141592653589793$

The relative approximation error in $\pi$ is:
$$\epsilon_{\pi} = \frac{|\pi - \pi_0|}{\pi} = \frac{|3.141592653589793 - 3.14|}{3.141592653589793} \approx 0.0507\%$$

Because volume functions scale linearly with $\pi$ ($V \propto \pi$), relative error propagates directly into the calculated volume:
$$\frac{|V_{\text{exact}} - V_{\text{approx}}|}{V_{\text{exact}}} = \epsilon_{\pi} \approx 5.07 \times 10^{-4}$$

For a sphere with radius $r = 100 \text{ m}$:
$$V_{\text{exact}} = \frac{4}{3} \pi (100)^3 \approx 4{,}188{,}790.20 \text{ m}^3$$
$$V_{\text{approx}} = \frac{4}{3} (3.14) (100)^3 \approx 4{,}186{,}666.67 \text{ m}^3$$
$$\Delta V \approx 2{,}123.53 \text{ m}^3$$

In high-precision applications (such as CAD, aerospace engineering, or scientific computing), this discrepancy is unacceptable, emphasizing the need for standard library constants like `math.pi`.

---

## 3. Classification Model Evaluation: Precision, Recall, and F1-Score

### 3.1 The Binary Confusion Matrix

In binary classification, model predictions are partitioned into four mutually exclusive outcomes against ground truth:

```text
                           ACTUAL CONDITION
                       Positive            Negative
PREDICTED   Positive | True Positive (TP)  | False Positive (FP) | -> Total Predicted Positive (TP + FP)
CONDITION   Negative | False Negative (FN) | True Negative (TN)  | -> Total Predicted Negative (FN + TN)
                       -----------------------------------------
                     | Total Actual Pos    | Total Actual Neg    |
                     | (TP + FN)           | (FP + TN)           |
```

- **True Positive ($TP$):** An instance correctly predicted as positive.
- **False Positive ($FP$):** A negative instance incorrectly labeled as positive (Type I error).
- **False Negative ($FN$):** A positive instance incorrectly labeled as negative (Type II error).
- **True Negative ($TN$):** An instance correctly predicted as negative.

### 3.2 Precision, Recall, and the Precision-Recall Trade-off

From the Confusion Matrix, two primary metrics measure detection quality:

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

- **Precision** measures predictive purity: among all samples flagged as positive, what proportion was genuinely positive? High precision is essential when false alarms incur high costs (e.g., spam filtering, automated trading transactions).
- **Recall** (also known as Sensitivity or True Positive Rate) measures coverage: among all actual positive samples in the universe, what proportion did the model successfully find? High recall is critical when missed detections are dangerous (e.g., cancer diagnosis, malware detection).

There is an inherent trade-off between the two. Increasing the decision threshold raises precision but lowers recall; lowering the threshold captures more positives (higher recall) at the cost of more false alarms (lower precision).

### 3.3 The Harmonic Mean and the F1-Score

To summarize both dimensions in a single scalar, machine learning uses the **$F_1$-score**, defined as the harmonic mean of Precision and Recall:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

#### Why the Harmonic Mean?

The Pythagorean means of two positive numbers $x$ and $y$ are ordered as follows:
$$\text{Harmonic Mean} \le \text{Geometric Mean} \le \text{Arithmetic Mean}$$
$$\frac{2}{\frac{1}{x} + \frac{1}{y}} \le \sqrt{x y} \le \frac{x + y}{2}$$

Consider a trivial baseline model that classifies every sample as positive in a scenario where 1% of samples are truly positive:
- $\text{Recall} = 1.0$ (it caught all true positives).
- $\text{Precision} = 0.01$ (only 1% of predicted positives were actual positives).

Evaluating with the **Arithmetic Mean**:
$$\bar{A} = \frac{1.0 + 0.01}{2} = 0.505 \quad (50.5\%)$$
This arithmetic average suggests moderate capability, completely masking the 99% false alarm rate.

Evaluating with the **Harmonic Mean ($F_1$)**:
$$F_1 = 2 \cdot \frac{1.0 \cdot 0.01}{1.0 + 0.01} = \frac{0.02}{1.01} \approx 0.0198 \quad (1.98\%)$$
The harmonic mean is dominated by the smaller value. It requires **both** Precision and Recall to be high to achieve a respectable score.

### 3.4 Defensive Type and Boundary Validation

In production libraries, functions accepting discrete count parameters must defend against invalid types (such as floats, strings, or booleans) and unphysical values (non-positive counts).

```python
def calculate_f1_score(tp, fp, fn):
    # Enforce strict integer typing
    # Note: in Python, bool is a subclass of int, so we verify type explicitly
    if type(tp) is not int:
        print("tp must be int")
        return None
    if type(fp) is not int:
        print("fp must be int")
        return None
    if type(fn) is not int:
        print("fn must be int")
        return None

    # Enforce strictly positive boundary condition
    if not (tp > 0 and fp > 0 and fn > 0):
        print("tp and fp and fn must be greater than zero")
        return None

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"precision is {precision}")
    print(f"recall is {recall}")
    print(f"f1-score is {f1_score}")
    return precision, recall, f1_score
```

---

## 4. Neural Network Activation Functions and Dynamic Dispatch

### 4.1 The Necessity of Non-Linearity

An artificial neural network computes a sequence of affine transformations:
$$z_1 = W_1 x + b_1, \quad z_2 = W_2 z_1 + b_2, \quad \dots, \quad z_k = W_k z_{k-1} + b_k$$

Without an intervening non-linear activation function $\sigma(\cdot)$, any multi-layer deep network collapses into a single affine transformation:
$$z_k = W_k (W_{k-1} \dots (W_1 x + b_1) \dots + b_{k-1}) + b_k = W^* x + b^*$$

Where $W^* = \prod_{i=k}^1 W_i$. A 100-layer linear network has no more expressive power than a single linear regression model. Non-linear activation functions bend the feature space, allowing the network to satisfy the **Universal Approximation Theorem** (Cybenko, 1989; Hornik, 1991).

### 4.2 Three Classical Activation Functions

```text
1. Binary Step            2. Sigmoid                      3. ELU (alpha = 0.01)
     f(x)                      f(x)                             f(x)
      |                         |                                |
  1.0 +-------+             1.0 +         .---               1.0 +        /
      |       |                 |       .'                       |       /
      |       |             0.5 +      /                     0.0 +------+----
  0.0 +---+---+             0.0 +----+                           |     /
      +---|---+                 +----+---                        |    /
         0.0                       0.0                              0.0
```

#### 1. Binary Step Function (Heaviside)
$$f_{\text{step}}(x) = \begin{cases} 1 & \text{if } x \ge 0 \\ 0 & \text{if } x < 0 \end{cases}$$

- **Range:** $\{0, 1\}$.
- **Derivative:** $f'(x) = 0$ for all $x \ne 0$, and undefined at $x = 0$.
- **Limitations:** Because the derivative is zero almost everywhere, backpropagation cannot compute gradients ($\frac{\partial L}{\partial W} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial W} = 0$). Modern gradient-based training cannot use this activation.

#### 2. Sigmoid (Logistic) Function
$$\sigma(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{1 + e^x}$$

- **Range:** $(0, 1)$.
- **Derivative:** Elegantly expressed in terms of its own output:
  $$\sigma'(x) = \sigma(x) (1 - \sigma(x))$$
- **Limitations:**
  1. **Vanishing Gradient:** For large positive or negative inputs ($|x| > 4$), the derivative approaches zero ($\sigma'(x) \to 0$). Backpropagating through many sigmoid layers exponentially attenuates gradient magnitudes.
  2. **Non-Zero Centered:** Outputs are strictly positive, causing zig-zag dynamics during gradient updates.

#### 3. Exponential Linear Unit (ELU)
Proposed by Clevert et al. (2015) to accelerate deep learning training:
$$f_{\text{elu}}(x) = \begin{cases} x & \text{if } x \ge 0 \\ \alpha (e^x - 1) & \text{if } x < 0 \end{cases}$$

- **Range:** $(-\alpha, \infty)$.
- **Derivative:**
  $$f'_{\text{elu}}(x) = \begin{cases} 1 & \text{if } x \ge 0 \\ \alpha e^x = f(x) + \alpha & \text{if } x < 0 \end{cases}$$
- **Advantages:** For positive inputs ($x \ge 0$), its derivative is $1$, avoiding vanishing gradients. For negative inputs ($x < 0$), it smoothly saturates to $-\alpha$, pushing mean activations closer to zero and improving noise robustness.

### 4.3 Safe String Parsing and Dispatch Architecture

In command-line tools or web endpoints, inputs arrive as raw strings. The function must safely confirm that numerical strings can be parsed to `float` without unhandled exceptions:

```python
def is_number(n: str) -> bool:
    try:
        float(n)
        return True
    except (ValueError, TypeError):
        return False
```

```text
User Input: x_str, function_name
           |
           v
    is_number(x_str)?
       /       \
     NO         YES
     /           \
Print error:      Is function_name valid?
"x must be a      {'binary', 'sigmoid', 'elu'}
number"               /               \
                    NO                 YES
                    /                   \
        Print error:                     Cast x = float(x_str)
        "{name} is not supported"        Evaluate formula
                                         Print "{name}: f({x})={result}"
```

---

## 5. Regression Loss Functions and Monte Carlo Error Analysis

### 5.1 Supervised Regression Objective

In continuous target prediction, we are given a dataset of target-prediction pairs $(y_i, \hat{y}_i)$ for $i \in \{1, \dots, n\}$. A loss metric maps these discrepancies into a scalar cost representing model error.

### 5.2 Mathematical Formulation of Error Metrics

| Metric | Formula | Derivative w.r.t. $\hat{y}_i$ | Robustness to Outliers |
| :--- | :--- | :--- | :--- |
| **Absolute Error (AE)** | $\text{AE} = \|y_i - \hat{y}_i\|$ | $-\text{sgn}(y_i - \hat{y}_i)$ | High (Linear penalty) |
| **Squared Error (SE)** | $\text{SE} = (y_i - \hat{y}_i)^2$ | $-2(y_i - \hat{y}_i)$ | Low (Quadratic penalty) |
| **Mean Absolute Error (MAE)** | $\frac{1}{n}\sum_{i=1}^n \|y_i - \hat{y}_i\|$ | $-\frac{1}{n}\text{sgn}(y_i - \hat{y}_i)$ | High (Estimates Median) |
| **Mean Squared Error (MSE)** | $\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$ | $-\frac{2}{n}(y_i - \hat{y}_i)$ | Low (Estimates Mean) |
| **Root Mean Squared Error (RMSE)** | $\sqrt{\text{MSE}}$ | $\frac{1}{2\text{RMSE}} \frac{\partial \text{MSE}}{\partial \hat{y}_i}$ | Low (Same scale as $y$) |

```text
Error Penalty Comparison: L1 vs L2
Loss L
  |
8 +          . (SE: quadratic)
  |         /
6 +        /
  |       /     . (AE: linear)
4 +      /    .'
  |     /   .'
2 +    /  .'
  |   / .'
0 +--+-'---+---+---+---+ |y - y_hat|
  0    1   2   3   4   5
```

### 5.3 Monte Carlo Simulation Architecture

Monte Carlo simulation assesses stochastic system behavior by drawing independent samples from known probability distributions. In this lab, we generate $n$ synthetic pairs:
$$\hat{y}_i \sim \text{Uniform}(0, 10), \quad y_i \sim \text{Uniform}(0, 10)$$

Step-by-step procedure:
1. Validate that the sample count string consists solely of digits using `num_samples_str.isnumeric()`.
2. Parse the target loss name (`'MAE'`, `'MSE'`, or `'RMSE'`).
3. Iterate $i$ from $0$ to $n-1$:
   - Generate predicted and ground-truth values via `random.uniform(0.0, 10.0)`.
   - Compute step error: $\text{AE}$ for MAE; $\text{SE}$ for MSE and RMSE.
   - Accumulate total loss.
   - Print per-sample trace: `f"{loss_name} sample-{i} | pred: {p:.4f} | target: {t:.4f} | loss: {l:.4f}"`.
4. Divide the accumulated sum by $n$. For RMSE, apply `math.sqrt()`.

---

## 6. Combinatorics, Recurrences, and Sequences

### 6.1 Pascal's Triangle via Multiplicative Step Ratios

Pascal's triangle is an array of binomial coefficients:
$$\binom{n}{r} = \frac{n!}{(n-r)! r!}$$

```text
Level 0:       1
Level 1:      1 1
Level 2:     1 2 1
Level 3:    1 3 3 1
Level 4:   1 4 6 4 1
```

A common naive implementation computes factorials directly using $n!$, which risks integer overflow in unoptimized languages and redundant operations ($O(n)$ per coefficient).

Alternatively, consider the ratio between consecutive elements in the same row:
$$\binom{n}{r+1} = \frac{n!}{(n-(r+1))! (r+1)!} = \frac{n!}{(n-r-1)! (r+1) r!}$$

Using the identity $(n-r)! = (n-r)(n-r-1)!$:
$$\frac{\binom{n}{r+1}}{\binom{n}{r}} = \frac{n!}{(n-r-1)! (r+1) r!} \cdot \frac{(n-r)! r!}{n!} = \frac{n-r}{r+1}$$

Therefore:
$$\binom{n}{r+1} = \binom{n}{r} \cdot \frac{n-r}{r+1}$$

Using this recurrence, the $r+1$-th term is computed from the $r$-th term in $O(1)$ operations, with no factorial evaluations:
```python
value = 1
for pos in range(row + 1):
    print(value, end=" ")
    value = value * (row - pos) // (pos + 1)
```

### 6.2 Fibonacci Sequence with $O(1)$ Auxiliary Space

The Fibonacci sequence is defined by the second-order linear recurrence:
$$F_0 = 0, \quad F_1 = 1, \quad F_n = F_{n-1} + F_{n-2} \quad (n \ge 2)$$

```text
Recurrence Tree (Exponential naive recursion O(2^n)):
                      F(5)
                   /        \
               F(4)          F(3)
              /    \        /    \
            F(3)   F(2)   F(2)   F(1)

Iterative State Machine (Linear time O(n), O(1) space):
[ n1=0, n2=1 ] -> [ n1=1, n2=1 ] -> [ n1=1, n2=2 ] -> [ n1=2, n2=3 ] -> ...
```

By maintaining only two sliding state variables (`n1, n2`), we avoid call-stack overhead and compute $F_n$ in strictly $O(n)$ time and $O(1)$ auxiliary space.

---

## 7. Taylor Series Polynomial Expansions

### 7.1 Mathematical Framework

A Taylor series represents a smooth function $f(x) \in C^\infty$ as an infinite polynomial expansion of its derivatives. When expanded around $a = 0$, it is called a **Maclaurin series**:
$$f(x) = \sum_{n=0}^\infty \frac{f^{(n)}(0)}{n!} x^n = f(0) + f'(0)x + \frac{f''(0)}{2!}x^2 + \frac{f'''(0)}{3!}x^3 + \dots$$

### 7.2 Trigonometric and Hyperbolic Polynomial Expansions

#### 1. Sine Function ($\sin x$)
Since derivatives of $\sin(x)$ cycle $(\sin, \cos, -\sin, -\cos)$ with values at zero $(0, 1, 0, -1)$, only odd powers survive:
$$\sin(x) = \sum_{n=0}^\infty (-1)^n \frac{x^{2n+1}}{(2n+1)!} = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \dots$$

#### 2. Cosine Function ($\cos x$)
Even powers survive:
$$\cos(x) = \sum_{n=0}^\infty (-1)^n \frac{x^{2n}}{(2n)!} = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \frac{x^6}{6!} + \dots$$

#### 3. Hyperbolic Sine ($\sinh x$)
$$\sinh(x) = \frac{e^x - e^{-x}}{2} = \sum_{n=0}^\infty \frac{x^{2n+1}}{(2n+1)!} = x + \frac{x^3}{3!} + \frac{x^5}{5!} + \frac{x^7}{7!} + \dots$$

#### 4. Hyperbolic Cosine ($\cosh x$)
$$\cosh(x) = \frac{e^x + e^{-x}}{2} = \sum_{n=0}^\infty \frac{x^{2n}}{(2n)!} = 1 + \frac{x^2}{2!} + \frac{x^4}{4!} + \frac{x^6}{6!} + \dots$$

### 7.3 Truncation Error Bounds

By Taylor's Theorem with Lagrange Remainder, truncating the expansion at order $N$ yields a remainder:
$$R_N(x) = \frac{f^{(2N+3)}(\xi)}{(2N+3)!} x^{2N+3} \quad \text{for some } \xi \in (0, x)$$

Because $|\cos(\xi)| \le 1$ and $|\sin(\xi)| \le 1$, the absolute approximation error is strictly bounded by the first omitted term:
$$|R_N(x)| \le \frac{|x|^{2N+3}}{(2N+3)!}$$

Due to the rapid factorial growth in the denominator, the series converges quickly for moderate $|x|$.

---

## 8. Integer Arithmetic: Digit Extraction and Number Reversal

### 8.1 Positional Numeral Arithmetic

Any positive integer $n \in \mathbb{Z}^+$ can be uniquely written in base-10 positional notation:
$$n = \sum_{k=0}^{m-1} d_k 10^k = d_{m-1} 10^{m-1} + \dots + d_1 10^1 + d_0 10^0$$
Where $d_k \in \{0, 1, \dots, 9\}$ and $m = \lfloor \log_{10} n \rfloor + 1$ is the number of decimal digits.

To reverse the sequence of digits without relying on string serialization (`str(n)[::-1]`), we use repeated modulo and integer division operations:

```text
Algorithm State Trace for n = 79648:
Iteration | n (Input) | remainder = n % 10 | rev = rev * 10 + remainder | n = n // 10
----------+-----------+--------------------+----------------------------+------------
Init      | 79648     | -                  | 0                          | -
1         | 79648     | 8                  | 0 * 10 + 8 = 8             | 7964
2         | 7964      | 4                  | 8 * 10 + 4 = 84            | 796
3         | 796       | 6                  | 84 * 10 + 6 = 846          | 79
4         | 79        | 9                  | 846 * 10 + 9 = 8469        | 7
5         | 7         | 7                  | 8469 * 10 + 7 = 84697      | 0 (Exit)
```

```python
def reverse_integer(n: int) -> int:
    if n < 0:
        raise ValueError("n must be a non-negative integer")

    reversed_num = 0
    remaining = n
    while remaining > 0:
        remainder = remaining % 10
        reversed_num = (reversed_num * 10) + remainder
        remaining //= 10

    return reversed_num
```

### 8.2 Invariants and Edge Cases
- **Loop Invariant:** At the start of each iteration, $\text{reversed\_num} \cdot 10^k + \text{remaining}$ reconstructs the original integer under base-10 decomposition.
- **Trailing Zeros:** For inputs like $12460$, the least significant digit is $0$. It becomes the leading digit in the accumulator ($0 \times 10 + 0 = 0$), which is naturally absorbed as standard integers do not retain leading zeros ($12460 \to 6421$).
- **Complexity:** Exactly $\lfloor \log_{10} n \rfloor + 1$ iterations. Time complexity is $O(\log_{10} n)$ with $O(1)$ auxiliary memory.

---

## 9. Practice Exercises and Implementation Review

Complete the exercises across the accompanying notebooks in `notebooks/`:

1. **Exercise 1 (Volumes):** Implement `calc_cube_volume`, `calc_rectangular_prism_volume`, `calc_cylinder_volume`, and `calc_sphere_volume`. Verify inputs with unit tests.
2. **Exercise 2 (F1-Score):** Implement `calculate_f1_score(tp, fp, fn)`. Ensure explicit error handling for invalid types (`type is not int`) and invalid values ($\le 0$).
3. **Exercise 3 (Activations):** Implement `evaluate_activation(x_str, function_name)`. Validate numerical parsing via `is_number()` and support `binary`, `sigmoid`, and `elu`.
4. **Exercise 4 (Losses):** Implement `simulate_regression_loss(num_samples_str, loss_name)`. Verify numeric input via `.isnumeric()` and simulate stochastic losses over uniform random distributions.
5. **Exercise 5 (Combinatorics):** Generate Pascal's triangle rows using ratio updates ($\binom{n}{r+1} = \binom{n}{r} \frac{n-r}{r+1}$) and output the Fibonacci sequence with $O(1)$ memory.
6. **Exercise 6 (Taylor Series):** Approximate $\sin, \cos, \sinh, \cosh$ using Taylor expansions and evaluate convergence error against `math`.
7. **Exercise 7 (Digit Reversal):** Reverse integers using pure modulo arithmetic (`% 10` and `// 10`).

---

## 10. References and Further Reading

- Python Documentation: [Built-in Functions — `print()`](https://docs.python.org/3/library/functions.html#print)
- Python Enhancement Proposal: [PEP 498 — Literal String Interpolation](https://peps.python.org/pep-0498/)
- Cybenko, G. (1989). *Approximation by superpositions of a sigmoidal function*. Mathematics of Control, Signals and Systems, 2(4), 303–314.
- Clevert, D. A., Unterthiner, T., & Hochreiter, S. (2015). *Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)*. arXiv:1511.07289.
- Fawcett, T. (2006). *An introduction to ROC analysis*. Pattern Recognition Letters, 27(8), 861–874.
- Knuth, D. E. (1997). *The Art of Computer Programming, Volume 1: Fundamental Algorithms* (3rd ed.). Addison-Wesley.
