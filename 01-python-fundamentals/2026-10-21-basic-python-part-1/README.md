# Basic Python (Part 1)

[Back to Module 01](../README.md) · [Back to the master roadmap](../../README.md)

**Session date:** 2026-10-21 · **Module:** 01 — Python Fundamentals & Algorithmic Thinking
**Format:** Guided reading, executable notebooks, and mathematical case studies

## Overview

This session establishes the programming foundation for the rest of the curriculum. It starts with the question every language must answer — how do we represent numbers, text, and objects with bits? — and then builds a working subset of Python: variables, scalar types, operators, input and output, conditionals, functions, modules, and numerical computation.

The emphasis is not on memorizing syntax. The goal is to understand what the interpreter is doing, why floating-point arithmetic behaves the way it does, and how small mathematical ideas become reusable programs. The session closes with three ideas that recur throughout machine learning: activation functions, numerical stability, and stable softmax.

## Learning outcomes

By the end of this session, you should be able to:

- Explain how bits, types, and metadata represent numbers and text.
- Write and run a Python program in a virtual environment or a notebook.
- Choose appropriate scalar types and convert between them safely.
- Use arithmetic, comparison, and logical operators correctly.
- Read input, format output, and document functions with type hints and docstrings.
- Implement conditionals, including the ReLU activation function.
- Translate piecewise mathematical definitions into Python functions.
- Approximate $\pi$ with inscribed polygons and Riemann sums.
- Estimate derivatives with finite differences.
- Implement the sigmoid, hyperbolic tangent, Newton's method, and stable softmax.
- Create labelled plots with Matplotlib and inspect images with OpenCV.

## Session map

| Notebook | Topics |
| :--- | :--- |
| [01 — Built-in functions](notebooks/01-built-in-functions.ipynb) | `print`, `type`, `input`, conversion, temperature, geometry |
| [02 — Math and random](notebooks/02-math-and-random.ipynb) | `math`, `random`, seeding, sign, division, modulo |
| [03 — User-defined functions](notebooks/03-user-defined-functions.ipynb) | Parameters, defaults, docstrings, scope, sigmoid, Newton's method |
| [04 — Numerical stability](notebooks/04-numerical-stability.ipynb) | IEEE 754, underflow, overflow, softmax, log-sum-exp |
| [05 — Matplotlib](notebooks/05-matplotlib-visualization.ipynb) | Line plots, activation functions, convergence plots |
| [06 — OpenCV](notebooks/06-opencv-image-basics.ipynb) | Image arrays, BGR/RGB, grayscale, resize, crop |

The sample image used by the OpenCV notebook is stored at [`assets/tree.jpg`](assets/tree.jpg).

## 1. What a programming language must represent

A programming language needs a way to encode the things a program manipulates and a way to interpret those encodings unambiguously.

### 1.1 Bits need types

With $n$ bits there are $2^n$ distinct patterns. Three bits produce eight patterns, but those patterns can mean different things depending on the type attached to them:

- unsigned integers: $0$ through $7$;
- signed integers: a range such as $-4$ through $3$;
- characters: one of eight code points;
- three unrelated boolean flags.

The bit pattern alone is not enough. A type supplies the interpretation: signedness, width, encoding, and the operations that are valid. This is why C distinguishes `unsigned char` from `char`, and why Python attaches a type to every object.

### 1.2 Fixed-width and arbitrary-precision integers

C integers have a fixed width. An 8-bit unsigned integer represents $0$ through $255$; an 8-bit signed integer typically represents $-128$ through $127$. Overflow wraps around or is undefined depending on the language.

Python integers are arbitrary precision. They grow as needed, so `2 ** 1000` is exact. The trade-off is that arithmetic on very large integers is slower and uses more memory than fixed-width machine arithmetic.

### 1.3 Floating-point numbers

Most languages represent real numbers with floating point. Python's `float` is an IEEE 754 binary64 value: one sign bit, eleven exponent bits, and fifty-two fraction bits. It is the same representation as a C `double`.

A binary64 value is stored as

$$
(-1)^s \times (1.f)_{2} \times 2^{e-1023}
$$

for normal numbers, where $s$ is the sign, $f$ is the fractional part, and $e$ is the biased exponent. The finite precision explains results such as:

```python
>>> 0.1 + 0.2
0.30000000000000004
>>> 0.1 + 0.2 == 0.3
False
```

Use `math.isclose` when comparing computed floating-point values.

### 1.4 Text

Text is stored as numbers through an encoding. ASCII maps 128 characters to the integers 0–127; Unicode extends the mapping to the world's writing systems. Python 3 strings are sequences of Unicode code points. They become bytes only when encoded, for example with UTF-8:

```python
message = "Python"
print(message.encode("utf-8"))
print(message.encode("utf-8").decode("utf-8"))
```

Confusing text with bytes is a common source of corrupted output and broken file handling. Decode bytes when reading; encode strings when writing.

### 1.5 Objects and names

In Python, a variable is a name bound to an object. The object has an identity, a type, and a value. Assignment binds the name; it does not copy the object.

```python
first = [1, 2, 3]
second = first
second.append(4)
print(first)
```

Both names refer to the same list, so the output is `[1, 2, 3, 4]`. Use `id()` to inspect identity and `is` to test identity. Use `==` to test value equality.

## 2. Environment and first program

### 2.1 Virtual environments

A virtual environment isolates the packages of one project from the system interpreter. From the repository root:

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS or Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the packages used in this session:

```bash
python -m pip install numpy matplotlib opencv-python jupyterlab
```

The complete repository stack is in [`requirements.txt`](../../requirements.txt).

### 2.2 A first program

Create `main.py`:

```python
print("My first Python program")
```

Run it from a terminal:

```bash
python main.py
```

In a notebook, place the same code in a cell and execute it with `Shift+Enter`. The notebook kernel keeps state between cells, which is convenient for exploration but can hide ordering mistakes. Restart the kernel and run all cells before trusting a notebook result.

### 2.3 Comments and style

Comments explain why code exists, not what the syntax already says. Use four spaces per indentation level, lowercase `snake_case` names, and a blank line between top-level definitions. These conventions come from PEP 8, the standard Python style guide.

## 3. Variables, names, and scalar types

The assignment statement has the form

```python
variable_name = expression
```

Names should be meaningful. They may contain letters, digits, and underscores, cannot begin with a digit, and cannot be keywords such as `if`, `for`, `class`, or `return`.

The four scalar types used in this session are:

| Type | Examples | Purpose |
| :--- | :--- | :--- |
| `int` | `1`, `0`, `-42`, `10**30` | Exact integers of arbitrary size |
| `float` | `1.5`, `-3.21`, `1.0`, `2e-3` | IEEE 754 double-precision numbers |
| `str` | `"Python"`, `'data'`, `"42"` | Unicode text |
| `bool` | `True`, `False` | Logical truth values |

`type(value)` reports the class of a value:

```python
number_of_days = 10
distance_km = 20.5
greeting = "Hello"
is_enrolled = True

print(type(number_of_days))
print(type(distance_km))
print(type(greeting))
print(type(is_enrolled))
```

Output:

```text
<class 'int'>
<class 'float'>
<class 'str'>
<class 'bool'>
```

Python is dynamically typed: a name can be rebound to a value of a different type. The interpreter checks operations at runtime, so type errors appear when the offending line executes, not when the file is loaded. Type hints document intent and support static analysis, but they are not enforced at runtime.

### 3.1 Input and conversion

`input(prompt)` always returns a string:

```python
value_text = input("Value: ")
print(type(value_text))
```

Convert explicitly:

```python
value_int = int(value_text)
value_float = float(value_text)
```

`int("3.5")` raises `ValueError`; parse it as a float first if truncation is intended:

```python
number = int(float("3.5"))
```

Handle invalid input when it comes from a user:

```python
raw_value = "not a number"
try:
    value = float(raw_value)
except ValueError:
    print(f"Invalid number: {raw_value!r}")
```

## 4. Operators

### 4.1 Arithmetic operators

| Operator | Meaning | Example |
| :---: | :--- | :--- |
| `+` | Addition | `2 + 3 == 5` |
| `-` | Subtraction | `3 - 2 == 1` |
| `*` | Multiplication | `2 * 3 == 6` |
| `/` | True division | `9 / 4 == 2.25` |
| `//` | Floor division | `9 // 4 == 2` |
| `%` | Modulo (remainder) | `9 % 4 == 1` |
| `**` | Exponentiation | `3 ** 2 == 9` |

`/` always returns a float, even when both operands are integers. `//` rounds toward negative infinity, and `%` follows the sign of the divisor:

```python
print(-9 // 4)  # -3
print(-9 % 4)  # 3
print(divmod(-9, 4))  # (-3, 3)
```

### 4.2 Precedence and associativity

Exponentiation binds more tightly than unary minus, and it is right-associative:

```python
print(-(2**2))  # -4, because it is -(2 ** 2)
print((-2) ** 2)  # 4
print(2**3**2)  # 512, because it is 2 ** (3 ** 2)
```

Parentheses are cheap. Use them when the intended grouping is not obvious.

### 4.3 Comparison and logical operators

| Operator | Meaning |
| :---: | :--- |
| `==` | Equal to |
| `!=` | Not equal to |
| `>` | Greater than |
| `<` | Less than |
| `>=` | Greater than or equal to |
| `<=` | Less than or equal to |
| `and`, `or`, `not` | Boolean composition |

Comparisons return booleans and can be chained:

```python
age = 25
print(18 <= age < 65)
```

`and` and `or` short-circuit. `and` returns the first falsy operand or the last operand; `or` returns the first truthy operand or the last operand. This behavior is useful for defaults, but it can surprise readers when non-boolean values are involved. Prefer explicit conditionals when clarity matters.

## 5. Built-in functions and formatted output

Python provides a set of always-available functions:

| Function | Purpose |
| :--- | :--- |
| `print(*objects)` | Write values to standard output |
| `type(object)` | Return the class of an object |
| `input(prompt)` | Read a line from standard input as a string |
| `int(value)` | Convert to an integer |
| `float(value)` | Convert to a floating-point number |
| `abs(number)` | Absolute value |
| `round(number, ndigits)` | Round to a given precision |
| `min(iterable)` | Smallest element |
| `max(iterable)` | Largest element |
| `sum(iterable)` | Sum of elements |
| `len(iterable)` | Number of elements |

Example:

```python
values = [-2.5, 4.0, 1.5]
print(abs(-7))
print(round(3.14159, 2))
print(min(values))
print(max(values))
print(sum(values))
print(len("Python"))
```

Formatted string literals, or f-strings, interpolate expressions and control formatting:

```python
celsius = 37.0
fahrenheit = (9 / 5) * celsius + 32
print(f"{celsius:.1f} degrees Celsius = {fahrenheit:.1f} degrees Fahrenheit")
```

The format specification `.1f` means fixed-point notation with one digit after the decimal point.

## 6. The `math` and `random` modules

### 6.1 Mathematical functions

The `math` module provides scalar mathematical functions:

```python
import math

print(math.fabs(-2))  # 2.0
print(math.exp(2))  # 7.38905609893065
print(math.log(4))  # 1.3862943611198906
print(math.log(math.e))  # 1.0
print(math.sqrt(4))  # 2.0
print(math.sin(2))  # 0.9092974268256817
print(math.cos(2))  # -0.4161468365471424
print(math.pi)  # 3.141592653589793
print(math.e)  # 2.718281828459045
```

Angles are measured in radians. Convert with `math.radians(degrees)` and `math.degrees(radians)`.

### 6.2 Random numbers

The `random` module generates pseudo-random values. It is suitable for simulations and teaching, not for cryptography.

```python
import random

random.seed(42)

print(random.random())  # float in [0, 1)
print(random.randint(1, 9))  # integer in [1, 9]
print(random.uniform(-1.0, 1.0))
print(random.choice(["red", "green", "blue"]))
```

Setting the seed makes a sequence reproducible. This is essential when debugging a simulation or comparing experiments.

A correct sign function does not divide by the input, because zero has no sign in that formula:

```python
def sign(number: float) -> int:
    """Return -1, 0, or 1 according to the sign of number."""
    if number > 0:
        return 1
    if number < 0:
        return -1
    return 0
```

## 7. Conditionals

### 7.1 `if`, `elif`, and `else`

A conditional selects a block of code based on a boolean expression:

```python
def classify_temperature(celsius: float) -> str:
    """Return a short description of a temperature."""
    if celsius < 0:
        return "freezing"
    if celsius < 20:
        return "cool"
    if celsius < 30:
        return "warm"
    return "hot"
```

`elif` is used when exactly one branch should run:

```python
def describe_parity(number: int) -> str:
    if number % 2 == 0:
        return "even"
    else:
        return "odd"
```

Every condition is followed by a colon, and the body must be indented. Mixing tabs and spaces causes confusing errors; use four spaces consistently.

### 7.2 Truthiness

Python treats several values as false: `False`, `None`, numeric zero, and empty containers such as `""`, `[]`, and `{}`. Everything else is truthy. Prefer explicit tests when the distinction matters:

```python
items = []
if not items:
    print("The list is empty")
```

### 7.3 The ReLU activation function

The rectified linear unit is a piecewise function used throughout neural networks:

$$
\operatorname{ReLU}(x) =
\begin{cases}
0 & \text{if } x \le 0,\\
x & \text{if } x > 0.
\end{cases}
$$

```python
def relu(x: float) -> float:
    """Return the rectified linear unit of x."""
    if x > 0:
        return x
    return 0.0
```

`max(0.0, x)` is equivalent. Applied to a vector element by element, ReLU replaces negative values with zero and leaves positive values unchanged.

### 7.4 From branches to arithmetic

Conditionals are easy to read, but vectorized numerical code often replaces them with arithmetic. Three useful identities from the session are:

**Two-point interpolation.** If $a=0$ should produce $b^2$ and $a=K$ should produce $\sqrt{b}$, then

$$
y = \frac{a}{K}\sqrt{b} + \frac{K-a}{K}b^2.
$$

**Positive/negative switch.** If $a>0$ should produce $b^2$ and $a<0$ should produce $\sqrt{b}$, then

$$
y = \frac{b^2+\sqrt{b}}{2} + \frac{b^2-\sqrt{b}}{2}\cdot\frac{|a|}{a}.
$$

The expression is undefined at $a=0$, where a separate branch is still required.

**One-hot selection.** For the three cases $a=0 \mapsto b^2$, $a=1 \mapsto \sqrt{b}$, and $a=2 \mapsto b$, define

$$
\mathbf{u} = \begin{bmatrix} b^2 \\ \sqrt{b} \\ b \end{bmatrix},
\qquad
\mathbf{v} = \text{one-hot}(a),
\qquad
y = \mathbf{v}^\top \mathbf{u}.
$$

The one-hot vector selects one component by multiplication. This is the same idea used by categorical cross-entropy targets and mixture models.

## 8. User-defined functions

### 8.1 Anatomy

A function wraps a computation behind a name:

```python
def compute_rectangle_area(height: float, width: float) -> float:
    """Return the area of a rectangle.

    Args:
        height: The height of the rectangle.
        width: The width of the rectangle.

    Returns:
        The area, computed as height multiplied by width.
    """
    return height * width
```

The pieces are:

- `def` — the keyword that begins a function definition;
- `compute_rectangle_area` — the function name, lowercase with underscores and usually a verb;
- `height`, `width` — parameters that receive arguments;
- `-> float` — a return type hint;
- the docstring — a description used by `help()` and documentation tools;
- the indented body;
- `return` — the value handed back to the caller.

Naming the function after the action it performs makes call sites readable. A function that returns a value should usually be named with a verb phrase, such as `compute_area`, `normalize_scores`, or `load_image`.

### 8.2 Arguments, defaults, and keyword arguments

```python
def compute_rectangle_area_with_defaults(
    height: float = 0.0,
    width: float = 0.0,
) -> float:
    """Return the area of a rectangle, using zero for omitted dimensions."""
    return height * width


print(compute_rectangle_area_with_defaults())
print(compute_rectangle_area_with_defaults(5, 6))
print(compute_rectangle_area_with_defaults(width=6))
```

Default arguments are evaluated once, when the function is defined. Never use a mutable object as a default:

```python
def append_bad(item, items=[]):  # The same list is reused across calls.
    items.append(item)
    return items


def append_good(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

### 8.3 Scope

Python resolves names using the local, enclosing, global, and built-in scopes, often summarized as LEGB:

1. **Local** — names assigned inside the current function.
2. **Enclosing** — names in an outer function for a nested function.
3. **Global** — names at the module level.
4. **Built-in** — names such as `len` and `print`.

Assignment inside a function creates a local name unless the name is declared `global` or `nonlocal`. Reading a local name before its assignment raises `UnboundLocalError`, even if a global name with the same spelling exists:

```python
value = 10


def broken() -> int:
    result = value * 2
    value = 1
    return result
```

The assignment to `value` makes it local for the entire function body, so the read on the previous line fails.

Prefer explicit parameters over reading globals. A function that depends only on its arguments is easier to test, reuse, and reason about.

### 8.4 The sigmoid function

The logistic sigmoid is

$$
\sigma(x) = \frac{1}{1+e^{-x}}.
$$

It maps $\mathbb{R}$ into $(0,1)$ and is strictly increasing. Its derivative is

$$
\sigma'(x) = \sigma(x)\bigl(1-\sigma(x)\bigr).
$$

```python
import math


def compute_sigmoid(x: float) -> float:
    """Return the logistic sigmoid of x."""
    return 1.0 / (1.0 + math.exp(-x))
```

Passing `x` as a parameter is essential. A function that reads a global `x` works only for one value and is difficult to test.

### 8.5 Newton's method for the square root

Newton's method finds a root of $f(x)=0$ by following the tangent line:

$$
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}.
$$

For $f(x)=x^2-N$, the update is

$$
x_{n+1} = x_n - \frac{x_n^2-N}{2x_n}
= \frac{1}{2}\left(x_n + \frac{N}{x_n}\right).
$$

Starting from a positive estimate, the sequence converges to $\sqrt{N}$ for $N>0$. Convergence is quadratic near the root: the number of correct digits roughly doubles each iteration.

```python
def newton_sqrt(
    number: float,
    tolerance: float = 1e-12,
    max_iterations: int = 100,
) -> float:
    """Return the square root of number using Newton's method."""
    if number < 0:
        raise ValueError("number must be non-negative")
    if number == 0:
        return 0.0

    estimate = number / 2.0
    for _ in range(max_iterations):
        next_estimate = 0.5 * (estimate + number / estimate)
        if abs(next_estimate - estimate) < tolerance:
            return next_estimate
        estimate = next_estimate

    return estimate
```

The tolerance handles floating-point convergence; the iteration cap prevents an infinite loop if the tolerance is unreachable.

## 9. Mathematical case studies

### 9.1 Temperature conversion

The linear maps are

$$
F = \frac{9}{5}C + 32,
\qquad
C = \frac{5}{9}(F-32).
$$

```python
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a Celsius temperature to Fahrenheit."""
    return (9 / 5) * celsius + 32


def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Convert a Fahrenheit temperature to Celsius."""
    return (fahrenheit - 32) * 5 / 9
```

Composing the two functions should return the original value up to floating-point rounding. Test that property explicitly rather than assuming it.

### 9.2 Circumference and area of a unit circle

A regular $m$-gon inscribed in the unit circle approximates the circle from the inside. The central angle subtended by one side is $2\pi/m$. The side length is

$$
s_m = 2\sin\left(\frac{\pi}{m}\right),
$$

so the perimeter is

$$
P_m = 2m\sin\left(\frac{\pi}{m}\right) \longrightarrow 2\pi.
$$

Each isosceles triangle with two unit sides and included angle $2\pi/m$ has area

$$
A_{\triangle} = \frac{1}{2}\sin\left(\frac{2\pi}{m}\right),
$$

so the polygon area is

$$
A_m = \frac{m}{2}\sin\left(\frac{2\pi}{m}\right) \longrightarrow \pi.
$$

```python
import math


def polygon_perimeter(m: int) -> float:
    """Return the perimeter of a regular m-gon inscribed in a unit circle."""
    return 2 * m * math.sin(math.pi / m)


def polygon_area(m: int) -> float:
    """Return the area of a regular m-gon inscribed in a unit circle."""
    return 0.5 * m * math.sin(2 * math.pi / m)


for sides in (6, 12, 60, 600):
    print(sides, polygon_perimeter(sides), polygon_area(sides))
```

The area sequence approaches $\pi$ and the perimeter sequence approaches $2\pi$, both from below. The error decreases roughly as $1/m^2$ in each case, because the Taylor expansion of $\sin$ starts with a cubic correction.

### 9.3 Riemann sums

The upper half of the unit circle is the graph of

$$
f(x) = \sqrt{1-x^2}, \qquad -1 \le x \le 1.
$$

Its area is $\pi/2$, so twice the integral is $\pi$:

$$
\pi = 2\int_{-1}^{1}\sqrt{1-x^2}\,dx.
$$

A midpoint Riemann sum with $n$ intervals approximates the integral as

$$
A_n = \sum_{i=1}^{n} f(x_i)\,\Delta x,
\qquad
x_i = -1 + \left(i-\frac{1}{2}\right)\Delta x,
\qquad
\Delta x = \frac{2}{n}.
$$

```python
import numpy as np


def riemann_pi(segment_count: int) -> float:
    """Approximate pi with a midpoint Riemann sum on the unit semicircle."""
    edges = np.linspace(-1.0, 1.0, segment_count + 1)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    heights = np.sqrt(np.maximum(0.0, 1.0 - midpoints**2))
    return 2.0 * np.sum(widths * heights)


for n in (10, 100, 1_000, 10_000):
    print(n, riemann_pi(n))
```

For smooth integrands, the midpoint rule has error $O(1/n^2)$. The semicircle has a square-root singularity in its derivative at the endpoints, so the observed convergence rate is slightly lower, but the estimate still approaches $\pi$. The `np.maximum` guard protects against tiny negative values produced by floating-point rounding near $x=\pm 1$.

### 9.4 Derivatives with finite differences

The derivative of $f$ at $x$ is

$$
f'(x) = \lim_{h\to 0}\frac{f(x+h)-f(x)}{h}.
$$

A computer cannot take a limit, so it chooses a small $h$. The forward difference

$$
D_h^+ f(x) = \frac{f(x+h)-f(x)}{h}
$$

has truncation error $O(h)$. The central difference

$$
D_h f(x) = \frac{f(x+h)-f(x-h)}{2h}
$$

has truncation error $O(h^2)$ but subtracts nearly equal numbers, which amplifies floating-point roundoff. For binary64 arithmetic, a forward difference is typically best around $h\approx 10^{-8}$, and a central difference around $h\approx 10^{-5}$.

```python
def f(x: float) -> float:
    """Return x squared plus 2x."""
    return x**2 + 2 * x


def derivative_forward(x: float, h: float = 1e-6) -> float:
    """Estimate f'(x) with a forward difference."""
    return (f(x + h) - f(x)) / h


def derivative_central(x: float, h: float = 1e-6) -> float:
    """Estimate f'(x) with a central difference."""
    return (f(x + h) - f(x - h)) / (2 * h)


for step in (1e-2, 1e-4, 1e-6, 1e-8):
    print(step, derivative_forward(2.0, step), derivative_central(2.0, step))
```

For $f(x)=x^2+2x$, the exact derivative is $f'(x)=2x+2$, so $f'(2)=6$. The table shows the trade-off: large steps have truncation error, while extremely small steps have roundoff error.

### 9.5 Sigmoid and hyperbolic tangent

The sigmoid and hyperbolic tangent are S-shaped activation functions:

$$
\sigma(x) = \frac{1}{1+e^{-x}},
\qquad
\tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}.
$$

Their ranges and derivatives are:

| Function | Range | Derivative |
| :--- | :--- | :--- |
| $\sigma(x)$ | $(0,1)$ | $\sigma(x)(1-\sigma(x))$ |
| $\tanh(x)$ | $(-1,1)$ | $1-\tanh^2(x)$ |

Both saturate for large $|x|$: the derivative approaches zero, which contributes to vanishing gradients in deep networks. ReLU avoids saturation on the positive side but is not differentiable at zero; frameworks define a subgradient there.

### 9.6 Euler's number and compound interest

If an initial amount $A$ earns interest at annual rate $r$, compounded $n$ times per year, the amount after $t$ years is

$$
A_t = A\left(1+\frac{r}{n}\right)^{nt}.
$$

Set $m=n/r$. Then

$$
A_t = A\left(1+\frac{1}{m}\right)^{mrt}
\longrightarrow A e^{rt}
\quad \text{as } m\to\infty,
$$

because

$$
e = \lim_{m\to\infty}\left(1+\frac{1}{m}\right)^m.
$$

```python
import math


def compound_balance(
    principal: float,
    annual_rate: float,
    years: float,
    compounds_per_year: int,
) -> float:
    """Return the balance under periodic compound interest."""
    return principal * (1 + annual_rate / compounds_per_year) ** (compounds_per_year * years)


print(compound_balance(1000.0, 0.05, 10, 1))
print(compound_balance(1000.0, 0.05, 10, 12))
print(1000.0 * math.exp(0.05 * 10))
```

The three results approach each other as the compounding frequency increases. The exponential limit is the continuous-compounding model.

## 10. Numerical stability and softmax

### 10.1 Underflow and overflow

IEEE 754 binary64 has a finite exponent range. Python's `float` behaves as follows:

```python
print(1e-100)  # 1e-100
print(1e-1000)  # 0.0, underflow
print(1e100)  # 1e+100
print(1e1000)  # inf, overflow
```

`math.exp(1000.0)` raises `OverflowError` instead of returning `inf`. Underflow is often silent: a probability that should be tiny becomes exactly zero, and taking its logarithm then fails with `ValueError: math domain error`.

### 10.2 Softmax

Softmax turns a vector of scores into a probability distribution:

$$
\operatorname{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}.
$$

The outputs satisfy

$$
0 < \operatorname{softmax}(x_i) < 1,
\qquad
\sum_{i=1}^{n}\operatorname{softmax}(x_i)=1.
$$

A naive implementation exponentiates the raw scores:

```python
import math


def softmax_naive(values: list[float]) -> list[float]:
    """Return softmax probabilities without numerical protection."""
    exponentials = [math.exp(value) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]
```

For `[1.0, 2.0, 3.0]` it works. For `[1001.0, 1002.0, 1003.0]`, `math.exp(1001.0)` overflows before normalization can occur. Even when it does not raise, `exp` can produce `inf`, and `inf / inf` yields `nan`.

### 10.3 Stable softmax

Let $m=\max_i x_i$. Multiply the numerator and denominator by $e^{-m}$:

$$
\frac{e^{x_i}}{\sum_j e^{x_j}}
=
\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}.
$$

This is algebraically identical but numerically safe: every exponent is at most zero, so every term lies in $(0,1]$ and the largest term is exactly one.

```python
def softmax(values: list[float]) -> list[float]:
    """Return a numerically stable softmax distribution."""
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


print(softmax([1.0, 2.0, 3.0]))
print(softmax([1001.0, 1002.0, 1003.0]))
print(softmax([1.0, 1001.0, 1002.0]))
```

The stable version also handles scores that are all equal: the shifted values are all zero, every exponential is one, and the result is a uniform distribution.

### 10.4 Vectorized implementation

NumPy expresses the same computation without a Python loop:

```python
import numpy as np


def softmax_numpy(values: np.ndarray) -> np.ndarray:
    """Return a numerically stable softmax for a one-dimensional array."""
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()
```

For log-probabilities, use the log-sum-exp identity:

$$
\log\sum_i e^{x_i} = m + \log\sum_i e^{x_i-m}.
$$

`scipy.special.logsumexp` implements this pattern and avoids both overflow and underflow:

```python
from scipy.special import logsumexp

scores = np.array([1.0, 1001.0, 1002.0])
log_probabilities = scores - logsumexp(scores)
print(log_probabilities)
print(np.exp(log_probabilities))
```

## 11. Visualization with Matplotlib

Matplotlib turns arrays into figures. A useful plot has a title, labelled axes, a readable scale, and a legend when more than one series is present.

```python
import matplotlib.pyplot as plt

x_values = [0, 1, 2, 3]
y_values = [1, 3, 4, 2]

figure, axis = plt.subplots(figsize=(6, 4))
axis.plot(x_values, y_values, marker="o", label="sample series")
axis.set_title("A simple line plot")
axis.set_xlabel("x")
axis.set_ylabel("y")
axis.set_xticks(x_values)
axis.grid(True, alpha=0.3)
axis.legend()
plt.show()
```

The object-oriented interface (`figure, axis = plt.subplots()`) is clearer than the implicit `pyplot` state for anything beyond a quick sketch. When a figure contains multiple panels, use `plt.subplots(1, 2)` and index the axes.

Activation functions are easy to visualize with `numpy.linspace`:

```python
import numpy as np

x = np.linspace(-6.0, 6.0, 400)
sigmoid = 1.0 / (1.0 + np.exp(-x))
tanh = np.tanh(x)

figure, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(x, sigmoid, label="sigmoid")
axes[0].plot(x, sigmoid * (1 - sigmoid), label="derivative")
axes[0].set_title("Sigmoid")
axes[1].plot(x, tanh, label="tanh")
axes[1].plot(x, 1 - tanh**2, label="derivative")
axes[1].set_title("Tanh")
for axis in axes:
    axis.set_xlabel("x")
    axis.set_ylabel("value")
    axis.grid(True, alpha=0.3)
    axis.legend()
figure.tight_layout()
plt.show()
```

For convergence plots, a logarithmic y-axis makes multiplicative error visible across several orders of magnitude.

## 12. Image basics with OpenCV

An image is a numerical array. OpenCV loads color images as three-channel arrays in blue-green-red (BGR) order. Matplotlib expects red-green-blue (RGB), so convert before displaying:

```python
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

image_path = Path("..") / "assets" / "tree.jpg"
image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

if image_bgr is None:
    raise FileNotFoundError(f"Could not read image: {image_path.resolve()}")

print(image_bgr.shape)  # (height, width, channels)
print(image_bgr.dtype)  # uint8

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

figure, axis = plt.subplots(figsize=(8, 6))
axis.imshow(image_rgb)
axis.set_title("Sample image loaded with OpenCV")
axis.axis("off")
plt.show()
```

Common operations:

```python
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
height, width = image_bgr.shape[:2]
resized = cv2.resize(
    image_bgr,
    (width // 2, height // 2),
    interpolation=cv2.INTER_AREA,
)
cropped = image_bgr[: height // 2, : width // 2]
```

`cv2.resize` takes `(width, height)`, while array indexing takes rows before columns. This order reversal is a frequent source of bugs.

`cv2.imshow` opens a native window and requires a GUI backend. It blocks until a key is pressed and fails in headless environments. Notebook-based work should use Matplotlib for portable display.

## 13. Exercises

Work through these after reading the notebooks. Write a short explanation for each result, not only the code.

1. **Bounds.** Implement `clamp(value, lower, upper)` without using `min` or `max`. Then implement it with both built-ins and compare readability.
2. **Temperature round-trip.** Write Celsius-to-Fahrenheit and Fahrenheit-to-Celsius functions. Test five values and report the maximum absolute error of the round-trip.
3. **Right triangles.** Read three side lengths as floats, sort them, and use `math.isclose` to decide whether they form a right triangle. Explain why sorting simplifies the test.
4. **Dice simulation.** Simulate 10,000 rolls of a six-sided die with a fixed seed. Report the frequency and relative frequency of each face. How close are the frequencies to $1/6$?
5. **Polygon estimates.** Compute the perimeter and area approximations of $\pi$ for $m=4,8,16,\dots,4096$. Plot the absolute error on a logarithmic scale. Estimate the slope of the error curve.
6. **Riemann sums.** Repeat the semicircle approximation with left endpoints, midpoints, and trapezoids. Compare the convergence rates.
7. **Derivative table.** For $f(x)=x^2+2x$, compute forward and central differences at $x=2$ for $h=10^{-2},10^{-4},10^{-6},10^{-8},10^{-10}$. Identify where roundoff starts to dominate.
8. **Stable softmax.** Implement softmax for a list of floats without using `max`. Can you make it stable using sorting instead? What is the cost?
9. **Log probabilities.** Compute the log-softmax of `[1000.0, 1001.0, 1002.0]` with `logsumexp`. Verify that exponentiating the result gives a valid probability distribution.
10. **Image operations.** Load the sample tree image, convert it to grayscale, crop the central half, and resize it to $256\times256$. Display the original and the result side by side.

## 14. Reproducing this session

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib opencv-python jupyterlab
python -m jupyter lab
```

Open the notebooks in order. Each notebook is self-contained except for the sample image path, which is relative to the notebook directory. Restart the kernel and run all cells before relying on a result.

The notebooks intentionally contain no saved outputs. This keeps the repository small and makes the results reproducible from the code alone.

## 15. Key takeaways

- Bits are meaningless without a type; Python attaches a type to every object.
- Python integers are arbitrary precision; floats are finite-precision IEEE 754 values.
- `/` returns a float, `//` floors, and `%` follows the divisor's sign.
- `input()` returns a string; convert explicitly and handle invalid input.
- Conditionals express piecewise mathematics; one-hot vectors and interpolation can express the same logic arithmetically.
- Functions should take explicit parameters, document their behavior, and avoid hidden global state.
- Floating-point equality is fragile; use tolerances.
- Exponentiation overflows quickly; stable softmax shifts scores by their maximum.
- Work in log space when probabilities underflow.
- Numerical approximations trade truncation error against roundoff error.
- A plot needs a title, labelled axes, and a scale that makes the pattern visible.
- Images are arrays; channel order and shape conventions differ between libraries.

## References

- Python documentation: <https://docs.python.org/3/>
- `math` module: <https://docs.python.org/3/library/math.html>
- `random` module: <https://docs.python.org/3/library/random.html>
- PEP 8, Style Guide for Python Code: <https://peps.python.org/pep-0008/>
- Matplotlib documentation: <https://matplotlib.org/stable/>
- OpenCV Python tutorials: <https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html>
- NumPy documentation: <https://numpy.org/doc/stable/>
- IEEE 754 floating-point standard: <https://standards.ieee.org/ieee/754/6210/>
- *Python Crash Course* by Eric Matthes.
- *Fluent Python* by Luciano Ramalho.
- *Calculus: Early Transcendentals* by James Stewart.
