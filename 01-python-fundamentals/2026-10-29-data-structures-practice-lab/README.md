# Data Structures Practice Lab: Clean Code, Execution Tracing, Frequency Analysis, and Sliding Window Kernels

[Back to Module 01](../README.md) · [Back to the master roadmap](../../README.md)

**Session date:** 2026-10-29 · **Module:** 01 — Python Fundamentals & Algorithmic Thinking  
**Format:** Comprehensive study guide, mathematical foundations, formal specifications, and executable practice notebooks

---

## Overview

In production artificial intelligence pipelines and computational engineering, algorithmic correctness depends as much on code hygiene and execution tracing as it does on theoretical complexity. Machine learning engineers frequently encounter cryptic failure modes—not because their mathematical models are flawed, but because mutable default arguments corrupt state across training epochs, sequence mutations disrupt iteration loops, operator precedence distorts loss calculations, or unindexed sliding windows introduce quadratic bottlenecks.

This practice lab provides a rigorous, hands-on consolidation of core Python mechanics, defensive programming practices, and fundamental algorithmic patterns. Grounded in first principles, this session bridges foundational syntax and computational efficiency across six essential engineering domains:

1. **Defensive Debugging and Critical Runtime Pitfalls:** We analyze the internal mechanics of recursion limits and call stack exhaustion (`RecursionError`), sequence mutation hazards during iteration (infinite loops and skipped items caused by list index shifting), dictionary lookup traps (`KeyError`), and string interpolation bugs.
2. **PEP 8 Code Quality, Code Smells, and Clean Refactoring:** We examine how clean code conventions protect systems against subtle bugs. We eliminate magic numbers, avoid shadowing built-in identifiers, eliminate dead code, dismantle mutable global variables, resolve the infamous mutable default argument trap (`def fn(x, items=[])`), and correct algebraic operator precedence errors in mathematical formulas.
3. **Code Reading Comprehension and Mental Execution Tracing:** We develop systematic trace-table techniques to mentally simulate the Python runtime across custom range strides, string tokenization, early break conditions, matrix transposition mapping ($M_{ji} = X_{ij}$), numerical clamping functions, and adaptive Bubble Sort invariants.
4. **Frequency Analysis and Hash Table Data Structures:** We implement character-level and word-level occurrence counters. We dissect string sanitization, case normalization, and punctuation translation tables, comparing algorithmic paradigms across imperative dictionaries, `.get()` fallbacks, `collections.defaultdict`, and `collections.Counter`.
5. **1D Sliding Window Maximum (Kernel Max):** We formulate the sliding window maximum problem over discrete 1D arrays. Moving from naive $O(N \cdot k)$ sub-array slicing to anchor-radius indexing, we formalize boundary-aware clamped windows and derive the optimal $O(N)$ amortized algorithm using a monotonic decreasing double-ended queue.
6. **1D Prefix Sums, Integral Arrays, and Moving Averages:** We introduce discrete integration via cumulative sum tables. We prove the constant-time $O(1)$ Range Sum Theorem ($\sum_{x=a}^b f(x) = F(b) - F(a-1)$), formulate Simple Moving Averages (SMA) in $O(1)$ time per query, explore Exponential Moving Averages (EMA), and connect 1D integral arrays to multi-dimensional integral images in computer vision.

---

## Learning outcomes

By the end of this session, you should be able to:

- Diagnose and resolve `RecursionError` by verifying base conditions and inspecting the runtime call stack limit with `sys.getrecursionlimit()`.
- Explain why mutating a Python `list` during `for` loop iteration corrupts iteration state, and apply non-mutating alternatives (list comprehensions and slicing copies).
- Prevent `KeyError` exceptions when accumulating frequencies by using `dict.get()`, `dict.setdefault()`, or `collections.defaultdict`.
- Identify and refactor common PEP 8 violations: operator spacing, bracket padding, naming conventions, and redundant imports.
- Explain why Python binds default argument expressions at function definition time rather than invocation time, and apply the `None` sentinel pattern for mutable defaults.
- Detect operator precedence bugs in mathematical formulas (e.g., $(-b + \sqrt{\Delta}) / 2 \cdot a$ vs. $(-b + \sqrt{\Delta}) / (2a)$).
- Construct execution trace tables to track loop variable states, condition checks, and memory mutations.
- Transpose 2D nested matrices using explicit coordinate mapping ($M[j][i] = X[i][j]$) and list comprehensions.
- Implement numerical clamping functions to saturate signals within closed intervals $[x_{\min}, x_{\max}]$.
- Trace the internal comparison and swap counts of adaptive Bubble Sort, verifying that early termination occurs when an iteration produces zero swaps.
- Build robust character and word frequency statistics functions with full lowercasing and punctuation stripping using `str.maketrans()`.
- Compare time and space complexities between iterative hash map insertion and specialized container classes (`Counter`).
- Implement 1D sliding window maximum routines across brute-force slicing, queue simulation, and anchor-radius indexing.
- Handle boundary conditions for sliding kernels using truncated/clamped windows and edge replication padding.
- Implement the optimal $O(N)$ Monotonic Deque algorithm for sliding window maximum and mathematically prove its amortized linear runtime.
- Construct 1D prefix sum (integral) arrays in $O(N)$ time and execute arbitrary interval sum queries in $O(1)$ time.
- Compute Simple Moving Averages (SMA) in constant time per window using prefix sum lookups, and evaluate variance reduction on noisy telemetry data.

---

## Session map

| Notebook | Focus | Key Topics |
| :--- | :--- | :--- |
| [01 — Python Pitfalls & Refactoring](notebooks/01-python-pitfalls-and-refactoring.ipynb) | Runtime Pitfalls & Clean Code | `RecursionError`, list mutation bugs, f-strings, `KeyError`, mutable defaults, operator precedence, PEP 8 |
| [02 — Code Reading & Execution Tracing](notebooks/02-code-reading-and-execution-tracing.ipynb) | Mental Simulation & Tracing | Trace tables, range strides, string slicing, matrix transposition, numerical clamping, Bubble Sort tracing |
| [03 — Frequency Analysis & Hash Maps](notebooks/03-character-and-word-frequency-counter.ipynb) | Vocabulary & Term Counting | Character counting, text file ingestion, punctuation stripping, `dict.get()`, `defaultdict`, `Counter` |
| [04 — Sliding Window Max Kernel](notebooks/04-sliding-window-max-kernel.ipynb) | 1D Kernel Max & Monotonic Queues | Slicing baseline, anchor-radius formulation, boundary awareness (clamped/padded), $O(N)$ monotonic deque |
| [05 — Prefix Sums & Moving Averages](notebooks/05-prefix-sums-and-moving-averages.ipynb) | Discrete Integration & Smoothing | Integral array recurrence, $O(1)$ interval range queries, SMA, EMA, telemetry noise reduction |

---

## 1. Anatomy of Runtime Pitfalls and Debugging Mechanics

### 1.1 Unbounded Recursion and the Call Stack

In Python, every function invocation allocates a **stack frame** on the process call stack. A stack frame encapsulates local variable bindings, argument references, and the return instruction pointer. Because stack memory is finite, CPython enforces a hard recursion limit (typically $1{,}000$ frames) accessible via `sys.getrecursionlimit()`.

When a recursive algorithm fails to establish a reachable **base condition**, successive frames accumulate until the interpreter raises `RecursionError: maximum recursion depth exceeded`:

```text
+---------------------------------------+
| Frame 1000: recursive_fn(n)          |  <-- Stack overflow! RecursionError
+---------------------------------------+
| ...                                   |
+---------------------------------------+
| Frame 2: recursive_fn(n)             |
+---------------------------------------+
| Frame 1: recursive_fn(n)             |
+---------------------------------------+
| Main Process Scope                    |
+---------------------------------------+
```

```python
import sys


# Buggy: Missing base condition
def buggy_recurse(n: int) -> int:
    return buggy_recurse(n)


# Defensive fix: Explicit base condition and input validation
def safe_factorial(n: int) -> int:
    if not isinstance(n, int) or n < 0:
        raise ValueError("Input must be a non-negative integer.")
    if n in (0, 1):
        return 1
    return n * safe_factorial(n - 1)
```

---

### 1.2 The Hazards of Mutating Sequences During Iteration

A ubiquitous bug in beginner and intermediate Python scripts is modifying a `list` while iterating over it using a `for` loop. Under the hood, Python's list iterator maintains an internal integer offset index $i$, incrementing it by 1 at each step until $i \ge \text{len}(\text{list})$.

#### Hazard A: Appending Inside the Loop (Infinite Loop)

```python
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    if num == 3:
        numbers.append(3)  # Infinite loop! The sequence length grows continuously
```

At step $i = 2$, `numbers[2] == 3`, triggering `append(3)`. The list length expands from 5 to 6. When the loop advances to index $i = 5$, it encounters the newly appended `3`, appends another `3`, and continues indefinitely.

#### Hazard B: Removing Inside the Loop (Skipped Elements)

```python
fruits = ["apple", "banana", "banana", "peach"]
for fruit in fruits:
    if fruit == "banana":
        fruits.remove(fruit)  # BUG: skips the second 'banana'!
```

```text
Step 0: index i = 0, item = "apple"   --> Keep.
Step 1: index i = 1, item = "banana"  --> fruits.remove("banana").
        The array shifts left: ["apple", "banana", "peach"].
Step 2: index i = 2, item = "peach"   --> The second "banana" was shifted to index 1 and was SKIPPED!
```

#### Safe Idioms

1. **List Comprehension (Pure Functional):**
   ```python
   clean_fruits = [f for f in fruits if f != "banana"]
```
2. **Iterating Over a Shallow Copy:**
   ```python
   for fruit in fruits[:]:
       if fruit == "banana":
           fruits.remove(fruit)
```

---

### 1.3 Key Lookup Failures (`KeyError`) in Hash Maps

Python dictionaries are implemented as open-addressing hash tables. Attempting to access an uninitialized key with an in-place operator like `inventory[item] += 1` evaluates the right-hand lookup `inventory[item]` first. If `item` is absent, the lookup raises `KeyError`.

```python
# Unsafe
counts = {}
for ch in "hello":
    # counts[ch] += 1  # Raises KeyError on 'h'

    # Pattern 1: Explicit membership check
    if ch in counts:
        counts[ch] += 1
    else:
        counts[ch] = 1

    # Pattern 2: dict.get() with default fallback
    counts[ch] = counts.get(ch, 0) + 1

    # Pattern 3: dict.setdefault()
    counts.setdefault(ch, 0)
    counts[ch] += 1
```

---

## 2. PEP 8 Code Quality, Code Smells, and Clean Refactoring

Readable, disciplined code reduces cognitive load and prevents software bugs. In this section, we review fundamental PEP 8 conventions and architectural refactorings.

### 2.1 Whitespace and Formatting Conventions

| Violation / Bad Practice | Correct PEP 8 Practice | Rational Rule |
| :--- | :--- | :--- |
| `fruits= [ 'apple' ,"lemon"]` | `fruits = ['apple', 'lemon']` | Consistent space around assignment; no space inside brackets; single space after comma |
| `f2 = x/y + z` | `f2 = x / y + z` | Spaces around binary operators for readability |
| `'text' . upper()` | `'text'.upper()` | No space before attribute dot access |
| `print ('hello')` | `print('hello')` | No space before function call opening parenthesis |
| `weights = [ 1.0, 2.5 ]` | `weights = [1.0, 2.5]` | Avoid extraneous whitespace immediately inside brackets |
| `print(x)#comment` | `print(x)  # Inline comment` | Inline comments require at least 2 spaces before `#` and 1 space after `#` |
| `import math, os, sys` | `import math`<br>`import os`<br>`import sys` | Individual imports on separate lines for clarity and diff hygiene |

---

### 2.2 Variable Shadowing and Magic Numbers

```python
# Bad: Shadowing built-in functions
len = 3  # Overwrites the built-in len() function!
# Later calls like len([1, 2]) will raise TypeError: 'int' object is not callable

# Bad: Magic numbers embedded in business logic
area = radius * radius * 3.14159

# Clean: Named constants and standard library precision
import math

PI = math.pi
area = (radius**2) * PI
```

---

### 2.3 The Mutable Default Argument Trap

In Python, function default arguments are evaluated **once**, at function definition time (when the `def` statement executes), not at call time. If a default argument is mutable (such as a `list` or `dict`), mutations persist across all subsequent function invocations:

```python
# DANGEROUS: The list is instantiated once at import time
def add_observation(val: float, observations: list[float] = []) -> list[float]:
    observations.append(val)
    return observations


# Call 1:
res1 = add_observation(10.0)  # [10.0]
# Call 2:
res2 = add_observation(20.0)  # [10.0, 20.0]  <-- UNEXPECTED SHARED STATE!
```

#### The Canonical Solution: Sentinel `None`

```python
def add_observation_safe(val: float, observations: list[float] | None = None) -> list[float]:
    if observations is None:
        observations = []
    observations.append(val)
    return observations
```

---

### 2.4 Operator Precedence in Mathematical Formulas

In arithmetic expressions involving division and multiplication, Python evaluates operators of equal precedence strictly from **left to right**.

Consider the quadratic formula roots:
$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

```python
# CRITICAL BUG:
x = (-b + math.sqrt(delta)) / 2 * a
# Evaluates as: (((-b + sqrt(delta)) / 2) * a)
# Divides by 2, then multiplies the result by a!

# CORRECT:
x = (-b + math.sqrt(delta)) / (2 * a)
```

---

### 2.5 Dictionary Dispatch Tables vs. Sprawling `if-elif` Chains

In machine learning and control systems, branching on discrete string identifiers (e.g., optimizer types, loss functions, activation names) via long `if-elif-else` ladders increases cyclomatic complexity:

```python
from typing import Callable

# Clean Dictionary Dispatch Table: O(1) direct lookup
Operator = Callable[[float, float], float]

MATH_OPERATIONS: dict[str, Operator] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b if b != 0 else float("nan"),
}


def execute_operation(op: str, a: float, b: float) -> float:
    action = MATH_OPERATIONS.get(op)
    if action is None:
        raise ValueError(f"Unsupported operation: {op}")
    return action(a, b)
```

---

## 3. Mental Execution Tracing and Code Comprehension

Accurate code reading requires mentally simulating variable assignments, condition checks, and memory states using structured **Trace Tables**.

### 3.1 Trace Table: Loop Accumulation with Early Termination

Consider the following snippet:

```python
accumulator = 0
for step in range(5):
    accumulator += step
    if accumulator > 0 and accumulator % 3 == 0:
        break
print(accumulator)
```

| Step Index | Loop Variable `step` | `accumulator += step` | Condition `accumulator > 0 and accumulator % 3 == 0` | Action |
| :---: | :---: | :---: | :---: | :--- |
| Initial | — | $0$ | — | Enter loop |
| 0 | $0$ | $0 + 0 = 0$ | $0 > 0 \land 0 \equiv 0 \implies \text{False}$ | Continue |
| 1 | $1$ | $0 + 1 = 1$ | $1 > 0 \land 1 \equiv 1 \implies \text{False}$ | Continue |
| 2 | $2$ | $1 + 2 = 3$ | $3 > 0 \land 3 \pmod 3 == 0 \implies \text{True}$ | **Break** loop |

**Result:** The loop halts after step 2. Final printed value is `3`.

---

### 3.2 2D Matrix Transposition: Explicit Nested Index Mapping

Given a matrix $X \in \mathbb{R}^{R \times C}$, its transpose $X^T \in \mathbb{R}^{C \times R}$ maps element $(r, c)$ to $(c, r)$:

$$X^T[c][r] = X[r][c]$$

```text
Original Matrix X (2 rows x 3 columns):
[ [1,  2,  3],
  [4,  5,  6] ]

Transposed Matrix X^T (3 rows x 2 columns):
[ [1, 4],
  [2, 5],
  [3, 6] ]
```

```python
def transpose_matrix(X: list[list[float]]) -> list[list[float]]:
    rows = len(X)
    cols = len(X[0])
    # Preallocate transposed matrix with zeros
    transposed = [[0.0 for _ in range(rows)] for _ in range(cols)]
    for r in range(rows):
        for c in range(cols):
            transposed[c][r] = X[r][c]
    return transposed
```

---

### 3.3 Numerical Saturation and Value Clamping

In signal acquisition and neural network gradient clipping, values must often be bounded within a specified range $[v_{\min}, v_{\max}]$:

$$\text{clamp}(x, v_{\min}, v_{\max}) = \max(v_{\min}, \min(x, v_{\max}))$$

```python
def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(value, max_val))
```

---

## 4. Frequency Analysis and Hash Table Data Structures

### 4.1 Character Frequency Counting

Given a string $S$ containing characters from $[a-zA-Z]$, count how often each character occurs regardless of case.

#### Algorithm Steps

1. Normalize input string to lowercase: $S_{\text{norm}} = S.\text{lower}()$.
2. Initialize an empty hash map: $H \leftarrow \{\}$.
3. For each character $c \in S_{\text{norm}}$:
   - If $c$ is alphabetic, increment $H[c] = H.\text{get}(c, 0) + 1$.
4. Return $H$.

```text
Input: "Happiness"
Normalization: "happiness"
Trace:
'h' -> 1
'a' -> 1
'p' -> 2
'i' -> 1
'n' -> 1
'e' -> 1
's' -> 2
Result: {'h': 1, 'a': 1, 'p': 2, 'i': 1, 'n': 1, 'e': 1, 's': 2}
```

---

### 4.2 Word Frequency Counting from External Text Files

In natural language processing, converting raw prose into a bag-of-words representation requires text sanitization:

```text
Raw Text: "AI is a topic that is very interesting. However, it is not easy."
                      |
                      v  [str.maketrans() removes punctuation: . , ! ?]
Sanitized: "ai is a topic that is very interesting  however  it is not easy "
                      |
                      v  [.split() tokenizes on arbitrary whitespace]
Tokens:    ['ai', 'is', 'a', 'topic', 'that', 'is', 'very', 'interesting', 'however', 'it', 'is', 'not', 'easy']
                      |
                      v  [Hash map frequency accumulation]
Vocabulary: {'is': 3, 'ai': 1, 'a': 1, 'topic': 1, 'that': 1, 'very': 1, ...}
```

```python
import string
from pathlib import Path


def count_words_in_file(path: Path | str) -> dict[str, int]:
    text = Path(path).read_text(encoding="utf-8")
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    tokens = text.translate(translator).lower().split()

    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts
```

---

## 5. 1D Sliding Window Maximum (Kernel Max)

### 5.1 Formal Problem Definition

Given an input array $A = [a_0, a_1, \dots, a_{N-1}]$ of $N$ real numbers and a window size $k \in [1, N]$, compute the maximum value for each contiguous subarray of length $k$ as the window shifts right by 1 index at each step.

The total number of valid full windows is:
$$W = N - k + 1$$

For each window index $j \in [0, W - 1]$:
$$M[j] = \max_{0 \le t < k} A[j + t] = \max(A[j : j + k])$$

```text
Example: A = [3, 4, 5, 1, -44, 5, 10, 12, 33, 1], k = 3
Window 0: [3, 4, 5]            --> max = 5
Window 1:    [4, 5, 1]         --> max = 5
Window 2:       [5, 1, -44]    --> max = 5
Window 3:          [1, -44, 5] --> max = 5
Window 4:             [-44, 5, 10] --> max = 10
Window 5:                 [5, 10, 12] --> max = 12
Window 6:                    [10, 12, 33] --> max = 33
Window 7:                        [12, 33, 1] --> max = 33

Output: [5, 5, 5, 5, 10, 12, 33, 33]
```

---

### 5.2 Anchor-Radius Formulation and Boundary Awareness

In digital signal processing and 1D convolution kernels, filters are often centered at an **anchor index** $i$ with half-width **radius** $r = \lfloor k / 2 \rfloor$.
For odd window sizes $k = 2r + 1$, an anchor $i$ spans the symmetric interval $[i - r, i + r]$.

When anchor $i$ is near the array boundaries ($i < r$ or $i \ge N - r$), the nominal kernel exceeds the array bounds. Two standard boundary treatments are used:

1. **Clamped / Truncated Window:**
   $$\text{Window}(i) = A[\max(0, i - r) : \min(N, i + r + 1)]$$
2. **Replication Padding:**
   Prepend $r$ copies of $a_0$ to the left and append $r$ copies of $a_{N-1}$ to the right, preserving input dimensionality.

```python
def sliding_max_clamped(arr: list[float], k: int) -> list[float]:
    n = len(arr)
    radius = k // 2
    return [max(arr[max(0, i - radius) : min(n, i + radius + 1)]) for i in range(n)]
```

---

### 5.3 Optimal $O(N)$ Monotonic Deque Algorithm

While naive sub-array slicing consumes $O(N \cdot k)$ time, the optimal algorithm runs in strict **linear $O(N)$ time** using a **Monotonic Decreasing Double-Ended Queue (Deque)**.

#### Algorithmic Invariants

1. The deque stores **indices** of elements from the input array.
2. The values corresponding to the stored indices are maintained in **strictly decreasing order**:
   $$A[\text{dq}[0]] > A[\text{dq}[1]] > \dots > A[\text{dq}[m-1]]$$
3. The element at the front of the deque (`dq[0]`) always stores the maximum value for the active window.

```text
Processing element A[i]:
  1. Evict Outdated Indices:
     If dq[0] < i - k + 1, dq.popleft() (index has slid out of window).
  2. Maintain Monotonicity:
     While dq and A[dq[-1]] <= A[i]:
         dq.pop() (elements smaller than current can never be window maximum again).
  3. Push Current Index:
     dq.append(i).
  4. Record Output:
     If i >= k - 1, output A[dq[0]].
```

#### Amortized Complexity Proof

Every array index $i \in \{0, 1, \dots, N-1\}$ is appended to the deque exactly once and popped from the deque at most once. The total number of deque operations across the entire algorithm is bounded by $2N$. Therefore, the amortized time complexity is $O(N)$, independent of window size $k$.

```python
from collections import deque


def sliding_window_max_monotonic(num_list: list[int | float], k: int) -> list[int | float]:
    if not num_list or k <= 0:
        return []
    if k > len(num_list):
        return [max(num_list)]

    dq: deque[int] = deque()
    result: list[int | float] = []

    for i, val in enumerate(num_list):
        if dq and dq[0] < i - k + 1:
            dq.popleft()

        while dq and num_list[dq[-1]] <= val:
            dq.pop()

        dq.append(i)

        if i >= k - 1:
            result.append(num_list[dq[0]])

    return result
```

---

## 6. 1D Prefix Sums, Integral Arrays, and Moving Averages

### 6.1 The Integral Array and Discrete Integration

Let $f(x)$ be a discrete sequence of $N$ real numbers indexed by $x \in \{0, 1, \dots, N-1\}$. We define the **Prefix Sum Array** (also known as the **Integral Array**) $F(x)$ as the cumulative sum from index $0$ to $x$:

$$F(x) = \sum_{t=0}^x f(t)$$

#### Recurrence Relation
$$F(0) = f(0)$$
$$F(x) = F(x - 1) + f(x) \quad \text{for } x \ge 1$$

Constructing $F(x)$ requires a single linear pass over the array: $O(N)$ time and $O(N)$ auxiliary space.

```text
x:    0    1    2    3    4    5    6    7    8    9
f(x): 1    8    5    7    3    5    8    3    2    9
      |  + |  + |  + |  + |  + |  + |  + |  + |  + |
F(x): 1    9   14   21   24   29   37   40   42   51
```

---

### 6.2 The $O(1)$ Range Sum Theorem

**Theorem:** For any arbitrary query range $[a, b]$ where $0 \le a \le b < N$:
$$\sum_{x=a}^b f(x) = F(b) - F(a - 1) \quad (\text{with } F(-1) \equiv 0)$$

#### Proof
By definition:
$$F(b) = \sum_{x=0}^b f(x) = \sum_{x=0}^{a-1} f(x) + \sum_{x=a}^b f(x)$$
Subtracting $\sum_{x=0}^{a-1} f(x) = F(a-1)$ from both sides yields:
$$\sum_{x=a}^b f(x) = F(b) - F(a - 1) \quad \blacksquare$$

#### Example from Lecture Slides
- Calculate the area under $f(x)$ for interval $x \in [4, 6]$:
  $$F(6) = 37$$
  $$F(3) = 21$$
  $$\text{Area } A = F(6) - F(3) = 37 - 21 = 16$$
- Direct sum verification: $f(4) + f(5) + f(6) = 3 + 5 + 8 = 16$. The query executes in exact $O(1)$ arithmetic operations.

---

### 6.3 Simple Moving Average (SMA) via Prefix Sums

The Simple Moving Average (SMA) of window size $k$ smooths high-frequency fluctuations in streaming signals:

$$\text{SMA}_k(i) = \frac{1}{k} \sum_{t=i - k + 1}^i f(t)$$

Using the prefix sum table $F$, each SMA term is computed in $O(1)$ time:
$$\text{SMA}_k(i) = \frac{F(i) - F(i - k)}{k}$$

```python
def moving_average_prefix(arr: list[float], k: int) -> list[float]:
    if not arr or k <= 0 or k > len(arr):
        return []

    # Build prefix sum table in O(N)
    prefix = [0.0] * len(arr)
    prefix[0] = float(arr[0])
    for i in range(1, len(arr)):
        prefix[i] = prefix[i - 1] + float(arr[i])

    # Compute all window averages in O(1) each -> O(N) total
    sma = []
    for i in range(k - 1, len(arr)):
        window_sum = prefix[i] if i == k - 1 else prefix[i] - prefix[i - k]
        sma.append(window_sum / k)
    return sma
```

---

## 7. Comparative Computational Complexity

| Algorithm / Operation | Time Complexity | Auxiliary Space | Deterministic / Optimal | Primary Engineering Application |
| :--- | :---: | :---: | :---: | :--- |
| **Character Frequency (`dict.get`)** | $O(L)$ | $O(\Sigma)$ | Optimal | String tokenization, alphabet profiling |
| **Word Frequency (`translate` + `split`)** | $O(N + W)$ | $O(V)$ | Optimal | Vocabulary building, document statistics |
| **Sliding Window Max (Naive Slicing)** | $O(N \cdot k)$ | $O(1)$ | Non-optimal | Small window prototypes ($k \le 3$) |
| **Sliding Window Max (Monotonic Deque)**| $O(N)$ | $O(k)$ | Optimal | Real-time telemetry, 1D CNN pooling |
| **Prefix Sum Array Construction** | $O(N)$ | $O(N)$ | Optimal | Precomputing cumulative density functions |
| **Interval Range Sum Query** | $O(1)$ | $O(1)$ | Optimal | Real-time interval aggregation, bounding boxes |
| **Simple Moving Average (via Prefix Sum)**| $O(N)$ | $O(N)$ | Optimal | Sensor filtering, signal trend detection |
