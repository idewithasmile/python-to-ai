# Brute Force and Exhaustive Search

[Back to Module 01](../README.md) · [Back to the master roadmap](../../README.md)

**Session date:** 2026-10-28 · **Module:** 01 — Python Fundamentals & Algorithmic Thinking  
**Format:** Guided reading, executable notebooks, formal complexity proofs, and combinatorial optimization labs

---

## Overview

In computer science and machine learning, algorithm design paradigms dictate how problems are formulated, decomposed, and solved. Before applying sophisticated heuristics, divide-and-conquer strategies, dynamic programming, or gradient-based optimizations, software engineers and AI practitioners begin with the foundational baseline: **Brute Force** and **Exhaustive Search**.

The **brute force** approach is a direct, straightforward methodology that solves a given computational problem in the most obvious way, directly adhering to the problem statement and definitions of the concepts involved. While brute force algorithms are rarely optimal for large-scale datasets, they provide three vital software engineering and research advantages:

1. **Theoretical correctness and completeness:** Because brute force algorithms systematically examine the problem definition or enumerate the entire candidate solution space, they guarantee discovering the global optimum (or proving that no solution exists).
2. **Gold standard baseline:** Advanced algorithms (such as heuristic search, metaheuristics, branch-and-bound, or deep neural networks) require an exact benchmark to quantify approximation error, precision loss, and computational acceleration.
3. **Simplicity and low implementation overhead:** When input sizes $n$ are small (e.g., configuring micro-architectures, tuning 3 to 4 hyperparameters, or sorting tiny sub-arrays in hybrid sort algorithms like Timsort), the low constant factors and absence of memory allocation overhead make brute force faster and more reliable than complex algorithms.

This session develops the theory, mechanics, mathematical analysis, and practical implementations of brute force and exhaustive search across three core computational paradigms:

1. **Searching Problems:** We analyze **Linear Search** (sequential scan) across unsorted and sorted arrays, implement the **Sentinel Linear Search** optimization to eliminate per-iteration boundary checks, and connect linear scanning to exact nearest-neighbor search (**Flat Index**) in vector databases.
2. **Sorting Problems:** We dissect **Selection Sort** (in-place minima extraction) and **Bubble Sort** (adjacent comparison and bubbling). We formally derive the arithmetic series for comparison counts ($\frac{n(n-1)}{2}$), correct common textbook misprints, analyze write-optimality ($O(n)$ swaps in Selection Sort), connect Bubble Sort swaps to permutation inversions, and examine adaptive early-termination strategies.
3. **Exhaustive Search & Combinatorial Optimization:** We define exhaustive search over discrete state spaces, formulate permutation generation algorithms (iterative insertion, recursive backtracking, and Heap's algorithm), and solve the NP-hard **Travelling Salesman Problem (TSP)** from first principles.
4. **Machine Learning Applications:** We bridge combinatorial search to practical artificial intelligence: solving the 0/1 Knapsack Problem via power-set generation, implementing **Best Subset Selection** for sparse linear modeling across $2^p - 1$ feature subsets, and building a multi-parameter **Grid Search** tuner.

---

## Learning outcomes

By the end of this session, you should be able to:

- Formulate algorithms using the brute force design paradigm based directly on mathematical definitions.
- Implement standard and sentinel linear search in Python with full type annotations.
- Construct line-by-line cost-and-frequency tables to derive exact Best, Worst, and Average case time complexities for search routines.
- Implement Selection Sort from first principles, maintaining loop invariants across in-place sub-array partitions.
- Mathematically derive the exact comparison count $\sum_{i=1}^{n-1} i = \frac{n(n-1)}{2}$ for quadratic sorting algorithms.
- Implement standard and adaptive Bubble Sort with early termination flags, reducing best-case runtime from $\Theta(n^2)$ to $\Theta(n)$.
- Prove that the total number of adjacent swaps executed by Bubble Sort equals the exact inversion count of the input array.
- Contrast Selection Sort and Bubble Sort across runtime, memory writes (swaps), algorithmic stability, and cache locality.
- Generate all $n!$ permutations of a sequence using iterative insertion, recursive backtracking, and Heap's algorithm.
- Generate all $2^n$ subsets of a set using bit manipulation (binary masks) and recursive combinations.
- Formulate the Travelling Salesman Problem (TSP) using graph distance matrices and solve it to global optimality via exhaustive permutation evaluation.
- Analyze the super-exponential scaling wall of factorial ($n!$) and exponential ($2^n$) algorithms.
- Apply exhaustive search to machine learning workflows, including 0/1 Knapsack resource allocation, Best Subset Selection in regression, and Cartesian Grid Search hyperparameter optimization.

---

## Session map

| Notebook | Focus | Key Topics |
| :--- | :--- | :--- |
| [01 — Linear Search](notebooks/01-linear-search-and-sequential-scan.ipynb) | Sequential Scanning & Cost Counting | RAM model, sentinel optimization, cost-frequency tables, Best/Average/Worst cases, Flat Index exact KNN |
| [02 — Selection Sort](notebooks/02-selection-sort-and-in-place-minima.ipynb) | In-Place Minima Selection | Suffix scan, in-place swap, loop invariants, $n(n-1)/2$ derivation, write-optimality ($O(n)$ swaps), instability proof |
| [03 — Bubble Sort](notebooks/03-bubble-sort-and-adaptive-swapping.ipynb) | Adjacent Swaps & Adaptive Flags | Pairwise bubbling, early stopping flag, inversion count theorem, stability preservation, Cocktail Shaker Sort |
| [04 — Permutations](notebooks/04-permutation-generation-and-combinatorics.ipynb) | Combinatorial Generation | Iterative insertion, recursive backtracking, Heap's algorithm, power set generation ($2^n$), growth comparisons |
| [05 — TSP Solver](notebooks/05-travelling-salesman-problem.ipynb) | Travelling Salesman Problem | Complete graph adjacency matrix, $(n-1)!$ tour space, path cost calculation, global optimality proof, scaling barrier |
| [06 — ML Applications](notebooks/06-exhaustive-search-lab-and-ml-applications.ipynb) | Applied Combinatorial Search | 0/1 Knapsack, Best Subset Selection ($2^p - 1$ models), Cartesian Grid Search, paradigm trade-offs |

---

## 1. The Brute Force Paradigm & Direct Algorithm Design

### 1.1 Core Principles

The term **Brute Force** denotes an algorithm design strategy that solves a problem solely based on the problem statement and the definitions of the mathematical concepts involved. 

```text
+------------------------------------------------------------------------+
|                          BRUTE FORCE PARADIGM                          |
+------------------------------------------------------------------------+
|  1. Inspect Problem Definition                                         |
|     - What is the explicit mathematical specification of the input?    |
|     - What predicate or objective function defines the valid output?   |
|  2. Direct Translation into Computation                                |
|     - Map the mathematical formula directly into procedural loops.     |
|     - Do not introduce pre-sorting, caching, or algebraic shortcuts.  |
|  3. Exhaustive Verification                                            |
|     - Check candidates sequentially or evaluate the full search space. |
|     - Return the first match or the global extremum.                   |
+------------------------------------------------------------------------+
```

### 1.2 Classical Introductory Examples

#### Example 1: Integer Exponentiation ($A^n$)

- **Problem:** Given a non-zero base $A \in \mathbb{R}$ and a non-negative integer exponent $n \in \mathbb{N}_0$, compute $A^n$.
- **Definition:** $A^n = \underbrace{A \times A \times \dots \times A}_{n \text{ factors}}$ with $A^0 = 1$.
- **Brute Force Algorithm:** Initialize `result = 1.0` and multiply by $A$ sequentially $n$ times:

```python
def power_brute_force(base: float, exponent: int) -> float:
    result = 1.0
    for _ in range(exponent):
        result *= base
    return result
```

- **Analysis:** Requires exactly $n$ multiplications ($O(n)$ time) and $O(1)$ auxiliary memory. (By contrast, Exponentiation by Squaring reduces this to $O(\log n)$ by exploiting $A^n = (A^{n/2})^2$).

#### Example 2: Factorial Computation ($n!$)

- **Problem:** Compute the factorial of a non-negative integer $n \in \mathbb{N}_0$.
- **Definition:** $n! = \prod_{i=1}^n i = 1 \times 2 \times \dots \times n$ with $0! = 1$.
- **Brute Force Algorithm:** Accumulate the running product of integers from $1$ up to $n$ in a simple linear loop ($O(n)$ time).

```python
def factorial_brute_force(n: int) -> int:
    product = 1
    for i in range(1, n + 1):
        product *= i
    return product
```

---

## 2. Searching Problems: Linear Search (Sequential Scan)

### 2.1 Problem Specification

- **Input:** A sequence $A = \langle a_0, a_1, \dots, a_{n-1} \rangle$ of $n$ elements and a query value `key`.
- **Output:** The index $i \in [0, n-1]$ such that $A[i] = \text{key}$, or $-1$ if $\text{key} \notin A$.

### 2.2 Algorithm Mechanics

Linear Search evaluates elements sequentially starting at index $0$. At each step $i$, it evaluates whether $A[i] == \text{key}$. If true, execution halts immediately and returns index $i$. If the index reaches $n$ without a match, it returns $-1$.

```text
Sequence:   [  1,   2,  10,  15,  23 ]     Target: key = 23
Index:         0    1    2    3    4

Step 0: Check A[0] == 23 ->  1 == 23 (False) -> Increment index
Step 1: Check A[1] == 23 ->  2 == 23 (False) -> Increment index
Step 2: Check A[2] == 23 -> 10 == 23 (False) -> Increment index
Step 3: Check A[3] == 23 -> 15 == 23 (False) -> Increment index
Step 4: Check A[4] == 23 -> 23 == 23 (True)  -> Return Index 4
```

### 2.3 Formal Cost-and-Frequency Analysis

Under the Random Access Machine (RAM) model, each primitive operation executes in unit time $c_i$:

```python
def linear_search(array: list[int], key: int) -> int:
    n = len(array)  # cost c1, 1 time
    i = 0  # cost c2, 1 time
    while i < n:  # cost c3, executed (k + 1) times
        if array[i] == key:  # cost c4, executed k times
            return i  # cost c5, executed 1 time (if found)
        i += 1  # cost c6, executed (k - 1) times
    return -1  # cost c7, executed 1 time (if not found)
```

Where $k$ denotes the number of elements examined before termination ($1 \le k \le n$):

1. **Best Case ($k = 1$):**
   The query `key` matches the very first element $A[0]$.
   $$T_{\text{best}}(n) = c_1 + c_2 + 2c_3 + c_4 + c_5 = \Theta(1)$$
2. **Worst Case ($k = n$ and key absent):**
   The query `key` is not present in the sequence. The loop runs all $n$ iterations and evaluates the loop condition $n + 1$ times.
   $$T_{\text{worst}}(n) = c_1 + c_2 + (n+1)c_3 + n c_4 + n c_6 + c_7 = c_{\text{loop}} \cdot n + c_{\text{init}} = \Theta(n)$$
3. **Average Case:**
   Assume the query `key` is present in the sequence with probability $p$, and uniformly distributed across all $n$ index positions with equal probability $p/n$. If absent (probability $1 - p$), the algorithm inspects $n$ elements.
   $$\mathbb{E}[K] = \sum_{i=1}^n \left( \frac{p}{n} \cdot i \right) + (1 - p) \cdot n = \frac{p}{n} \frac{n(n+1)}{2} + (1 - p)n = p \frac{n+1}{2} + (1 - p)n$$
   When the target is guaranteed to be present ($p = 1$):
   $$\mathbb{E}[K] = \frac{n+1}{2} = \Theta(n)$$

### 2.4 Sentinel Linear Search Optimization

In standard linear search, every single iteration evaluates two boolean predicates:
1. `i < n` (Array bounds check)
2. `array[i] == key` (Target equality check)

In tight performance-critical loops, the array bounds check accounts for roughly half of all branch instructions. The **Sentinel Search** technique eliminates the bounds check by temporarily writing `key` to the end of the array (at index $n-1$), guaranteeing that the search will terminate without index overflow:

```python
def sentinel_linear_search(array: list[int], key: int) -> int:
    n = len(array)
    if n == 0:
        return -1

    last_val = array[-1]
    if last_val == key:
        return n - 1

    array[-1] = key  # Place sentinel
    i = 0
    while array[i] != key:  # Only one comparison per iteration!
        i += 1

    array[-1] = last_val  # Restore original value

    if i < n - 1 or last_val == key:
        return i
    return -1
```

---

## 3. Sorting Problems: Selection Sort

### 3.1 Problem Specification

- **Input:** An unordered sequence of $n$ comparable elements $A = \langle a_0, a_1, \dots, a_{n-1} \rangle$.
- **Output:** A permutation $A' = \langle a'_0, a'_1, \dots, a'_{n-1} \rangle$ such that $a'_0 \le a'_1 \le \dots \le a'_{n-1}$.

### 3.2 Algorithmic Strategy

Selection Sort divides the array into two logical partitions:
1. A **sorted prefix** occupying indices $[0 \dots \text{step} - 1]$.
2. An **unsorted suffix** occupying indices $[\text{step} \dots n - 1]$.

At each iteration `step`, the algorithm scans the entire unsorted suffix to identify the index of the minimum element, and then swaps that minimum element into index `step`.

```text
Initial Array: [  7,   2,   5,   4,  17 ]
---------------------------------------------------------------------
Step 0: Scan [7, 2, 5, 4, 17] -> Min is 2 at idx 1 -> Swap(7, 2)
        Result: [ 2 |  7,   5,   4,  17 ]
Step 1: Scan [7, 5, 4, 17]    -> Min is 4 at idx 3 -> Swap(7, 4)
        Result: [ 2,   4 |  5,   7,  17 ]
Step 2: Scan [5, 7, 17]       -> Min is 5 at idx 2 -> In place
        Result: [ 2,   4,   5 |  7,  17 ]
Step 3: Scan [7, 17]          -> Min is 7 at idx 3 -> In place
        Result: [ 2,   4,   5,   7 | 17 ]
---------------------------------------------------------------------
Final Sorted Array: [ 2, 4, 5, 7, 17 ]
```

### 3.3 Implementation

```python
def selection_sort(array: list[int]) -> list[int]:
    n = len(array)
    for step in range(n):
        min_idx = step
        for i in range(step + 1, n):
            if array[i] < array[min_idx]:
                min_idx = i

        array[step], array[min_idx] = array[min_idx], array[step]

    return array
```

### 3.4 Mathematical Derivation of Comparison Count

In Selection Sort, the number of comparisons is strictly invariant with respect to the initial ordering of the data. Whether the array is already sorted, reversely sorted, or random:

- For `step = 0`: Inner loop compares $A[i]$ with $A[\text{min\_idx}]$ for $i \in [1 \dots n-1]$ $\implies n - 1$ comparisons.
- For `step = 1`: Inner loop compares for $i \in [2 \dots n-1]$ $\implies n - 2$ comparisons.
- For `step = k`: Inner loop performs $n - 1 - k$ comparisons.
- For `step = n - 2`: Inner loop performs $1$ comparison.

Summing across all outer loop cycles:
$$C(n) = \sum_{k=0}^{n-2} (n - 1 - k) = (n - 1) + (n - 2) + \dots + 2 + 1 = \sum_{j=1}^{n-1} j$$

Using the closed-form arithmetic progression formula $\sum_{j=1}^m j = \frac{m(m+1)}{2}$ with $m = n - 1$:
$$C(n) = \frac{(n - 1)((n - 1) + 1)}{2} = \frac{n(n - 1)}{2} = \frac{n^2 - n}{2}$$

> [!IMPORTANT]
> **Errata Note on Slide Materials:**
> The introductory slide notes state the sum as $n(n-2)/2$. This is a typographical slip in the slide deck. As rigorously proven above via the arithmetic series $\sum_{j=1}^{n-1} j$, the exact closed form is $\frac{n(n-1)}{2}$. For example, for $n = 5$:
> $$\text{Actual comparisons} = 4 + 3 + 2 + 1 = 10$$
> Using $\frac{n(n-1)}{2} = \frac{5 \times 4}{2} = 10$, which matches ground truth.

### 3.5 Write Optimality and Algorithmic Stability

- **Memory Writes ($O(n)$):** While Selection Sort executes $\Theta(n^2)$ comparisons, it performs at most $n - 1$ array swap operations (writes to memory). This property makes Selection Sort attractive in embedded systems where writes to non-volatile flash memory are substantially slower or wear out physical storage cells.
- **Algorithmic Instability:** Selection Sort is **not stable**. A long-range swap can leap over identical keys and invert their relative sequence:
  ```text
  Input: [(2, "alpha"), (2, "beta"), (1, "gamma")]
  Step 0: Min is (1, "gamma") at idx 2.
          Swap with index 0:
  Result: [(1, "gamma"), (2, "beta"), (2, "alpha")]
  ```
  The original relative ordering of `(2, "alpha")` before `(2, "beta")` was destroyed.

---

## 4. Sorting Problems: Bubble Sort

### 4.1 Algorithmic Strategy

Bubble Sort operates on a complementary brute-force intuition: rather than selecting the minimum and placing it at the front, it repeatedly passes through the array, comparing adjacent elements $(A[i], A[i+1])$ and swapping them if $A[i] > A[i+1]$.

Through successive passes, larger elements gradually migrate ("bubble up") to the end of the array:
- Pass $0$ guarantees that the global maximum settles at index $n - 1$.
- Pass $1$ guarantees that the second largest element settles at index $n - 2$.
- Pass $k$ guarantees that the suffix $[n - 1 - k \dots n - 1]$ is fully sorted.

```text
Initial Array: [  7,   2,   5,   4,  17 ]
---------------------------------------------------------------------
Pass 0:
  i = 0: Compare (7, 2)  -> 7 > 2  -> Swap  -> [ 2, 7, 5, 4, 17 ]
  i = 1: Compare (7, 5)  -> 7 > 5  -> Swap  -> [ 2, 5, 7, 4, 17 ]
  i = 2: Compare (7, 4)  -> 7 > 4  -> Swap  -> [ 2, 5, 4, 7, 17 ]
  i = 3: Compare (7, 17) -> 7 < 17 -> OK    -> [ 2, 5, 4, 7, 17 ]
  Largest element 17 settled at end.

Pass 1:
  i = 0: Compare (2, 5)  -> 2 < 5  -> OK    -> [ 2, 5, 4, 7, 17 ]
  i = 1: Compare (5, 4)  -> 5 > 4  -> Swap  -> [ 2, 4, 5, 7, 17 ]
  i = 2: Compare (5, 7)  -> 5 < 7  -> OK    -> [ 2, 4, 5, 7, 17 ]
  Second largest element 7 settled.

Pass 2:
  i = 0: Compare (2, 4)  -> OK
  i = 1: Compare (4, 5)  -> OK
  No swaps performed!
---------------------------------------------------------------------
Final Sorted Array: [ 2, 4, 5, 7, 17 ]
```

### 4.2 Implementation: Standard vs Adaptive (Early Termination)

In unoptimized Bubble Sort, all $n - 1$ passes are executed regardless of whether the array is already sorted. By introducing a boolean flag `swapped = False`, the algorithm detects when a pass completes without making any exchanges, terminating immediately in $O(n)$ time.

```python
def bubble_sort(array: list[int]) -> list[int]:
    n = len(array)
    for step in range(n):
        swapped = False
        for i in range(0, n - step - 1):
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                swapped = True
        # Early termination: array is already sorted
        if not swapped:
            break
    return array
```

### 4.3 Inversions and Swap Equivalence Theorem

Let $A = \langle a_0, a_1, \dots, a_{n-1} \rangle$ be a sequence of distinct numbers.

> [!NOTE]
> **Definition (Inversion):** An inversion of sequence $A$ is an ordered pair of indices $(i, j)$ such that $i < j$ and $A[i] > A[j]$.

**Theorem:** *The total number of adjacent swaps performed by Bubble Sort on sequence $A$ is strictly equal to the number of inversions in $A$.*

**Proof sketch:**
1. Any adjacent swap of $A[i]$ and $A[i+1]$ with $A[i] > A[i+1]$ transforms the pair into $A'[i] < A'[i+1]$. This single operation resolves exactly one inversion between index $i$ and index $i+1$.
2. For all other indices $k \notin \{i, i+1\}$, the relative order between $A[k]$ and the swapped elements remains unchanged. Hence, no other inversions are created or destroyed.
3. Because the sorted array has $0$ inversions, and each swap decrements the inversion count by exactly $1$, the total number of swaps must equal the initial inversion count $I(A)$. $\blacksquare$

### 4.4 Selection Sort vs Bubble Sort Comparison

| Attribute | Selection Sort | Bubble Sort (Adaptive) |
| :--- | :--- | :--- |
| **Best-Case Time** | $\Theta(n^2)$ | $\Theta(n)$ (Already sorted, single pass) |
| **Average-Case Time** | $\Theta(n^2)$ | $\Theta(n^2)$ |
| **Worst-Case Time** | $\Theta(n^2)$ | $\Theta(n^2)$ |
| **Comparisons** | $\frac{n(n-1)}{2}$ (Strictly invariant) | Range: $n-1$ (best) to $\frac{n(n-1)}{2}$ (worst) |
| **Memory Writes (Swaps)** | At most $n - 1 = O(n)$ | Equal to inversions = up to $\frac{n(n-1)}{2} = O(n^2)$ |
| **Algorithmic Stability** | **Unstable** (Long-range jumps) | **Stable** (Swaps only strictly out-of-order adjacent items) |
| **Cache Locality** | Reads linear, writes minimal | High adjacent spatial locality, high cache-line reuse |

---

## 5. Exhaustive Search & Combinatorial Enumeration

### 5.1 The Nature of Exhaustive Search

When a computational problem seeks an optimal element or configuration among a discrete set of combinatorial possibilities, **Exhaustive Search** represents the combinatorial extension of brute force.

```text
+---------------------------------------------------------------+
|                   EXHAUSTIVE SEARCH PIPELINE                  |
+---------------------------------------------------------------+
|  1. Solution Space Generation                                 |
|     Systematically generate every candidate solution in the  |
|     discrete state space (permutations, subsets, or trees).  |
|                                |                              |
|                                v                              |
|  2. Constraint Filtering & Objective Evaluation               |
|     Evaluate each candidate against problem constraints.     |
|     Compute cost/fitness: J(s) for candidate s.               |
|                                |                              |
|                                v                              |
|  3. Global Extrema Retention                                  |
|     Track s* = argmin J(s). Update current best solution.     |
|                                |                              |
|                                v                              |
|  4. Output Global Optimum                                     |
|     Return verified optimal configuration s*.                 |
+---------------------------------------------------------------+
```

### 5.2 Combinatorial Search Spaces

1. **Permutations ($n!$):**
   Used when solution states correspond to an ordering or arrangement of $n$ distinct entities (e.g., Travelling Salesman Problem, Job-Shop Scheduling).
   $$P(n) = n! = n \times (n-1) \times \dots \times 1$$
2. **Subsets / Power Set ($2^n$):**
   Used when solution states correspond to binary inclusion/exclusion decisions for $n$ items (e.g., 0/1 Knapsack Problem, Feature Subset Selection, Boolean Satisfiability SAT).
   $$|\mathcal{P}(S)| = 2^n$$
3. **Combinations ($\binom{n}{k}$):**
   Used when selecting an unordered subset of fixed size $k$ from $n$ candidate choices.
   $$\binom{n}{k} = \frac{n!}{k!(n - k)!}$$

### 5.3 Permutation Generation Algorithms

#### 1. Iterative Insertion Method

Starting from a singleton list $[[x_0]]$, iteratively take element $x_i$ and insert it into all $i + 1$ possible slot positions of each existing permutation:

```python
def permute_iterative(nums: list[int]) -> list[list[int]]:
    result = []
    n = len(nums)
    i = 0
    while i < n:
        if not result:
            result.append([nums[i]])
        else:
            new_perms = []
            for perm in result:
                for j in range(i + 1):
                    new_p = perm.copy()
                    new_p.insert(j, nums[i])
                    new_perms.append(new_p)
            result = new_perms
        i += 1
    return result
```

#### 2. Recursive Backtracking (In-Place Swapping)

Generates all permutations using depth-first search on the decision tree by swapping candidates into the active index position and backtracking:

```python
def permute_backtrack(nums: list[int]) -> list[list[int]]:
    result = []
    a = nums.copy()

    def backtrack(start: int):
        if start == len(a):
            result.append(a.copy())
            return
        for i in range(start, len(a)):
            a[start], a[i] = a[i], a[start]  # Swap
            backtrack(start + 1)
            a[start], a[i] = a[i], a[start]  # Undo swap (backtrack)

    backtrack(0)
    return result
```

### 5.4 Super-Exponential Growth: The Factorial Barrier

To understand why exhaustive search faces a computational wall, consider the time required to evaluate candidate solutions on a processor executing $10^9$ operations per second ($1 \text{ GHz}$):

| Input Size ($n$) | Linear ($n$) | Quadratic ($n^2$) | Exponential ($2^n$) | Factorial ($n!$) |
| :--- | :--- | :--- | :--- | :--- |
| **5** | $5 \text{ ns}$ | $25 \text{ ns}$ | $32 \text{ ns}$ | $120 \text{ ns}$ |
| **10** | $10 \text{ ns}$ | $100 \text{ ns}$ | $1 \text{ }\mu\text{s}$ | $3.63 \text{ ms}$ |
| **12** | $12 \text{ ns}$ | $144 \text{ ns}$ | $4.1 \text{ }\mu\text{s}$ | $479 \text{ ms}$ |
| **15** | $15 \text{ ns}$ | $225 \text{ ns}$ | $32.8 \text{ }\mu\text{s}$ | $1.31 \times 10^3 \text{ s}$ ($\sim 22 \text{ min}$) |
| **20** | $20 \text{ ns}$ | $400 \text{ ns}$ | $1.05 \text{ ms}$ | $2.43 \times 10^{18} \text{ ops}$ ($\sim 77 \text{ years}$) |
| **30** | $30 \text{ ns}$ | $900 \text{ ns}$ | $1.07 \text{ s}$ | $2.65 \times 10^{32} \text{ ops}$ ($> 10^{15} \times \text{age of universe}$) |

---

## 6. Case Study: The Travelling Salesman Problem (TSP)

### 6.1 Formal Problem Statement

Let $G = (V, E, W)$ be a complete weighted graph where $V = \{0, 1, \dots, n-1\}$ represents a set of $n$ cities, and $W: V \times V \to \mathbb{R}^+$ defines the non-negative symmetric distance matrix $D$ between city pairs with $D_{ii} = 0$ and $D_{ij} = D_{ji}$.

The **Travelling Salesman Problem** asks for a tour (Hamiltonian cycle) that visits every city in $V$ exactly once and returns to the initial city such that the total path distance is minimized:
$$\min_{\pi \in \Pi(V \setminus \{s\})} L(\pi) = D_{s, \pi(1)} + \sum_{i=1}^{n-2} D_{\pi(i), \pi(i+1)} + D_{\pi(n-1), s}$$

### 6.2 Search Space Size

- Fixing the starting city $s \in V$ avoids cyclic rotational duplicates (e.g., $A \to B \to C \to A$ is equivalent to $B \to C \to A \to B$).
- The remaining $n - 1$ cities can be ordered in $(n - 1)!$ permutations.
- In symmetric TSP ($D_{ij} = D_{ji}$), reversing the entire cycle yields the identical travel cost (e.g., $A \to B \to C \to D \to A$ costs the same as $A \to D \to C \to B \to A$). Hence, there are exactly $\frac{(n-1)!}{2}$ geometrically distinct tours.
- Exhaustive search evaluates all $(n - 1)!$ candidate sequences to find the absolute global minimum.

### 6.3 Complete Evaluation on the Lecture Benchmark

Consider the 4-city graph $V = \{A, B, C, D\}$ with pairwise distance matrix:

$$\mathbf{D} = \begin{pmatrix}
0 & 2 & 8 & 5 \\
2 & 0 & 3 & 4 \\
8 & 3 & 0 & 7 \\
5 & 4 & 7 & 0
\end{pmatrix}$$

```text
         (A)
       /  |  \
    2 /   |8  \ 5
     /    |    \
   (B)----|----(D)
     \    |    /
    3 \   |7  / 4
       \  |  /
         (C)
```

With starting city $s = A$ (index $0$), we generate all $3! = 6$ permutations of cities $\{B, C, D\}$:

| # | Candidate Tour | Path Cost Decomposition | Total Distance |
| :--- | :--- | :--- | :--- |
| 1 | $A \to B \to C \to D \to A$ | $D_{AB} + D_{BC} + D_{CD} + D_{DA} = 2 + 3 + 7 + 5$ | **17 (Optimal)** |
| 2 | $A \to B \to D \to C \to A$ | $D_{AB} + D_{BD} + D_{DC} + D_{CA} = 2 + 4 + 7 + 8$ | **21** |
| 3 | $A \to C \to B \to D \to A$ | $D_{AC} + D_{CB} + D_{BD} + D_{DA} = 8 + 3 + 4 + 5$ | **20** |
| 4 | $A \to C \to D \to B \to A$ | $D_{AC} + D_{CD} + D_{DB} + D_{BA} = 8 + 7 + 4 + 2$ | **21** |
| 5 | $A \to D \to B \to C \to A$ | $D_{AD} + D_{DB} + D_{BC} + D_{CA} = 5 + 4 + 3 + 8$ | **20** |
| 6 | $A \to D \to C \to B \to A$ | $D_{AD} + D_{DC} + D_{CB} + D_{BA} = 5 + 7 + 3 + 2$ | **17 (Optimal)** |

Both Tour 1 ($A \to B \to C \to D \to A$) and Tour 6 ($A \to D \to C \to B \to A$) achieve the minimal distance $17$. They are reverse reflections of the same Hamiltonian cycle.

### 6.4 Python Implementation

```python
from itertools import permutations
from sys import maxsize


def solve_tsp_exhaustive(distances: list[list[int]], start: int = 0) -> tuple[int, list[int]]:
    n = len(distances)
    cities = [i for i in range(n) if i != start]
    min_cost = maxsize
    best_tour = []

    for perm in permutations(cities):
        cost = 0
        current = start
        for next_city in perm:
            cost += distances[current][next_city]
            current = next_city
        cost += distances[current][start]

        if cost < min_cost:
            min_cost = cost
            best_tour = [start] + list(perm) + [start]

    return min_cost, best_tour
```

---

## 7. Exhaustive Search in Applied Machine Learning

While exhaustive search cannot scale to billion-parameter neural networks, it serves as an indispensable tool in core machine learning workflows.

### 7.1 Best Subset Selection in Feature Selection

Given a response variable $y \in \mathbb{R}^n$ and $p$ potential explanatory features $X_1, X_2, \dots, X_p$, greedy methods (Forward Selection, Backward Elimination, Lasso) make local or heuristic choices. By contrast, **Best Subset Selection** exhaustively fits an Ordinary Least Squares (OLS) model for every non-empty subset of features:
$$\text{Total models} = \sum_{k=1}^p \binom{p}{k} = 2^p - 1$$

For each subset $S \subseteq \{1, \dots, p\}$, it calculates the residual sum of squares and model selection criteria like the Bayesian Information Criterion (BIC):
$$\text{BIC}(S) = n \ln\left(\frac{\text{RSS}(S)}{n}\right) + |S| \ln(n)$$
Exhaustive search guarantees finding the globally optimal sparse feature subset, avoiding the local minima traps of step-wise selection.

### 7.2 Cartesian Grid Search for Hyperparameters

When tuning non-differentiable hyperparameters (learning rate schedule $\eta$, weight decay $\lambda$, decision tree depth $d$, dropout rate $p$), machine learning libraries use **Grid Search**.

Given candidate sets $\Theta_1, \Theta_2, \dots, \Theta_m$, Grid Search evaluates the objective loss across the entire Cartesian product:
$$\Theta^* = \arg\min_{\theta \in \Theta_1 \times \Theta_2 \times \dots \times \Theta_m} \mathcal{L}_{\text{validation}}(f_\theta)$$

---

## 8. Summary of Algorithmic Trade-offs

| Paradigm | Completeness | Optimality Guarantee | Typical Time Complexity | Typical Memory Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **Brute Force / Exhaustive** | **Guaranteed** | **Global Optimum** | Exponential ($2^n$) or Factorial ($n!$) | $O(n)$ (Depth-first) |
| **Dynamic Programming** | Guaranteed | Global Optimum | Polynomial (e.g., $O(n^2 2^n)$ TSP) | Exponential ($O(n 2^n)$) |
| **Branch and Bound** | Guaranteed | Global Optimum | Worst $O(n!)$, Average much lower | Exponential (Breadth-first) |
| **Greedy Heuristics** | Heuristic | Local Optimum only | Polynomial ($O(n^2)$ or $O(n \log n)$) | $O(n)$ |
| **Gradient Descent** | Continuous | Stationary point (Local minimum) | $O(N \cdot D \cdot \text{epochs})$ | $O(\text{weights})$ |

---

## References

1. **Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.** (2009). *Introduction to Algorithms* (3rd ed.). The MIT Press. Chapter 2 (Getting Started) and Chapter 34 (NP-Completeness).
2. **Sedgewick, R., & Wayne, K.** (2011). *Algorithms* (4th ed.). Addison-Wesley Professional. Chapter 2 (Sorting).
3. **Goodrich, M. T., Tamassia, R., & Goldwasser, M. H.** (2013). *Data Structures and Algorithms in Python*. John Wiley & Sons. Chapter 3 (Algorithm Analysis) and Chapter 12 (Sorting and Selection).
4. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer. Chapter 3 (Linear Methods for Regression: Subset Selection).
