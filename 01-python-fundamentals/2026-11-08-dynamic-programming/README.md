# Dynamic Programming and Divide-and-Conquer

[Back to Module 01](../README.md) · [Back to the master roadmap](../../README.md)

**Session date:** 2026-11-08 · **Module:** 01 — Python Fundamentals & Algorithmic Thinking  
**Format:** Textbook study guide, executable notebooks, call stack traces, and algorithmic optimization labs

---

## Overview

In algorithmic problem solving, computer science theory, and machine learning infrastructure, two multi-stage algorithmic paradigms govern efficient problem decomposition: **Divide-and-Conquer** and **Dynamic Programming (DP)**.

While both techniques decompose a complex problem instance into smaller sub-instances, they address fundamentally different computational structures:
- **Divide-and-Conquer** partitions a problem into **independent, non-overlapping subproblems**, solves each recursively, and combines their solutions. Classic examples include **Merge Sort** and **Quick Sort**, where recurrence relations such as $T(n) = 2T(n/2) + O(n)$ yield optimal $O(n \log n)$ time bounds through the **Master Theorem** and **Recursion-Tree analysis**.
- **Dynamic Programming** tackles problems exhibiting **overlapping subproblems** and **optimal substructure**. When naive recursion recomputes identical subproblems exponentially many times ($O(2^n)$), Dynamic Programming solves each unique subproblem exactly once, caching the intermediate solutions in a memoization table (top-down) or building them iteratively (bottom-up tabulation), collapsing exponential runtime to linear or polynomial time ($O(n)$ or $O(n^2)$).

This session develops the complete mathematical, architectural, and practical foundations of both paradigms:
1. **The Divide-and-Conquer Framework:** The three pillars (*Divide*, *Conquer*, *Combine*), algorithmic recurrence modeling $T(n) = a T(n/b) + D(n) + C(n)$, recursion-tree summation, and the Master Theorem.
2. **Merge Sort & In-Place Quick Sort:** Algorithmic walkthroughs, execution trees on lecture benchmark sequence `[7, 2, 5, 30, 16, 8, 4, 17]`, Lomuto and Hoare partition mechanics, and pivot strategies mitigating $O(n^2)$ worst-case degradation.
3. **Dynamic Programming Foundations:** The Principle of Optimality, contrasting divide-and-conquer with dynamic programming, call tree explosion in naive Fibonacci evaluation ($O(\phi^n)$), top-down memoization, and space-optimized bottom-up tabulation ($O(1)$ auxiliary space).
4. **Longest Increasing Subsequence (LIS):** Subproblem formulation, the $O(n^2)$ state transition recurrence, predecessor backpointer tracking (`trace_dp`) for optimal candidate reconstruction, and the optimal $O(n \log n)$ Patience Sorting algorithm.
5. **Applied Machine Learning Connections:** The 0/1 Knapsack problem for feature subset selection and model quantization under resource budgets, Levenshtein Edit Distance for ASR error evaluation (Word Error Rate / Character Error Rate) and OCR correction, and Dynamic Time Warping (DTW) for multimodal temporal alignment.
6. **Unified Algorithmic Taxonomy:** Formal comparison of Brute Force, Recursion, Two Pointers, Divide-and-Conquer, and Dynamic Programming, alongside the 7 primitive operations and asymptotic complexity analysis.

---

## Learning outcomes

By the end of this session, you should be able to:

- Deconstruct the three core phases of Divide-and-Conquer: **Divide**, **Conquer**, and **Combine**.
- Implement Merge Sort with clean linear merging ($O(n)$) and prove its $O(n \log n)$ time bound across best, average, and worst cases.
- Apply the **Recursion-Tree Method** to visualize level costs ($cn$), tree depth ($\log_b n$), and total work across recursive layers.
- Formulate recurrences in the standard Master Theorem form $T(n) = a T(n/b) + f(n)$ and evaluate Cases 1, 2, and 3.
- Implement Quick Sort using both the **Lomuto** single-direction partition scheme and the **Hoare** convergent two-pointer partition scheme.
- Explain why Quick Sort degrades to $O(n^2)$ on sorted/reverse-sorted arrays with deterministic end pivots, and implement randomized pivot selection to guarantee expected $O(n \log n)$ time.
- Identify the two mandatory prerequisites of Dynamic Programming: **Optimal Substructure** and **Overlapping Subproblems**.
- Contrast Divide-and-Conquer with Dynamic Programming across subproblem independence, state caching, and problem categories.
- Implement top-down memoization using explicit hash tables and Python's `@functools.lru_cache`.
- Construct bottom-up tabulation tables to compute Fibonacci sequences in $O(n)$ time and optimize auxiliary memory to $O(1)$.
- Formulate the dynamic programming recurrence for the **Longest Increasing Subsequence (LIS)**: $dp[i] = 1 + \max_{j < i, nums[j] < nums[i]} dp[j]$.
- Reconstruct the step-by-step state transition table for LIS on benchmark sequences.
- Implement predecessor pointer tracking (`trace_dp`) and recover optimal subsequence paths via reverse traversal.
- Implement the $O(n \log n)$ Patience Sorting LIS algorithm using binary search (`bisect_left`).
- Solve the 0/1 Knapsack problem using 2D tabulation and 1D memory compression, applying it to machine learning feature budget selection.
- Implement Levenshtein Edit Distance to measure string divergence, generate optimal alignment edit transcripts, and quantify speech-to-text error rates.
- Classify algorithmic problems across the five foundational paradigms (Brute Force, Recursion, Two Pointers, Divide-and-Conquer, Dynamic Programming).

---

## Session map

| Notebook | Focus | Key Topics |
| :--- | :--- | :--- |
| [01 · Divide-and-Conquer & Merge Sort](notebooks/01-divide-and-conquer-and-merge-sort.ipynb) | Multi-Branched Recursion & Master Theorem | Divide-Conquer-Combine schema, Merge procedure, recurrence $T(n) = 2T(n/2) + cn$, recursion-tree depth, Master Theorem Case 2 proof |
| [02 · Quick Sort & Partition Mechanics](notebooks/02-quicksort-and-partition-mechanics.ipynb) | In-Place Partitioning & Pivot Strategies | Lomuto partitioning, Hoare convergent pointers, randomized pivot selection, $O(n^2)$ worst-case avoidance, auxiliary stack space |
| [03 · Dynamic Programming Foundations](notebooks/03-dynamic-programming-fibonacci-and-memoization.ipynb) | Overlapping Subproblems & State Caching | Exponential call tree $O(\phi^n)$, top-down memoization, bottom-up DP table, state compression to $O(1)$ space, matrix exponentiation |
| [04 · Longest Increasing Subsequence](notebooks/04-longest-increasing-subsequence-and-path-reconstruction.ipynb) | Sequence Optimization & Path Backtracking | Subsequence vs substring, $O(n^2)$ DP formulation, state transition table, predecessor backpointers (`trace_dp`), $O(n \log n)$ Patience Sorting |
| [05 · Classic DP: Knapsack & Edit Distance](notebooks/05-classic-dp-knapsack-and-edit-distance.ipynb) | Applied ML Constraints & NLP Sequence Alignment | 0/1 Knapsack for feature selection, 1D memory optimization, Levenshtein distance, edit operations alignment, speech-to-text WER/CER |
| [06 · Algorithmic Lab & ML Applications](notebooks/06-algorithmic-paradigms-lab-and-ml-applications.ipynb) | Design Taxonomy & Time-Series Alignment | Dynamic Time Warping (DTW), comparison of 5 design paradigms, 7 primitive operations, full `unittest` suite |

---

## 1. The Divide-and-Conquer Paradigm

### 1.1 Fundamental Concept
Divide-and-Conquer is an algorithmic paradigm that resolves a computational problem by breaking it into smaller instances of the **exact same problem**. The execution model follows three explicit steps:

```
                      [ Original Problem (Size n) ]
                                    |
                    +---------------+---------------+
                    | (DIVIDE)                      | (DIVIDE)
                    v                               v
          [ Subproblem 1 (Size n/b) ]     [ Subproblem 2 (Size n/b) ]
                    |                               |
              (CONQUER /                      (CONQUER /
               RECURSION)                      RECURSION)
                    v                               v
          [ Solution 1 ]                  [ Solution 2 ]
                    +---------------+---------------+
                                    |
                                (COMBINE)
                                    v
                       [ Final Solution (Size n) ]
```

1. **Divide:** Partition the problem instance $S$ of size $n$ into $a \ge 1$ subproblems, each of smaller size $n/b$ (where $b > 1$).
2. **Conquer:** Solve each subproblem recursively. If a subproblem is sufficiently small ($n \le n_0$), solve it directly using an elementary base-case routine.
3. **Combine:** Synthesize the solutions of the $a$ subproblems into the solution for the original instance.

### 1.2 General Algorithmic Schema
In pseudocode, the universal Divide-and-Conquer template is expressed as:

```python
def divide_and_conquer(instance):
    if is_base_case(instance):
        return solve_base_case(instance)

    # 1. Divide
    subproblems = divide(instance)

    # 2. Conquer
    sub_solutions = [divide_and_conquer(sub) for sub in subproblems]

    # 3. Combine
    return combine(sub_solutions)
```

---

## 2. Merge Sort: Mechanics, Trace, and Recurrence

### 2.1 Algorithmic Formulation
The sorting problem takes an input sequence of $n$ elements $\langle a_1, a_2, \dots, a_n \rangle$ and produces a reordered permutation $\langle a'_1, a'_2, \dots, a'_n \rangle$ such that $a'_1 \le a'_2 \le \dots \le a'_n$.

Merge Sort applies Divide-and-Conquer as follows:
- **Divide:** Divide the $n$-element sequence into two subsequences of size $\lfloor n/2 \rfloor$ and $\lceil n/2 \rceil$. Time cost: $D(n) = O(1)$.
- **Conquer:** Recursively sort both halves using Merge Sort. Base case: an array of length $\le 1$ is already sorted.
- **Combine:** Merge the two sorted halves into a single sorted array. Time cost: $C(n) = O(n)$.

```python
def merge(s1: list[int], s2: list[int], s: list[int]) -> None:
    """Merge two sorted lists s1 and s2 into s."""
    i = j = 0
    while i + j < len(s):
        if j == len(s2) or (i < len(s1) and s1[i] <= s2[j]):
            s[i + j] = s1[i]
            i += 1
        else:
            s[i + j] = s2[j]
            j += 1


def merge_sort(s: list[int]) -> None:
    """Divide-and-conquer Merge Sort."""
    n = len(s)
    if n < 2:
        return
    mid = n // 2
    s1 = s[:mid]
    s2 = s[mid:]
    merge_sort(s1)
    merge_sort(s2)
    merge(s1, s2, s)
```

### 2.2 Step-by-Step Execution Trace on Lecture Sequence
Consider the lecture benchmark sequence:
$$\text{Input: } S = [7, 2, 5, 30, 16, 8, 4, 17]$$

```
Level 0:                         [7, 2, 5, 30, 16, 8, 4, 17]
                                  /                       \
Level 1:                 [7, 2, 5, 30]                 [16, 8, 4, 17]
                         /           \                 /            \
Level 2:             [7, 2]         [5, 30]         [16, 8]         [4, 17]
                     /    \         /     \         /     \         /     \
Level 3:           [7]    [2]     [5]    [30]     [16]    [8]      [4]    [17]
                    \     /        \     /         \     /         \     /
Merge Level 2:       [2, 7]         [5, 30]         [8, 16]         [4, 17]
                         \           /                 \            /
Merge Level 1:            [2, 5, 7, 30]                 [4, 8, 16, 17]
                                  \                       /
Merge Level 0:                   [2, 4, 5, 7, 8, 16, 17, 30]
```

---

## 3. Analysis of Divide-and-Conquer Recurrences

### 3.1 General Recurrence Model
Let $T(n)$ denote the running time of a Divide-and-Conquer algorithm on an instance of size $n$:
$$T(n) = \begin{cases} O(1) & \text{if } n \le n_0 \\ a T(n/b) + D(n) + C(n) & \text{if } n > n_0 \end{cases}$$

Where:
- $a$: Number of subproblems generated per recursive invocation ($a \ge 1$).
- $n/b$: Size of each subproblem ($b > 1$).
- $D(n)$: Time consumed dividing the problem into subproblems.
- $C(n)$: Time consumed combining subproblem solutions into the final answer.

For Merge Sort, $a = 2$, $b = 2$, $D(n) = O(1)$, and $C(n) = O(n)$. Thus:
$$T(n) = 2 T(n/2) + cn$$

### 3.2 The Recursion-Tree Method
In a recursion tree, each node represents the cost of a single subproblem:
1. **Tree Depth:** At depth $k$, subproblem size is $n / 2^k$. The base case $n / 2^k = 1$ is reached when $k = \log_2 n$. Hence, the tree height is $\log_2 n$.
2. **Nodes per Level:** At level $k$, there are $2^k$ nodes.
3. **Cost per Node:** Each node at level $k$ performs combine work proportional to its size: $c \cdot (n / 2^k)$.
4. **Cost per Level:** Total work at level $k$ is:
   $$\text{LevelCost}(k) = 2^k \times c \left(\frac{n}{2^k}\right) = cn$$
5. **Sum Across All Levels:**
   $$T(n) = \sum_{k=0}^{\log_2 n} cn = cn (\log_2 n + 1) = \Theta(n \log n)$$

```
Level 0:                    cn                       = cn
                          /    \
Level 1:             cn/2        cn/2                = cn
                    /   \       /   \
Level 2:         cn/4   cn/4   cn/4  cn/4            = cn
                 ...     ...    ...   ...            ...
Level log n:     c   c   c   c  ...   c              = cn
---------------------------------------------------------
Total Sum:       cn * (log2(n) + 1)                  = Theta(n log n)
```

### 3.3 The Master Theorem
The Master Theorem provides a cookbook solution for recurrences of the form:
$$T(n) = a T(n/b) + f(n), \quad a \ge 1, \, b > 1$$

Let $n_{\text{crit}} = n^{\log_b a}$ represent the watershed function comparing the cost of subproblems against the dividing/combining cost $f(n)$:

| Case | Condition | Intuition | Solution |
| :---: | :--- | :--- | :---: |
| **Case 1** | $f(n) = O(n^{\log_b a - \epsilon})$ for some $\epsilon > 0$ | Leaf costs dominate root cost | $T(n) = \Theta(n^{\log_b a})$ |
| **Case 2** | $f(n) = \Theta(n^{\log_b a} \log^k n)$ for $k \ge 0$ | Work evenly spread across all levels | $T(n) = \Theta(n^{\log_b a} \log^{k+1} n)$ |
| **Case 3** | $f(n) = \Omega(n^{\log_b a + \epsilon})$ and regularity condition holds | Root cost dominates leaf costs | $T(n) = \Theta(f(n))$ |

#### Application to Merge Sort:
- $a = 2, \, b = 2 \implies \log_b a = \log_2 2 = 1 \implies n^{\log_b a} = n^1$.
- $f(n) = cn = \Theta(n^1) = \Theta(n^{\log_b a} \log^0 n)$ where $k = 0$.
- **Case 2 applies:**
  $$T(n) = \Theta(n^1 \log^{0+1} n) = \Theta(n \log n)$$

---

## 4. Quick Sort: Partitioning Mechanics & Complexity

### 4.1 Quick Sort Strategy
Quick Sort is an in-place sorting algorithm that rearranges elements around a chosen **pivot** element:
- **Divide:** Rearrange array $S$ into two partitions: elements $\le x$ (left) and elements $> x$ (right). The pivot $x$ settles at its exact final sorted index.
- **Conquer:** Recursively sort the left and right partitions.
- **Combine:** Trivial ($O(1)$) because sorting is performed directly in-place.

### 4.2 Lomuto Partition Scheme
Lomuto's partition scheme uses the rightmost element as the pivot:

```python
def partition_lomuto(low: int, high: int, s: list[int]) -> int:
    pivot = s[high]
    pointer = low
    for i in range(low, high):
        if s[i] <= pivot:
            s[i], s[pointer] = s[pointer], s[i]
            pointer += 1
    s[pointer], s[high] = s[high], s[pointer]
    return pointer


def quicksort_lomuto(low: int, high: int, s: list[int]) -> None:
    if low < high:
        p = partition_lomuto(low, high, s)
        quicksort_lomuto(low, p - 1, s)
        quicksort_lomuto(p + 1, high, s)
```

#### Walkthrough on `[7, 2, 5, 30, 16, 8, 4, 17]`:
- Pivot = `17` (index 7). `pointer` = 0.
- `i = 0`: $s[0] = 7 \le 17 \implies$ swap $s[0]$ with $s[0]$, `pointer` becomes 1.
- `i = 1`: $s[1] = 2 \le 17 \implies$ swap $s[1]$ with $s[1]$, `pointer` becomes 2.
- `i = 2`: $s[2] = 5 \le 17 \implies$ swap $s[2]$ with $s[2]$, `pointer` becomes 3.
- `i = 3`: $s[3] = 30 > 17 \implies$ no swap.
- `i = 4`: $s[4] = 16 \le 17 \implies$ swap $s[4]$ with $s[3]$ (`pointer`), array becomes `[7, 2, 5, 16, 30, 8, 4, 17]`, `pointer` becomes 4.
- `i = 5`: $s[5] = 8 \le 17 \implies$ swap $s[5]$ with $s[4]$, array becomes `[7, 2, 5, 16, 8, 30, 4, 17]`, `pointer` becomes 5.
- `i = 6`: $s[6] = 4 \le 17 \implies$ swap $s[6]$ with $s[5]$, array becomes `[7, 2, 5, 16, 8, 4, 30, 17]`, `pointer` becomes 6.
- Final step: Swap $s[7]$ (`17`) with $s[6]$ (`30`). Array: `[7, 2, 5, 16, 8, 4, 17, 30]`. Pivot `17` is now at index 6.

### 4.3 Complexity Profiles: Merge Sort vs Quick Sort

| Metric | Merge Sort | Quick Sort (Deterministic) | Quick Sort (Randomized) |
| :--- | :---: | :---: | :---: |
| **Best-case Time** | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ |
| **Average-case Time** | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ |
| **Worst-case Time** | $O(n \log n)$ | $O(n^2)$ (sorted inputs) | $O(n \log n)$ (expected) |
| **Auxiliary Memory** | $O(n)$ (merged buffers) | $O(\log n)$ stack space | $O(\log n)$ stack space |
| **Algorithmic Stability** | Stable | Unstable | Unstable |

---

## 5. The Dynamic Programming Paradigm

### 5.1 The Principle of Optimality
Dynamic Programming was formalized by Richard Bellman in the 1950s. It rests on the **Principle of Optimality**:
> *An optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision.*

### 5.2 Mandatory Criteria for Dynamic Programming
1. **Optimal Substructure:** The optimal solution to a problem of size $n$ can be constructed directly from the optimal solutions to subproblems of size $< n$.
2. **Overlapping Subproblems:** The space of subproblems is small; naive recursion revisits the identical subproblems repeatedly instead of generating distinct subproblems.

### 5.3 Divide-and-Conquer vs Dynamic Programming

| Characteristic | Divide-and-Conquer | Dynamic Programming |
| :--- | :--- | :--- |
| **Subproblem Structure** | **Disjoint / Independent:** Subproblems do not share sub-subproblems. | **Overlapping:** Subproblems share identical sub-subproblems. |
| **Recomputation** | Solves each subproblem without needing a cache. | Caches solutions to prevent exponential redundant work. |
| **Primary Workflow** | Top-down tree branching (e.g., Merge Sort, Binary Search). | Top-down with Memoization OR Bottom-up with Tabulation. |
| **Typical Problem Type** | Sorting, Geometric hulls, Fast Fourier Transform (FFT). | Combinatorial optimization, Sequence alignment, Shortest paths. |

---

## 6. Fibonacci Numbers: Exponential Decay vs Polynomial Efficiency

### 6.1 The Exponential Call Tree
The Fibonacci sequence:
$$F(0) = 0, \quad F(1) = 1, \quad F(n) = F(n-1) + F(n-2) \quad (n \ge 2)$$

In naive recursion, computing $F(4)$ invokes $F(3)$ and $F(2)$. But $F(3)$ also invokes $F(2)$ and $F(1)$. The subproblem $F(2)$ is computed multiple times:

```
                        F(4)
                     /        \
                 F(3)          F(2)   <-- Duplicate computation!
                /    \        /    \
             F(2)    F(1)   F(1)   F(0)
            /    \
         F(1)    F(0)
```

The total number of recursive leaf calls is $F(n+1) \approx \frac{1}{\sqrt{5}} \phi^{n+1} = \Theta(\phi^n)$, where $\phi = \frac{1 + \sqrt{5}}{2} \approx 1.618$. Computing $F(50)$ requires over $40$ billion function calls!

### 6.2 Top-Down Dynamic Programming (Memoization)
By storing results in a hash table or array, every subproblem is evaluated at most once:

```python
memo = {0: 0, 1: 1}


def fibonacci_memo(n: int) -> int:
    if n not in memo:
        memo[n] = fibonacci_memo(n - 1) + fibonacci_memo(n - 2)
    return memo[n]
```
- **Time Complexity:** $O(n)$ (linear number of unique states).
- **Space Complexity:** $O(n)$ ($O(n)$ dictionary cache + $O(n)$ recursive call stack depth).

### 6.3 Bottom-Up Dynamic Programming (Tabulation)
Iteratively fills an array from index 0 up to $n$:

```python
def fibonacci_dp(n: int) -> tuple[int, list[int]]:
    if n == 0:
        return 0, [0]
    if n == 1:
        return 1, [0, 1]

    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n], dp
```

#### Trace for $F(4)$ and $F(7)$:
- $F(4)$ table: `[0, 1, 1, 2, 3]` $\implies F(4) = 3$.
- $F(7)$ table: `[0, 1, 1, 2, 3, 5, 8, 13]` $\implies F(7) = 13$.

### 6.4 State Compression to $O(1)$ Auxiliary Space
Notice that $F(i)$ only requires $F(i-1)$ and $F(i-2)$. All older values can be discarded:

```python
def fibonacci_o1(n: int) -> int:
    if n < 2:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev1 + prev2
    return prev1
```
- **Time Complexity:** $O(n)$.
- **Space Complexity:** $O(1)$ (only two scalar integer variables).

---

## 7. Longest Increasing Subsequence (LIS)

### 7.1 Problem Statement
Given an array of integers $nums = [a_0, a_1, \dots, a_{n-1}]$, find the length of the longest subsequence such that all elements are in strictly ascending order:
$$i_0 < i_1 < \dots < i_{k-1} \quad \text{and} \quad nums[i_0] < nums[i_1] < \dots < nums[i_{k-1}]$$

> **Subsequence vs Subarray:** A subarray consists of contiguous elements (e.g., `[2, 1]` in `[3, 10, 2, 1, 20]`). A subsequence maintains relative index order but can skip arbitrary elements (e.g., `[3, 10, 20]`).

### 7.2 Dynamic Programming Formulation ($O(n^2)$)
1. **State Definition:** Let $dp[i]$ denote the length of the longest strictly increasing subsequence that **ends at index $i$** (meaning $nums[i]$ is the last element).
2. **Base Case:** Every isolated element $nums[i]$ forms an increasing subsequence of length 1:
   $$dp[i] = 1 \quad \forall \, i \in [0, n-1]$$
3. **State Transition:** For each index $i$, inspect all previous indices $j < i$. If $nums[i] > nums[j]$, then $nums[i]$ can extend any valid increasing subsequence ending at $nums[j]$:
   $$dp[i] = 1 + \max \left(\{ dp[j] \mid 0 \le j < i \text{ and } nums[j] < nums[i] \} \cup \{0\}\right)$$
4. **Global Optimum:** The length of the LIS is $\max_{0 \le i < n} dp[i]$.

### 7.3 Step-by-Step State Transition Trace on Lecture Input
Consider $nums = [3, 10, 2, 1, 20]$:

| Index ($i$) | $nums[i]$ | Prior Valid $j$ ($nums[j] < nums[i]$) | $dp[i]$ Calculation | $dp[i]$ Value | `trace_dp[i]` | Subsequence Ending at $i$ |
| :---: | :---: | :--- | :--- | :---: | :---: | :--- |
| **0** | `3` | None ($j < 0$) | Base case: $1$ | **1** | `-1` | `[3]` |
| **1** | `10` | $j = 0$ ($nums[0] = 3 < 10$) | $\max(1, dp[0] + 1) = 2$ | **2** | `0` | `[3, 10]` |
| **2** | `2` | None ($nums[0]=3, nums[1]=10 > 2$) | Base case: $1$ | **1** | `-1` | `[2]` |
| **3** | `1` | None ($nums[0]=3, 10, 2 > 1$) | Base case: $1$ | **1** | `-1` | `[1]` |
| **4** | `20` | $j = 0 (3), 1 (10), 2 (2), 3 (1)$ | $\max(dp[0], dp[1], dp[2], dp[3]) + 1 = 2 + 1 = 3$ | **3** | `1` | `[3, 10, 20]` |

**Result:** $\max(dp) = 3$.

### 7.4 Backtracking Path Reconstruction
To reconstruct the optimal sequence rather than just its length, maintain an array `trace_dp` initialized to `-1`. When $dp[i]$ is updated by extending $dp[j]$, set `trace_dp[i] = j`:

```python
def longest_subsequence_with_trace(
    nums: list[int],
) -> tuple[int, list[int], list[int]]:
    n = len(nums)
    if n == 0:
        return 0, [], []

    dp = [1] * n
    trace_dp = [-1] * n

    for i in range(n):
        for j in range(i):
            if nums[i] > nums[j] and dp[i] < dp[j] + 1:
                dp[i] = dp[j] + 1
                trace_dp[i] = j

    max_len = max(dp)
    curr = dp.index(max_len)

    result = []
    while curr != -1:
        result.append(nums[curr])
        curr = trace_dp[curr]

    result.reverse()  # Reverse backward chain to get sequence order
    return max_len, dp, result
```

> **Pedagogical Note:** In raw lecture code, a call to `sorted(result)` was made after backtracking. While coincidentally working for strictly increasing numbers, sorting a path fails on non-monotonic or reverse-traced sequences. The mathematically sound approach is to simply reverse the reconstructed backward path `result.reverse()`.

### 7.5 Patience Sorting with Binary Search: $O(n \log n)$
By maintaining an array `tails` where `tails[k]` stores the smallest terminal value among all increasing subsequences of length $k+1$, each incoming number can be placed using binary search (`bisect_left`):
- If $x > \text{all tails}$, append $x$ (extends LIS length by 1).
- Otherwise, replace the first tail $\ge x$ with $x$ (preserves length, improves potential for future extensions).
- **Time Complexity:** $n$ iterations $\times O(\log n)$ binary search = $O(n \log n)$.

---

## 8. Classic DP Paradigms in Machine Learning

### 8.1 The 0/1 Knapsack Problem for Model Budgeting
In machine learning engineering, serving models on mobile or edge devices requires selecting an optimal subset of features or model layers under strict memory and latency constraints:
$$\max \sum_{i=1}^n v_i x_i \quad \text{subject to} \quad \sum_{i=1}^n w_i x_i \le W, \quad x_i \in \{0, 1\}$$

Using dynamic programming, the state $dp[w]$ represents the maximum utility attainable with capacity $w$. Iterating backwards prevents multiple reuse of the same item:

```python
def knapsack_01_optimized(weights: list[int], values: list[int], capacity: int) -> int:
    dp = [0] * (capacity + 1)
    for w_i, v_i in zip(weights, values):
        for c in range(capacity, w_i - 1, -1):
            dp[c] = max(dp[c], dp[c - w_i] + v_i)
    return dp[capacity]
```

### 8.2 Levenshtein Edit Distance for ASR & OCR Evaluation
In speech recognition (Automatic Speech Recognition - ASR) and Optical Character Recognition (OCR), transcript accuracy is evaluated using **Word Error Rate (WER)** and **Character Error Rate (CER)**:
$$\text{WER} = \frac{S + D + I}{N}$$
Where $S$ is substitutions, $D$ is deletions, $I$ is insertions, and $N$ is the number of ground truth tokens. The numerator is exactly the **Levenshtein Distance** between the hypothesis and reference token sequences:

$$D(i, j) = \begin{cases} D(i-1, j-1) & \text{if } s_1[i-1] == s_2[j-1] \\ 1 + \min(D(i-1, j), D(i, j-1), D(i-1, j-1)) & \text{if } s_1[i-1] \ne s_2[j-1] \end{cases}$$

---

## 9. Algorithm Design Taxonomy

Across modern algorithmic engineering and machine learning systems, five overarching algorithmic paradigms form the foundation of problem solving:

| Paradigm | Mechanism | Best Suited For | Canonical Examples | Time Complexity |
| :--- | :--- | :--- | :--- | :---: |
| **Brute Force** | Exhaustively enumerates states | Baseline verification, small search spaces | Linear Search, Traveling Salesperson | $O(n)$ to $O(n!)$ |
| **Recursion** | Decomposes problem into self-similar sub-instances | Tree traversals, inductive processes | Factorial, Tree traversals | $O(n)$ to $O(2^n)$ |
| **Two Pointers** | Convergent or sliding window markers | Monotonic, sorted, or contiguous arrays | Two Sum, Subarray Sum, Deduplication | $O(n)$ |
| **Divide-and-Conquer** | Independent subproblems solved recursively | Disjoint partition structures | Merge Sort, Quick Sort, Karatsuba | $O(n \log n)$ |
| **Dynamic Programming** | Overlapping subproblems cached in table | Combinatorial optimization with optimal substructure | Fibonacci, LIS, Knapsack, Edit Distance | $O(n)$ to $O(n^2)$ |

---

## 10. Computational Complexity Analysis Workflow

To systematically analyze the computational complexity of an algorithm:

```
[ Algorithmic Source / Pseudocode ]
               |
               v
[ Identify 7 Primitive Operations ]
  - Assigning a variable (x = 5)
  - Performing arithmetic (x + y)
  - Comparing two numbers (x < y)
  - Indexing into an array (arr[i])
  - Following an object reference (node.next)
  - Calling a function (foo(x))
  - Returning from a function (return res)
               |
               v
[ Formulate Step Count Function T(n) ]
               |
               v
[ Apply Asymptotic Notation ]
  - Big-O (Upper bound): T(n) <= c * g(n) for n >= n0
  - Big-Omega (Lower bound): T(n) >= c * g(n) for n >= n0
  - Big-Theta (Tight bound): c1 * g(n) <= T(n) <= c2 * g(n)
```

---

## 11. References and Further Reading

1. **Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009).** *Introduction to Algorithms* (3rd ed.). The MIT Press.
   - Chapter 2: Getting Started (Merge Sort analysis)
   - Chapter 4: Divide-and-Conquer (Recursion Tree & Master Theorem)
   - Chapter 7: Quicksort
   - Chapter 15: Dynamic Programming
2. **Goodrich, M. T., Tamassia, R., & Goldwasser, M. H. (2013).** *Data Structures and Algorithms in Python*. John Wiley & Sons.
   - Chapter 12: Sorting and Selection
   - Chapter 13: Text Processing (Longest Common Subsequence)
3. **Bellman, R. (1957).** *Dynamic Programming*. Princeton University Press.
4. **Levenshtein, V. I. (1966).** Binary codes capable of correcting deletions, insertions, and reversals. *Soviet Physics Doklady*, 10(8), 707–710.
