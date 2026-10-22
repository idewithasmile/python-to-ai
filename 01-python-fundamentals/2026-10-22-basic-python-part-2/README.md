# Basic Python (Part 2): Loops, Strings, and Files

[Back to Module 01](../README.md) · [Back to the master roadmap](../../README.md)

**Session date:** 2026-10-22 · **Module:** 01 — Python Fundamentals & Algorithmic Thinking
**Format:** Guided reading, executable notebooks, numerical case studies, and a practice lab

## Overview

Part 1 treated a program as a collection of values and functions. Part 2 adds the ingredient that makes programs dynamic: repetition. A loop lets a small piece of code stand in for a huge, or even unbounded, family of computations. Strings and files then give those computations something to consume and something to leave behind.

The session follows a deliberate progression:

1. **Control flow.** `for` loops traverse known collections; `while` loops repeat while a condition remains true. Both are built from the same four ingredients: initial state, test, body, and update.
2. **Numerical case studies.** Loops turn mathematical series and iterative methods into executable algorithms. We estimate $\pi$ and $e$, compute square roots with Newton's method, and simulate coin tossing.
3. **Text processing.** A string is an immutable sequence of Unicode code points. We learn to slice, format, search, split, join, and validate text.
4. **Persistent data.** Files connect a program to data that outlives a single run. We read, write, append, parse CSV records, and handle errors.
5. **Composition.** The final lab combines loops, strings, and files into small programs that a reader can test and reason about.

The emphasis is not on memorising syntax. It is on building a mental model precise enough to predict what a loop will print, why a string method returns a new object, and what can go wrong when a file is missing or encoded differently than expected.

## Learning outcomes

By the end of this session, you should be able to:

- Explain the execution model of `for` and `while` loops, including the role of the test, body, and update.
- Use `range` correctly with one, two, and three arguments, including negative steps.
- Control loop execution with `break`, `continue`, and the loop `else` clause.
- Write trace tables that predict the state of a loop before running it.
- Implement factorial, Euler's number, Gregory–Leibniz, Nilakantha, Monte Carlo $\pi$, and Newton's method.
- Recognise off-by-one errors, infinite loops, and quadratic string-building patterns.
- Create, index, slice, compare, format, search, and transform strings.
- Explain the difference between Unicode code points and encoded bytes.
- Read and write text files with explicit encodings and context managers.
- Parse CSV data with the standard library and validate malformed records.
- Handle common exceptions such as `ValueError`, `IndexError`, and `FileNotFoundError`.
- Combine loops, strings, and files into a reproducible text-processing pipeline.

## Session map

| Notebook | Topics |
| :--- | :--- |
| [01 — For loops](notebooks/01-for-loops.ipynb) | Iterables, `range`, trace tables, `break`, `continue`, `else`, nested loops, factorial, Euler's number, Newton's method |
| [02 — Numerical case studies](notebooks/02-numerical-case-studies.ipynb) | Gregory–Leibniz, Nilakantha, Monte Carlo $\pi$, convergence, error measurement, coin tossing, square roots |
| [03 — While loops](notebooks/03-while-loops.ipynb) | Condition-controlled iteration, sentinels, `while True`, loop invariants, input validation, Collatz sequences |
| [04 — Strings](notebooks/04-strings.ipynb) | Unicode, indexing, slicing, formatting, operators, methods, CSV lines, structured-text parsing |
| [05 — File I/O](notebooks/05-file-io.ipynb) | Paths, encodings, open modes, reading, writing, appending, context managers, CSV parsing, statistics |
| [06 — Practice lab](notebooks/06-practice-lab.ipynb) | FizzBuzz, series, Pascal's triangle, Fibonacci, GCD, primality, word frequency, report generation |

All datasets live in [`data/`](data/README.md). The notebooks use relative paths, so they run from a fresh checkout without mounting a cloud drive or editing absolute paths.

## 1. Iteration as repeated state change

A loop is a controlled repetition of a block of code. Every useful loop has four parts:

1. **Initialisation.** Establish the variables the loop depends on.
2. **Condition.** Decide whether another iteration is needed.
3. **Body.** Perform the work for the current iteration.
4. **Update.** Change the state so that progress is made toward termination.

A `for` loop usually hides the update inside the iterable protocol. A `while` loop makes the update explicit. In both cases, if the state never moves toward a failing condition, the loop does not terminate.

### 1.1 Trace tables

A trace table records the values of the important variables at each step. It is the cheapest debugging tool in programming and the fastest way to check an off-by-one error.

Consider:

```python
total = 0
for number in range(1, 5):
    total = total + number
```

| Iteration | `number` | `total` before | `total` after |
| :---: | :---: | :---: | :---: |
| 1 | 1 | 0 | 1 |
| 2 | 2 | 1 | 3 |
| 3 | 3 | 3 | 6 |
| 4 | 4 | 6 | 10 |

After the loop, `total` is `10`. The value `5` is never visited because `range(1, 5)` stops before its second argument. This is not a bug; it is the defining convention of half-open intervals.

### 1.2 `for` versus `while`

| Question | `for` | `while` |
| :--- | :--- | :--- |
| What controls repetition? | An iterable supplies the next value. | A boolean condition is re-evaluated. |
| Typical use | Traverse a known collection or a fixed number of steps. | Repeat until an event occurs or a tolerance is met. |
| Termination | The iterable is exhausted, or `break` runs. | The condition becomes false, or `break` runs. |
| Risk | Off-by-one errors in `range`. | Forgetting to update the state. |

Use a `for` loop when the number of iterations is known or when you are visiting every element of a collection. Use a `while` loop when the number of iterations depends on a condition that is discovered during execution.

## 2. `for` loops

### 2.1 Iterables

An iterable is an object that can produce a sequence of values, one at a time. Strings, lists, tuples, dictionaries, sets, files, and `range` objects are all iterable.

```python
for character in "Python":
    print(character)

for value in [10, 20, 30]:
    print(value)

for key in {"learning_rate": 0.1, "epochs": 20}:
    print(key)
```

Iterating over a dictionary yields its keys in insertion order. To visit values, use `.values()`; to visit both, use `.items()`.

```python
config = {"learning_rate": 0.1, "epochs": 20}

for key, value in config.items():
    print(f"{key}: {value}")
```

### 2.2 The anatomy of a `for` statement

```python
for element in iterable:
    # body
```

- `for` and `in` are keywords.
- `element` is a name bound to the current value on each iteration.
- `iterable` is any object that implements the iteration protocol.
- The colon opens an indented block.
- Indentation is syntax, not decoration. Mixing tabs and spaces raises `TabError`; inconsistent indentation raises `IndentationError`.

The loop variable remains bound after the loop finishes. If the iterable was empty, the name may be undefined, which is a common source of `NameError`.

```python
for item in []:
    pass

# print(item)  # NameError: item was never bound
```

### 2.3 `range`

`range` represents an immutable arithmetic progression. It is lazy: the values are produced on demand, so `range(10**9)` does not allocate a billion integers.

| Call | Values |
| :--- | :--- |
| `range(5)` | `0, 1, 2, 3, 4` |
| `range(2, 6)` | `2, 3, 4, 5` |
| `range(1, 10, 2)` | `1, 3, 5, 7, 9` |
| `range(5, 0, -1)` | `5, 4, 3, 2, 1` |
| `range(0)` | empty |

The general form is `range(start, stop, step)`, where `start` is inclusive, `stop` is exclusive, and `step` may be negative. A zero step raises `ValueError`.

```python
for i in range(5, 0, -1):
    print(i)
```

`range` objects support membership tests and indexing in constant time:

```python
numbers = range(0, 100, 5)
print(7 in numbers)  # False
print(numbers[3])  # 15
print(len(numbers))  # 20
```

### 2.4 `break`, `continue`, and `else`

`break` exits the nearest enclosing loop immediately. `continue` skips the rest of the current iteration and starts the next one. The optional `else` block runs only if the loop finishes without encountering `break`.

```python
for number in range(10):
    if number == 5:
        break
    print(number)
# Output: 0, 1, 2, 3, 4
```

```python
for number in range(10):
    if number == 5:
        continue
    print(number)
# Output: 0, 1, 2, 3, 4, 6, 7, 8, 9
```

```python
def contains_negative(values):
    for value in values:
        if value < 0:
            print("Found a negative value")
            break
    else:
        print("All values are non-negative")


contains_negative([3, 1, 4])
contains_negative([3, -1, 4])
```

The `else` clause is easy to misread. It belongs to the loop, not to the `if`. A comment such as `# no break` makes the intent explicit.

### 2.5 Nested loops

The body of a loop may contain another loop. The inner loop runs completely for every iteration of the outer loop. The total number of iterations is the product of the two counts.

```python
for row in range(1, 4):
    for column in range(1, 4):
        print(f"{row * column:3d}", end="")
    print()
```

`break` exits only the innermost loop. To exit both loops, use a flag, a function with `return`, or an exception. A function is usually the clearest option.

```python
def find_pair(matrix, target):
    for row in matrix:
        for value in row:
            if value == target:
                return True
    return False
```

### 2.6 Common `for` loop errors

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| Loop runs one time too many or too few | Confusing inclusive and exclusive bounds | Remember that `range(start, stop)` excludes `stop`. |
| `IndexError` while iterating | Using `range(len(items) + 1)` | Use `range(len(items))` or iterate the items directly. |
| Modification during iteration | Appending to or deleting from the collection being traversed | Iterate over a copy, or build a new collection. |
| Loop variable has an unexpected final value | The loop variable persists after the loop | Do not rely on it after an empty loop; initialise explicitly. |
| Infinite loop with a `for` | The iterable itself is infinite, such as a generator | Add a bound or a `break` condition. |

## 3. `while` loops

A `while` loop repeats as long as its condition evaluates to `True`.

```python
i = 0
while i < 5:
    print(i)
    i = i + 1
print("Done")
```

The update `i = i + 1` is essential. Without it, the condition never changes and the loop runs forever.

### 3.1 Sentinel-controlled loops

A sentinel is a special value that marks the end of input. The loop continues until the sentinel is observed.

```python
def sum_until_zero(values):
    total = 0
    index = 0
    while index < len(values) and values[index] != 0:
        total = total + values[index]
        index = index + 1
    return total


print(sum_until_zero([4, 7, 2, 0, 99]))
```

The order of the condition matters. Checking `values[index]` before `index < len(values)` would raise `IndexError` when the list is exhausted.

### 3.2 `while True` and `break`

An intentionally infinite loop with an explicit exit is often clearer than a complicated condition.

```python
import random

random.seed(7)
while True:
    number = random.randint(1, 10)
    print(f"Drew {number}")
    if number == 7:
        break
print("Stopped after drawing a seven")
```

Every `while True` should have an obvious exit path. If the exit depends on external input, document it and validate the input.

### 3.3 Loop invariants

A loop invariant is a statement that is true before the loop, remains true after every iteration, and, together with the exit condition, establishes the final result.

For a summation loop:

```python
values = [3, 1, 4, 1, 5]
total = 0
index = 0

while index < len(values):
    # Invariant: total equals the sum of values[0:index]
    total = total + values[index]
    index = index + 1

# Exit condition: index == len(values)
# Therefore total equals the sum of the entire list.
```

Invariants are useful when a loop is too large to trace by hand. They also guide the placement of updates: if an invariant mentions `values[0:index]`, the update must advance `index` exactly once per iteration.

### 3.4 The `continue` trap

A `continue` statement jumps to the next condition check. If the update appears after `continue`, it may never run.

```python
i = 0
while i < 5:
    if i == 2:
        continue
    print(i)
    i = i + 1
```

This loop prints `0` and `1`, then runs forever because `i` remains `2`. When using `continue` in a `while` loop, update the state before the `continue` or restructure the code.

```python
i = 0
while i < 5:
    i = i + 1
    if i == 2:
        continue
    print(i)
```

### 3.5 Input validation

A validation loop repeatedly asks for input until the value is acceptable. Separate the parsing step from the range check so that each error message is specific.

```python
def read_positive_integer(prompt):
    while True:
        raw = input(prompt)
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if value <= 0:
            print("Please enter a positive number.")
            continue
        return value
```

The `return` is the only exit. Every other branch either reports an error or continues. This keeps the control flow easy to audit.

## 4. Numerical case studies

Loops are the natural implementation of iterative mathematics. The examples below are small enough to verify by hand and rich enough to reveal real issues: slow convergence, floating-point error, and the difference between an exact formula and a numerical estimate.

### 4.1 Factorial and Euler's number

The factorial is defined by

$$
n! = \prod_{k=1}^{n} k,
$$

with $0! = 1$. A loop computes it in $O(n)$ multiplications.

```python
def factorial(n):
    if n < 0:
        raise ValueError("factorial is undefined for negative integers")
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result
```

Euler's number is the infinite series

$$
e = \sum_{k=0}^{\infty} \frac{1}{k!}.
$$

A truncated series gives an approximation:

```python
def estimate_e(terms):
    total = 0.0
    for k in range(terms):
        total += 1.0 / factorial(k)
    return total


print(estimate_e(15))
```

Factorials grow quickly. `factorial(171)` overflows a binary64 float, so the series must stop well before that if floating-point results are required. Python integers can represent the factorial exactly, but converting a huge integer to `float` raises `OverflowError`.

### 4.2 Estimating $\pi$ with series

The Gregory–Leibniz series is

$$
\pi = 4 \sum_{k=1}^{\infty} \frac{(-1)^{k+1}}{2k-1}
= 4\left(1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \cdots\right).
$$

```python
def estimate_pi_gregory(terms):
    total = 0.0
    for k in range(1, terms + 1):
        total += (-1) ** (k + 1) / (2 * k - 1)
    return 4.0 * total
```

The Nilakantha series converges much faster:

$$
\pi = 3 + 4 \sum_{k=0}^{\infty} \frac{(-1)^k}{(2k+2)(2k+3)(2k+4)}.
$$

```python
def estimate_pi_nilakantha(terms):
    total = 3.0
    for k in range(terms):
        numerator = (-1) ** k
        denominator = (2 * k + 2) * (2 * k + 3) * (2 * k + 4)
        total += 4.0 * numerator / denominator
    return total
```

Both series are alternating. The error of a truncated alternating series is bounded by the magnitude of the first omitted term. Gregory–Leibniz converges slowly: the error is roughly $1/(2n)$, so a thousand terms still leave about three correct decimal places. Nilakantha's terms decay cubically, so the same number of terms gives far more precision.

### 4.3 Monte Carlo estimation of $\pi$

Consider a square of side $2$ centred at the origin and the inscribed circle of radius $1$. Their areas are

$$
A_s = 2^2 = 4, \qquad A_c = \pi \cdot 1^2 = \pi.
$$

If points are sampled uniformly from the square, the probability of landing inside the circle is

$$
\frac{A_c}{A_s} = \frac{\pi}{4}.
$$

Therefore

$$
\pi \approx 4 \cdot \frac{N_c}{N},
$$

where $N_c$ is the number of points inside the circle and $N$ is the total number of points.

```python
import math
import random


def estimate_pi_monte_carlo(samples, seed=0):
    random.seed(seed)
    inside = 0
    for _ in range(samples):
        x = random.uniform(-1.0, 1.0)
        y = random.uniform(-1.0, 1.0)
        if x * x + y * y <= 1.0:
            inside += 1
    return 4.0 * inside / samples
```

The estimate is random, so repeat it with several seeds before drawing conclusions. The standard error of the proportion is $\sqrt{p(1-p)/N}$ with $p = \pi/4$, so the standard error of the $\pi$ estimate is approximately $1.64/\sqrt{N}$. Reducing the error by a factor of $10$ requires $100$ times as many samples.

### 4.4 Newton's method for square roots

To compute $\sqrt{N}$, solve $f(x) = x^2 - N = 0$. Newton's update is

$$
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
= x_n - \frac{x_n^2 - N}{2x_n}
= \frac{1}{2}\left(x_n + \frac{N}{x_n}\right).
$$

Starting from $x_0 = N/2$ is safe for $N > 0$.

```python
def sqrt_newton(number, tolerance=1e-12, max_iterations=100):
    if number < 0:
        raise ValueError("number must be non-negative")
    if number == 0:
        return 0.0

    estimate = number / 2.0
    for iteration in range(max_iterations):
        next_estimate = (estimate + number / estimate) / 2.0
        if abs(next_estimate - estimate) < tolerance:
            return next_estimate
        estimate = next_estimate

    raise RuntimeError("Newton's method did not converge")
```

Convergence is quadratic near the root: the number of correct digits roughly doubles each iteration. The stopping rule must not divide by zero; for $N = 0$ the special case returns immediately.

For $N = 9$, the sequence is

| $n$ | $x_n$ |
| :---: | :--- |
| 0 | 4.5 |
| 1 | 3.25 |
| 2 | 3.009615... |
| 3 | 3.000015... |
| 4 | 3.000000000039... |
| 5 | 3.0 |

### 4.5 Simulating coin tossing

A fair coin has two outcomes with equal probability:

$$
P(\text{heads}) = P(\text{tails}) = \frac{1}{2}.
$$

A simulation draws a uniform random number in $[0, 1)$ and counts values below $0.5$ as tails.

```python
import random


def simulate_coin_tosses(tosses, seed=0):
    random.seed(seed)
    heads = 0
    tails = 0
    for _ in range(tosses):
        if random.random() < 0.5:
            tails += 1
        else:
            heads += 1
    return heads, tails


heads, tails = simulate_coin_tosses(10_000)
print(heads, tails, heads + tails)
```

The law of large numbers says that the observed proportion approaches $1/2$ as the number of tosses grows. It does not say that the absolute difference between heads and tails shrinks; that difference typically grows like $\sqrt{N}$. This distinction between proportions and counts is a recurring theme in statistics.

### 4.6 Recursive counterparts

Factorial and Fibonacci have natural recursive definitions.

```python
def factorial_recursive(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 1
    return n * factorial_recursive(n - 1)
```

```python
def fibonacci_recursive(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)
```

The recursive factorial runs in $O(n)$ time and uses $O(n)$ stack space. The naive recursive Fibonacci runs in $O(\varphi^n)$ time, where $\varphi \approx 1.618$, because it recomputes the same subproblems repeatedly. An iterative version runs in $O(n)$ time and $O(1)$ space.

```python
def fibonacci_iterative(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    previous, current = 0, 1
    for _ in range(n):
        previous, current = current, previous + current
    return previous
```

Python's default recursion limit is typically 1000 frames. Deep recursion raises `RecursionError`. Use iteration when the depth is unbounded or when the recursive call is not structurally simpler.

## 5. Strings

### 5.1 Text is Unicode

A Python 3 `str` is a sequence of Unicode code points. A code point is an integer identifier for a character; it is not the same thing as a byte. Encoding converts code points to bytes, and decoding converts bytes back to code points.

```python
text = "café"
print(len(text))  # 4 code points
print(len(text.encode("utf-8")))  # 5 bytes
print(ord("A"))  # 65
print(chr(65))  # A
```

UTF-8 is the default encoding on modern Python installations, but you should still pass `encoding="utf-8"` when opening files explicitly. Relying on the platform default causes files to break when moved between Windows, Linux, and macOS.

### 5.2 Creating strings

Strings can be written with single quotes, double quotes, or triple quotes. Triple quotes preserve newlines and are convenient for documentation.

```python
single = "Python"
double = "Python"
multiline = """First line
Second line"""
```

Escape sequences insert characters that would otherwise be hard to type:

| Escape | Meaning |
| :--- | :--- |
| `\n` | newline |
| `\t` | tab |
| `\\` | backslash |
| `\'` | single quote |
| `\"` | double quote |

A raw string treats backslashes literally, which is useful for regular expressions and Windows paths.

```python
print("C:\\Users\\ada")
print(r"C:\Users\ada")
```

### 5.3 Indexing and slicing

Strings are ordered and zero-indexed. Negative indices count from the end.

```python
word = "Python"
print(word[0])  # P
print(word[-1])  # n
print(word[1:4])  # yth
print(word[:3])  # Pyt
print(word[3:])  # hon
print(word[::-1])  # nohtyP
```

Slicing uses `start:stop:step`, with `stop` exclusive. An out-of-range slice is clipped rather than raising an error; an out-of-range index raises `IndexError`.

Strings are immutable. Methods that appear to modify a string actually return a new string.

```python
word = "python"
upper_word = word.upper()
print(word)  # python
print(upper_word)  # PYTHON
```

```python
word = "python"
# word[0] = "P"  # TypeError: 'str' object does not support item assignment
```

### 5.4 Operators

| Operator | Meaning | Example | Result |
| :--- | :--- | :--- | :--- |
| `+` | concatenation | `"AI" + "!"` | `"AI!"` |
| `*` | repetition | `"ab" * 3` | `"ababab"` |
| `==`, `!=` | equality | `"a" == "A"` | `False` |
| `<`, `<=`, `>`, `>=` | lexicographic order | `"apple" < "banana"` | `True` |
| `in`, `not in` | substring test | `"th" in "python"` | `True` |

Lexicographic comparison uses Unicode code point order. This is why uppercase letters sort before lowercase letters: `"Z" < "a"` is `True`.

Repeated concatenation inside a loop is quadratic because each `+` creates a new string. Collect pieces in a list and join once.

```python
pieces = []
for number in range(5):
    pieces.append(str(number))
result = "".join(pieces)
print(result)
```

### 5.5 Formatting

Modern Python uses f-strings for almost all formatting.

```python
name = "Ada"
score = 0.87654
print(f"{name} scored {score:.2%}")
print(f"{name:>10}|")
print(f"{score:,.3f}")
```

The format specification after the colon has the form `[[fill]align][sign][width][,][.precision][type]`. Common type codes are `d` for integers, `f` for fixed-point floats, `e` for scientific notation, `%` for percentages, and `s` for strings.

The older `%` operator and `str.format` still appear in legacy code.

```python
print("Hello %s, you are %d years old." % ("Ada", 36))
print("Hello {name}, you are {age} years old.".format(name="Ada", age=36))
```

Mixing value types with the wrong conversion code raises `TypeError` or produces surprising output. f-strings avoid most of these problems because expressions are evaluated in place.

### 5.6 String methods

Methods fall into four broad groups.

#### Classification

| Method | Returns `True` when |
| :--- | :--- |
| `str.isdigit()` | Every character is a digit. |
| `str.isalpha()` | Every character is alphabetic. |
| `str.isalnum()` | Every character is alphabetic or numeric. |
| `str.isspace()` | Every character is whitespace and the string is non-empty. |
| `str.islower()` | All cased characters are lowercase. |
| `str.isupper()` | All cased characters are uppercase. |
| `str.istitle()` | The string is title-cased. |

```python
print("123".isdigit())  # True
print("12.3".isdigit())  # False
print("".isspace())  # False
print("   ".isspace())  # True
```

#### Case and layout

```python
text = "  the quick brown fox  "
print(text.strip())
print(text.title())
print(text.upper())
print("42".zfill(5))
print("python".center(12, "-"))
```

`strip`, `lstrip`, and `rstrip` remove whitespace from both ends, the left, or the right. They do not modify the original string.

#### Searching and replacing

```python
text = "the quick brown fox"
print(text.find("quick"))  # 4
print(text.find("slow"))  # -1
print(text.index("quick"))  # 4
print(text.count("o"))  # 2
print(text.startswith("the"))  # True
print(text.endswith("fox"))  # True
print(text.replace("fox", "dog"))
```

`find` returns `-1` when the substring is absent. `index` raises `ValueError`. Use `find` when absence is an expected outcome and `index` when absence is a programming error.

#### Splitting and joining

```python
record = "001,ada,1990-01-01"
fields = record.split(",")
print(fields)

joined = " | ".join(fields)
print(joined)
```

`split()` without arguments splits on arbitrary whitespace and discards empty strings. `split(",")` preserves empty fields, which is important for CSV data. `partition` splits into exactly three pieces around the first occurrence of a separator.

```python
print("key=value".partition("="))
# ('key', '=', 'value')
```

### 5.7 Parsing structured text

A CSV line can be parsed with `split` when the fields contain no commas, quotes, or embedded newlines.

```python
line = "001,ada,1990-01-01"
identifier, name, date_of_birth = line.split(",")
print(f"{name} has id {identifier}")
```

Tuple unpacking works only when the number of fields is exactly known. For real CSV files, use the `csv` module, which handles quoting and escaped delimiters correctly.

A simple XML-like record can be parsed with `find` and slicing:

```python
data = "<student><name>Ada</name><age>36</age></student>"
name_start = data.find("<name>") + len("<name>")
name_end = data.find("</name>")
name = data[name_start:name_end]
print(name)
```

This technique is fragile. It fails on nested tags, attributes, comments, and malformed input. For production code, use `xml.etree.ElementTree`, `json`, or an HTML parser rather than hand-written slicing.

### 5.8 Common string errors

| Error | Cause | Fix |
| :--- | :--- | :--- |
| `IndexError` | Index outside the string | Check `0 <= index < len(text)`. |
| `TypeError` | Concatenating a string and a number | Convert explicitly with `str(number)`. |
| `ValueError` | `index` could not find the substring | Use `find`, or handle the exception. |
| `UnicodeDecodeError` | Reading bytes with the wrong encoding | Open with `encoding="utf-8"` or the correct encoding. |
| Logic bug | Comparing with `is` instead of `==` | Use `==` for value equality. |

## 6. Files

### 6.1 Why files matter

Variables disappear when a process ends. A file is persistent storage managed by the operating system. Reading and writing files introduces new concerns: paths, permissions, encodings, newline conventions, partial writes, and concurrent access.

### 6.2 Paths

Use `pathlib` for path manipulation. It is clearer and more portable than string concatenation.

```python
from pathlib import Path

data_directory = Path("..") / "data"
text_file = data_directory / "text" / "hello_world.txt"
print(text_file)
print(text_file.exists())
```

A relative path is resolved against the current working directory. In a notebook launched from the `notebooks/` directory, `Path("..") / "data"` points to the session's `data/` folder; from the session directory itself, use `Path("data")`. If the working directory changes, relative paths break. The notebooks in this session use a small helper that checks both locations. For scripts, resolve paths against `Path(__file__).resolve().parent` rather than the current directory.

### 6.3 Open modes

| Mode | Meaning | If the file exists | If the file is missing |
| :--- | :--- | :--- | :--- |
| `"r"` | read text | opens it | `FileNotFoundError` |
| `"w"` | write text | truncates it | creates it |
| `"a"` | append text | appends to it | creates it |
| `"x"` | exclusive creation | `FileExistsError` | creates it |
| `"r+"` | read and write | opens without truncation | `FileNotFoundError` |
| `"b"` | binary mode | depends on the primary mode | depends on the primary mode |

Text mode performs newline translation and encoding. Binary mode returns and accepts `bytes`. Use binary mode for images, audio, and serialised formats that are not text.

### 6.4 Reading

The simplest approach reads the entire file into one string.

```python
with (Path("..") / "data" / "text" / "hello_world.txt").open(encoding="utf-8") as handle:
    content = handle.read()
print(content)
```

For line-oriented processing, iterate over the file object. This streams the file and does not load it all at once.

```python
path = Path("..") / "data" / "text" / "hello_world.txt"
with path.open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        print(line_number, line.rstrip("\n"))
```

`readlines()` returns a list of lines, including the trailing newline on each line. It is convenient for small files and dangerous for large ones.

```python
with path.open(encoding="utf-8") as handle:
    lines = handle.readlines()
print(len(lines))
```

### 6.5 Writing and appending

```python
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    output = Path(directory) / "notes.txt"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("first line\n")
        handle.write("second line\n")

    with output.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("third line\n")

    print(output.read_text(encoding="utf-8"))
```

`"w"` truncates an existing file. `"a"` appends. `write` does not add a newline automatically; include `\n` explicitly. `print(..., file=handle)` does add a newline.

The context manager closes the file even if an exception occurs. Relying on garbage collection to close files is a bug waiting to happen: data may not be flushed, and the operating system may run out of file descriptors.

### 6.6 CSV parsing

For simple, well-formed files, manual parsing is instructive.

```python
path = Path("..") / "data" / "area_price.csv"
with path.open(encoding="utf-8") as handle:
    header = next(handle).strip().split(",")
    rows = []
    for line in handle:
        fields = line.strip().split(",")
        rows.append(dict(zip(header, fields)))

print(header)
print(rows[0])
```

For real data, use the standard library.

```python
import csv

path = Path("..") / "data" / "iris.csv"
with path.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    records = list(reader)

print(reader.fieldnames)
print(records[0])
```

`newline=""` is required by the `csv` module so that it can handle quoted fields containing line breaks correctly. Do not open CSV files with `newline=None` and then pass them to `csv.reader`.

### 6.7 Computing statistics from a CSV

The following example computes the mean petal length without pandas.

```python
path = Path("..") / "data" / "iris_petal.csv"
lengths = []
with path.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        lengths.append(float(row["PetalLength"]))

mean_length = sum(lengths) / len(lengths)
print(f"count={len(lengths)}, mean={mean_length:.4f}")
print(f"min={min(lengths):.2f}, max={max(lengths):.2f}")
```

This pattern generalises: parse a field, validate it, accumulate it, then compute a summary. Keep parsing and aggregation separate so each stage can be tested independently.

### 6.8 Handling errors

File operations can fail for many reasons. Catch only the exceptions you can handle, and report enough context to diagnose the problem.

```python
path = Path("..") / "data" / "text" / "missing.txt"
try:
    content = path.read_text(encoding="utf-8")
except FileNotFoundError:
    print(f"File not found: {path}")
except PermissionError:
    print(f"Permission denied: {path}")
except UnicodeDecodeError as error:
    print(f"Could not decode {path}: {error}")
else:
    print(content)
```

Avoid a bare `except:`. It catches `KeyboardInterrupt` and `SystemExit`, hides programming errors, and makes debugging much harder.

### 6.9 Common file errors

| Error | Cause | Fix |
| :--- | :--- | :--- |
| `FileNotFoundError` | Wrong path or missing file | Print the resolved path and check `exists()`. |
| `PermissionError` | No read or write permission | Check permissions or choose another location. |
| `IsADirectoryError` | Opening a directory as a file | Supply a file path. |
| `UnicodeDecodeError` | Wrong encoding | Specify the correct encoding, often UTF-8. |
| `UnicodeEncodeError` | Characters not representable in the target encoding | Write UTF-8 or replace unsupported characters. |
| Lost data | File not closed or not flushed | Use a `with` block. |
| Empty file after writing | Opened with `"w"` and never wrote, or closed before writing | Check the control flow and flush order. |

## 7. Putting it together

The following pipeline reads a CSV file, validates each record, computes a summary, and writes a report. It uses only the standard library.

```python
import csv
from pathlib import Path
from tempfile import TemporaryDirectory


def load_students(path):
    students = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            identifier = row["student_id"].strip()
            name = row["name"].strip()
            date_of_birth = row["date_of_birth"].strip()
            if not identifier or not name or not date_of_birth:
                raise ValueError(f"missing field on line {line_number}")
            students.append(
                {
                    "student_id": identifier,
                    "name": name,
                    "date_of_birth": date_of_birth,
                }
            )
    return students


def build_report(students):
    lines = [f"student count: {len(students)}"]
    for student in sorted(students, key=lambda item: item["name"]):
        lines.append(f"{student['student_id']}: {student['name']}")
    return "\n".join(lines) + "\n"


students = load_students(Path("..") / "data" / "text" / "students.csv")
report = build_report(students)

with TemporaryDirectory() as directory:
    output = Path(directory) / "report.txt"
    output.write_text(report, encoding="utf-8")
    print(output.read_text(encoding="utf-8"))
```

The pipeline separates loading, validation, transformation, and output. Each function can be tested with a small fixture, and the temporary directory ensures that the example leaves no generated files in the repository.

## 8. Complexity and performance

Loops are where asymptotic behaviour becomes visible.

| Pattern | Complexity | Why |
| :--- | :--- | :--- |
| Summing a list | $O(n)$ | One pass over $n$ items. |
| Membership test in a list | $O(n)$ | Scans until a match is found. |
| Membership test in a set | $O(1)$ average | Hash-based lookup. |
| String concatenation in a loop | $O(n^2)$ | Each concatenation copies the accumulated string. |
| `"".join(parts)` | $O(n)$ | One allocation plus one copy per part. |
| Nested loop over $n \times n$ | $O(n^2)$ | The inner loop runs $n$ times for each outer iteration. |
| Reading a file line by line | $O(n)$ time, $O(1)$ extra space | Lines are processed as they are read. |
| Reading a whole file with `read()` | $O(n)$ time and memory | The entire contents are held in memory. |

The difference between $O(n)$ and $O(n^2)$ is invisible for ten items and decisive for ten million. Prefer `join` over repeated `+`, sets over lists for membership tests, and streaming over `read()` for large files.

## 9. Common errors and debugging checklist

When a loop produces the wrong result, check the following in order:

1. **Initialisation.** Are all accumulators and indices initialised before the loop?
2. **Bounds.** Is the stop value included or excluded? Does `range` use the intended step?
3. **Update.** Does every path through the body update the state that controls termination?
4. **Indentation.** Is the statement inside the loop or after it?
5. **Mutation.** Is the collection being modified while it is being iterated?
6. **Types.** Are values converted before arithmetic or comparison?
7. **Termination.** Can the condition become false? Is there a `break` on every success path?
8. **Floating point.** Is the comparison using an absolute tolerance rather than `==`?

| Exception | Typical cause |
| :--- | :--- |
| `SyntaxError` | Missing colon, parenthesis, or quote. |
| `IndentationError` | Inconsistent indentation. |
| `NameError` | Variable used before assignment or outside its scope. |
| `TypeError` | Unsupported operand types, such as `str + int`. |
| `ValueError` | Right type, invalid value, such as `int("abc")`. |
| `IndexError` | Sequence index out of range. |
| `KeyError` | Missing dictionary key. |
| `ZeroDivisionError` | Division by zero. |
| `FileNotFoundError` | Path does not exist. |
| `UnicodeDecodeError` | Wrong encoding while reading text. |
| `RecursionError` | Recursion depth exceeded. |

## 10. Practice lab

The practice notebook contains worked examples and self-check assertions. The problems below are ordered by difficulty.

### Warm-up

1. Print the integers from `1` to `20`, but skip multiples of `3` using `continue`.
2. Print the integers from `20` down to `1` using a negative `range` step.
3. Compute the sum of the first `n` squares and compare it with $(n(n+1)(2n+1))/6$.
4. Print a multiplication table for the numbers `1` through `5` using nested loops.

### Series and iteration

5. Estimate $e$ with `5`, `10`, and `20` terms. Print the absolute error against `math.e`.
6. Estimate $\pi$ with Gregory–Leibniz and Nilakantha using the same number of terms. Which converges faster?
7. Estimate $\pi$ with Monte Carlo using `1_000`, `10_000`, and `100_000` samples. Repeat each experiment with three seeds and report the mean error.
8. Implement `sqrt_newton(number, tolerance)` and stop when the change between successive estimates is below the tolerance.
9. Implement the hyperbolic sine series
   $$
   \sinh(x) = \sum_{k=0}^{\infty} \frac{x^{2k+1}}{(2k+1)!}.
   $$
10. Implement the cosine series
    $$
    \cos(x) = \sum_{k=0}^{\infty} \frac{(-1)^k x^{2k}}{(2k)!}.
    $$

### Strings and text

11. Write a function that returns the number of vowels in a string.
12. Write a function that reverses a string without using slicing.
13. Write a function that checks whether a string is a palindrome, ignoring case and spaces.
14. Parse a CSV line into a dictionary. Validate that it has exactly three fields.
15. Implement a Caesar cipher with a configurable shift. Decode the result and verify that it matches the input.
16. Count word frequencies in a text file and print the five most common words. Break ties alphabetically.

### Files and records

17. Read `data/text/students.csv` and print one formatted line per student.
18. Read `data/iris.csv`, compute the mean sepal length, and count the records for each species.
19. Read `data/iris_petal.csv`, compute the mean petal length for each label, and write a report to a temporary file.
20. Write a function that copies a text file, normalises line endings to `\n`, and preserves the original encoding.

### Algorithms

21. Compute Pascal's triangle up to row `n` without using `math.comb`.
22. Compute the Fibonacci sequence iteratively and recursively. Compare the running time for `n = 30`.
23. Implement Euclid's algorithm for the greatest common divisor.
24. Implement the Collatz sequence and return its length for a given starting value.
25. Implement the sieve of Eratosthenes and list all primes below `100`.

## 11. Cheat sheet

### Loops

```python
# for over a range
for i in range(5):
    print(i)

# for over a collection
for item in ["a", "b", "c"]:
    print(item)

# while with explicit update
i = 0
while i < 5:
    print(i)
    i += 1

# break and continue
for i in range(10):
    if i == 3:
        continue
    if i == 7:
        break
    print(i)

# loop else
for item in items:
    if item == target:
        break
else:
    print("not found")
```

### Strings

```python
text = "  Python for Data Science  "
print(text.strip())
print(text.lower())
print(text.split())
print("-".join(["a", "b", "c"]))
print(text.replace("Data", "Machine Learning"))
print(text.find("Data"))
print(text.startswith("  Py"))
print(f"{text.strip():.>30}")
```

### Files

```python
from pathlib import Path

path = Path("..") / "data" / "text" / "hello_world.txt"

with path.open(encoding="utf-8") as handle:
    for line in handle:
        print(line.rstrip("\n"))

with path.open(encoding="utf-8") as handle:
    content = handle.read()
```

### Numeric recipes

```python
import math

factorial = 1
for k in range(1, 11):
    factorial *= k

e_estimate = sum(1 / math.factorial(k) for k in range(15))

pi_estimate = 4 * sum((-1) ** (k + 1) / (2 * k - 1) for k in range(1, 10_001))

root = 9 / 2
for _ in range(10):
    root = (root + 9 / root) / 2
```

## 12. References

- Python Software Foundation, *The Python Tutorial*: <https://docs.python.org/3/tutorial/>
- Python Software Foundation, *Built-in Types*: <https://docs.python.org/3/library/stdtypes.html>
- Python Software Foundation, *Unicode HOWTO*: <https://docs.python.org/3/howto/unicode.html>
- Python Software Foundation, *csv — CSV File Reading and Writing*: <https://docs.python.org/3/library/csv.html>
- Python Software Foundation, *pathlib — Object-oriented filesystem paths*: <https://docs.python.org/3/library/pathlib.html>
- PEP 8, *Style Guide for Python Code*: <https://peps.python.org/pep-0008/>
- PEP 257, *Docstring Conventions*: <https://peps.python.org/pep-0257/>
- James, Witten, Hastie, and Tibshirani, *An Introduction to Statistical Learning*: <https://www.statlearning.com/>
- Downey, *Think Python*, 3rd edition: <https://allendowney.github.io/ThinkPython/>
- Ramalho, *Fluent Python*, 2nd edition.

## 13. Session checklist

- [ ] I can predict the output of a `for` loop with `range(start, stop, step)`.
- [ ] I can explain when a loop `else` block runs.
- [ ] I can write a `while` loop with a clear termination argument.
- [ ] I can estimate $\pi$ and $e$ with loops and measure the error.
- [ ] I can implement Newton's method and explain why it converges quickly.
- [ ] I can slice and format strings without consulting a reference.
- [ ] I can read, write, and append text files with explicit encodings.
- [ ] I can parse a CSV file with the standard library.
- [ ] I can identify the cause of an `IndexError`, `ValueError`, or `FileNotFoundError`.
- [ ] I completed at least ten practice-lab problems.
