# Object-Oriented Programming: Inheritance, Polymorphism, Access Control, and Deep Learning Architecture

[Back to Module 01](../README.md) · [Back to the master roadmap](../../README.md)

**Session date:** 2026-11-01 · **Module:** 01 — Python Fundamentals & Algorithmic Thinking  
**Format:** Theoretical exposition, UML architectural modeling, access modifier mechanics, polymorphic protocols, modular package engineering, and deep learning framework subclassing

---

## Overview

In modular software engineering, structuring complex domains requires more than encapsulating individual objects. As software systems evolve, independent entities inevitably share common structural traits, operational contracts, and behaviors. Attempting to build large applications solely with isolated classes results in catastrophic code duplication, fragile copy-paste maintenance, and fragmented interfaces.

To manage structural complexity at scale, the **Object-Oriented Programming (OOP)** paradigm provides three foundational mechanisms:
1. **Inheritance:** Establishing hierarchical **"is-a"** relationships where specialized subclasses inherit state and behavior from generalized superclasses, eliminating redundancy and centralizing shared logic.
2. **Access Control & Encapsulation:** Enforcing architectural boundaries between public interfaces and internal implementation details via public, protected (`_`), and private (`__`) members, safeguarded by runtime **name mangling**.
3. **Polymorphism & Dynamic Dispatch:** Enabling uniform interfaces across heterogeneous types through dynamic dispatch, **Duck Typing** ("if it walks like a duck and quacks like a duck, it is a duck"), and native **Operator Overloading**.

At the boundary of software engineering and Artificial Intelligence, modern deep learning libraries—most prominently **PyTorch** and **Keras**—are built entirely upon these object-oriented patterns. Neural networks are modeled not as static computation graphs or collections of standalone functions, but as object hierarchies inheriting from base module abstractions (`torch.nn.Module`, `tf.keras.layers.Layer`).

This session delivers a textbook-grade exposition of advanced Object-Oriented Programming in Python. We trace the paradigm from core inheritance mechanics and Abstract Base Classes (ABCs) to access control, Method Resolution Order (MRO) via C3 Linearization, Python package engineering, and the implementation of a zero-dependency deep learning layer hierarchy from first principles.

---

## Learning outcomes

By completing this session, you will be able to:

- Formulate and defend the architectural motivation for inheritance, eliminating redundant implementations across domain taxonomies.
- Implement single and multiple inheritance hierarchies, utilizing `super().__init__()` for deterministic constructor chaining.
- Resolve ambiguous method calls in multiple inheritance hierarchies by tracing the Method Resolution Order (MRO) and the C3 Linearization algorithm.
- Design strict behavioral contracts and formal interfaces using Python's Abstract Base Classes (`abc.ABC` and `@abstractmethod`).
- Distinguish between Python's access tiers: Public members, Protected conventions (`_single_underscore`), and Private members (`__double_underscore`).
- Explain the runtime mechanics of Python compiler name mangling (`_ClassName__attribute`) and demystify its role in preventing namespace collisions.
- Construct robust, validated class invariants using idiomatic `@property` getters, setters, and deleters.
- Differentiate between Subtype Polymorphism (nominal subclassing) and Ad-hoc Polymorphism (dynamic duck typing).
- Overload Python's special double-underscore ("dunder") methods (`__add__`, `__sub__`, `__mul__`, `__abs__`, `__eq__`, `__repr__`, `__str__`) to endow custom objects with native language syntax.
- Emulate method overloading in Python's single-namespace environment using optional arguments and `functools.singledispatch`.
- Architect production-grade Python packages, structuring directory layouts, managing `__init__.py` files, and specifying explicit public exports with `__all__`.
- Analyze the object-oriented design of deep learning frameworks, constructing a pure NumPy layer hierarchy (`Layer`, `Dense`, `ReLU`, `Softmax`, `Sequential`) and assembling a Multi-Layer Perceptron (MLP) for Fashion-MNIST image classification.

---

## Session map

| Notebook | Focus | Key Topics |
| :--- | :--- | :--- |
| [01 — Inheritance & Super](notebooks/01-inheritance-mechanics-and-super.ipynb) | Inheritance Mechanics & ABCs | "Is-a" taxonomy, `Math1` -> `Math2` code reuse, constructor chaining with `super()`, `Animal` / `Employee` hierarchies, `abc.ABC`, `@abstractmethod`, `Shape` models, C3 Linearization & MRO |
| [02 — Access Modifiers & Mangling](notebooks/02-access-modifiers-and-name-mangling.ipynb) | Access Control & Properties | "Consenting adults" philosophy, Public, Protected (`_`), Private (`__`), name mangling inspection via `__dict__`, `Student` -> `GraduateStudent` inheritance, `@property` data validation invariants |
| [03 — Polymorphism & Overloading](notebooks/03-polymorphism-duck-typing-and-overloading.ipynb) | Polymorphic Dispatch & Operators | Subtype polymorphism, Duck Typing protocols without inheritance, method overloading with `functools.singledispatch`, `Vector2D` operator overloading (`__add__`, `__mul__`, `__abs__`, `__eq__`), shape mass analytics |
| [04 — Modules & Packages](notebooks/04-modules-packages-and-architecture.ipynb) | Modular Architecture & Namespaces | Module mechanics, `sys.modules` cache, `sys.path`, package structure, `__init__.py` public facades, `__all__` export controls, relative vs absolute imports, digital audio DSP pipeline case study |
| [05 — Deep Learning Subclassing](notebooks/05-deep-learning-subclassing-pytorch-and-keras.ipynb) | Neural Network Layer Abstractions | OOP in PyTorch (`nn.Module`) & Keras (`Layer`), callable instances (`__call__`), NumPy layer hierarchy (`Dense`, `ReLU`, `Softmax`, `Sequential`), Fashion-MNIST 784-128-10 MLP forward pass |

---

## 1. Theoretical Foundations of Inheritance

### 1.1 The Problem of Code Duplication

In software architecture, the DRY (**Don't Repeat Yourself**) principle dictates that every piece of domain knowledge or logic must have a single, unambiguous representation within a system. 

Consider a mathematical computing package containing a base class `Math1` implementing parity checking and factorial calculations:

```text
+------------------------------------+
|               Math1                |
+------------------------------------+
| + is_even(number: int): bool       |
| + factorial(number: int): int      |
+------------------------------------+
```

Now consider a requirement for an advanced class `Math2` that requires both `is_even` and `factorial`, but adds an approximation of Euler's number $e$ via Taylor series expansion:

$$e = \sum_{k=0}^{\infty} \frac{1}{k!} = 1 + \frac{1}{1!} + \frac{1}{2!} + \frac{1}{3!} + \dots$$

Without inheritance, `Math2` must duplicate the entirety of `Math1`'s source code:

```text
+------------------------------------+       +------------------------------------+
|               Math1                |       |           DuplicateMath2           |
+------------------------------------+       +------------------------------------+
| + is_even(number: int): bool       |       | + is_even(number: int): bool       |  <-- DUPLICATED
| + factorial(number: int): int      |       | + factorial(number: int): int      |  <-- DUPLICATED
+------------------------------------+       | + estimate_euler(iters: int): float|
                                             +------------------------------------+
```

This duplication introduces grave architectural liabilities:
- **Maintenance Forking:** Bug fixes or algorithmic optimizations applied to `Math1.factorial()` do not propagate to `DuplicateMath2`.
- **Interface Fragmentation:** There is no structural guarantee that both classes maintain identical parameter signatures.
- **Cognitive Overhead:** Developers must inspect both implementations to determine whether subtle differences exist.

### 1.2 Resolving Redundancy via Inheritance

Inheritance establishes an **"is-a"** semantic relationship: `Math2` **is a** specialized type of `Math1`. The derived class inherits all attributes and methods of the base class without re-declaring them:

```text
+------------------------------------+
|               Math1                |
+------------------------------------+
| + is_even(number: int): bool       |
| + factorial(number: int): int      |
+------------------------------------+
                  ^
                  |  inherits ("is-a")
+------------------------------------+
|               Math2                |
+------------------------------------+
| + estimate_euler(iters: int): float|
+------------------------------------+
```

In Python, this relationship is expressed directly in the class definition header:

```python
class Math1:
    def is_even(self, number: int) -> bool:
        return number % 2 == 0

    def factorial(self, number: int) -> int:
        result = 1
        for i in range(1, number + 1):
            result *= i
        return result


class Math2(Math1):
    def estimate_euler(self, iterations: int) -> float:
        # Reusing the inherited factorial method directly
        return 1.0 + sum(1.0 / self.factorial(i) for i in range(1, iterations + 1))
```

---

## 2. Constructor Chaining and the Mechanics of `super()`

### 2.1 State Initialization in Derived Classes

When an instance of a subclass is created, its initialization process must ensure that both the superclass attributes and the subclass attributes are properly instantiated. 

If a subclass does not define an `__init__()` method, Python automatically executes the superclass's `__init__()`. However, when the subclass defines its own `__init__()`, it **overrides** the parent constructor. If the developer fails to explicitly invoke the parent constructor, the superclass attributes remain uninitialized, leading to runtime `AttributeError` exceptions.

```text
+--------------------------------------------------------------------------+
|                     CONSTRUCTOR INVOCATION FLOW                          |
+--------------------------------------------------------------------------+
|                                                                          |
|    1. Client invokes: Dog(name="Barnaby", age=4, breed="Retriever")      |
|                                |                                         |
|                                v                                         |
|    2. Python allocates object via Dog.__new__(Dog)                       |
|                                |                                         |
|                                v                                         |
|    3. Enters Dog.__init__(self, name, age, breed)                        |
|                                |                                         |
|                                v                                         |
|    4. Calls super().__init__(name, age)                                  |
|                                |                                         |
|                                v                                         |
|    5. Enters Animal.__init__(self, name, age)                            |
|       - Binds self.name = "Barnaby"                                      |
|       - Binds self.age = 4                                               |
|                                |                                         |
|                                v                                         |
|    6. Returns execution to Dog.__init__()                                |
|       - Binds self.breed = "Retriever"                                   |
|                                                                          |
+--------------------------------------------------------------------------+
```

### 2.2 The Role of `super()`

Python's built-in `super()` returns a proxy object that delegates attribute and method lookups to a parent or sibling class in the **Method Resolution Order (MRO)**:

```python
class Animal:
    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age

    def describe(self) -> str:
        return f"{self.name} (Age: {self.age})"


class Dog(Animal):
    def __init__(self, name: str, age: int, breed: str) -> None:
        # Constructor chaining: initialize superclass state first
        super().__init__(name, age)
        self.breed = breed

    def describe(self) -> str:
        # Reusing parent behavior while augmenting output
        parent_description = super().describe()
        return f"{parent_description}, Breed: {self.breed}"
```

`super()` provides critical architectural benefits:
1. **Decoupling from Parent Names:** If the superclass is renamed or swapped in a refactoring, child classes calling `super()` require fewer manual updates.
2. **Deterministic Cooperative Multiple Inheritance:** In diamond-shaped inheritance graphs, `super()` guarantees that each class in the hierarchy is visited exactly once according to C3 Linearization.

---

## 3. Abstract Base Classes (ABCs) and Interface Enforcement

### 3.1 The Contractual Need for Interfaces

In dynamically typed languages like Python, duck typing allows any object with matching method names to be used interchangeably. However, for mission-critical enterprise systems and large AI pipelines, informal conventions can permit silent interface drift:
- A developer might name a method `calculate_area()`, while another names it `compute_area()`, and a third names it `area()`.
- Client code expecting a common interface fails at runtime with `AttributeError`.

Python resolves this through **Abstract Base Classes** provided by the standard library's `abc` module.

### 3.2 Abstract Classes and Methods

An Abstract Base Class:
1. Inherits from `abc.ABC`.
2. Marks one or more methods with the `@abstractmethod` decorator.
3. **Cannot be instantiated directly.** Attempting to call `Shape()` raises a `TypeError`.
4. Forces all concrete subclasses to implement every abstract method before instantiation is permitted.

```python
import math
from abc import ABC, abstractmethod


class Shape(ABC):
    """Abstract base class defining the geometric contract."""

    def __init__(self, shape_id: str, color: str = "white") -> None:
        self.shape_id = shape_id
        self.color = color

    @abstractmethod
    def area(self) -> float:
        """Compute the enclosed surface area."""
        pass

    @abstractmethod
    def perimeter(self) -> float:
        """Compute the perimeter boundary length."""
        pass

    def compute_mass(self, density_g_cm2: float) -> float:
        """Concrete template method sharing calculation across all shapes."""
        return self.area() * density_g_cm2


class Circle(Shape):
    def __init__(self, shape_id: str, radius: float, color: str = "white") -> None:
        super().__init__(shape_id, color)
        self.radius = float(radius)

    def area(self) -> float:
        return math.pi * (self.radius**2)

    def perimeter(self) -> float:
        return 2.0 * math.pi * self.radius


class Rectangle(Shape):
    def __init__(self, shape_id: str, width: float, height: float, color: str = "white") -> None:
        super().__init__(shape_id, color)
        self.width = float(width)
        self.height = float(height)

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2.0 * (self.width + self.height)
```

---

## 4. Multiple Inheritance and Method Resolution Order (MRO)

### 4.1 The Diamond Problem

When a class inherits from multiple parent classes that share a common ancestor, a classical ambiguity arises known as the **Diamond Problem**:

```text
               +---------------+
               |       A       |
               | + execute()   |
               +-------+-------+
                       ^
           +-----------+-----------+
           |                       |
   +-------+-------+       +-------+-------+
   |       B       |       |       C       |
   | + execute()   |       | + execute()   |
   +-------+-------+       +-------+-------+
           ^                       ^
           +-----------+-----------+
                       |
               +-------+-------+
               |       D       |
               | (inherits B,C)|
               +---------------+
```

If class `D` inherits from both `B` and `C`, and client code invokes `d.execute()`, which implementation should Python run? `B.execute()` or `C.execute()`?

### 4.2 C3 Linearization

Python resolves this deterministically using the **C3 Linearization Algorithm** (introduced in Python 2.3). C3 enforces three critical guarantees:
1. **Subclasses precede parents:** A derived class is always checked before its superclasses.
2. **Order of declaration is preserved:** In `class D(B, C):`, class `B` is prioritized over class `C`.
3. **Monotonicity:** If class $X$ precedes class $Y$ in the resolution order of any parent, $X$ will precede $Y$ in the resolution order of all descendants.

Every class exposes its linearized resolution chain through the `__mro__` attribute:

```python
class A:
    def execute(self) -> str:
        return "A"


class B(A):
    def execute(self) -> str:
        return f"B -> {super().execute()}"


class C(A):
    def execute(self) -> str:
        return f"C -> {super().execute()}"


class D(B, C):
    def execute(self) -> str:
        return f"D -> {super().execute()}"


# Method Resolution Order inspection:
print([cls.__name__ for cls in D.__mro__])
# Output: ['D', 'B', 'C', 'A', 'object']

d = D()
print(d.execute())
# Output: D -> B -> C -> A
```

---

## 5. Access Modifiers, Encapsulation, and Name Mangling

### 5.1 Python's Encapsulation Philosophy

In languages such as Java or C++, access boundaries are enforced by compiler keywords:
- `private`: Accessible only within the declaring class.
- `protected`: Accessible within the declaring class and its subclasses.
- `public`: Accessible from any client code.

Python adopts the principle that **"we are all consenting adults here"**. The runtime does not restrict access to memory addresses or construct impenetrable security barriers. Instead, Python relies on clear naming conventions and compiler-level transformations.

```text
+-----------------------------------------------------------------------------+
|                          PYTHON ACCESS TIERS                                |
+-----------------------------------------------------------------------------+
| Tier      | Syntax            | Meaning & Contract                          |
+-----------+-------------------+---------------------------------------------+
| Public    | name              | Default. Unrestricted external read/write.  |
| Protected | _name             | Advisory convention. Internal to class and  |
|           |                   | subclasses. External mutation discouraged.  |
| Private   | __name            | Triggers compiler Name Mangling to prevent  |
|           |                   | accidental collisions in child classes.     |
+-----------------------------------------------------------------------------+
```

### 5.2 The Mechanics of Name Mangling

When the Python compiler encounters an attribute or method identifier with two leading underscores and at most one trailing underscore (`__attribute`), it automatically transforms the symbol in the class bytecode:

$$\text{\texttt{\_\_attribute}} \longrightarrow \text{\texttt{\_ClassName\_\_attribute}}$$

Let us verify this by inspecting an object's internal dictionary (`__dict__`):

```python
class BankAccount:
    def __init__(self, account_holder: str, balance: float) -> None:
        self.account_holder = account_holder  # Public
        self._routing_number = "021000021"  # Protected
        self.__balance = float(balance)  # Private (mangled)


account = BankAccount("Dr. Vance", 25000.0)

# Inspecting internal namespace:
print(account.__dict__)
# Output contains:
# 'account_holder': 'Dr. Vance'
# '_routing_number': '021000021'
# '_BankAccount__balance': 25000.0

# Direct access to __balance fails:
# account.__balance -> AttributeError: 'BankAccount' object has no attribute '__balance'

# Access via mangled identifier succeeds (proving no physical sandbox):
print(account._BankAccount__balance)  # 25000.0
```

### 5.3 Why Name Mangling Exists

The primary purpose of name mangling is **not** data security, but **inheritance safety**. 

If a base class `A` utilizes an internal state variable `__state`, and a derived class `B` also defines `__state`, without name mangling `B`'s initialization would silently overwrite `A`'s variable. With name mangling, `A` reads `self._A__state` while `B` reads `self._B__state`, ensuring that both classes operate safely without state interference.

### 5.4 Encapsulation via `@property`

To provide safe, controlled access to internal state without exposing raw variables or writing verbose Java-style `get_value()` and `set_value()` methods, Python provides the `@property` decorator:

```python
class AcademicStaff:
    def __init__(self, name: str, base_salary: float) -> None:
        self.name = name
        self.base_salary = base_salary  # Invokes property setter

    @property
    def base_salary(self) -> float:
        """Getter returning internal protected float."""
        return self._base_salary

    @base_salary.setter
    def base_salary(self, value: float) -> None:
        """Setter enforcing positive numeric invariant."""
        if not isinstance(value, (int, float)):
            raise TypeError("Salary must be numeric.")
        if value <= 0:
            raise ValueError(f"Base salary must be positive, got {value}.")
        self._base_salary = float(value)
```

---

## 6. Polymorphism, Duck Typing, and Operator Overloading

### 6.1 Subtype Polymorphism vs Duck Typing

Polymorphism allows code to treat different objects through a uniform operational interface:
1. **Subtype Polymorphism (Nominal):** Objects inherit from a common base class (`Person` -> `Doctor`, `Student`, `Researcher`). The runtime dynamically dispatches `perform_activity()` based on the concrete subclass instance.
2. **Ad-hoc Polymorphism (Duck Typing):** In Python, objects do not need a shared superclass. As long as an object provides the required method signature or protocol, client code can execute against it.

```python
class ConsoleLogger:
    def log(self, message: str) -> None:
        print(f"[CONSOLE] {message}")


class NetworkLogger:
    def log(self, message: str) -> None:
        # Simulated socket transmission
        pass


def execute_task(logger) -> None:
    # Duck typing: Expects any object that provides a .log() method
    logger.log("Task executed successfully.")
```

### 6.2 Operator Overloading

Operator overloading allows user-defined classes to intercept Python's built-in mathematical, comparison, and container operators by defining double-underscore ("dunder") methods:

```text
+--------------------------------------------------------------------------+
|                     COMMON OPERATOR DUNDER METHODS                       |
+--------------------------------------------------------------------------+
| Syntax            | Method                    | Mathematical Operation   |
+-------------------+---------------------------+--------------------------+
| a + b             | __add__(self, other)      | Addition / Translation   |
| a - b             | __sub__(self, other)      | Subtraction / Difference |
| a * b             | __mul__(self, other)      | Multiplication / Scaling |
| a == b            | __eq__(self, other)       | Equivalence Relation     |
| abs(a)            | __abs__(self)             | Euclidean Norm / Modulus |
| str(a)            | __str__(self)             | Human-readable String    |
| repr(a)           | __repr__(self)            | Unambiguous Code String  |
+--------------------------------------------------------------------------+
```

#### Vector2D Mathematical Implementation

Consider a 2D Euclidean spatial vector $\mathbf{v} = \begin{bmatrix} x \\ y \end{bmatrix}$:
- Vector Addition: $\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_x + v_x \\ u_y + v_y \end{bmatrix}$
- Scalar Multiplication: $k \mathbf{v} = \begin{bmatrix} k \cdot v_x \\ k \cdot v_y \end{bmatrix}$
- Euclidean Norm: $\|\mathbf{v}\| = \sqrt{x^2 + y^2}$

```python
import math


class Vector2D:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)

    def __repr__(self) -> str:
        return f"Vector2D(x={self.x}, y={self.y})"

    def __add__(self, other: object) -> "Vector2D":
        if isinstance(other, Vector2D):
            return Vector2D(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __mul__(self, other: object) -> "Vector2D | float":
        if isinstance(other, (int, float)):
            return Vector2D(self.x * other, self.y * other)
        if isinstance(other, Vector2D):
            # Dot product
            return self.x * other.x + self.y * other.y
        return NotImplemented

    def __abs__(self) -> float:
        return math.hypot(self.x, self.y)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Vector2D):
            return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)
        return False
```

---

## 7. Python Modules, Packages, and Clean Architecture

### 7.1 The Module Execution Model

In Python, every individual `.py` file functions as a **Module**. When a module is loaded:
1. Python executes the file top-to-bottom within its own namespace.
2. The module is cached in `sys.modules`. Subsequent imports return the cached object immediately without re-executing file statements.
3. The module's global variables become attributes on the module object (`module.variable`).

To differentiate between a file being run as a standalone script versus being imported as a library, Python populates the special variable `__name__`:
```python
if __name__ == "__main__":
    # Executed only when invoked directly from the CLI
    main()
```

### 7.2 Packages and `__init__.py`

A **Package** is a file system directory containing Python modules and an `__init__.py` file. The `__init__.py` file serves three essential functions:
1. **Package Declaration:** Signals to Python's import machinery that the directory should be treated as an importable package.
2. **Public API Facade:** Selectively exposes internal classes, allowing consumers to import top-level symbols cleanly:
   ```python
   # Inside audiokit/__init__.py
   from .effects.reverb import ConvolutionReverb
   from .filters.lowpass import LowPassFilter
```
3. **Export Control with `__all__`:** Declares which symbols are imported when a user runs `from package import *`:
   ```python
   __all__ = ["LowPassFilter", "ConvolutionReverb"]
```

### 7.3 Relative vs Absolute Imports

- **Absolute Imports:** Specify the full path from the project root directory:
  ```python
  from audiokit.filters.lowpass import LowPassFilter
```
- **Explicit Relative Imports:** Use leading dots to indicate relative position:
  - `from . import signal` (same directory)
  - `from .equalizer import ParametricEqualizer` (sibling module)
  - `from ..signal import AudioSignal` (parent package module)

---

## 8. Deep Learning Framework OOP Architecture (PyTorch & Keras)

### 8.1 Why Deep Learning Uses Subclassing

Modern deep learning frameworks rejected flat, functional architectures in favor of object-oriented subclassing. A neural layer is fundamentally a stateful entity:
- **State:** Trainable weights $\mathbf{W}$, biases $\mathbf{b}$, running mean/variance, optimizer states.
- **Behavior:** The forward computational pass ($f(\mathbf{X}) = \sigma(\mathbf{X}\mathbf{W} + \mathbf{b})$), backward gradient calculations, serialization, device transfers (`.to(device)`).

Subclassing base module abstractions (`torch.nn.Module`, `tf.keras.layers.Layer`) guarantees that:
1. Parameters registered inside submodules are automatically discovered and aggregated for backpropagation.
2. Pre- and post-forward execution hooks (profiling, gradient caching) execute reliably via the callable instance protocol (`__call__`).
3. Complex nested architectures (Transformers, ResNets, GANs) can be composed hierarchically.

### 8.2 Building a NumPy Neural Layer Hierarchy from Scratch

To demonstrate how PyTorch's `nn.Module` operates under the hood, we construct a pure NumPy layer hierarchy:

```text
                           +--------------------------+
                           |       Layer (ABC)        |
                           +--------------------------+
                           | - _parameters: dict      |
                           | - _sublayers: dict       |
                           | + total_parameters: int  |
                           | + __call__(inputs)       |
                           | + forward(inputs)*       |
                           +------------+-------------+
                                        ^
          +-----------------+-----------+-----------+-----------------+
          |                 |                       |                 |
+---------+-------+ +-------+-------+       +-------+-------+ +-------+-------+
|      Dense      | |     ReLU      |       |    Softmax    | |  Sequential   |
+-----------------+ +---------------+       +---------------+ +---------------+
| - weights: ndarray| | + forward() |       | + forward()   | | - layers: list|
| - biases: ndarray | +-------------+       +---------------+ | + forward()   |
| + forward()     |                                           +---------------+
+-----------------+
```

#### Base `Layer` Implementation
```python
from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__
        self._parameters: dict[str, np.ndarray] = {}
        self._sublayers: dict[str, Layer] = {}

    def register_parameter(self, name: str, tensor: np.ndarray) -> None:
        self._parameters[name] = tensor

    def register_sublayer(self, name: str, layer: Layer) -> None:
        self._sublayers[name] = layer

    @property
    def total_parameters(self) -> int:
        direct = sum(p.size for p in self._parameters.values())
        children = sum(l.total_parameters for l in self._sublayers.values())
        return direct + children

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        # Callable instance delegates to forward method
        return self.forward(inputs)

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass
```

#### Subclassed `Dense`, `ReLU`, and `Softmax`
```python
class Dense(Layer):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        # He normal initialization: std = sqrt(2 / in_features)
        std = np.sqrt(2.0 / in_features)
        self.register_parameter("weights", np.random.randn(in_features, out_features) * std)
        self.register_parameter("biases", np.zeros(out_features))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.dot(inputs, self._parameters["weights"]) + self._parameters["biases"]


class ReLU(Layer):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, inputs)


class Softmax(Layer):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Numerically stable softmax with row-max subtraction
        shifted = inputs - np.max(inputs, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
```

### 8.3 Fashion-MNIST Multi-Layer Perceptron (MLP) Case Study

We now compose these layers into a complete neural network architecture for the 10-class Fashion-MNIST benchmark:
- **Input:** $28 \times 28$ grayscale images flattened into a 784-dimensional vector.
- **Hidden Layer:** Dense layer projecting $784 \longrightarrow 128$ dimensions, activated by ReLU.
- **Output Layer:** Dense layer projecting $128 \longrightarrow 10$ dimensions, normalized by Softmax.

```python
class FashionMNISTMLP(Layer):
    def __init__(self) -> None:
        super().__init__(name="FashionMNIST_MLP")
        self.fc1 = Dense(28 * 28, 128)
        self.act1 = ReLU()
        self.fc2 = Dense(128, 10)
        self.classifier = Softmax()

        # Register child sublayers
        self.register_sublayer("fc1", self.fc1)
        self.register_sublayer("act1", self.act1)
        self.register_sublayer("fc2", self.fc2)
        self.register_sublayer("classifier", self.classifier)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim == 3 and inputs.shape[1:] == (28, 28):
            inputs = inputs.reshape(inputs.shape[0], -1)
        x = self.fc1(inputs)
        x = self.act1(x)
        logits = self.fc2(x)
        return self.classifier(logits)
```

Total Trainable Parameters Calculation:
$$\text{Weights}_{\text{fc1}} = 784 \times 128 = 100,352$$
$$\text{Biases}_{\text{fc1}} = 128$$
$$\text{Weights}_{\text{fc2}} = 128 \times 10 = 1,280$$
$$\text{Biases}_{\text{fc2}} = 10$$
$$\text{Total Parameters} = 100,352 + 128 + 1,280 + 10 = 101,770$$

---

## 9. Architectural Comparison: Paradigms & Conventions

| Architectural Dimension | Procedural Programming | Object-Oriented Programming | Deep Learning OOP Subclassing |
| :--- | :--- | :--- | :--- |
| **Primary Decomposition** | Functions & subroutines | Classes & collaborating objects | Modules, Layers, and Computational Graphs |
| **State Management** | Global variables, passed dictionaries | Encapsulated instance attributes | Registered Parameter Tensors (`weights`, `biases`) |
| **Interface Polymorphism** | Manual `if/elif` type branching | Dynamic dispatch & Duck Typing | Callable instance protocol (`__call__` -> `forward`) |
| **Code Reuse** | Library utility functions | Inheritance hierarchies & ABCs | Layer composition & Subclassed model hierarchies |
| **Access Control** | None (open namespace) | Protected (`_`) & Private (`__`) mangling | Protected submodule/parameter registries |

---

## 10. Practical Exercises and Mastery Drills

1. **Polygon Hierarchy Expansion:** In [01 — Inheritance & Super](notebooks/01-inheritance-mechanics-and-super.ipynb), implement a concrete `RegularPolygon` subclass of `Shape` taking `num_sides: int` and `side_length: float`. Implement exact formulas for area ($A = \frac{n s^2}{4 \tan(\pi / n)}$) and perimeter ($P = n s$).
2. **Strict Descriptor Validation:** In [02 — Access Modifiers & Mangling](notebooks/02-access-modifiers-and-name-mangling.ipynb), construct a reusable descriptor class `ValidatedNumber` that validates minimum and maximum boundaries across arbitrary instance variables.
3. **Polynomial Operator Overloading:** In [03 — Polymorphism & Overloading](notebooks/03-polymorphism-duck-typing-and-overloading.ipynb), implement a `Polynomial` class where `p1 + p2` adds polynomial coefficients, `p1 * p2` computes polynomial convolution, and `p(x)` evaluates the polynomial via Horner's method using `__call__`.
4. **Dynamic Plugin Architecture:** In [04 — Modules & Packages](notebooks/04-modules-packages-and-architecture.ipynb), build a dynamic plugin loader that searches a target directory for `.py` files, inspects them using `importlib`, and automatically registers any class inheriting from `BaseDSPModule`.
5. **Convolutional Layer Subclassing:** In [05 — Deep Learning Subclassing](notebooks/05-deep-learning-subclassing-pytorch-and-keras.ipynb), subclass `Layer` to implement a 2D spatial convolution layer `Conv2D(in_channels, out_channels, kernel_size, stride)` in pure NumPy.

---

## References & Further Reading

1. **Python Documentation:** *The Python Standard Library — `abc` — Abstract Base Classes*, Python Software Foundation.
2. **Python Enhancement Proposals:** 
   - PEP 253: Subtyping Built-in Types.
   - PEP 3119: Introducing Abstract Base Classes.
   - PEP 420: Implicit Namespace Packages.
3. **Barrett, K., et al.:** *A Monotonic Superclass Linearization for Dylan (The C3 Linearization Algorithm)*, OOPSLA 1996.
4. **Gamma, E., Helm, R., Johnson, R., Vlissides, J.:** *Design Patterns: Elements of Reusable Object-Oriented Software*, Addison-Wesley, 1994.
5. **Paszke, A., et al.:** *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, NeurIPS 2019.
6. **Chollet, F.:** *Deep Learning with Python (2nd Edition)*, Manning Publications, 2021.
