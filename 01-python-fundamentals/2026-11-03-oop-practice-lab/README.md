# Object-Oriented Programming Practice Lab: Encapsulation, Hierarchies, Aggregation, and Bounded Data Structures

[Back to Module 01](../README.md) · [Back to the master roadmap](../../README.md)

**Session date:** 2026-11-03 · **Module:** 01 — Python Fundamentals & Algorithmic Thinking  
**Format:** Comprehensive textbook-grade study guide, UML architectural models, mathematical and computational complexity proofs, and executable practice notebooks

---

## Overview

In production machine learning systems, deep learning frameworks, and large-scale computational backends, object-oriented programming (OOP) serves as the primary paradigm for structuring software. Whether designing reusable tensor layers in PyTorch (`torch.nn.Module`), configuring modular training pipelines in Hugging Face Transformers, or engineering streaming replay buffers in deep reinforcement learning, robust OOP architecture ensures software maintainability, extensibility, and correctness.

Without rigorous object-oriented design, software systems suffer from tightly coupled components, uncontrolled side effects, violated data invariants, and brittle codebases. This practice lab provides an intensive, textbook-grade consolidation of object-oriented principles, structural design patterns, and fundamental bounded data structures.

This session bridges core OOP theory and practical algorithmic engineering across six pillars:

1. **Object-Oriented Fundamentals & Encapsulation Mechanics:** We review the fundamental distinction between classes (abstract templates and behavior definitions) and objects (concrete stateful instances). We examine how Python enforces data encapsulation, contrasts public, protected (`_single_underscore`), and private (`__double_underscore`) members, activates compiler-level name mangling (`_ClassName__attribute`), and leverages properties (`@property`, `@setter`) to maintain invariants without sacrificing idiomatic access.
2. **Data Abstraction vs. Encapsulation:** We formalize the architectural divergence between data abstraction (design-time decisions focusing on *what* an object exposes through abstract interfaces and contracts) and data encapsulation (implementation-time mechanisms focusing on *how* internal representations and mechanics are hidden and bundled). We demonstrate how Python's Abstract Base Classes (`abc.ABC` and `@abstractmethod`) enforce strict interface conformance.
3. **Inheritance Taxonomies and Method Resolution Order:** We dissect the `is-a` relationship, single vs. multiple inheritance, multilevel hierarchies, hierarchical branching, and hybrid taxonomies. We inspect the C3 Linearization algorithm that governs Python's Method Resolution Order (MRO) and demonstrate cooperative multiple inheritance using `super()`.
4. **Polymorphic Dispatch and Dynamic Typing:** We explore runtime polymorphism, dynamic typing, and Python's "duck typing" philosophy (*"if it walks like a duck and quacks like a duck, it's a duck"*). We examine method overriding, operator overloading via special dunder methods (`__len__`, `__getitem__`, `__repr__`, `__eq__`), and clean emulation of method overloading using single-dispatch decorators.
5. **Structural Relationships: Aggregation vs. Composition:** We formalize the distinction between composition (strong whole-part ownership where component lifecycles are bound to their container) and aggregation (loose association where components maintain independent lifecycles and can outlive their host container). We implement Exercise 1: the `Device` and `Manufacturer` system, verifying how loose coupling improves modular testing and hardware inventory modeling.
6. **Characteristics of OOP — Municipal Directory System:** We build Exercise 2: a municipal `Ward` directory managing polymorphic residents (`Student`, `Teacher`, `Doctor`) derived from an abstract `Person` base. We implement aggregate query engines, role filtering, demographic statistics, and mathematical age sorting.
7. **Bounded Linear Data Structures (Stacks & Queues):** We develop memory-bounded abstract data types (ADTs) under strict capacity limits. We implement Exercise 3: a bounded `Stack` governed by Last-In, First-Out (LIFO) discipline with explicit overflow and underflow exception handling. We implement Exercise 4: a bounded `Queue` governed by First-In, First-Out (FIFO) discipline, contrasting the $O(N)$ linear memory shifts of naive Python list implementations with the optimal $O(1)$ amortized operations of circular array ring buffers.
8. **Applied Machine Learning Architecture:** We connect foundational OOP patterns directly to modern AI systems. We build a mini deep-learning sequential layer container mimicking PyTorch's execution hierarchy, construct an experience replay buffer for reinforcement learning using circular queue pointers, and evaluate computation graphs using stack-based reverse-mode traversal.

---

## Learning outcomes

By the end of this session, you should be able to:

- Distinguish formally between class definitions (structural blueprints) and object instances (concrete memory allocations).
- Implement defensive encapsulation in Python using private naming conventions (`__attribute`), analyze compiler name mangling (`_ClassName__attribute`), and write idiomatic property getters and setters.
- Explain the philosophical and architectural differences between data abstraction (design-time interface hiding) and data encapsulation (implementation-time data bundling).
- Construct abstract base classes using `abc.ABC` and `@abstractmethod` to enforce mandatory method contracts across derived domain entities.
- Architect clean inheritance hierarchies (`is-a` relationships) and trace the execution path of cooperative multiple inheritance using `super()` and the C3 Linearization MRO.
- Implement runtime polymorphism and method overriding, allowing client containers to interact with polymorphic derived instances via unified interfaces.
- Explain duck typing and leverage Python special dunder methods (`__repr__`, `__str__`, `__len__`, `__eq__`, `__iter__`) to integrate custom domain classes into Python's native syntax.
- Formulate the structural differences between object composition (strong lifetime coupling) and object aggregation (weak reference association), identifying when to use each in production architectures.
- Implement the `Manufacturer` and `Device` aggregation model, verifying that manufacturer components can exist independently and be shared across multiple device instances.
- Build a hierarchical municipal management system (`Ward`, `Person`, `Student`, `Teacher`, `Doctor`), implementing domain queries, role-specific filters, and mean attribute calculations.
- Prove mathematically that sorting residents by age in ascending order is equivalent to sorting their Gregorian year of birth ($	ext{YoB}$) in descending order: $	ext{Age} = Y_{	ext{current}} - 	ext{YoB} \implies 	ext{YoB} = Y_{	ext{current}} - 	ext{Age}$.
- Define the formal Abstract Data Type (ADT) specification for a Bounded Stack, implementing state invariants for `push`, `pop`, `top`, `is_empty`, and `is_full`.
- Implement robust exception handling for stack overflow (`OverflowError`) and underflow (`IndexError`), verifying zero memory leaks or partial mutations during exceptional states.
- Define the formal ADT specification for a Bounded Queue, contrasting the algorithmic consequences of naive list-based deletions ($O(N)$ due to element shifting) versus circular array indexing ($O(1)$ pointer arithmetic).
- Implement a high-performance circular array ring buffer utilizing modulo arithmetic: $	ext{rear} \leftarrow (	ext{rear} + 1) \pmod C$, tracking front and rear pointers in constant time.
- Connect linear bounded data structures to deep learning systems: formulate an Experience Replay Buffer for Q-learning agents utilizing circular queue mechanics.
- Implement an autograd computation graph evaluator using stack-based execution tapes for reverse-mode topological differentiation.

---

## Session map

| Notebook | Focus | Key Topics |
| :--- | :--- | :--- |
| [01 · Encapsulation & Aggregation Lab](notebooks/01-encapsulation-and-aggregation-lab.ipynb) | Class Mechanics & Structural Association | State vs. behavior, public/protected/private, name mangling, `@property`, `Manufacturer` & `Device` aggregation vs composition, defensive copies |
| [02 · Ward Hierarchy & Polymorphism](notebooks/02-ward-people-hierarchy-and-polymorphism.ipynb) | Inheritance & Municipal Directory System | `Person` base class, `Student`, `Teacher`, `Doctor` derived classes, method overriding, `Ward` aggregation container, `count_doctor`, `sort_age`, `ave_teacher_year_of_birth` |
| [03 · Bounded Stack Data Structure](notebooks/03-bounded-stack-data-structure-and-applications.ipynb) | LIFO Abstract Data Type & Exception Invariants | Bounded capacity, `push`, `pop`, `top`, `is_empty`, `is_full`, `OverflowError`, `IndexError`, parentheses matching, reverse-polish notation |
| [04 · Bounded Queue & Circular Buffers](notebooks/04-bounded-queue-and-circular-ring-buffers.ipynb) | FIFO Abstract Data Type & Modulo Indexing | Naive list `pop(0)` $O(N)$ bottleneck vs circular ring buffer $O(1)$, front/rear modulo pointer math, `collections.deque` benchmark, throughput profiling |
| [05 · Applied OOP: ML Buffers & Layers](notebooks/05-applied-oop-ml-replay-buffers-and-layers.ipynb) | Deep Learning Modules & RL Buffers | Subclassing layer framework (`Layer`, `Linear`, `ReLU`, `SequentialContainer`), RL Experience Replay Buffer with uniform sampling, reverse-mode execution tape |

---

## 1. Object-Oriented Fundamentals & The Encapsulation Engine

### 1.1 Classes, Objects, and Memory Representation

In computer science, an **object** is an entity that bundles state (data members or attributes) and behavior (functions or methods) into a cohesive runtime unit. A **class** is an abstract blueprint or type descriptor that specifies what data attributes an object possesses and what operations may be performed upon it.

```text
+-------------------------------------------------------+
|                       CLASS                           |
|  - Blueprint / Type Descriptor                        |
|  - Defines attributes: name: str, price: float        |
|  - Defines methods: describe(), discount(pct: float)  |
+-------------------------------------------------------+
                           |
            Instantiation  | (Allocates heap memory,
                           |  invokes __new__ & __init__)
                           v
+-------------------------------------------------------+
|                   OBJECT INSTANCE                     |
|  - Concrete runtime allocation                        |
|  - Possesses distinct identity (id(obj))              |
|  - Contains unique state values:                      |
|      self.name = 'EdgeNode-X1'                        |
|      self.price = 1299.99                             |
|  - Shares method code across all instances            |
+-------------------------------------------------------+
```

When an object is instantiated in Python via `obj = MyClass(args)`:
1. `MyClass.__new__(cls, *args, **kwargs)` allocates memory for the new instance on the heap and returns a raw instance.
2. `MyClass.__init__(self, *args, **kwargs)` initializes the freshly allocated instance, binding initial attribute values to `self`.
3. Every instance maintains an internal reference dictionary `__dict__` storing its mutable instance attributes, unless optimized via `__slots__`.

### 1.2 Data Encapsulation and Access Modifiers

Encapsulation serves two primary software engineering objectives:
1. **Bundling:** Unifying related data variables and the procedures that mutate them within the same architectural boundary.
2. **Information Hiding:** Shielding internal implementation details and data representations from external modification, preventing outside consumers from driving an object into an invalid or corrupt state.

Unlike compiled languages like C++ or Java that enforce compile-time access controls (`public`, `protected`, `private`), Python operates under a community convention commonly summarized as *"we are all consenting adults here"*. However, Python provides syntactic conventions and compiler transformations to establish distinct levels of accessibility:

| Modifier Level | Syntax Convention | Scope & Visibility | Python Runtime Mechanism |
| :--- | :--- | :--- | :--- |
| **Public** | `self.attribute` | Everywhere | Standard dictionary key in `instance.__dict__` |
| **Protected** | `self._attribute` | Internal class & derived subclasses | Soft convention; signals private intent to linters and IDEs |
| **Private** | `self.__attribute` | Enclosing class only | Compiler-enforced **Name Mangling**: rewritten to `_ClassName__attribute` |

```python
class StorageAccount:
    """Demonstrates access modifier levels and compiler name mangling."""

    def __init__(self, account_id: str, balance: float) -> None:
        self.account_id: str = account_id  # Public: accessible freely
        self._audit_code: str = "AUDIT-2026"  # Protected: intended for subclasses
        self.__balance: float = float(balance)  # Private: mangled by Python compiler

    def get_balance(self) -> float:
        """Public getter method providing controlled read access."""
        return self.__balance

    def deposit(self, amount: float) -> None:
        """Public mutator enforcing defensive validation invariants."""
        if amount <= 0.0:
            raise ValueError(f"Deposit amount must be strictly positive, received {amount}")
        self.__balance += amount
```

When the Python compiler encounters an attribute prefixed with at least two leading underscores and at most one trailing underscore (e.g., `__balance`), it automatically rewrites the identifier:
$$	ext{mangled\_name} = 	ext{"\_"} + 	ext{ClassName} + 	ext{"\_\_"} + 	ext{attribute\_name}$$

For example, `__balance` defined inside class `StorageAccount` is mangled to `_StorageAccount__balance`. While this identifier can technically still be accessed via its mangled name, the mechanism prevents accidental shadowing or overriding in derived classes and communicates private encapsulation.

### 1.3 Modern Property Decorators (`@property`)

To maintain clean syntax while enforcing encapsulation invariants, Python provides the property descriptor protocol via the `@property` decorator:

```python
class TemperatureSensor:
    def __init__(self, celsius: float) -> None:
        self.celsius = celsius  # Invokes the @celsius.setter directly

    @property
    def celsius(self) -> float:
        """Return the temperature reading in degrees Celsius."""
        return self._celsius

    @celsius.setter
    def celsius(self, value: float) -> None:
        """Validate thermodynamic absolute zero boundary before updating state."""
        if value < -273.15:
            raise ValueError(f"Temperature {value}°C is below absolute zero (-273.15°C)!")
        self._celsius = float(value)

    @property
    def fahrenheit(self) -> float:
        """Computed derived property: converts Celsius to Fahrenheit."""
        return (self._celsius * 9.0 / 5.0) + 32.0
```

Properties provide zero-cost syntactic abstraction: client code reads `sensor.celsius` or `sensor.fahrenheit` as if they were simple public variables, while the underlying engine executes validation logic and dynamic transformations.

---

## 2. Data Abstraction vs. Data Encapsulation

Software engineering frequently conflates **data abstraction** and **data encapsulation**. While complementary, they address fundamentally different phases of software design:

```text
+-----------------------------------------------------------------------------------------+
|                                    DATA ABSTRACTION                                     |
|  - Addressed at the DESIGN / ARCHITECTURAL level.                                       |
|  - Focuses on WHAT the object does rather than HOW it accomplishes it.                  |
|  - Exposes essential behavioral interfaces while hiding underlying algorithmic details. |
|  - Implemented via: Abstract Base Classes (abc.ABC, @abstractmethod), Protocols.        |
+-----------------------------------------------------------------------------------------+
                                           vs.
+-----------------------------------------------------------------------------------------+
|                                   DATA ENCAPSULATION                                    |
|  - Addressed at the IMPLEMENTATION level.                                               |
|  - Focuses on HOW the object protects its internal state and bundles behavior.          |
|  - Hides internal data representations to prevent unauthorized tampering.               |
|  - Implemented via: Access modifiers (public, protected _, private __), properties.     |
+-----------------------------------------------------------------------------------------+
```

### 2.1 Abstract Base Classes (`abc.ABC`)

In Python, formal abstraction is declared using the standard library's `abc` module. By inheriting from `abc.ABC` and marking abstract methods with `@abstractmethod`, a class cannot be instantiated directly unless all abstract methods are overridden by the derived subclass:

```python
from abc import ABC, abstractmethod


class HardwareInterface(ABC):
    """Abstract base interface defining the mandatory contract for hardware controllers."""

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the hardware peripheral. Must return True if successful."""
        pass

    @abstractmethod
    def read_telemetry(self) -> dict[str, float]:
        """Poll telemetry sensors and return readings as a dictionary."""
        pass
```

Attempting to instantiate `HardwareInterface()` directly raises `TypeError: Can't instantiate abstract class HardwareInterface without an implementation for abstract methods 'initialize', 'read_telemetry'`. This guarantees architectural safety across large engineering teams.

---

## 3. Structural Relationships: Aggregation vs. Composition

When designing relationships between collaborating classes, object-oriented systems differentiate between **inheritance** (`is-a`), **composition** (`has-a`, strong whole-part ownership), and **aggregation** (`has-a`, loose association).

```text
               OBJECT-ORIENTED STRUCTURAL RELATIONSHIPS
               
       +------------------------------------------------+
       |               Inheritance (is-a)               |
       |  Subclass derives from Base Class              |
       |  E.g., Student is-a Person                     |
       +------------------------------------------------+
                               |
                               v
       +------------------------------------------------+
       |              Association (has-a)               |
       +------------------------------------------------+
             /                                    \
            v                                      v
+-------------------------------+      +-------------------------------+
|       Composition (has-a)     |      |       Aggregation (has-a)     |
| - Strong whole-part ownership |      | - Loose association / shared  |
| - Child lifetime is bound to  |      | - Child exists independently  |
|   the parent lifecycle.       |      |   of the parent container.    |
| - If parent is destroyed,     |      | - If parent is destroyed,     |
|   the child is also destroyed.|      |   child continues to exist!   |
| - E.g., Person and Heart;     |      | - E.g., Device and            |
|   NeuralNet and Layer weights |      |   Manufacturer; Ward & People |
+-------------------------------+      +-------------------------------+
```

### 3.1 Exercise 1 Architectural Deep Dive: `Manufacturer` and `Device`

In Exercise 1 from the lecture curriculum, we examine the aggregation relationship between `Manufacturer` and `Device`:

- A `Manufacturer` represents an industrial producer with an identifier (`identity: int`) and geographic facility (`location: str`).
- A `Device` represents a commercial hardware product with a `name: str`, `price: float`, and an associated `manufacturer: Manufacturer`.

#### Aggregation vs. Composition Implementation Comparison

Under **Aggregation**, the `Manufacturer` is created outside the device and passed in as a reference. Multiple devices can share the exact same manufacturer instance:

```python
# --- AGGREGATION PATTERN ---
class Manufacturer:
    def __init__(self, identity: int, location: str) -> None:
        self.__identity: int = identity
        self.__location: str = location

    def describe(self) -> str:
        return f"Manufacturer ID: {self.__identity}, Location: {self.__location}"


class Device:
    def __init__(self, name: str, price: float, manufacturer: Manufacturer) -> None:
        self.__name: str = name
        self.__price: float = float(price)
        self.__manufacturer: Manufacturer = manufacturer  # Holds reference to external object!

    def describe(self) -> str:
        return (
            f"Device: {self.__name}, Price: ${self.__price:.2f} | {self.__manufacturer.describe()}"
        )


# Shared lifecycle: manufacturer exists independently
mfg = Manufacturer(identity=101, location="Munich, Germany")
dev1 = Device(name="Sensor-A", price=120.0, manufacturer=mfg)
dev2 = Device(name="Sensor-B", price=180.0, manufacturer=mfg)

del dev1  # Deleting dev1 does NOT destroy mfg! dev2 and mfg continue to exist.
```

Under **Composition**, the `Device` would construct its own internal `Manufacturer` within its constructor:
```python
# --- COMPOSITION PATTERN ---
class DeviceComposed:
    def __init__(self, name: str, price: float, identity: int, location: str) -> None:
        self.__name: str = name
        self.__price: float = float(price)
        self.__manufacturer = Manufacturer(identity, location)  # Tightly bound lifecycle
```
If `DeviceComposed` is destroyed, its internal `__manufacturer` object is also destroyed by garbage collection.

---

## 4. Characteristics of OOP: Municipal Directory System

Exercise 2 models an administrative municipal directory (`Ward`) managing polymorphic human residents.

```text
                             +-------------------------+
                             |         Person          |
                             |  - name: str            |
                             |  - yob: int             |
                             |  + describe(): str      |
                             |  + get_yob(): int       |
                             +-------------------------+
                                          ^
                                          |  is-a (Inheritance)
                    +---------------------+---------------------+
                    |                     |                     |
        +-----------------------+ +--------------------+ +---------------------+
        |        Student        | |      Teacher       | |       Doctor        |
        | - grade: str          | | - subject: str     | | - specialist: str   |
        | + describe(): str     | | + describe(): str  | | + describe(): str   |
        +-----------------------+ +--------------------+ +---------------------+
                    ^                     ^                     ^
                    |                     |                     |
                    +---------------------+---------------------+
                                          |
                                          | (list[Person])
                                          | Aggregation
                             +-------------------------+
                             |          Ward           |
                             |  - name: str            |
                             |  - list_people: list    |
                             |  + add_person(p)        |
                             |  + count_doctor()       |
                             |  + sort_age()           |
                             |  + ave_teacher_yob()    |
                             +-------------------------+
```

### 4.1 Class Hierarchy and Polymorphic Method Overriding

All citizens share common demographic attributes: `name` and `yob` (Year of Birth). The base class `Person` implements default description logic:

```python
class Person:
    def __init__(self, name: str, yob: int) -> None:
        self._name: str = name
        self._yob: int = int(yob)

    def get_yob(self) -> int:
        return self._yob

    def describe(self) -> str:
        return f"Person - Name: {self._name} - YoB: {self._yob}"
```

Derived classes override `describe()` to specialize their output:
- `Student`: `f"Student - Name: {self._name} - YoB: {self._yob} - Grade: {self.__grade}"`
- `Teacher`: `f"Teacher - Name: {self._name} - YoB: {self._yob} - Subject: {self.__subject}"`
- `Doctor`: `f"Doctor - Name: {self._name} - YoB: {self._yob} - Specialist: {self.__specialist}"`

Because each derived class conforms to the `Person` interface, the container class `Ward` can iterate over its resident collection and invoke `.describe()` polymorphically without inspecting concrete types.

### 4.2 Mathematical Relationship: Sorting by Age vs. Year of Birth

Exercise 2 requires sorting residents by **age in ascending order**.

Let $Y_{	ext{current}}$ denote the fixed current calendar year, and let $	ext{YoB}_i$ denote the year of birth of individual $i$. The age $A_i$ is given by:
$$A_i = Y_{	ext{current}} - 	ext{YoB}_i$$

Now, consider two individuals $i$ and $j$ such that individual $i$ is strictly younger than individual $j$:
$$A_i < A_j$$

Substituting the age formula:
$$Y_{	ext{current}} - 	ext{YoB}_i < Y_{	ext{current}} - 	ext{YoB}_j$$

Subtracting $Y_{	ext{current}}$ from both sides:
$$-	ext{YoB}_i < -	ext{YoB}_j$$

Multiplying by $-1$ reverses the inequality:
$$	ext{YoB}_i > 	ext{YoB}_j$$

$$	herefore \quad A_i \le A_j \iff 	ext{YoB}_i \ge 	ext{YoB}_j$$

**Theorem:** Sorting a collection of individuals in **ascending order of age** is mathematically identical to sorting them in **descending order of Year of Birth ($	ext{YoB}$)**.

```python
def sort_age(self) -> None:
    """Sort residents in ascending order of age (descending order of YoB)."""
    self.__list_people.sort(key=lambda p: p.get_yob(), reverse=True)
```

---

## 5. Bounded Stack Abstract Data Type

A **Stack** is a linear data structure governed by the **Last-In, First-Out (LIFO)** operational principle: the last element pushed onto the stack is the first element retrieved during a pop operation.

```text
                  BOUNDED STACK (Capacity = 4)
                  
   Initial: Empty               Push(31)             Push(12)             Push(10)
   +------------+           +------------+       +------------+       +------------+
 3 |   [Empty]  |         3 |   [Empty]  |     3 |   [Empty]  |     3 |   [Empty]  |
 2 |   [Empty]  |         2 |   [Empty]  |     2 |   [Empty]  |     2 |     10     | <- TOP
 1 |   [Empty]  |         1 |   [Empty]  |     1 |     12     | <-TOP 1 |     12     |
 0 |   [Empty]  |         0 |     31     | <-TOP 0|    31     |     0 |     31     |
   +------------+           +------------+       +------------+       +------------+
   is_empty()=True          len=1                len=2                len=3
   
   Push(20) [Full!]             Pop() -> 20          Top() -> 10          Push(99) -> Overflow!
   +------------+           +------------+       +------------+       +------------+
 3 |     20     | <- TOP  3 |   [Empty]  |     3 |   [Empty]  |     3 |  OVERFLOW  |
 2 |     10     |         2 |     10     | <-TOP 2|    10     | <-TOP 2|  EXCEPTION |
 1 |     12     |         1 |     12     |     1 |     12     |     1 |     12     |
 0 |     31     |         0 |     31     |     0 |     31     |     0 |     31     |
   +------------+           +------------+       +------------+       +------------+
   is_full()=True           len=3                len=3 (no change)    State preserved
```

### 5.1 Formal ADT Specification and State Invariants

Let $S = \langle e_0, e_1, \dots, e_{k-1} angle$ denote a sequence of elements stored in the stack, where $k = |S|$ is the current element count, and $C \in \mathbb{Z}^+$ is the pre-defined maximum capacity ($C \ge 1$).

The Bounded Stack invariant guarantees:
$$0 \le |S| \le C \quad orall t \ge 0$$

- **`is_empty()`:** Returns $	ext{True} \iff |S| = 0$.
- **`is_full()`:** Returns $	ext{True} \iff |S| = C$.
- **`push(value)`:**
  $$	ext{Precondition:} \quad |S| < C$$
  $$	ext{Postcondition:} \quad S' = \langle e_0, \dots, e_{k-1}, 	ext{value} angle \quad 	ext{and} \quad |S'| = k + 1$$
  $$	ext{Exception:} \quad 	ext{If } |S| = C, 	ext{ raises } 	exttt{OverflowError} 	ext{ and leaves } S 	ext{ unchanged.}$$
- **`pop()`:**
  $$	ext{Precondition:} \quad |S| > 0$$
  $$	ext{Postcondition:} \quad S' = \langle e_0, \dots, e_{k-2} angle, \quad |S'| = k - 1, \quad 	ext{returns } e_{k-1}$$
  $$	ext{Exception:} \quad 	ext{If } |S| = 0, 	ext{ raises } 	exttt{IndexError} 	ext{ (Underflow).}$$
- **`top()`:** Returns $e_{k-1}$ without mutating $S$. Raises `IndexError` if $|S| = 0$.

### 5.2 Complexity Analysis

Using Python's dynamic list as the underlying container:
- Appending to the end of a list (`push`) runs in amortized $O(1)$ time. Because capacity $C$ is bounded, internal reallocation ceases once capacity is reached.
- Removing from the end of a list (`pop()`) runs in worst-case $O(1)$ time.
- Reading the final index (`top() -> arr[-1]`) runs in worst-case $O(1)$ time.
- Total auxiliary memory is strictly bounded by $O(C)$.

---

## 6. Bounded Queue Abstract Data Type & Circular Ring Buffers

A **Queue** is a linear data structure governed by the **First-In, First-Out (FIFO)** operational principle: elements are inserted at the rear and removed from the front.

```text
                  FIFO DISCIPLINE (Capacity = 4)
                  
   Enqueue(31)         Enqueue(12)         Enqueue(10)         Enqueue(20) [Full!]
   +----+             +----+----+         +----+----+----+     +----+----+----+----+
   | 31 |             | 31 | 12 |         | 31 | 12 | 10 |     | 31 | 12 | 10 | 20 |
   +----+             +----+----+         +----+----+----+     +----+----+----+----+
   ^    ^             ^         ^         ^              ^     ^                   ^
 Front Rear         Front      Rear     Front           Rear Front                Rear
 
   Dequeue() -> 31              Dequeue() -> 12              Front() -> 10
   +----+----+----+             +----+----+                  +----+----+
   | 12 | 10 | 20 |             | 10 | 20 |                  | 10 | 20 | (no change)
   +----+----+----+             +----+----+                  +----+----+
   ^              ^             ^         ^                  ^         ^
 Front           Rear         Front      Rear              Front      Rear
```

### 6.1 The $O(N)$ Deletion Bottleneck in Naive Lists

When implementing a queue using a standard Python list `list.pop(0)`:
```python
class NaiveListQueue:
    def dequeue(self):
        return self.__queue.pop(0)  # WARNING: O(N) LINEAR TIME!
```

In Python's CPython implementation, a `list` is a contiguous array of pointers. Deleting element 0 forces the runtime to shift all remaining $N - 1$ elements one memory position to the left via `memmove`:
$$	ext{Cost of } M 	ext{ dequeue operations} = \sum_{i=1}^M O(i) = O(M^2)$$

For large queues or real-time streaming buffers, this introduces an unacceptable quadratic bottleneck.

### 6.2 The Circular Array Ring Buffer Solution ($O(1)$ Time)

To achieve true $O(1)$ worst-case time for both enqueue and dequeue operations without dynamic memory copies, we implement a **Circular Ring Buffer**:
- Allocate a fixed-size array of size $C$: `self._buffer = [None] * capacity`.
- Maintain two index markers: `_front` (index of current head) and `_rear` (index where next element will be written).
- Maintain an explicit counter: `_count` ($0 \le 	ext{\_count} \le C$).
- Use modulo arithmetic to wrap indices around array boundaries:
$$	ext{rear}_{	ext{next}} = (	ext{rear} + 1) \pmod C$$
$$	ext{front}_{	ext{next}} = (	ext{front} + 1) \pmod C$$

```text
                 CIRCULAR ARRAY RING BUFFER (Capacity = 4)
                 
           Index 0: [ 31 ]                  Index 0: [ None ]
          /               \                /                \
Index 3: [ 20 ]       Index 1: [ 12 ]    Index 3: [ 20 ]   Index 1: [ 12 ]
          \               /                \                /
           Index 2: [ 10 ]                  Index 2: [ 10 ]
           
      Full Buffer (_count = 4)          After Dequeue() (_count = 3)
      Front = 0, Rear = 0 (wrapped)     Front advances to (0+1)%4 = 1
```

Every operation (`enqueue`, `dequeue`, `front`) computes simple index additions and modulo arithmetic, executing in deterministic $O(1)$ worst-case time.

---

## 7. Applied Machine Learning and Systems Architecture

The abstract concepts explored in this practice lab form the computational bedrock of modern artificial intelligence pipelines:

### 7.1 Layer Subclassing and Modular Networks
Deep learning libraries such as PyTorch model complex neural architectures by subclassing base modules (`torch.nn.Module`). Layers encapsulate their trainable weight tensors, register sub-modules via aggregation, and expose unified forward call contracts (`forward(*inputs)`).

### 7.2 Experience Replay Buffers in Reinforcement Learning
Deep Q-Networks (DQN) break temporal correlations in consecutive agent trajectories by storing transitions:
$$e_t = (s_t, a_t, r_t, s_{t+1}, d_t)$$
The replay memory is precisely a bounded circular queue. When capacity $C$ is reached, new experiences overwrite the oldest transitions at the front pointer in $O(1)$ time, while minibatch samplers query random indices across the valid buffer window.

### 7.3 Autograd Reverse-Mode Differentiation Tapes
Reverse-mode automatic differentiation evaluates vector-Jacobian products by recording forward operations onto an execution tape. During backward propagation, operators are unwound in reverse order using an execution Stack, ensuring child gradients are fully accumulated before parent nodes are evaluated.

---

## 8. Algorithmic Complexity Comparison

| Data Structure / Operation | Primary Method | Time Complexity | Auxiliary Space | Exception on Limit |
| :--- | :--- | :--- | :--- | :--- |
| **Bounded Stack** | `push(x)` | $O(1)$ | $O(C)$ | `OverflowError` when full |
| **Bounded Stack** | `pop()` | $O(1)$ | $O(1)$ | `IndexError` when empty |
| **Bounded Stack** | `top()` | $O(1)$ | $O(1)$ | `IndexError` when empty |
| **Naive Queue (List)** | `enqueue(x)` | $O(1)$ amortized | $O(C)$ | `OverflowError` when full |
| **Naive Queue (List)** | `dequeue()` | $O(N)$ linear memory shift | $O(1)$ | `IndexError` when empty |
| **Circular Ring Queue** | `enqueue(x)` | $O(1)$ deterministic | $O(C)$ | `OverflowError` when full |
| **Circular Ring Queue** | `dequeue()` | $O(1)$ deterministic | $O(1)$ | `IndexError` when empty |
| **Circular Ring Queue** | `front()` | $O(1)$ deterministic | $O(1)$ | `IndexError` when empty |

---

## References and Further Reading

1. **Python Software Foundation:** *The Python Data Model & Special Method Names (Dunder Methods).* Available at: [https://docs.python.org/3/reference/datamodel.html](https://docs.python.org/3/reference/datamodel.html)
2. **Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022):** *Introduction to Algorithms (4th ed.).* MIT Press. Chapter 10: Elementary Data Structures (Stacks and Queues).
3. **Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994):** *Design Patterns: Elements of Reusable Object-Oriented Software.* Addison-Wesley. (Composite, Strategy, and Template Method patterns).
4. **Mnih, V., et al. (2015):** *Human-level control through deep reinforcement learning.* Nature, 518(7540), 529–533. (Replay buffer architecture).
5. **Paszke, A., et al. (2019):** *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019. (Object-oriented module subclassing).
