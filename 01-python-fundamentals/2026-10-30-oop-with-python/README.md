# Object-Oriented Programming in Python

[Back to Module 01](../README.md) · [Back to the master roadmap](../../README.md)

**Session date:** 2026-10-30 · **Module:** 01 — Python Fundamentals & Algorithmic Thinking  
**Format:** Theoretical exposition, UML class modeling, executable notebooks, pointer data structures, and deep learning layer abstractions

---

## Overview

In the evolution of software engineering, programming paradigms dictate how computational problems are conceptualized, decomposed, and maintained. Early programming models relied on **Procedural Programming** (Procedure-Oriented Programming, or POP), organizing code into linear sequences of functional subroutines operating directly on passive, mutable data. While procedural programming is conceptually straightforward for small scripts, it degrades severely as systems grow in size and complexity: functions become tightly coupled to mutable global data structures, state changes cause non-local side effects, and extending existing features requires modifying established code paths.

To resolve these architectural limitations, the industry converged on **Object-Oriented Programming** (OOP). OOP reimagines software systems as collections of autonomous, interacting entities called **Objects**. Each object bundles **state** (data attributes) and **behavior** (methods) into a unified, encapsulated capsule. Rather than passing data through sequences of global functions, objects communicate by invoking methods on one another, hiding internal state representations behind well-defined public interfaces.

This session develops a textbook-grade, first-principles foundation in Object-Oriented Programming with Python. We explore the paradigm from core syntax and runtime mechanics to high-level architecture:

1. **Paradigms Compared:** We examine why procedural code fails at scale through a case study of a pizza preparation pipeline. We contrast top-down functional decomposition with bottom-up object modeling and formalize the distinction between functions and methods.
2. **Classes, Instances, and Attribute Mechanics:** We inspect Python's object model: the class as an in-memory blueprint, instance instantiation, the role of `__init__`, the namespace separation between class attributes (`Class.__dict__`) and instance attributes (`instance.__dict__`), attribute shadowing, and the `self` binding mechanism.
3. **Encapsulation & Access Modifiers:** We examine Python's philosophy of data hiding. We analyze public attributes, protected conventions (`_single_underscore`), and private attributes (`__double_underscore`) governed by compiler name mangling (`_ClassName__attribute`). We then transition to idiomatic encapsulation using Python's `@property` decorator to enforce type and range invariants without breaking attribute-style access.
4. **Object Relationships (UML Modeling):** We formalize the object relationship taxonomy:
   - **Association:** Peer-to-peer communication without lifecycle coupling (e.g., a `Mechanic` servicing a `Car`).
   - **Aggregation:** Weak containment ("has-a") where components maintain independent lifecycles and can be shared concurrently (e.g., a `Person` aggregating an independent `Date` of birth; a `Team` aggregating `Employee` references).
   - **Composition:** Strong containment ("part-of") where the composite whole owns the lifecycle of its components, triggering cascade destruction (e.g., a `Company` owning `Department` structures; an `Employee` owning an internal `CompensationProfile`).
5. **The Iteration Protocol & Custom Linked Structures:** We dissect Python's iteration contract (`__iter__` and `__next__`), desugar the `for` loop syntax into explicit while-loops with `StopIteration` handling, and build stateful custom iterator classes. We apply these principles to construct a **Singly Linked List** from first principles, comparing node-based pointer chaining with contiguous dynamic arrays.
6. **Object Sorting & Rich Comparisons:** We examine how custom domain objects are organized and ordered in Python collections, implementing key functions, `operator.attrgetter`, and natural dunder ordering (`__lt__`, `__eq__`, `@functools.total_ordering`).
7. **Applied AI Case Study:** We demonstrate how modern deep learning frameworks (such as PyTorch's `nn.Module`) are built upon object-oriented composition and callable instances (`__call__`). We construct a zero-dependency, modular neural network architecture from scratch using NumPy.

---

## Learning outcomes

By the end of this session, you should be able to:

- Articulate the failure modes of Procedural Programming (global namespace pollution, tight coupling, fragile side-effects) and explain how OOP addresses them.
- Define Python classes with explicit constructors (`__init__`) and understand the difference between object allocation (`__new__`) and initialization (`__init__`).
- Contrast class attributes with instance attributes, map attribute lookup order through `__dict__`, and avoid attribute shadowing bugs.
- Explain the role of the `self` parameter in instance method dispatch and demonstrate how Python desugars `obj.method(x)` into `Class.method(obj, x)`.
- Implement callable class instances using the `__call__` dunder method.
- Apply Python encapsulation conventions: public attributes, protected members (`_name`), and private members with name mangling (`__name`).
- Build robust, validated data models using the `@property` getter and setter decorators.
- Diagram, categorize, and implement the three primary object relationships: Association, Aggregation, and Composition.
- Prove lifecycle independence in Aggregation versus cascade lifecycle ownership in Composition.
- Explain the Python Iteration Protocol contract (`__iter__` returning an iterator; `__next__` returning elements and raising `StopIteration`).
- Desugar standard `for` loops into equivalent procedural while-loops using `iter()` and `next()`.
- Implement a reusable, multi-pass iterable container and contrast it with a one-shot consumable iterator.
- Implement a Singly Linked List from first principles with an encapsulated `Node` class, supporting $O(1)$ prepend, $O(n)$ append, search, deletion, and iterable traversal.
- Analyze the time and space complexity trade-offs between node-based linked lists and contiguous dynamic arrays (`list`).
- Sort collections of custom objects using lambda keys, `operator.attrgetter`, and rich comparison dunder methods (`__lt__`, `__eq__`, `@total_ordering`).
- Build a modular, object-oriented neural network layer hierarchy (`Module`, `Linear`, `ReLU`, `Sequential`) in pure Python and NumPy.

---

## Session map

| Notebook | Focus | Key Topics |
| :--- | :--- | :--- |
| [01 — Classes & Attributes](notebooks/01-classes-objects-and-attributes.ipynb) | Class Mechanics & Encapsulation | Procedural vs OOP, `__init__`, class vs instance attributes, `__dict__`, `self` dispatch, name mangling, `@property` getters/setters |
| [02 — Object Relationships](notebooks/02-object-relationships-aggregation-composition.ipynb) | Association, Aggregation & Composition | UML notation, lifecycle dependencies, `Person`/`Date` aggregation, `Company`/`Team`/`Employee` hierarchy, `organization_roster.json` parsing |
| [03 — Custom Iterators](notebooks/03-custom-iterators-and-iteration-protocol.ipynb) | Python Iteration Protocol | `__iter__` and `__next__` contract, `StopIteration`, `for` loop desugaring, stateful `ArithmeticProgression`, multi-pass iterables vs generators |
| [04 — Singly Linked List](notebooks/04-singly-linked-list-and-node-chaining.ipynb) | Node Chaining & Linked Structures | `Node` class, manual pointer chaining, `SinglyLinkedList` container, $O(1)$ prepend, $O(n)$ append/delete, Pythonic dunder integration |
| [05 — Sorting & AI Abstraction](notebooks/05-object-sorting-and-ai-model-abstraction.ipynb) | Object Sorting & Neural Network Architecture | Custom object sorting (`Square`, `Book`), `@total_ordering`, `books_catalog.json` parsing, NumPy `Module` hierarchy (`Linear`, `ReLU`, `Sequential`) |

---

## 1. The Paradigm Shift: Procedural vs Object-Oriented Programming

### 1.1 The Procedural Programming Model (POP)

In Procedural Programming (POP), computational logic is structured as a top-down hierarchy of functions operating on raw data structures:

```text
+------------------------------------------------------------------------+
|                    PROCEDURAL ARCHITECTURE (POP)                       |
+------------------------------------------------------------------------+
|                                                                        |
|                     +----------------------+                           |
|                     |     Main Program     |                           |
|                     +----------+-----------+                           |
|                                |                                       |
|             +------------------+------------------+                    |
|             v                                     v                    |
|      +---------------+                     +---------------+           |
|      | Subroutine A  |                     | Subroutine B  |           |
|      +-------+-------+                     +-------+-------+           |
|              |                                     |                   |
|              v                                     v                   |
|      +---------------+                     +---------------+           |
|      |  Function A1  |                     |  Function B1  |           |
|      +-------+-------+                     +-------+-------+           |
|              |                                     |                   |
|              +-----------------+-------------------+                   |
|                                |                                       |
|                                v                                       |
|                    [ Mutable Global State ]                            |
|                    - pizza_name: str                                   |
|                    - dough_type: str                                   |
|                    - toppings: list                                    |
|                    - bake_time: int                                    |
+------------------------------------------------------------------------+
```

#### The Pizza Preparation Failure Case

Consider a bakery order management system. Procedurally, preparing a New York Style Sauce & Cheese pizza involves sequential steps:
1. `prepare_ingredients()`: Select thin crust dough, add marinara sauce, add grated Reggiano cheese.
2. `bake()`: Bake for 25 minutes at 180°C.
3. `cut()`: Slice into diagonal triangles.
4. `box()`: Pack in a standard pizza box.

If the bakery now adds a Chicago Style Deep Dish pizza:
- Dough becomes extra-thick crust; sauce becomes crushed plum tomato; topping becomes shredded Mozzarella cheese; baking time increases to 35 minutes; slicing changes to square cuts.

In procedural code, accommodating both types requires either:
1. **Branching global state:** Introducing flags (`if style == "NY": ... elif style == "CHICAGO": ...`) inside every subroutine. As new pizza types are added, every single preparation function must be modified, violating the **Open/Closed Principle**.
2. **Duplicating functions:** Creating `prepare_ny()`, `prepare_chicago()`, `bake_ny()`, `bake_chicago()`, resulting in massive code duplication and maintenance friction.
3. **Fragile global state:** Any function can modify global variables at any time, making asynchronous or multi-threaded execution impossible without extensive race conditions.

### 1.2 The Object-Oriented Programming Model (OOP)

OOP replaces the top-down decomposition with a bottom-up model composed of autonomous, self-contained objects:

```text
+------------------------------------------------------------------------+
|                  OBJECT-ORIENTED ARCHITECTURE (OOP)                    |
+------------------------------------------------------------------------+
|                                                                        |
|    +-----------------------------+     +--------------------------+    |
|    |      Object: NY Pizza       |     |   Object: Chicago Pizza  |    |
|    +-----------------------------+     +--------------------------+    |
|    | State:                      |     | State:                   |    |
|    | - dough: "Thin Crust"       |     | - dough: "Thick Crust"   |    |
|    | - sauce: "Marinara"         |     | - sauce: "Plum Tomato"   |    |
|    | - toppings: ["Reggiano"]    |     | - toppings: ["Mozzarella"]|   |
|    | - bake_time: 25             |     | - bake_time: 35          |    |
|    | - cut_style: "Diagonal"     |     | - cut_style: "Square"    |    |
|    +-----------------------------+     +--------------------------+    |
|    | Behavior:                   |     | Behavior:                |    |
|    | + prepare()                 |     | + prepare()              |    |
|    | + bake()                    |     | + bake()                 |    |
|    | + cut()                     |     | + cut()                  |    |
|    | + box()                     |     | + box()                  |    |
|    +--------------+--------------+     +------------+-------------+    |
|                   |                                 |                  |
|                   +----------------+----------------+                  |
|                                    | (Message Passing)                 |
|                                    v                                   |
|                   +---------------------------------+                  |
|                   |       Order Dispatcher          |                  |
|                   +---------------------------------+                  |
+------------------------------------------------------------------------+
```

In OOP:
- Each pizza is an instance of a `Pizza` class.
- The state (dough, sauce, toppings, baking time, slicing style) is encapsulated inside the instance.
- Behaviors (`prepare()`, `bake()`, `cut()`, `box()`) are methods bound to the instance, operating directly on its private state.
- Adding new pizza types requires zero changes to existing order fulfillment routines.

### 1.3 Detailed Comparison: POP vs OOP

| Dimension | Procedural Programming (POP) | Object-Oriented Programming (OOP) |
| :--- | :--- | :--- |
| **Primary Unit** | Functions / Subroutines | Classes / Objects |
| **Approach** | Top-down functional decomposition | Bottom-up architectural modeling |
| **Data Handling** | Data moves freely between functions; frequently global | Data is encapsulated within objects; accessed via methods |
| **State Security** | Low; any function can mutate global state | High; internal state is guarded behind access boundaries |
| **Reusability** | Difficult; functions rely on external state schemas | High; classes can be instantiated, composed, or subclassed |
| **Extensibility** | Adding features requires editing existing routines | Adding features involves creating new classes or subclasses |
| **Communication** | Direct function calls with passed arguments | Method invocation and message passing between objects |
| **Primary Languages** | C, Pascal, Fortran, standard Bash | Python, C++, Java, Rust (traits), Swift |

### 1.4 Function vs Method: The Critical Distinction

In Python, the distinction between a **function** and a **method** is fundamental:

- A **Function** is an independent, callable block of code defined outside any class. It accepts arguments explicitly and has no implicit binding to an object instance:
  ```python
  def compute_distance(x1: float, y1: float, x2: float, y2: float) -> float:
      return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
```
- A **Method** is a function defined within a class body. It is bound to an instance of the class (or to the class itself in class methods), and receives that instance implicitly as its first parameter (`self`):
  ```python
  class Point:
      def __init__(self, x: float, y: float) -> None:
          self.x = x
          self.y = y

      def distance_to(self, other: "Point") -> float:
          return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
```

When you evaluate `p1.distance_to(p2)`, Python's descriptor protocol automatically intercepts the attribute access on the instance `p1`, binds the underlying function `Point.distance_to` to `p1`, and translates the call to:
$$\text{Point.distance\_to}(p1, p2)$$

---

## 2. Classes, Instances, and Attribute Mechanics

### 2.1 The Class as a Blueprint

A **Class** is an abstract blueprint or user-defined type specification. It dictates what data fields an object will hold and what operations can be performed on it. An **Object** is a concrete instance of that class allocated in memory.

```text
    +-----------------------------------------------+
    |                  CLASS: Dog                   |
    |  - species: "Canis familiaris" (Class Attr)   |
    |  - __init__(self, name, breed, age)           |
    |  - bark(self)                                 |
    +-----------------------------------------------+
                           |
            +--------------+--------------+
            | Instantiation               | Instantiation
            v                             v
+-----------------------+     +-----------------------+
|  Instance: dog1       |     |  Instance: dog2       |
|  - name: "Buddy"      |     |  - name: "Milo"       |
|  - breed: "Retriever" |     |  - breed: "Beagle"    |
|  - age: 3             |     |  - age: 2             |
|  - [ref -> Dog class] |     |  - [ref -> Dog class] |
+-----------------------+     +-----------------------+
```

### 2.2 Class Attributes vs Instance Attributes

Understanding the distinction between class and instance attributes is vital to avoiding subtle state pollution bugs in Python:

1. **Class Attributes:** Defined directly in the class block. They reside in `ClassName.__dict__` and are shared by all instances of the class. They are ideal for constants, configuration defaults, or cross-instance counters.
2. **Instance Attributes:** Bound dynamically to a specific instance via `self.attribute_name` (usually in `__init__`). They reside in `instance.__dict__` and are private to that specific object.

```python
class Dog:
    # Class attribute (shared)
    species: str = "Canis familiaris"
    total_dogs: int = 0

    def __init__(self, name: str) -> None:
        # Instance attribute (unique)
        self.name: str = name
        Dog.total_dogs += 1
```

#### Namespace Lookup and Shadowing

When accessing an attribute via `instance.attr`:
1. Python inspects `instance.__dict__`.
2. If absent, it searches `instance.__class__.__dict__`.
3. If absent, it traverses the Method Resolution Order (MRO) base classes.
4. If nowhere found, it calls `__getattr__` or raises `AttributeError`.

**The Shadowing Trap:**
```python
d1 = Dog("Buddy")
d2 = Dog("Milo")

d1.species = "Canis lupus"  # Creates instance attribute 'species' in d1.__dict__!
# d1.species -> "Canis lupus"
# d2.species -> "Canis familiaris" (still references Dog.species)
# Dog.species -> "Canis familiaris" (unchanged)
```

### 2.3 Object Construction: `__new__` vs `__init__`

Creating an object in Python is a two-phase process:
1. **`__new__(cls, *args, **kwargs)`:** The allocator. A static method that actually allocates raw memory for the new object and returns the freshly created instance.
2. **`__init__(self, *args, **kwargs)`:** The initializer. Called immediately after `__new__` has returned the instance, receiving that instance as `self` to initialize its attributes.

```text
    User calls: Point(3, 4)
              |
              v
    1. Point.__new__(Point, 3, 4)  ---> Allocates memory, returns new instance `inst`
              |
              v
    2. Point.__init__(inst, 3, 4)  ---> Initializes `inst.x = 3`, `inst.y = 4`
              |
              v
    Returns initialized `inst` to caller
```

### 2.4 The `self` Parameter

Unlike C++ or Java (which use an implicit `this` pointer), Python makes the instance receiver explicit in method signatures. The parameter is conventionally named `self`.

Why explicit `self`?
- **Clarity over Magic:** Python's design philosophy ("Explicit is better than implicit") ensures that reading a method makes it immediately clear whether an attribute belongs to the local function scope or the object instance.
- **Dynamic Dispatch:** Makes rebinding methods or dynamically attaching functions to classes uniform and predictable.

### 2.5 Callable Instances: The `__call__` Dunder

Python allows class instances to behave like functions by implementing the special dunder method `__call__(self, *args, **kwargs)`:

```python
class Polynomial:
    def __init__(self, a: float, b: float, c: float) -> None:
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x: float) -> float:
        # f(x) = a*x^2 + b*x + c
        return self.a * (x**2) + self.b * x + self.c


f = Polynomial(2.0, 3.0, 1.0)
print(f(5.0))  # Invokes f.__call__(5.0) -> 2*(25) + 3*(5) + 1 = 66.0
```

This pattern is the foundational building block for stateful function objects and neural network layers in modern AI libraries.

---

## 3. Encapsulation and Access Modifiers

### 3.1 The Three Visibility Levels in Python

**Encapsulation** bundles state and behavior while restricting unauthorized external access to internal representations. Unlike languages with rigid compiler keywords (`public`, `private`, `protected`), Python enforces access through naming conventions:

```text
+-------------------+---------------------+---------------------------------------------------+
| Visibility Level  | Naming Convention   | Accessibility & Enforcement                       |
+-------------------+---------------------+---------------------------------------------------+
| Public            | name                | Accessible from anywhere.                         |
| Protected         | _name               | Intended for internal & subclass use only.        |
|                   |                     | (Convention; not enforced by interpreter).        |
| Private           | __name              | Internal to defining class. Interpreter activates |
|                   |                     | Name Mangling: rewritten to _ClassName__name.     |
+-------------------+---------------------+---------------------------------------------------+
```

### 3.2 Name Mangling Mechanics

When an attribute has at least two leading underscores and at most one trailing underscore (e.g., `self.__balance`), the Python bytecode compiler automatically mangles the identifier:

$$\text{\_\_balance} \implies \text{\_ClassName\_\_balance}$$

```python
class Account:
    def __init__(self, initial_deposit: float) -> None:
        self.__balance = initial_deposit


acc = Account(500.0)
# acc.__balance -> Raises AttributeError
# acc._Account__balance -> Returns 500.0 (mangled name)
```

**Purpose of Name Mangling:**
Name mangling was not designed as an impenetrable security boundary against malicious callers ("We are all consenting adults here"). Rather, its primary purpose is to **prevent accidental name collisions** in inheritance hierarchies when a subclass defines an attribute with the same name as a private attribute in a superclass.

### 3.3 Pythonic Encapsulation with `@property`

In languages like C++ or Java, developers write boilerplate getters and setters (`getBalance()`, `setBalance(val)`). In Python, this pattern is considered an anti-pattern because it pollutes public interfaces.

Instead, Python provides the `@property` descriptor decorator, enabling clean attribute-access syntax while running validation logic behind the scenes:

```python
class Temperature:
    def __init__(self, celsius: float) -> None:
        self.celsius = celsius  # Routes through setter validation

    @property
    def celsius(self) -> float:
        """The celsius property getter."""
        return self._celsius

    @celsius.setter
    def celsius(self, value: float) -> None:
        """The celsius property setter with physical invariant validation."""
        if value < -273.15:
            raise ValueError(f"Temperature {value}°C is below absolute zero (-273.15°C)!")
        self._celsius = float(value)

    @property
    def fahrenheit(self) -> float:
        """Computed property: derived on the fly without state synchronization."""
        return (self._celsius * 9.0 / 5.0) + 32.0
```

---

## 4. Object Relationships: Association, Aggregation, and Composition

### 4.1 UML Primer: The Unified Modeling Language

In software architecture, UML Class Diagrams visualize class definitions and relationships using standardized graphical notations:

```text
+------------------------------------+
|             ClassName              |
+------------------------------------+
| - private_attribute: type          |
| # protected_attribute: type        |
| + public_attribute: type           |
+------------------------------------+
| + public_method(arg: type): return |
| - private_method(): void           |
+------------------------------------+
```

The relationships between classes are classified into four primary categories:

```text
General Relationship Hierarchy:

               +-----------------------+
               |      Association      |  (Plain line: "uses-a")
               +-----------+-----------+
                           |
                           v
               +-----------------------+
               |      Aggregation      |  (Open diamond: "has-a", shared)
               +-----------+-----------+
                           |
                           v
               +-----------------------+
               |      Composition      |  (Filled diamond: "part-of", owned)
               +-----------------------+
```

### 4.2 Association ("Uses-A")

An **Association** is a relationship where two objects communicate and collaborate, but neither object owns or manages the lifecycle of the other. Both objects exist independently.

- **Example:** A `Mechanic` services a `Car`. The mechanic can service multiple cars; a car can be serviced by multiple mechanics over its lifetime. If the car is decommissioned and deleted, the mechanic continues working at the garage.

```text
+--------------+                 services                  +--------------+
|   Mechanic   | ----------------------------------------> |     Car      |
+--------------+                                           +--------------+
| + name       |                                           | + vin        |
| + phone      |                                           | + model      |
+--------------+                                           +--------------+
```

### 4.3 Aggregation ("Has-A", Shared Lifecycle)

**Aggregation** is a specialized form of association representing a whole-part container relationship with **weak ownership**:
- The component part can exist independently of the container whole.
- The component part is typically instantiated outside the container and passed into it as a reference.
- Deleting the container does **not** destroy the component part.
- The component part can be shared simultaneously across multiple containers.

- **Example 1: Person and Date of Birth:** A `Person` has a `Date` of birth. The `Date` object (representing day, month, year) is an independent calendar concept. Two different people can share the same birth date instance.
- **Example 2: Team and Employee:** A `Team` aggregates `Employee` members. If a team is dissolved, the employees remain employed at the company. An employee can belong simultaneously to multiple cross-functional teams (e.g., Sarah Connor in both Core Algorithms and Computer Vision).

```text
+--------------+                                           +--------------+
|     Team     | <>--------------------------------------> |   Employee   |
+--------------+               aggregates                  +--------------+
| + team_name  |                                           | + emp_id     |
| + members    |                                           | + name       |
+--------------+                                           +--------------+
```

### 4.4 Composition ("Part-Of", Strict Lifecycle Ownership)

**Composition** is a strong form of aggregation representing an intimate whole-part relationship with **exclusive lifecycle ownership**:
- The component part is fundamentally a part of the whole.
- The component part cannot exist without the container whole.
- The component part is created by the container whole (typically within `__init__`).
- Deleting the container whole triggers **cascade deletion** of all composed parts.

- **Example 1: Company and Department:** A `Company` composes `Department` structures. If the legal entity `Company` is liquidated, its departments cease to exist.
- **Example 2: Employee and CompensationProfile:** An `Employee` owns their internal `Salary` / `CompensationProfile`. The compensation profile has no independent identity outside the employee; deleting the employee deletes their salary records.

```text
+--------------+                                           +-----------------------+
|   Employee   | *---------------------------------------> |  CompensationProfile  |
+--------------+                 composes                  +-----------------------+
| + emp_id     |                                           | - base_monthly: float |
| - salary     |                                           | - annual_bonus: float |
+--------------+                                           +-----------------------+
```

### 4.5 Relationship Decision Framework

When designing object-oriented domain models, apply the following diagnostic decision tree:

```text
                      Do entities communicate?
                                 |
                                 v Yes
                  Is there a whole-to-part relationship?
                     /                           \
               No   /                             \ Yes
                   v                               v
           [ Association ]             Can the part exist if the
        (Mechanic and Car)             whole is destroyed?
                                          /          \
                                    Yes  /            \ No
                                        v              v
                                 [ Aggregation ]  [ Composition ]
                                 (Team/Employee)  (Company/Dept)
```

---

## 5. The Iteration Protocol and Custom Data Structures

### 5.1 The Iteration Protocol Contract

Iteration in Python is decoupled from specific concrete collections through the **Iteration Protocol**. The protocol consists of two distinct roles:

1. **Iterable (`__iter__`):** Any object capable of returning its elements one at a time. It implements `__iter__()` which must return an **Iterator** object.
2. **Iterator (`__next__` and `__iter__`):** A stateful stream cursor. It implements:
   - `__next__()`: Returns the next element from the sequence. When elements are exhausted, it raises the built-in `StopIteration` exception.
   - `__iter__()`: Returns `self`, allowing an iterator to be used anywhere an iterable is accepted.

### 5.2 Desugaring the `for` Loop

When Python compiles and executes a standard loop:
```python
for item in iterable:
    process(item)
```

The virtual machine translates this into the following explicit state-machine sequence:
```python
_iterator = iter(iterable)  # 1. Calls iterable.__iter__()
while True:
    try:
        item = next(_iterator)  # 2. Calls _iterator.__next__()
    except StopIteration:  # 3. Catches boundary signal
        break  # 4. Exits cleanly
    process(item)
```

### 5.3 Building a Custom Stateful Iterator

Let us construct an `ArithmeticProgression` iterator generating numbers $a_k = a_0 + k \cdot d$ up to $N$ steps:

```python
class ArithmeticProgression:
    """Stateful iterator generating an arithmetic sequence: start, start + step, ..."""

    def __init__(self, start: float, step: float, max_terms: int) -> None:
        if max_terms < 0:
            raise ValueError("max_terms must be non-negative.")
        self.start = float(start)
        self.step = float(step)
        self.max_terms = int(max_terms)
        self._current_term = 0
        self._current_value = self.start

    def __iter__(self) -> "ArithmeticProgression":
        return self

    def __next__(self) -> float:
        if self._current_term >= self.max_terms:
            raise StopIteration

        val = self._current_value
        self._current_value += self.step
        self._current_term += 1
        return val
```

### 5.4 Singly Linked List: OOP Pointer Chaining

Python's built-in `list` is a contiguous dynamic array. While offering $O(1)$ indexed access, inserting elements at the head (`insert(0, val)`) requires shifting all $n$ existing elements, incurring an $O(n)$ time penalty.

A **Singly Linked List** solves this by storing items in discrete `Node` objects chained by pointer references:

```text
Head
  |
  v
+-------------+      +-------------+      +-------------+
| Data: 10    | ---> | Data: 20    | ---> | Data: 30    | ---> None
| Next: [ref] |      | Next: [ref] |      | Next: None  |
+-------------+      +-------------+      +-------------+
```

#### Node and Container Implementation

```python
from typing import Any, Optional


class Node:
    def __init__(self, data: Any = None) -> None:
        self.data: Any = data
        self.next: Optional["Node"] = None


class SinglyLinkedList:
    def __init__(self) -> None:
        self.head: Optional[Node] = None
        self._size: int = 0

    def prepend(self, data: Any) -> None:
        """Insert at head in O(1) time."""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
        self._size += 1

    def append(self, data: Any) -> None:
        """Insert at tail in O(n) time."""
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
        else:
            curr = self.head
            while curr.next is not None:
                curr = curr.next
            curr.next = new_node
        self._size += 1

    def __iter__(self):
        """Conform to the Python iteration protocol."""
        curr = self.head
        while curr is not None:
            yield curr.data
            curr = curr.next

    def __len__(self) -> int:
        return self._size
```

#### Complexity Comparison: Linked List vs Dynamic Array

| Operation | Singly Linked List | Dynamic Array (`list`) | Architectural Cause |
| :--- | :---: | :---: | :--- |
| **Prepend (`insert(0)`)** | $O(1)$ | $O(n)$ | Linked list updates 1 pointer; array shifts $n$ items |
| **Append (`insert(end)`)** | $O(n)$ ($O(1)$ with tail ref) | $O(1)$ amortized | Array reserves spare capacity at buffer tail |
| **Index Lookup (`list[k]`)** | $O(k)$ | $O(1)$ | Array computes RAM memory address directly via offset calculation |
| **Delete arbitrary key** | $O(n)$ search + $O(1)$ unlink | $O(n)$ search + $O(n)$ shift | Linked list unlinks node; array shifts elements left |
| **Cache Locality** | Poor (scattered heap nodes) | Excellent (contiguous memory buffer) | CPU hardware prefetcher accelerates contiguous memory |

---

## 6. Collections, Custom Sorting, and Rich Comparisons

When storing domain objects in lists, invoking `list.sort()` or `sorted()` requires defining comparison semantics. Python supports three approaches:

### Approach A: Lambda Key Functions
```python
# Sort squares by side length
squares.sort(key=lambda s: s.side)
```

### Approach B: `operator.attrgetter`
```python
import operator

# Faster, C-accelerated attribute extraction
sorted_books = sorted(books, key=operator.attrgetter("price"), reverse=True)
```

### Approach C: Natural Ordering with Rich Comparison Dunders
To make objects naturally comparable using `<`, `<=`, `>`, `>=`, `==`, implement `__lt__` and `__eq__`, decorated with `@functools.total_ordering`:

```python
from functools import total_ordering
from typing import Any


@total_ordering
class Book:
    def __init__(self, title: str, price: float, rating: float) -> None:
        self.title = title
        self.price = float(price)
        self.rating = float(rating)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Book):
            return NotImplemented
        return (self.rating, -self.price) == (other.rating, -other.price)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Book):
            return NotImplemented
        # Primary sort: higher rating first; Secondary: lower price first
        return (-self.rating, self.price) < (-other.rating, other.price)
```

---

## 7. OOP in Modern Machine Learning Architectures

In modern deep learning libraries (such as PyTorch, Keras, and Flax), neural networks are structured as hierarchical, object-oriented systems:

```text
+------------------------------------------------------------------------+
|                      NEURAL NETWORK OOP HIERARCHY                      |
+------------------------------------------------------------------------+
|                                                                        |
|                       +----------------------+                         |
|                       |   Abstract: Module   |                         |
|                       +----------+-----------+                         |
|                                  |                                     |
|          +-----------------------+-----------------------+             |
|          | Inherits                                      | Inherits    |
|          v                                               v             |
|  +--------------------+                          +-------------------+ |
|  |    Linear Layer    |                          |    ReLU Layer     | |
|  +--------------------+                          +-------------------+ |
|  | - weights: ndarray |                          | + forward(x)      | |
|  | - biases: ndarray  |                          +-------------------+ |
|  | + forward(x)       |                                                |
|  +--------------------+                                                |
|          ^                                                             |
|          | Composed into                                               |
|          v                                                             |
|  +-------------------------------------------------------------------+ |
|  |                   Composite Container: Sequential                 | |
|  +-------------------------------------------------------------------+ |
|  | - _submodules: [Linear(in, hid), ReLU(), Linear(hid, out)]        | |
|  | + forward(x): chains output through all composed layers          | |
|  | + __call__(x): invokes forward() directly                         | |
|  +-------------------------------------------------------------------+ |
+------------------------------------------------------------------------+
```

### Pure Python/NumPy Implementation

```python
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np


class Module(ABC):
    """Abstract base class for all neural network layers."""

    def __init__(self) -> None:
        self._parameters: Dict[str, np.ndarray] = {}
        self._submodules: Dict[str, "Module"] = {}

    def register_parameter(self, name: str, param: np.ndarray) -> None:
        self._parameters[name] = param

    @property
    def parameters(self) -> List[np.ndarray]:
        params = list(self._parameters.values())
        for sub in self._submodules.values():
            params.extend(sub.parameters)
        return params

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Linear(Module):
    """Dense matrix transformation: y = x @ W + b."""

    def __init__(self, in_features: int, out_features: int, seed: int = 42) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (in_features + out_features))
        weights = rng.uniform(-limit, limit, size=(in_features, out_features))
        biases = np.zeros((1, out_features))
        self.register_parameter("weights", weights)
        self.register_parameter("biases", biases)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self._parameters["weights"]) + self._parameters["biases"]


class ReLU(Module):
    """Rectified linear activation."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)


class Sequential(Module):
    """Container executing composed submodules in forward order."""

    def __init__(self, *layers: Module) -> None:
        super().__init__()
        for idx, layer in enumerate(layers):
            self._submodules[f"layer_{idx}"] = layer

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self._submodules.values():
            out = layer(out)
        return out
```

This elegant design demonstrates how OOP principles—encapsulation of weights, behavioral polymorphism, composition of layers, and callable dunder interfaces—directly power the architectures of modern artificial intelligence.

---

## Summary & Practice Checklist

- [x] Contrasted Procedural Programming (POP) and Object-Oriented Programming (OOP).
- [x] Implemented classes, constructors (`__init__`), and examined instance versus class attributes.
- [x] Analyzed method binding, `self` dispatch, and callable instances via `__call__`.
- [x] Applied encapsulation conventions: public, protected (`_`), private (`__`), and `@property` descriptors.
- [x] Implemented and contrasted Association, Aggregation, and Composition with UML diagrams.
- [x] Dissected the Iteration Protocol (`__iter__`, `__next__`, `StopIteration`) and desugared `for` loops.
- [x] Constructed a Singly Linked List from first principles with full dunder protocol support.
- [x] Implemented rich comparison sorting (`__lt__`, `__eq__`, `@total_ordering`) on domain records.
- [x] Built a modular deep learning layer hierarchy in pure Python and NumPy.

---

## References & Further Reading

- Python Documentation: *Classes and the Python Data Model*, <https://docs.python.org/3/reference/datamodel.html>
- Gamma, Erich, Richard Helm, Ralph Johnson, and John Vlissides. *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley, 1994.
- Ramalho, Luciano. *Fluent Python: Clear, Concise, and Effective Programming*. 2nd Edition, O'Reilly Media, 2022.
- Martin, Robert C. *Clean Code: A Handbook of Agile Software Craftsmanship*. Prentice Hall, 2008.
- Cormen, Thomas H., Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. *Introduction to Algorithms*. 3rd Edition, MIT Press, 2009.
