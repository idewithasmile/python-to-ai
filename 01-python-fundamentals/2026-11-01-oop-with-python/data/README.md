# Object-Oriented Modeling Datasets: Geometric Shapes, Academic Staff, and Audio DSP Pipeline

This directory contains structured datasets used in object-oriented programming labs for testing inheritance hierarchies, abstract base classes, access modifier boundaries, polymorphic operator overloading, and modular package architecture.

## Overview of Datasets

| File | Format | Records | Description | Primary Use Case |
| :--- | :---: | :---: | :--- | :--- |
| `shapes_collection.json` | JSON | 8 items | Heterogeneous geometric shapes with parametric dimensions and material properties | Abstract Base Classes (`Shape`), method overriding (`area`, `perimeter`), mass computation, and operator overloading |
| `staff_hierarchy.json` | JSON | 5 staff records | Academic research institute hierarchy across multiple roles | Class inheritance (`Person` -> `Staff` -> `Researcher`), protected/private attribute encapsulation, and polymorphic compensation calculations |
| `audio_pipeline_config.json` | JSON | 4 DSP modules | Digital signal processing filter and effect module pipeline configuration | Simulating modular Python package architectures (`audiokit`), dynamic module importing, and uniform polymorphic invocation |

---

## 1. `shapes_collection.json`

### Purpose and Provenance

`shapes_collection.json` provides parametric definitions for various 2D geometric shapes (circles, rectangles, squares, triangles) alongside material density properties. It is designed for:
1. Instantiating concrete subclasses inheriting from an abstract `Shape` base class.
2. Verifying polymorphic calculations of geometric area and perimeter.
3. Overloading arithmetic operators (`__add__`, `__mul__`) to combine shapes and scale geometric structures.
4. Sorting heterogeneous shape collections by area, perimeter, or total mass.

### Schema Definition

| Field | Type | Nullable | Description |
| :--- | :---: | :---: | :--- |
| `id` | `string` | No | Unique alphanumeric shape identifier (e.g., `"SHP-001"`) |
| `type` | `string` | No | Geometric category: `"circle"`, `"rectangle"`, `"square"`, or `"triangle"` |
| `parameters` | `object` | No | Dimensional parameters specific to the geometric type |
| `parameters.radius` | `float` | Yes | Radius for circle shapes ($r > 0$) |
| `parameters.width` | `float` | Yes | Width for rectangle shapes ($w > 0$) |
| `parameters.height` | `float` | Yes | Height for rectangle or triangle shapes ($h > 0$) |
| `parameters.side` | `float` | Yes | Side length for square shapes ($s > 0$) |
| `parameters.base` | `float` | Yes | Base length for triangle shapes ($b > 0$) |
| `parameters.side_a` | `float` | Yes | Length of side $a$ for triangle perimeter calculation |
| `parameters.side_b` | `float` | Yes | Length of side $b$ for triangle perimeter calculation |
| `parameters.side_c` | `float` | Yes | Length of side $c$ for triangle perimeter calculation |
| `color` | `string` | No | Color name descriptor |
| `material_density_g_cm2` | `float` | No | Surface area density in grams per square centimeter |

### Loading Example

```python
import json
from pathlib import Path

shapes_path = Path("data/shapes_collection.json")
with open(shapes_path, mode="r", encoding="utf-8") as f:
    shapes = json.load(f)

print(f"Loaded {len(shapes)} shape specifications.")
```

---

## 2. `staff_hierarchy.json`

### Purpose and Provenance

`staff_hierarchy.json` models academic and research personnel across faculty, postdoctoral scholars, graduate researchers, and assistants. It is designed for:
1. Implementing multi-tiered inheritance structures (`Person` -> `Staff` -> `Faculty` / `PostdoctoralResearcher` / `GraduateResearcher`).
2. Practicing access modifier conventions: public contact details, protected base salary (`_base_salary`), and private tax/payroll credentials (`__tax_id`).
3. Demonstrating method overriding: specialized `compute_compensation()` and `perform_duties()` methods across academic ranks.

### Schema Definition

| Field Path | Type | Nullable | Description |
| :--- | :---: | :---: | :--- |
| `institution` | `string` | No | Name of the parent academic institution |
| `founded_year` | `integer` | No | Year of establishment |
| `departments[]` | `list[string]` | No | Academic departments |
| `staff_records[].id` | `string` | No | Staff identification badge code |
| `staff_records[].name` | `string` | No | Full legal name of the staff member |
| `staff_records[].role` | `string` | No | Role designation (`"Faculty"`, `"PostdoctoralResearcher"`, etc.) |
| `staff_records[].department` | `string` | No | Assigned academic department |
| `staff_records[].base_salary` | `float` | No | Monthly base salary in USD ($> 0.0$) |
| `staff_records[].bonus_rate` | `float` | No | Annual performance bonus multiplier ($0.0 \le r \le 1.0$) |
| `staff_records[].research_focus` | `string` | No | Primary scientific research topic |
| `staff_records[].publications` | `integer` | No | Total peer-reviewed publications count |
| `staff_records[].teaching_load_hours` | `integer` | No | Weekly classroom and lab instruction hours |

### Loading Example

```python
import json
from pathlib import Path

staff_path = Path("data/staff_hierarchy.json")
with open(staff_path, mode="r", encoding="utf-8") as f:
    data = json.load(f)

records = data["staff_records"]
print(f"Loaded {len(records)} staff records from {data['institution']}.")
```

---

## 3. `audio_pipeline_config.json`

### Purpose and Provenance

`audio_pipeline_config.json` specifies a digital signal processing pipeline with chainable filter and effect modules. It is designed for:
1. Demonstrating modular Python package architecture (`audiokit.formats`, `audiokit.filters`, `audiokit.effects`).
2. Illustrating the Python iteration and execution protocol across dynamically discovered DSP units.
3. Practicing duck typing and polymorphism: any audio module implementing `process(audio_signal)` can be seamlessly placed in the pipeline.

### Schema Definition

| Field Path | Type | Nullable | Description |
| :--- | :---: | :---: | :--- |
| `pipeline_title` | `string` | No | Name of the audio processing pipeline |
| `version` | `string` | No | Semantic version of the configuration |
| `supported_extensions[]` | `list[string]` | No | Supported audio file formats (`.wav`, `.aiff`, etc.) |
| `global_audio_specs.sample_rate_hz` | `integer` | No | Audio sampling rate in Hertz (e.g., $44100$) |
| `global_audio_specs.bit_depth` | `integer` | No | Bit quantization depth (e.g., $24$) |
| `global_audio_specs.channels` | `integer` | No | Number of audio channels ($1$ for mono, $2$ for stereo) |
| `dsp_modules[].module_name` | `string` | No | Dotted module path (e.g., `"audiokit.filters.lowpass"`) |
| `dsp_modules[].class_name` | `string` | No | Target DSP processing class name |
| `dsp_modules[].parameters` | `object` | No | Module-specific hyperparameters |
| `dsp_modules[].enabled` | `boolean` | No | Flag indicating whether this processing stage is active |

### Loading Example

```python
import json
from pathlib import Path

config_path = Path("data/audio_pipeline_config.json")
with open(config_path, mode="r", encoding="utf-8") as f:
    config = json.load(f)

active_modules = [m for m in config["dsp_modules"] if m["enabled"]]
print(f"Pipeline: {config['pipeline_title']} | Active DSP Stages: {len(active_modules)}")
```
