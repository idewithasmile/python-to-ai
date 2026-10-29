# Practice Lab Datasets: Text Corpora and Sensor Streams

This directory contains lightweight sample datasets used in the practice lab exercises for text processing, character/word frequency analysis, sliding window filtering, and moving average calculation.

## Overview of Datasets

| File | Format | Records | Description | Primary Use Case |
| :--- | :---: | :---: | :--- | :--- |
| `sample_text.txt` | Plain Text (UTF-8) | 6 sentences | Structured computer science prose discussing Python, algorithms, and clean code | Word tokenization, text normalization, and vocabulary frequency counting |
| `sensor_stream.csv` | Comma-Separated Values | 50 rows | Simulated environmental telemetry stream sampled at 2 Hz (0.5-second intervals) | 1D kernel maximum sliding windows, boundary-aware filtering, and prefix sum moving averages |

---

## 1. `sample_text.txt`

### Purpose and Provenance

`sample_text.txt` serves as an offline, deterministic benchmark corpus for string processing algorithms. It features mixed casing, varied sentence lengths, multi-syllable terms, and standard punctuation (commas, periods).

### Content Summary

- **Encoding:** UTF-8
- **Total lines:** 6 lines
- **Total words:** ~75 words
- **Unique vocabulary:** ~50 words

### Loading Example (Standard Library)

```python
from pathlib import Path

file_path = Path("data/sample_text.txt")
text = file_path.read_text(encoding="utf-8")
print(f"Total characters: {len(text)}")
```

---

## 2. `sensor_stream.csv`

### Purpose and Provenance

`sensor_stream.csv` represents a discrete 1D time series generated from a continuous environmental monitor. It models diurnal cyclic fluctuations and high-frequency noise, making it suitable for testing:
1. Local peak detection via sliding window kernel maxima.
2. Signal smoothing via Simple Moving Averages (SMA).
3. Constant-time $O(1)$ interval range queries using prefix sums (integral arrays).

### Schema Definition

| Column | Type | Unit | Nullable | Description |
| :--- | :---: | :---: | :---: | :--- |
| `step` | `int` | discrete step index | No | Non-negative sequential counter ($0, 1, 2, \dots, 49$) |
| `timestamp_sec` | `float` | seconds | No | Elapsed time from acquisition start, incrementing by $0.5\,\text{s}$ |
| `temperature_celsius` | `float` | $^{\circ}\text{C}$ | No | Ambient temperature measurement (range: $21.8 - 25.1^{\circ}\text{C}$) |
| `pressure_kpa` | `float` | $\text{kPa}$ | No | Barometric pressure measurement (range: $101.0 - 102.2\,\text{kPa}$) |

### Sample Records

```csv
step,timestamp_sec,temperature_celsius,pressure_kpa
0,0.0,22.1,101.3
1,0.5,22.4,101.2
2,1.0,22.8,101.4
3,1.5,23.5,101.5
4,2.0,24.2,101.8
```

### Loading Example (Standard Library `csv`)

```python
import csv
from pathlib import Path

csv_path = Path("data/sensor_stream.csv")
records = []
with open(csv_path, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        records.append(
            {
                "step": int(row["step"]),
                "timestamp_sec": float(row["timestamp_sec"]),
                "temperature": float(row["temperature_celsius"]),
                "pressure": float(row["pressure_kpa"]),
            }
        )

print(f"Loaded {len(records)} telemetry observations.")
```
