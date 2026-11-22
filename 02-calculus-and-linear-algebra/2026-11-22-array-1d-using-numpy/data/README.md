# Industrial IoT Sensor Readings Dataset

## Overview

This dataset contains high-frequency single-channel telemetry captured from an industrial robotic actuator test bench. It provides realistic 1D numerical sequences for practicing array ingestion, data cleaning, outlier clipping, conditional masking, and vectorized calibration using NumPy.

## File Inventory

- `sensor_readings_1d.csv`: Comma-separated telemetry measurements containing 60 sequential timestamp observations.

## Data Schema

| Column Name | Data Type | Units | Description | Expected Normal Range |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp_sec` | Integer | Seconds ($s$) | Elapsed recording time from test start | $[0, 59]$ |
| `temperature_celsius` | Float | Celsius ($^\circ\text{C}$) | Core actuator temperature | $[22.0, 28.0]$ |
| `voltage_mv` | Float | Millivolts ($\text{mV}$) | Bus supply voltage | $[3280.0, 3320.0]$ |
| `vibration_g` | Float | G-force ($g$) | Tri-axial composite acceleration | $[0.02, 0.08]$ |

## Anomaly Profiles

The dataset includes intentional real-world instrumentation anomalies for demonstration:
1. **Temperature transient spike** at $t = 14$ ($41.9^\circ\text{C}$) and sensor dropout at $t = 42$ ($10.8^\circ\text{C}$).
2. **Voltage power surge** at $t = 27$ ($3798.5\text{ mV}$).
3. **Mechanical shock resonance** at $t = 35$ ($1.24\text{ g}$).

## Ingestion with NumPy

```python
import numpy as np

# Load numerical columns skipping header
data = np.genfromtxt(
    "../data/sensor_readings_1d.csv",
    delimiter=",",
    skip_header=1,
)

timestamps = data[:, 0]
temperatures = data[:, 1]
voltages = data[:, 2]
vibrations = data[:, 3]

# Apply physical safety thresholding via np.clip
clamped_temp = np.clip(temperatures, a_min=20.0, a_max=30.0)

# Identify extreme voltage anomalies via np.where
surge_mask = np.where(voltages > 3400.0, 1, 0)
```
