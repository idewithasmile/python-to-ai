# Validation Datasets & Benchmarks

This directory contains curated tabular datasets designed to benchmark, audit, and demonstrate validation strategies, split topologies, and data leakage diagnostics.

---

## Dataset Catalog

| File | Modality | Samples | Dimensions | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `tabular_housing_validation.csv` | Tabular Continuous | 1,000 | 9 columns | Bias-variance trade-off, polynomial regression, K-Fold CV, Out-Of-Fold predictions |
| `customer_churn_stratified.csv` | Tabular Categorical | 1,200 | 8 columns | Stratified K-Fold vs. naive K-Fold on class-imbalanced targets (~18% positive) |
| `clinical_patient_records.csv` | Grouped Longitudinal | 780+ | 7 columns | GroupKFold cross-validation to prevent intra-subject data leakage |
| `sensor_telemetry_timeseries.csv` | Temporal Sequential | 1,000 | 5 columns | TimeSeriesSplit (walk-forward validation) preventing temporal lookahead bias |
| `covariate_shift_benchmark.csv` | Domain Adaptation | 1,000 | 4 columns | Adversarial validation to detect train-test covariate distribution shift |

---

## Detailed Schema Documentation

### 1. `tabular_housing_validation.csv`

Used for evaluating continuous response estimators, model capacity curves, and quantifying empirical bias-variance decomposition.

- `med_income` (`float`): Median income in block group (in tens of thousands USD). Range: $[0.50, 15.00]$.
- `house_age` (`float`): Median house age in years. Range: $[1.0, 52.0]$.
- `ave_rooms` (`float`): Average number of rooms per household.
- `ave_bedrooms` (`float`): Average number of bedrooms per household.
- `population` (`int`): Block group population.
- `ave_occupancy` (`float`): Average number of household members.
- `latitude` (`float`): Decimal degrees latitude (Northern hemisphere).
- `longitude` (`float`): Decimal degrees longitude (Western hemisphere).
- `target_price` (`float`): Median house value in hundreds of thousands USD (continuous target variable).

### 2. `customer_churn_stratified.csv`

Designed for binary classification validation under class imbalance, auditing how fold distributions shift under random partitioning versus stratified partitioning.

- `credit_score` (`int`): Customer credit rating score ($350 - 850$).
- `age` (`int`): Customer age in years.
- `tenure` (`int`): Years of active account tenure ($0 - 10$).
- `balance` (`float`): Account savings balance in USD.
- `num_products` (`int`): Number of financial products purchased ($1 - 4$).
- `is_active_member` (`int`): Binary indicator ($1$ for active engagement, $0$ otherwise).
- `estimated_salary` (`float`): Estimated annual gross earnings in USD.
- `churn` (`int`): Target label ($1$ if churned within 90 days, $0$ otherwise; class prior $\approx 18\%$).

### 3. `clinical_patient_records.csv`

Longitudinal records across 200 distinct patients (`PAT_0001` through `PAT_0200`). Each patient has multiple sequential hospital visits.

- `patient_id` (`string`): Unique identifier for patient cluster grouping.
- `visit_number` (`int`): Sequential index of hospital checkup ($1 - 5$).
- `systolic_bp` (`float`): Systolic blood pressure in mmHg.
- `glucose` (`float`): Fasting plasma glucose level in mg/dL.
- `bmi` (`float`): Body Mass Index ($\text{kg}/\text{m}^2$).
- `cholesterol` (`float`): Total serum cholesterol in mg/dL.
- `high_risk_flag` (`int`): Binary clinical indicator ($1$ for high cardiovascular risk, $0$ for baseline).

> [!IMPORTANT]
> When partitioning this dataset, splitting across observations rather than `patient_id` produces severe data leakage because repeated measurements from identical physiological baselines enter both the training and validation splits simultaneously. `GroupKFold` must be utilized.

### 4. `sensor_telemetry_timeseries.csv`

Time series telemetry capturing sensor readings sampled sequentially across 1,000 hourly timestamps.

- `timestamp` (`string`): ISO-formatted date-time stamp (`YYYY-MM-DD HH:MM:SS`).
- `ambient_temp_c` (`float`): Ambient machinery operating temperature in degrees Celsius.
- `vibration_rms` (`float`): Root-mean-square vibration acceleration in Gs.
- `power_draw_kw` (`float`): Instantaneous electrical power consumption in kilowatts.
- `load_pct` (`float`): Engine operating capacity percentage ($40.0\% - 95.0\%$).

> [!NOTE]
> Standard K-Fold violates temporal causality by training on future timestamps to predict past timestamps. A walk-forward rolling window (`TimeSeriesSplit`) ensures that for any validation fold index $k$, training data is strictly bounded by $t < t_{\text{val}}$.

### 5. `covariate_shift_benchmark.csv`

Synthetic benchmark containing 500 training instances and 500 testing instances where feature distributions differ systematically ($P_{\text{train}}(X) \neq P_{\text{test}}(X)$) while the decision boundary remains consistent.

- `feature_1` (`float`): Continuous predictor 1.
- `feature_2` (`float`): Continuous predictor 2.
- `target` (`int`): Binary ground truth outcome label.
- `split_origin` (`string`): Ground truth partition assignment (`train` vs. `test`). Used in Adversarial Validation to evaluate if a classifier can distinguish the partition origin.

