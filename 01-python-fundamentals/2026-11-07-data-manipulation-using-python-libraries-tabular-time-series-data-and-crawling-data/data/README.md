# Datasets & Benchmarks for Data Manipulation and Crawling

This directory contains curated datasets supporting tabular data manipulation, time-series feature engineering, NLP vocabulary indexing, and web crawling extraction pipelines.

---

## Dataset Catalog

| File | Modality | Samples / Size | Dimensions / Format | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `advertising_benchmark.csv` | Tabular Continuous | 200 rows | 5 columns (`id`, `tv`, `radio`, `newspaper`, `sales`) | First-principles tabular calculations, matrix representations, list-of-lists conversions, budget contribution modeling |
| `server_telemetry_timeseries.csv` | Temporal Continuous | 720 rows (30 days) | 6 columns (`timestamp`, `cpu`, `memory`, `requests`, `errors`, `latency`) | Temporal indexing, hourly-to-daily resampling, rolling statistics, lagged features, causal alignment |
| `sample_articles_corpus.json` | Semi-Structured Text | 5 articles | JSON records (`id`, `url`, `title`, `abstract`, `body`, `author`, etc.) | Text normalization, punctuation stripping, vocabulary compilation, index-based tokenization |
| `crawled_news_archive.zip` | Multimodal Archive | 10 files | `.zip` package with `corpus/*.txt` and `thumbnails/*.png` | Simulated crawled corpus archiving, multimodal asset verification, and batch extraction |

---

## Detailed Schema Documentation

### 1. `advertising_benchmark.csv`

The benchmark marketing mix dataset tracking advertising channel expenditures and corresponding sales revenue across 200 distinct regional consumer markets.

- `id` (`int`): Sequential observation identifier ($1$ to $200$).
- `tv` (`float`): Advertising budget invested in television promotions (in thousands of dollars, range: $[0.70, 296.40]$).
- `radio` (`float`): Advertising budget invested in radio broadcast commercials (in thousands of dollars, range: $[0.00, 49.60]$).
- `newspaper` (`float`): Advertising budget invested in print newspaper ads (in thousands of dollars, range: $[0.30, 114.00]$).
- `sales` (`float`): Sales volume generated across the regional market (in thousands of product units, range: $[1.60, 27.00]$).

#### Summary Statistics (Verified Exact)

| Feature | Sum | Slice Sum (Rows 2–11) | Min | Max | Mean | Median | Budget Share |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tv` | $29,408.50$ | $854.90$ | $0.70$ | $296.40$ | $147.04$ | $149.75$ | $73.206\%$ |
| `radio` | $4,652.80$ | $249.10$ | $0.00$ | $49.60$ | $23.26$ | $22.90$ | $11.582\%$ |
| `newspaper` | $6,110.80$ | $387.80$ | $0.30$ | $114.00$ | $30.55$ | $25.75$ | $15.212\%$ |
| `sales` | $2,804.50$ | $107.30$ | $1.60$ | $27.00$ | $14.02$ | $12.90$ | N/A |

---

### 2. `server_telemetry_timeseries.csv`

High-resolution telemetry recorded hourly from an enterprise cluster over a 30-day monitoring window ($720$ consecutive hourly steps from `2026-10-01T00:00:00Z` to `2026-10-30T23:00:00Z`).

- `timestamp` (`string` / `ISO-8601`): UTC timestamp formatted as `YYYY-MM-DDTHH:MM:SSZ`.
- `cpu_utilization_pct` (`float`): Average CPU utilization across compute nodes ($5.0\% - 98.5\%$), displaying diurnal load cycles.
- `memory_utilization_pct` (`float`): System RAM usage percentage ($20.0\% - 95.0\%$) exhibiting gradual trend accumulation and diurnal oscillations.
- `request_rate_per_sec` (`float`): Ingress HTTP request throughput per second ($10.0 - 250.0$).
- `error_count` (`int`): Non-2xx/3xx HTTP status response events observed during the hour.
- `p95_latency_ms` (`float`): 95th percentile request round-trip latency in milliseconds.

#### Feature Engineering Properties

- **Regular cadence:** Exactly $1$ observation per hour ($\Delta t = 1\,\text{h}$), ideal for frequency inference (`freq='h'`).
- **Resampling capability:** Supports downsampling to $6\,\text{h}$, $12\,\text{h}$, and $24\,\text{h}$ (daily) intervals using mean, max, or sum aggregations.
- **Causality requirement:** When creating rolling moving averages and exponential smoothing features, past windows $[t-w, t-1]$ must be used without incorporating time $t$ or future lookaheads.

---

### 3. `sample_articles_corpus.json`

A structured JSON corpus of technology and data engineering news articles curated for text cleaning, normalization, vocabulary building, and token vectorization.

- `id` (`string`): Unique document identifier (e.g., `article_00001`).
- `url` (`string`): Source web address of the published story.
- `category` (`string`): Domain topic category (`technology`, `distributed-systems`, `natural-language-processing`, `web-scraping`, `computer-vision`).
- `title` (`string`): Headline string with standard typography.
- `abstract` (`string`): Summary paragraph preceding main article text.
- `body` (`string`): Full body text spanning multiple sentences and technical terms.
- `author` (`string`): Journalist or researcher credited with the piece.
- `published_at` (`string` / `ISO-8601`): Canonical publication timestamp.
- `thumbnail_url` (`string`): Associated thumbnail image URL.

---

### 4. `crawled_news_archive.zip`

A standard archive replicating the filesystem layout generated by web harvesting scripts.

- `corpus/`: Subdirectory containing plain text files (`article_00001.txt` ... `article_00005.txt`) structured as `Title \n\n Abstract \n\n Body \n\n Author`.
- `thumbnails/`: Subdirectory containing associated image assets (`article_00001.png` ... `article_00005.png`) validated with 3-channel RGB dimensions ($160 \times 100$).
