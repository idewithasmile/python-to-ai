# Data Manipulation, Time-Series Feature Engineering, and Web Crawling Pipelines

[Back to Module 01](../README.md) · [Back to the master roadmap](../../README.md)

**Session date:** 2026-11-07 · **Module:** 01 — Python Fundamentals & Algorithmic Thinking  
**Format:** Theoretical exposition, mathematical formulations, architectural topologies, executable code primitives, and production harvesting pipelines

---

## Overview

Modern machine learning systems are rarely fed pristine, pre-packaged tensors. In production environments and data science competitions, over $80\%$ of engineering effort is expended upstream of model training: ingesting heterogeneous raw data formats, cleaning tabular matrices, formulating leak-free temporal features from high-velocity time-series streams, converting unstructured text into dense discrete vocabularies, and scraping web corpora to enrich training distributions.

A failure in data manipulation cascades downstream. Improperly handled missing values introduce silent distortion; temporal operations that look ahead into future time steps cause catastrophic generalization collapse upon deployment; unnormalized text inflates vocabulary dimensionality by orders of magnitude; and brittle, unthrottled web scrapers trigger rate limits, IP bans, or corrupted multimodal datasets.

This session establishes a first-principles foundation for data manipulation across three critical data modalities—**Tabular Data**, **Time-Series Streams**, and **Web-Scraped Text & Media**:

1. **Tabular Data Primitives & Algorithmic Statistics:** We examine 2D grid representations of observations and features, convert between high-level DataFrame abstractions and pure Python list-of-lists matrices, and derive first-principles implementations of core statistical aggregators (sum, bounded slice summation, min/max bounds, arithmetic mean, odd/even median, and marketing budget contribution shares) on real-world advertising benchmark data.
2. **Time-Series Manipulation & Causal Feature Engineering:** We formalize temporal sequences, analyze resampling paradigms (downsampling aggregations versus upsampling interpolations), formulate rolling window statistics (SMA, EMA, rolling volatility), and construct leak-free temporal features (lags, first differences, and cyclical trigonometric hour/day projections) while enforcing strict causality constraints.
3. **Text Normalization & Index-Based Vocabulary Encoding:** We explore the mathematical necessity of numerical text representations, build modular normalization pipelines (lowercasing, punctuation stripping, whitespace collapsing), construct dynamic vocabulary dictionaries with reserved special tokens (`<PAD>`, `<UNK>`), and vectorize raw natural language strings into integer sequences and Term Frequency vectors.
4. **Web Crawling, DOM Traversal, & Multimodal Harvesters:** We dissect the client-server HTTP transaction lifecycle, traverse HTML Document Object Model (DOM) hierarchies using XPath and CSS selectors, formulate polite scraping protocols (rate limiting, jitter, error retries, robots.txt etiquette), extract structured article metadata (title, lead, body paragraphs, author), normalize crawled image assets (handling palette mode `P` and alpha channels into standard 3-channel RGB), and package datasets into reproducible compressed archives.

---

## Learning outcomes

By the end of this session, you should be able to:

- Distinguish between row-major and column-major memory layouts and explain when to operate on pure Python matrices versus vectorized arrays.
- Ingest tabular CSV datasets and convert high-level DataFrames into nested Python lists using `.values.tolist()`.
- Implement first-principles algorithms for column summation, bounded index slicing ($L[\text{start}:\text{end}]$), minimum, maximum, mean, and parity-aware median without external statistical libraries.
- Compute multi-channel budget allocation and percentage contribution metrics across advertising media mixes.
- Parse ISO-8601 timestamps, infer series frequencies, and formulate causal time-series transformations without lookahead bias.
- Implement downsampling aggregations and contrast forward-filling (`ffill`) versus backward-filling (`bfill`) in terms of data leakage risks.
- Formulate and compute rolling window statistics: Simple Moving Average (SMA), Exponential Moving Average (EMA), and rolling standard deviation.
- Construct cyclical calendar features using trigonometric projections ($\sin$, $\cos$) for hour-of-day and day-of-week seasonality.
- Design and execute text normalization pipelines that sanitize typographical noise, strip punctuation, and collapse whitespace.
- Compile a vocabulary dictionary from a multi-document corpus, map tokens to contiguous integer IDs, and manage out-of-vocabulary (OOV) tokens with `<UNK>` fallback.
- Vectorize natural language sentences into fixed-length integer arrays with customizable truncation and padding.
- Compute Bag-of-Words (BoW) count vectors and normalized Term Frequency (TF) distributions.
- Trace the HTTP request-response cycle, handle status codes, and configure robust request headers (User-Agent, timeouts).
- Navigate HTML DOM trees to extract targeted nodes using absolute and relative XPath expressions as well as CSS selectors.
- Build an automated scraper that harvests paginated article catalogs and extracts structured fields: title, abstract, body paragraphs, and author credentials.
- Harvest web image assets, sanitize varying color modes (converting palette `P` and `RGBA` to standard `RGB`), resize thumbnails, and package text and image corpora into compressed `.zip` bundles.

---

## Session map

| Notebook | Focus | Key Topics |
| :--- | :--- | :--- |
| [01 — Tabular Data & Primitives](notebooks/01-tabular-data-and-statistical-primitives.ipynb) | Tabular Handling & First-Principles Primitives | DataFrame to list-of-lists conversion, column summation, slice sums (rows 2–11), min/max, mean, odd/even median, budget share on advertising benchmark |
| [02 — Time-Series Manipulation](notebooks/02-timeseries-manipulation-and-temporal-features.ipynb) | Temporal Sequences & Causal Engineering | Hourly telemetry ingestion, downsampling to daily metrics, SMA/EMA rolling filters, rolling volatility, lags, differences, cyclical sine/cosine calendar features |
| [03 — Text Preprocessing & Encoding](notebooks/03-text-preprocessing-and-index-encoding.ipynb) | NLP Normalization & Vocabulary Indexing | Case folding, punctuation removal, whitespace normalization, corpus vocabulary compilation, special tokens (`<PAD>`, `<UNK>`), vectorization, BoW and TF |
| [04 — Web Scraping & Archiving](notebooks/04-web-scraping-and-corpus-curation.ipynb) | Web Harvesting & Multimodal Pipelines | HTTP lifecycle, DOM tree parsing, XPath/CSS selection, structured article extraction, Pillow image color mode conversion (`P` $\to$ `RGB`), compressed archive packaging |

---

## 1. Tabular Data Structures & First-Principles Ingestion

Tabular data is the predominant format in business intelligence, classical machine learning, and structured benchmarking. Formally, a tabular dataset is a 2D matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ comprising $N$ empirical observations (rows) evaluated across $D$ distinct measurable features (columns):

$$\mathbf{X} = \begin{bmatrix} x_{1, 1} & x_{1, 2} & \cdots & x_{1, D} \\ x_{2, 1} & x_{2, 2} & \cdots & x_{2, D} \\ \vdots & \vdots & \ddots & \vdots \\ x_{N, 1} & x_{N, 2} & \cdots & x_{N, D} \end{bmatrix}$$

```
+-----------------------------------------------------------------------+
|                             TABULAR DATA MATRIX                       |
+-------------------+-----------------+-----------------+---------------+
| Observation (Row) | Feature 1 (TV)  | Feature 2 (Rad) | Target (Sale) |
+-------------------+-----------------+-----------------+---------------+
| Sample 1          |      230.1      |       37.8      |      22.1     |
| Sample 2          |       44.5      |       39.3      |      10.4     |
| ...               |       ...       |       ...       |      ...      |
| Sample N          |      232.1      |        8.6      |      13.4     |
+-------------------+-----------------+-----------------+---------------+
```

### 1.1 Ingestion and In-Memory Data Representations

When importing data from persistent storage (such as `.csv` files), data can be managed at several abstraction tiers:

1. **High-Level DataFrames (`pandas.DataFrame`):** Provides column indexing, rich query operations, and integration with C-extensions. Under the hood, columns of identical data types are stored in columnar memory blocks.
2. **Dense Homogeneous Arrays (`numpy.ndarray`):** Represents data as a contiguous row-major (`C_CONTIGUOUS`) or column-major (`F_CONTIGUOUS`) buffer in physical memory.
3. **Nested List-of-Lists (`list[list[float]]`):** The native Python representation where the outer list contains references to inner lists representing each row.

To transition from a DataFrame to native Python collections:

```python
import pandas as pd

# Read tabular CSV into DataFrame
df = pd.read_csv("advertising_benchmark.csv")

# Extract numeric values as a nested list of lists
advertising_lst = df[["tv", "radio", "newspaper", "sales"]].values.tolist()
```

Each record in `advertising_lst` is a list `[tv, radio, newspaper, sales]` accessed by index: `row[0]` for TV, `row[1]` for Radio, `row[2]` for Newspaper, and `row[3]` for Sales.

---

### 1.2 First-Principles Algorithmic Aggregations

While high-level libraries provide `.sum()`, `.mean()`, and `.median()`, understanding their algorithmic implementations is critical for interview readiness, custom CUDA kernels, and resource-constrained embedded runtimes.

#### 1. Column Summation
$$\text{Sum}(X_c) = \sum_{i=1}^N x_{i, c}$$

```python
def calculate_column_sum(matrix: list[list[float]], col_idx: int) -> float:
    return sum(row[col_idx] for row in matrix)
```

#### 2. Bounded Row Slice Summation
In many streaming and batch processing workflows, aggregations are restricted to specific contiguous index windows $[a, b)$:
$$\text{SliceSum}(X_c, a, b) = \sum_{i=a}^{b-1} x_{i, c}$$

In Python's 0-indexed slicing notation, rows $2^{\text{nd}}$ through $11^{\text{th}}$ (inclusive of row 11) correspond to `start_idx = 1` and `end_idx = 11`:

```python
def calculate_slice_sum(
    matrix: list[list[float]], col_idx: int, start_idx: int, end_idx: int
) -> float:
    return sum(row[col_idx] for row in matrix[start_idx:end_idx])
```

#### 3. Extremum Bounds (Min & Max)
Finding the minimum and maximum requires an $O(N)$ linear pass over the column elements:

```python
def calculate_column_min_max(matrix: list[list[float]], col_idx: int) -> tuple[float, float]:
    col_values = [row[col_idx] for row in matrix]
    return min(col_values), max(col_values)
```

#### 4. Arithmetic Mean
$$\bar{x}_c = \frac{1}{N} \sum_{i=1}^N x_{i, c}$$

```python
def calculate_column_mean(matrix: list[list[float]], col_idx: int) -> float:
    col_values = [row[col_idx] for row in matrix]
    return sum(col_values) / len(col_values)
```

#### 5. Median Calculation with Parity Handling
The median represents the 50th percentile of an ordered distribution. Given a sorted sequence $X_{(1)} \le X_{(2)} \le \dots \le X_{(N)}$:

$$\text{Median}(X) = \begin{cases} X_{(\frac{N+1}{2})}, & \text{if } N \text{ is odd} \\ \frac{X_{(\frac{N}{2})} + X_{(\frac{N}{2} + 1)}}{2}, & \text{if } N \text{ is even} \end{cases}$$

In Python 0-indexed arrays:
- If $N$ is odd ($N \pmod 2 = 1$): index is $N // 2$.
- If $N$ is even ($N \pmod 2 = 0$): average of elements at index $N // 2 - 1$ and $N // 2$.

```python
def calculate_column_median(matrix: list[list[float]], col_idx: int) -> float:
    sorted_values = sorted(row[col_idx] for row in matrix)
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
```

---

### 1.3 Marketing Mix Budget Contribution Modeling

In multi-channel marketing analytics, budget attribution quantifies the relative capital allocated to each promotional vehicle. Let $B_c = \sum_{i=1}^N x_{i, c}$ represent total expenditure in channel $c \in \{\text{TV}, \text{Radio}, \text{Newspaper}\}$. Total marketing investment is:

$$B_{\text{total}} = \sum_{c \in \text{channels}} B_c = B_{\text{TV}} + B_{\text{Radio}} + B_{\text{Newspaper}}$$

The percentage contribution of channel $c$ is:

$$\text{Contribution Share}_c = \frac{B_c}{B_{\text{total}}} \times 100\%$$

#### Numerical Benchmark Verification (`advertising_benchmark.csv`)

| Metric | TV Channel ($c=0$) | Radio Channel ($c=1$) | Newspaper Channel ($c=2$) | Sales Target ($c=3$) |
| :--- | :--- | :--- | :--- | :--- |
| **Sum** | $29,408.50$ | $4,652.80$ | $6,110.80$ | $2,804.50$ |
| **Slice Sum (Rows 2–11)** | $854.90$ | $249.10$ | $387.80$ | $107.30$ |
| **Minimum** | $0.70$ | $0.00$ | $0.30$ | $1.60$ |
| **Maximum** | $296.40$ | $49.60$ | $114.00$ | $27.00$ |
| **Mean ($\mu$)** | $147.0425$ | $23.2640$ | $30.5540$ | $14.0225$ |
| **Median** | $149.75$ | $22.90$ | $25.75$ | $12.90$ |
| **Budget Share** | **$73.206\%$** | **$11.582\%$** | **$15.212\%$** | — |

$$\text{Total Budget} = 29,408.50 + 4,652.80 + 6,110.80 = 40,172.10$$

$$\text{TV Share} = \frac{29408.5}{40172.1} \times 100\% = 73.206\%$$
$$\text{Radio Share} = \frac{4652.8}{40172.1} \times 100\% = 11.582\%$$
$$\text{Newspaper Share} = \frac{6110.8}{40172.1} \times 100\% = 15.212\%$$

---

## 2. Time-Series Manipulation & Causal Feature Engineering

A time-series dataset is an ordered temporal sequence where observations are causally indexed by timestamps:
$$\mathcal{S} = \{(t_1, \mathbf{x}_1), (t_2, \mathbf{x}_2), \dots, (t_T, \mathbf{x}_T)\}, \quad t_1 < t_2 < \dots < t_T$$

```
Temporal Ordering & Causal Direction
─────────────────────────────────────────────────────────────► (Time t)
Past Observations [t-k, ..., t-1]  │  Current [t]  │  Future [t+1, ...]
(Permissible for Feature Creation) │  (Target y_t) │  (STRICTLY FORBIDDEN)
```

> [!CAUTION]
> **The Principle of Temporal Causality:** Any feature engineered for observation $t$ must depend strictly on information recorded at or before time $t-1$ (for forecasting) or time $t$ (for state monitoring). Using future points $t+1, t+2$ constitutes **lookahead leakage**, producing models that test with near-zero error but fail immediately in production.

---

### 2.1 Resampling and Temporal Aggregations

Real-world telemetry arrives at irregular or high-frequency intervals. Resampling aligns temporal series to a consistent cadence:

- **Downsampling:** Reducing frequency (e.g., $1\,\text{hour} \to 24\,\text{hours}$). Requires an aggregation function:
  $$\bar{y}_{\text{day}} = \frac{1}{24} \sum_{h=1}^{24} y_h, \quad y_{\text{peak}} = \max_{1 \le h \le 24} y_h$$
- **Upsampling:** Increasing frequency (e.g., $1\,\text{day} \to 1\,\text{hour}$). Generates missing intermediate intervals that must be populated using:
  - **Forward-Fill (`ffill`):** Propagates the latest known measurement forward. Causal and leak-free.
  - **Backward-Fill (`bfill`):** Propagates future values backward. **Leaking future data into the past—never use in forecasting training sets!**
  - **Linear / Spline Interpolation:** Estimates points between two known anchors. Acceptable for retrospective historical analysis, but invalid for real-time edge streaming.

---

### 2.2 Rolling Filters and Volatility

Rolling statistics evaluate moving statistical summaries over a backward-looking historical window of length $w$:

#### 1. Simple Moving Average (SMA)
$$\text{SMA}_t(w) = \frac{1}{w} \sum_{i=0}^{w-1} y_{t-i}$$

#### 2. Exponential Moving Average (EMA)
Assigns geometrically decreasing weights to older observations, controlled by smoothing constant $\alpha \in (0, 1]$:
$$\text{EMA}_t(\alpha) = \alpha y_t + (1 - \alpha) \text{EMA}_{t-1}, \quad \text{where } \alpha = \frac{2}{w + 1}$$

#### 3. Rolling Volatility (Standard Deviation)
$$\sigma_t(w) = \sqrt{\frac{1}{w-1} \sum_{i=0}^{w-1} (y_{t-i} - \text{SMA}_t(w))^2}$$

Rolling volatility detects regime shifts, sensor anomalies, and turbulent operational states in server infrastructure.

---

### 2.3 Cyclical Trigonometric Feature Encodings

Calendar metrics such as hour of day ($h \in \{0, 1, \dots, 23\}$) or day of week ($d \in \{0, 1, \dots, 6\}$) are periodic. Treating them as ordinary linear integers creates an artificial discontinuity between $23:00$ and $00:00$ ($\Delta = 23$ instead of $\Delta = 1$).

To preserve cyclical topology, we project the periodic scalar onto the 2D Cartesian unit circle using sine and cosine transformations:

$$h_{\sin} = \sin\left(\frac{2\pi \cdot h}{24}\right), \quad h_{\cos} = \cos\left(\frac{2\pi \cdot h}{24}\right)$$

```
        (0, 1) [Hour 6:00]
            ▲
            │
(-1, 0) ────┼────► (1, 0) [Hour 0:00 / 24:00]
[Hour 18:00]│
            ▼
        (0, -1) [Hour 12:00]
```

This ensures that the Euclidean distance between $23:00$ and $00:00$ is identical to the distance between $12:00$ and $13:00$, allowing distance-based and linear models to learn smooth temporal dynamics.

---

## 3. Text Normalization & Index-Based Vocabulary Encoding

Natural language documents cannot be ingested directly into linear classifiers or deep neural networks. They must be parsed, cleaned, tokenized, and mapped to integer indices or embedding vectors.

```
+───────────────────────────+
|     Raw Text Corpus       |  "Trời hôm nay đẹp quá!"
+─────────────┬─────────────+
              │
              ▼
+───────────────────────────+
|    Text Normalization     |  Case folding, punctuation stripping,
+─────────────┬─────────────+  whitespace collapsing: "trời hôm nay đẹp quá"
              │
              ▼
+───────────────────────────+
|    Tokenization           |  Token list: ["trời", "hôm", "nay", "đẹp", "quá"]
+─────────────┬─────────────+
              │
              ▼
+───────────────────────────+
|   Vocabulary Mapping      |  Vocabulary Dictionary V:
+─────────────┬─────────────+  {"<PAD>": 0, "<UNK>": 1, "hôm": 2, "nay": 9, ...}
              │
              ▼
+───────────────────────────+
|   Index Vectorization     |  Discrete Index Array: [14, 2, 9, 36, 89]
+───────────────────────────+
```

### 3.1 Text Normalization Pipeline

A production text normalization pipeline executes three fundamental stages:

1. **Case Folding:** Converts characters to lowercase to eliminate false distinct tokens (e.g., `Intelligence`, `intelligence`, and `INTELLIGENCE` map to the same token `intelligence`).
2. **Punctuation Stripping:** Eliminates punctuation glyphs (`!`, `?`, `#`, `<`, `>`, etc.) using fast character translation tables (`str.maketrans`) or regex:
   ```python
   import re, string

   translator = str.maketrans("", "", string.punctuation)
   clean_text = raw_text.translate(translator)
```
3. **Whitespace Normalization:** Replaces tabs, line breaks, and consecutive spaces with a single whitespace delimiter:
   ```python
   clean_text = re.sub(r"\s+", " ", clean_text).strip()
```

#### Example Transformation
- **Input:** `'#ArTiFIciAL. In?>TeLLi!g@ENce'`
- **Output:** `'artificial intelligence'`

---

### 3.2 Vocabulary Compilation and Special Tokens

Given a training corpus of $M$ documents, the vocabulary $\mathcal{V}$ is compiled by harvesting all unique tokens. In industrial systems, the vocabulary includes reserved **special tokens**:

- `<PAD>` (Index 0): Padding token used to extend variable-length sequences to uniform tensor batch dimensions.
- `<UNK>` (Index 1): Unknown token used as a fallback for out-of-vocabulary (OOV) words encountered during inference.
- `<BOS>` (Index 2): Beginning of Sequence indicator.
- `<EOS>` (Index 3): End of Sequence indicator.

```python
class Vocabulary:
    def __init__(self):
        self.token_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_token = {0: "<PAD>", 1: "<UNK>"}

    def add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            return idx
        return self.token_to_id[token]

    def get_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id["<UNK>"])
```

---

### 3.3 Sequence Vectorization & Bag-of-Words

#### Vectorize Algorithm
Maps a tokenized string into an array of integer IDs with length enforcement:

```python
def vectorize(
    text: str,
    vocabulary: Vocabulary,
    max_length: int | None = None,
    pad: bool = False,
) -> list[int]:
    tokens = text_normalize(text).split()
    encoded = [vocabulary.get_id(t) for t in tokens]

    if max_length is not None:
        if len(encoded) > max_length:
            encoded = encoded[:max_length]  # Truncation
        elif pad and len(encoded) < max_length:
            pad_id = vocabulary.token_to_id["<PAD>"]
            encoded.extend([pad_id] * (max_length - len(encoded)))  # Padding
    return encoded
```

#### Bag-of-Words (BoW) & Term Frequency (TF)
Rather than maintaining sequential word order, Bag-of-Words models represent a document as a count vector $\mathbf{v} \in \mathbb{R}^{|\mathcal{V}|}$ where component $v_j$ counts the occurrences of word $j$.

Term Frequency (TF) normalizes counts by document length:
$$\text{TF}(w, d) = \frac{f_{w, d}}{\sum_{w' \in d} f_{w', d}}$$

---

## 4. Web Crawling, DOM Traversal, & Multimodal Harvesters

When pre-compiled datasets are unavailable, machine learning engineers build automated web harvesters to scrape text corpora and computer vision training sets directly from web properties.

```
Web Harvesting Architecture
┌─────────────────────────────────────────────────────────────┐
│ 1. Initialize HTTP Client / Headless Browser               │
│    - Set custom User-Agent headers, timeouts, and retries   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Request Category / Catalog Index Page                    │
│    - Iterate over paginated routes (e.g. page_idx)           │
│    - Extract article permalinks via //div[@class='...']//a  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Fetch Detail Articles & Parse DOM Tree                   │
│    - Title:     <h1> tag                                    │
│    - Abstract:  <h2> / lead paragraph                       │
│    - Body:      Iterate over <p> tags                       │
│    - Author:    <span class="name"> metadata                │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Multimodal Image Asset Acquisition                       │
│    - Extract <img src="..."> URLs                           │
│    - Download binary bytes via stream                       │
│    - Normalize color space (Palette P / RGBA -> RGB)        │
│    - Validate dimensions and save to thumbnails/            │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Archival Packaging                                       │
│    - Save plain text corpus (*.txt) and metadata JSON       │
│    - Compress into production *.zip archive                 │
└─────────────────────────────────────────────────────────────┘
```

### 4.1 Client-Server HTTP Lifecycle & Etiquette

Every web interaction executes an HTTP request-response transaction:
1. **Client GET Request:** Transmits request line and headers:
   ```http
   GET /thoi-su-page1 HTTP/1.1
   Host: vietnamnet.vn
   User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
   Accept: text/html,application/xhtml+xml
```
2. **Server Response:** Returns status line, headers, and body:
   - `200 OK`: Successful retrieval.
   - `301 / 302`: Resource moved (follow redirect).
   - `403 Forbidden` / `429 Too Many Requests`: Client blocked or rate-limited.
   - `500 / 503`: Server internal failure.

#### Crawling Etiquette & Resilience Rules
- **Rate Limiting & Jitter:** Always inject random pauses (`time.sleep(1.0 + random.uniform(0.5, 1.5))`) between consecutive requests to prevent server denial-of-service.
- **Robots.txt Exclusion Standard:** Inspect `https://target.com/robots.txt` to verify whether automated scrapers are permitted on target subpaths.
- **Exponential Backoff:** If a request receives a timeout or `503`, retry after $2^k$ seconds rather than hammering the server immediately.

---

### 4.2 DOM Navigation with XPath and CSS Selectors

HTML documents are hierarchical trees of nodes. Elements are targeted via two primary querying syntaxes:

| Syntax | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| **CSS Selector** | Selector | `div.content-detail > h1` | Selects `<h1>` immediately nested inside `<div class="content-detail">` |
| **XPath Absolute** | Path | `/html/body/div[2]/h1` | Rigid traversal from root node; fragile to layout modifications |
| **XPath Relative** | Path | `//div[@class='topStory-15nd']//a` | Matches any `<a>` nested within a `<div>` matching the class attribute |

#### Parsing News Articles via XPath
- **Main Container:** `'//div[@class="content-detail"]'`
- **Article Title:** `'//div[@class="content-detail"]//h1'`
- **Lead / Abstract:** `'//div[@class="content-detail"]//h2'`
- **Author Tag:** `'//span[@class="name"]'`
- **Body Paragraphs:** `'//div[@id="maincontent"]//p'`

---

### 4.3 Multimodal Image Normalization and Color Spaces

Computer vision models (e.g., ResNet, YOLOv8, ViT) assume standardized 3-channel RGB image inputs:
$$\mathbf{I} \in \mathbb{R}^{H \times W \times 3}, \quad \text{channels} = \{\text{Red}, \text{Green}, \text{Blue}\}$$

However, web media contains varied image color representations:
- **Palette Mode (`P`):** 8-bit index mode where each pixel value points to a 256-color palette lookup table. Ingesting palette images directly into tensor pipelines causes severe channel dimension errors.
- **Alpha Transparency (`RGBA`, `LA`):** 4-channel representation containing transparency masks that must be converted before convolution.
- **Grayscale (`L`):** Single-channel luminance.

#### Sanitization Function
```python
from io import BytesIO
from PIL import Image


def sanitize_image(image_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(image_bytes)) as img:
        # Convert palette P or RGBA modes into standard 3-channel RGB
        if img.mode in ("P", "RGBA", "LA") or img.mode != "RGB":
            img = img.convert("RGB")
        return img.copy()
```

---

### 4.4 Archival Storage & Dataset Bundling

To ensure reproducibility across data science teams, crawled assets must be persisted with structured atomic writes and packaged into compressed archives:

```python
import zipfile


def bundle_corpus(output_zip: str, text_dir: str, image_dir: str) -> None:
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for txt_file in Path(text_dir).glob("*.txt"):
            zf.write(txt_file, arcname=f"corpus/{txt_file.name}")
        for img_file in Path(image_dir).glob("*.png"):
            zf.write(img_file, arcname=f"thumbnails/{img_file.name}")
```

---

## 5. Practical Exercises & Verification Lab

### Exercise 1: First-Principles Statistical Auditing
Using `advertising_benchmark.csv`, verify that:
1. The budget contribution share of TV is strictly between $73.0\%$ and $73.5\%$.
2. The median of Newspaper advertising spend ($25.75$) differs from its arithmetic mean ($30.5540$), demonstrating right-skewness in promotional outlays.
3. The slice sum of Sales between row indices $1$ and $11$ ($L[1:11]$) is exactly $107.30$.

### Exercise 2: Causal Leakage Audit in Rolling Moving Averages
Given a 100-step telemetry sequence $Y = [y_0, y_1, \dots, y_{99}]$:
1. Write a function that computes a 5-step rolling mean centered at $t$ ($\frac{1}{5}\sum_{k=-2}^2 y_{t+k}$).
2. Explain mathematically why this centered window introduces lookahead bias.
3. Formulate the strictly backward-looking causal rolling mean and verify that index $t$ uses only indices $t-4$ through $t$.

### Exercise 3: Out-of-Vocabulary Fallback & Vectorization
Given the vocabulary compiled in Notebook 03:
1. Vectorize the sentence `"Quantum computing accelerates gradient descent optimization."`
2. Count how many tokens trigger the `<UNK>` token identifier (index 1).
3. Pad the resulting vector to length 12 using the `<PAD>` token identifier (index 0).

---

## Summary Checklist

- [x] Ingest tabular CSV data and convert to native list-of-lists matrices.
- [x] Implement first-principles column summation, bounded slice sums, min/max, mean, and parity-aware median.
- [x] Calculate marketing budget contribution percentages across promotional channels.
- [x] Resample hourly time-series data to daily aggregates and compute rolling SMA, EMA, and volatility.
- [x] Create causal lag features, differences, and cyclical trigonometric sine/cosine encodings without lookahead bias.
- [x] Construct an end-to-end text normalization pipeline stripping punctuation and normalizing whitespace.
- [x] Compile corpus vocabularies with reserved `<PAD>` and `<UNK>` tokens and vectorize text into index arrays.
- [x] Build web harvesters navigating HTML DOM trees via XPath and CSS selectors.
- [x] Normalize scraped image assets across color modes (`P` $\to$ `RGB`) and package multimodal corpora into compressed `.zip` archives.
