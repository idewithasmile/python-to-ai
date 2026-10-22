# Session Data

This directory contains the small, redistributable datasets used by the notebooks in [Basic Python (Part 2)](../README.md). Every file is deliberately tiny so that the examples run offline and remain easy to inspect with a text editor.

## Files and schemas

### `area_price.csv`

A four-row toy regression table used to demonstrate reading numeric CSV data.

| Column | Type | Description |
| :--- | :--- | :--- |
| `area` | float | Floor area in hundreds of square feet. |
| `price` | float | Sale price in hundreds of thousands of dollars. |

Example row: `6.7,9.1`.

### `advertising.csv`

The classic Advertising dataset with 200 observations. It is commonly distributed with *An Introduction to Statistical Learning* by James, Witten, Hastie, and Tibshirani.

| Column | Type | Description |
| :--- | :--- | :--- |
| `TV` | float | Television advertising budget, in thousands of dollars. |
| `Radio` | float | Radio advertising budget, in thousands of dollars. |
| `Newspaper` | float | Newspaper advertising budget, in thousands of dollars. |
| `Sales` | float | Product sales, in thousands of units. |

Missing values: none. The target column is `Sales`.

### `iris.csv`

A 30-row extract of the Iris flower measurements. The full dataset is 150 rows, but this extract is enough for file-reading and parsing exercises.

| Column | Type | Description |
| :--- | :--- | :--- |
| `Id` | integer | Row identifier from the original dataset. |
| `SepalLength` | float | Sepal length in centimetres. |
| `SepalWidth` | float | Sepal width in centimetres. |
| `PetalLength` | float | Petal length in centimetres. |
| `PetalWidth` | float | Petal width in centimetres. |
| `Species` | string | One of `Iris-setosa`, `Iris-versicolor`, or `Iris-virginica`. |

Unknown values: none. The `Species` column is the label.

### `iris_petal.csv`

A 150-row petal-only extract of the same Iris measurements. The header was normalised for this session.

| Column | Type | Description |
| :--- | :--- | :--- |
| `PetalLength` | float | Petal length in centimetres. |
| `PetalWidth` | float | Petal width in centimetres. |
| `Label` | integer | Encoded species: `0` = setosa, `1` = versicolor, `2` = virginica. |

Unknown values: none. This file is useful for practising numeric aggregation without parsing species names.

### `iris_demo.csv`

A four-row teaching sample with the same schema as `iris_petal.csv`. It is intentionally small enough to trace by hand.

### `text/hello_world.txt`

A three-line UTF-8 text fixture used for the basic read, `readlines`, and line-iteration examples.

### `text/students.csv`

A three-row CSV fixture used for split, join, and record-parsing examples.

| Column | Type | Description |
| :--- | :--- | :--- |
| `student_id` | string | Zero-padded identifier, for example `001`. |
| `name` | string | Lowercase first name. |
| `date_of_birth` | string | Date in `DD-MM-YYYY` format. |

## Attribution and licence notes

- The Advertising data are a widely redistributed teaching dataset from *An Introduction to Statistical Learning*.
- The Iris data originate from R. A. Fisher's 1936 paper and are in the public domain as a canonical teaching dataset.
- The text fixtures are synthetic and contain no personal information.

## Conventions

- Files are UTF-8 encoded and newline-terminated.
- The first row is always a header.
- Numeric columns contain no thousands separators or currency symbols.
- Dates remain strings in this session; parsing dates is deferred to the pandas module.
- Do not add credentials, personal data, large files, or generated outputs here. Notebook outputs belong in an ignored `outputs/` directory.
