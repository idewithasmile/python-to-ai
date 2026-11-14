# Session Data Schema — `sql_store`

This directory contains the reproducible relational dataset used across the `Database - SQL (1)` notebooks and exercises.

## Files

| File | Format | Description |
| :--- | :--- | :--- |
| `sql_store_seed.sql` | SQL DDL + DML | Creates and populates the four `sql_store` tables (`customers`, `products`, `orders`, `order_items`) in SQLite/MySQL-compatible SQL. |

## Table Schemas

### 1. `customers` (12 rows)

Stores customer profile, contact, regional, and loyalty-point metadata.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `customer_id` | `INTEGER` | `PRIMARY KEY` | Unique customer identifier |
| `first_name` | `TEXT` | `NOT NULL` | Customer first name |
| `last_name` | `TEXT` | `NOT NULL` | Customer last name |
| `birth_date` | `TEXT` | `NOT NULL` | ISO-8601 date (`YYYY-MM-DD`) |
| `phone` | `TEXT` | Nullable | Phone number (`NULL` when missing) |
| `address` | `TEXT` | `NOT NULL` | Street address |
| `city` | `TEXT` | `NOT NULL` | City name |
| `state` | `TEXT` | `NOT NULL` | Two-letter US state abbreviation |
| `points` | `INTEGER` | `NOT NULL` | Loyalty points balance |

### 2. `products` (13 rows)

Stores catalog inventory and unit pricing.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `product_id` | `INTEGER` | `PRIMARY KEY` | Unique product identifier |
| `name` | `TEXT` | `NOT NULL` | Product display name |
| `quantity_in_stock` | `INTEGER` | `NOT NULL` | Available units in inventory |
| `unit_price` | `REAL` | `NOT NULL` | Price per unit (USD) |

### 3. `orders` (14 rows)

Stores customer order headers and fulfillment timestamps.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `order_id` | `INTEGER` | `PRIMARY KEY` | Unique order identifier |
| `customer_id` | `INTEGER` | `NOT NULL`, `FOREIGN KEY -> customers(customer_id)` | Ordering customer |
| `order_date` | `TEXT` | `NOT NULL` | Order placement date (`YYYY-MM-DD`) |
| `status` | `INTEGER` | `NOT NULL` | Order status code (`1` = processed, `2` = shipped) |
| `comments` | `TEXT` | Nullable | Optional fulfillment note |
| `shipped_date` | `TEXT` | Nullable | Date shipped (`YYYY-MM-DD`) or `NULL` |
| `shipper_id` | `INTEGER` | Nullable | Assigned shipper ID or `NULL` |

### 4. `order_items` (22 rows)

Stores line-item quantities and unit prices per order.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `order_id` | `INTEGER` | `PRIMARY KEY`, `FOREIGN KEY -> orders(order_id)` | Parent order |
| `product_id` | `INTEGER` | `PRIMARY KEY`, `FOREIGN KEY -> products(product_id)` | Purchased product |
| `quantity` | `INTEGER` | `NOT NULL` | Units ordered |
| `unit_price` | `REAL` | `NOT NULL` | Unit price at time of order |
