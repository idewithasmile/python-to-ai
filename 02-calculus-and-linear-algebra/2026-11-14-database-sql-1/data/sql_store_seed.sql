-- Schema and synthetic sample dataset for the sql_store relational database
-- Designed for single-table SQL practice, pattern matching, aggregation, and window functions.

DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    birth_date TEXT NOT NULL,
    phone TEXT,
    address TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    points INTEGER NOT NULL
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    quantity_in_stock INTEGER NOT NULL,
    unit_price REAL NOT NULL
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    status INTEGER NOT NULL,
    comments TEXT,
    shipped_date TEXT,
    shipper_id INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

CREATE TABLE order_items (
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    PRIMARY KEY (order_id, product_id),
    FOREIGN KEY (order_id) REFERENCES orders (order_id),
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);

INSERT INTO customers (customer_id, first_name, last_name, birth_date, phone, address, city, state, points) VALUES
(1, 'Babara', 'MacCaffrey', '1986-03-28', '781-932-9754', '0 Sage Terrace', 'Waltham', 'MA', 2273),
(2, 'Ines', 'Brushfield', '1986-04-13', '804-427-9456', '14187 Commercial Trail', 'Hampton', 'VA', 947),
(3, 'Freddi', 'Boagey', '1985-02-07', '719-724-7869', '251 Springs Junction', 'Colorado Springs', 'CO', 2967),
(4, 'Ambur', 'Roseburgh', '1974-04-14', '407-231-8017', '30 Arapahoe Terrace', 'Orlando', 'FL', 457),
(5, 'Clemmie', 'Betchley', '1973-11-07', NULL, '5 Spohn Circle', 'Arlington', 'TX', 3675),
(6, 'Elka', 'Twiddell', '1991-09-04', '312-480-8498', '7 Manley Drive', 'Chicago', 'IL', 3073),
(7, 'Ilene', 'Dowson', '1964-08-30', '615-641-4759', '50 Lillian Crossing', 'Nashville', 'TN', 1672),
(8, 'Thacher', 'Naseby', '1993-07-17', '941-527-3977', '538 Mosinee Center', 'Sarasota', 'FL', 205),
(9, 'Romola', 'Rumgay', '1992-05-23', '559-181-3744', '3520 Ohio Trail', 'Visalia', 'CA', 1486),
(10, 'Levy', 'Mynett', '1969-10-13', '404-246-3370', '68 Lawn Avenue', 'Atlanta', 'GA', 796),
(11, 'Alana', 'Bundy', '2000-02-29', NULL, '124 Oakridge Avenue', 'Richmond', 'VA', 1850),
(12, 'Ivan', 'Seabrook', '1996-06-15', '305-812-9104', '890 Harbor Trail', 'Miami', 'FL', 3120);

INSERT INTO products (product_id, name, quantity_in_stock, unit_price) VALUES
(1, 'Foam Dinner Plate', 70, 1.21),
(2, 'Pork - Bacon,back Peameal', 49, 4.65),
(3, 'Lettuce - Romaine, Heart', 38, 3.35),
(4, 'Brocolinni - Gaylan, Chinese', 90, 4.53),
(5, 'Sauce - Ranch Dressing', 94, 1.63),
(6, 'Petit Baguette', 14, 2.39),
(7, 'Sweet Pea Sprouts', 98, 3.29),
(8, 'Island Oasis - Raspberry', 26, 0.74),
(9, 'Longan', 67, 2.26),
(10, 'Broom - Push', 6, 1.09),
(11, 'Apple Juice - Organic', 64, 4.00),
(12, 'Olive Oil - Extra Virgin', 32, 12.00),
(13, 'Espresso Roast Beans', 80, 9.50);

INSERT INTO orders (order_id, customer_id, order_date, status, comments, shipped_date, shipper_id) VALUES
(1, 6, '2019-01-30', 1, NULL, NULL, NULL),
(2, 7, '2018-08-02', 2, NULL, '2018-08-03', 4),
(3, 8, '2017-12-01', 1, NULL, NULL, NULL),
(4, 2, '2017-01-22', 1, NULL, NULL, NULL),
(5, 5, '2017-08-25', 2, NULL, '2017-08-26', 3),
(6, 10, '2018-11-18', 1, 'Handle with care', NULL, NULL),
(7, 2, '2018-09-22', 2, NULL, '2018-09-23', 4),
(8, 5, '2018-06-08', 1, NULL, NULL, NULL),
(9, 10, '2017-07-05', 2, 'Leave at front porch', '2017-07-06', 1),
(10, 6, '2018-04-22', 2, NULL, '2018-04-23', 2),
(11, 5, '2019-03-10', 2, NULL, '2019-03-11', 2),
(12, 5, '2019-05-19', 1, NULL, NULL, NULL),
(13, 12, '2026-11-08', 1, 'Weekend rush order', NULL, NULL),
(14, 12, '2026-11-13', 1, 'Recent order', NULL, NULL);

INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
(1, 4, 4, 3.74),
(2, 1, 2, 9.10),
(2, 4, 4, 1.66),
(2, 6, 2, 2.94),
(3, 3, 10, 9.12),
(4, 3, 7, 6.99),
(4, 10, 7, 6.40),
(5, 2, 3, 9.89),
(6, 1, 4, 8.65),
(6, 2, 4, 3.28),
(6, 3, 4, 7.46),
(6, 5, 1, 3.45),
(7, 3, 7, 9.17),
(8, 5, 2, 6.94),
(8, 8, 2, 8.59),
(9, 6, 5, 7.28),
(10, 1, 10, 6.01),
(10, 9, 9, 4.28),
(11, 11, 2, 4.00),
(12, 12, 1, 12.00),
(13, 13, 3, 9.50),
(14, 11, 5, 4.00);
