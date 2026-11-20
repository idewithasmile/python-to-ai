-- Schema and synthetic sample dataset for the enterprise relational database
-- Supports multi-table inner joins, self joins, outer joins, compound keys,
-- set operations, and ER diagram implementation.

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS invoices;
DROP TABLE IF EXISTS clients;
DROP TABLE IF EXISTS payment_methods;
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS offices;
DROP TABLE IF EXISTS order_item_notes;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS order_statuses;
DROP TABLE IF EXISTS shippers;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;

-- 1. Core Retail Store Tables

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

CREATE TABLE shippers (
    shipper_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE order_statuses (
    order_status_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    status INTEGER NOT NULL,
    comments TEXT,
    shipped_date TEXT,
    shipper_id INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id) ON UPDATE CASCADE,
    FOREIGN KEY (status) REFERENCES order_statuses (order_status_id) ON UPDATE CASCADE,
    FOREIGN KEY (shipper_id) REFERENCES shippers (shipper_id) ON UPDATE CASCADE
);

CREATE TABLE order_items (
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    PRIMARY KEY (order_id, product_id),
    FOREIGN KEY (order_id) REFERENCES orders (order_id) ON UPDATE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products (product_id) ON UPDATE CASCADE
);

CREATE TABLE order_item_notes (
    note_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    note TEXT NOT NULL,
    FOREIGN KEY (order_id, product_id) REFERENCES order_items (order_id, product_id) ON UPDATE CASCADE
);

-- 2. Human Resources Hierarchy Tables (Self Join Practice)

CREATE TABLE offices (
    office_id INTEGER PRIMARY KEY,
    address TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT NOT NULL
);

CREATE TABLE employees (
    employee_id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    job_title TEXT NOT NULL,
    salary INTEGER NOT NULL,
    reports_to INTEGER,
    office_id INTEGER NOT NULL,
    FOREIGN KEY (reports_to) REFERENCES employees (employee_id) ON UPDATE CASCADE,
    FOREIGN KEY (office_id) REFERENCES offices (office_id) ON UPDATE CASCADE
);

-- 3. Invoicing and Payments System Tables (Multi-Way Join Practice)

CREATE TABLE payment_methods (
    payment_method_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE clients (
    client_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    address TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    phone TEXT NOT NULL
);

CREATE TABLE invoices (
    invoice_id INTEGER PRIMARY KEY,
    number TEXT NOT NULL,
    client_id INTEGER NOT NULL,
    invoice_total REAL NOT NULL,
    payment_total REAL NOT NULL DEFAULT 0.00,
    invoice_date TEXT NOT NULL,
    due_date TEXT NOT NULL,
    payment_date TEXT,
    FOREIGN KEY (client_id) REFERENCES clients (client_id) ON UPDATE CASCADE
);

CREATE TABLE payments (
    payment_id INTEGER PRIMARY KEY,
    client_id INTEGER NOT NULL,
    invoice_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    amount REAL NOT NULL,
    payment_method INTEGER NOT NULL,
    FOREIGN KEY (client_id) REFERENCES clients (client_id) ON UPDATE CASCADE,
    FOREIGN KEY (invoice_id) REFERENCES invoices (invoice_id) ON UPDATE CASCADE,
    FOREIGN KEY (payment_method) REFERENCES payment_methods (payment_method_id) ON UPDATE CASCADE
);

-- Data Seeding

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

INSERT INTO shippers (shipper_id, name) VALUES
(1, 'Hettinger LLC'),
(2, 'Cognizant Fast'),
(3, 'Apex Logistics'),
(4, 'Speedy Express'),
(5, 'Federal Freight');

INSERT INTO order_statuses (order_status_id, name) VALUES
(1, 'Processed'),
(2, 'Shipped'),
(3, 'Delivered');

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
(9, 10, 1, 4.55),
(10, 1, 10, 9.40),
(10, 9, 9, 3.39),
(11, 3, 3, 9.89),
(12, 1, 2, 1.21),
(13, 11, 5, 4.00),
(14, 13, 2, 9.50);

INSERT INTO order_item_notes (note_id, order_id, product_id, note) VALUES
(1, 1, 4, 'Fragile - pack with double bubble wrap'),
(2, 2, 1, 'Standard bulk dinner plate packaging'),
(3, 2, 4, 'Temperature sensitive - cold pack included'),
(4, 2, 6, 'Fresh bakery item - same day dispatch'),
(5, 5, 2, 'Meat product - maintain refrigeration below 4C'),
(6, 6, 1, 'Gift wrapped per customer comment'),
(7, 10, 1, 'Commercial restaurant tier packing'),
(8, 13, 11, 'Organic certified seal inspected');

INSERT INTO offices (office_id, address, city, state) VALUES
(1, '55 East Beaver Street', 'Columbus', 'OH'),
(2, '2395 University Parkway', 'Sarasota', 'FL'),
(3, '350 5th Avenue', 'New York', 'NY'),
(4, '600 Montgomery Street', 'San Francisco', 'CA');

INSERT INTO employees (employee_id, first_name, last_name, job_title, salary, reports_to, office_id) VALUES
(37270, 'Yovonnda', 'Maggard', 'Executive Vice President', 110000, NULL, 1),
(33391, 'Darcy', 'Northey', 'Account Executive', 62000, 37270, 1),
(37851, 'Sayer', 'Gullis', 'Development Lead', 94000, 37270, 1),
(40448, 'Mindy', 'Crissil', 'Financial Analyst', 77000, 37270, 1),
(56274, 'Keriann', 'Alloisi', 'Staff Software Engineer', 82000, 37851, 1),
(63196, 'Alaster', 'Scutchin', 'Software Engineer', 68000, 37851, 1),
(67009, 'North', 'Denham', 'Junior Analyst', 45000, 40448, 1),
(70123, 'Elena', 'Rostova', 'Associate Product Manager', 58000, 33391, 2);

INSERT INTO payment_methods (payment_method_id, name) VALUES
(1, 'Credit Card'),
(2, 'Cash'),
(3, 'PayPal'),
(4, 'Wire Transfer');

INSERT INTO clients (client_id, name, address, city, state, phone) VALUES
(1, 'Vinte Enterprises', '3 Nevada Parkway', 'Syracuse', 'NY', '315-252-7305'),
(2, 'MyWorks Co', '34267 Cordelia Center', 'Atlanta', 'GA', '404-327-8177'),
(3, 'Yadel Logistics', '0963 Mayer Road', 'San Francisco', 'CA', '415-144-6037'),
(4, 'Kwimbee Tech', '81674 Westerfield Circle', 'Waltham', 'MA', '781-932-9754'),
(5, 'Topico Systems', '53406 Sunbrook Alley', 'Portland', 'OR', '503-455-8291'),
(6, 'Zephyr Analytics', '7224 Continental Pass', 'Denver', 'CO', '303-918-4421');

INSERT INTO invoices (invoice_id, number, client_id, invoice_total, payment_total, invoice_date, due_date, payment_date) VALUES
(1, 'INV-2019-001', 2, 101.79, 101.79, '2019-03-09', '2019-03-29', '2019-03-12'),
(2, 'INV-2019-002', 5, 175.32, 175.32, '2019-06-11', '2019-07-01', '2019-06-25'),
(3, 'INV-2019-003', 3, 147.99, 0.00, '2019-07-31', '2019-08-20', NULL),
(4, 'INV-2019-004', 1, 152.21, 152.21, '2019-03-08', '2019-03-28', '2019-03-27'),
(5, 'INV-2019-005', 5, 169.36, 169.36, '2019-07-18', '2019-08-07', '2019-07-23'),
(6, 'INV-2019-006', 4, 180.45, 0.00, '2019-08-01', '2019-08-21', NULL),
(7, 'INV-2019-007', 3, 130.44, 130.44, '2019-01-11', '2019-01-31', '2019-01-15'),
(8, 'INV-2019-008', 2, 167.22, 167.22, '2019-02-12', '2019-03-04', '2019-02-18'),
(9, 'INV-2019-009', 1, 148.41, 148.41, '2019-04-03', '2019-04-23', '2019-04-10'),
(10, 'INV-2019-010', 6, 210.00, 100.00, '2019-09-01', '2019-09-21', '2019-09-15');

INSERT INTO payments (payment_id, client_id, invoice_id, date, amount, payment_method) VALUES
(1, 5, 2, '2019-06-25', 175.32, 1),
(2, 1, 4, '2019-03-27', 152.21, 1),
(3, 2, 8, '2019-02-18', 167.22, 2),
(4, 5, 5, '2019-07-23', 169.36, 1),
(5, 3, 7, '2019-01-15', 130.44, 3),
(6, 2, 1, '2019-03-12', 101.79, 1),
(7, 1, 9, '2019-04-10', 148.41, 1),
(8, 6, 10, '2019-09-15', 100.00, 4);
