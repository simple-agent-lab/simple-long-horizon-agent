#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
export WORKSPACE

mkdir -p "$WORKSPACE"

# Create initial SQLite database with some tables and data
sqlite3 "$WORKSPACE/app.db" <<EOF
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL,
    email TEXT
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    amount REAL NOT NULL
);

INSERT INTO users (username, email) VALUES
('alice', 'alice@example.com'),
('bob', 'bob@example.com'),
('carol', 'carol@example.com');

INSERT INTO orders (user_id, amount) VALUES
(1, 19.99),
(2, 5.49),
(1, 15.00);
EOF

# Generate migration_spec.json with multiple operations
python3 - <<'PYEOF'
import json, os

WORKSPACE = os.environ.get('WORKSPACE', os.getcwd())

migration_spec = [
    {
        "operation": "add_table",
        "table_name": "products",
        "columns": [
            {"name": "product_id", "type": "INTEGER", "constraints": "PRIMARY KEY"},
            {"name": "name", "type": "TEXT", "constraints": "NOT NULL"},
            {"name": "price", "type": "REAL", "constraints": "NOT NULL"}
        ]
    },
    {
        "operation": "add_column",
        "table_name": "users",
        "column": {"name": "signup_date", "type": "TEXT", "constraints": "DEFAULT NULL"}
    },
    {
        "operation": "add_index",
        "table_name": "orders",
        "index_name": "idx_orders_user_id",
        "columns": ["user_id"]
    },
    {
        "operation": "rename_column",
        "table_name": "orders",
        "old_name": "amount",
        "new_name": "total_amount"
    },
    {
        "operation": "add_column",
        "table_name": "products",
        "column": {"name": "stock", "type": "INTEGER", "constraints": "DEFAULT 0"}
    },
    {
        "operation": "add_index",
        "table_name": "products",
        "index_name": "idx_products_name",
        "columns": ["name"]
    }
]

with open(os.path.join(WORKSPACE, "migration_spec.json"), "w") as f:
    json.dump(migration_spec, f, indent=2)
PYEOF
