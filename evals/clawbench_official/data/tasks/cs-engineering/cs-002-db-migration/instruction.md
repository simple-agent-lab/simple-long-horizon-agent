# SQLite Database Schema Migration

## Description

You are provided with a SQLite database file `app.db` and a migration specification file `migration_spec.json` in the workspace. Your task is to:

1. Read the migration operations from `migration_spec.json`. The operations can be of the following types:
   - `add_table`: Add a new table with specified columns.
   - `add_column`: Add a new column to an existing table.
   - `add_index`: Add an index on specified columns of a table.
   - `rename_column`: Rename a column in an existing table.

2. Apply these migration operations sequentially to the SQLite database `app.db`.

3. Verify that the schema changes have been applied correctly by querying the database schema.

4. Write a report file `migration_report.json` that contains:
   - `applied_operations`: List of operations that were applied.
   - `verification_results`: For each operation, whether it was successfully applied (`true` or `false`).

## Files

- `workspace/migration_spec.json`: JSON file containing a list of migration operations.
- `workspace/app.db`: The SQLite database file to be migrated.
- `workspace/migration_report.json`: The output JSON report file you must create.

## Migration Operations Format

Each operation in `migration_spec.json` is an object with a mandatory `operation` field and other fields depending on the operation type:

- `add_table`:
  ```json
  {
    "operation": "add_table",
    "table_name": "table_name",
    "columns": [
      {"name": "col1", "type": "TEXT", "constraints": "PRIMARY KEY"},
      {"name": "col2", "type": "INTEGER", "constraints": "NOT NULL"}
    ]
  }
  ```

- `add_column`:
  ```json
  {
    "operation": "add_column",
    "table_name": "existing_table",
    "column": {"name": "new_col", "type": "TEXT", "constraints": "DEFAULT 'N/A'"}
  }
  ```

- `add_index`:
  ```json
  {
    "operation": "add_index",
    "table_name": "existing_table",
    "index_name": "idx_col",
    "columns": ["col1", "col2"]
  }
  ```

- `rename_column`:
  ```json
  {
    "operation": "rename_column",
    "table_name": "existing_table",
    "old_name": "old_col",
    "new_name": "new_col"
  }
  ```

## Requirements

- Use SQLite commands and SQL syntax to perform the migrations.
- Handle errors gracefully; if an operation cannot be applied, mark it as failed in the verification results but continue with the next operations.
- The verification step must confirm the existence and correctness of the schema changes.
- The output JSON report must be valid and well-structured.

## Evaluation

Your solution will be evaluated by:

- Correctness of applied migrations.
- Accuracy of verification results.
- Proper formatting and completeness of the `migration_report.json`.

Good luck!