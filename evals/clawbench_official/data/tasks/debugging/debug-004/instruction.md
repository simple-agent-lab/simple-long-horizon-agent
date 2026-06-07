# Task: Multi-File Bug Localization

You are given three Python files in the workspace that form a small application:

- `workspace/main.py` — the entry point
- `workspace/utils.py` — utility functions
- `workspace/config.py` — configuration constants

Running `main.py` fails with an `ImportError`. The bug is **not** in `main.py` or `config.py` — it is in `utils.py`, which imports a function name that doesn't exist in `config.py`.

## Bug Description

`utils.py` imports `get_settings` from `config.py`, but `config.py` only exports `get_config`. This wrong import name causes `utils.py` (and therefore `main.py`) to fail on import.

Additionally, `utils.py` calls `get_settings()` internally, which also needs to be changed to `get_config()`.

## Requirements

1. Read all three files and identify the root cause.
2. Fix `utils.py` so that:
   - It imports `get_config` instead of `get_settings`
   - All internal calls use `get_config()` instead of `get_settings()`
3. Save all three files (even if unchanged) to the workspace:
   - `workspace/main.py`
   - `workspace/utils.py`
   - `workspace/config.py`
4. After fixing, `python workspace/main.py` must run successfully and print the expected report.

## Expected Output

When `main.py` runs with `process_data({"users": 150, "orders": 340, "revenue": 52000}, "json")`:
- `validate_format("json")` returns `True`
- The output report is printed with the app name `DataProcessor`

## Output

Save all fixed files to `workspace/`.
