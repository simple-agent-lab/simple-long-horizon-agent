# Task: Fix Runtime Crashes on Edge Cases

You are given a Python file at `workspace/processor.py` that **crashes when given edge-case inputs**. The code works fine with well-formed data but raises unhandled exceptions with incomplete or missing data.

## Bugs to Fix

1. **NoneType access** — `get_user_name(user)` crashes with `TypeError` when `user` is `None`
2. **KeyError** — `get_config_value(config, key)` crashes with `KeyError` when the key doesn't exist in the dict
3. **IndexError** — `get_first_element(items)` crashes with `IndexError` when the list is empty

## Requirements

1. Read `workspace/processor.py` and identify all three crash-causing patterns.
2. Add proper guards/handling so the code never raises unhandled exceptions:
   - `get_user_name(None)` → return `"Unknown"`
   - `get_user_name({"first_name": "John"})` (missing last_name) → use `"Unknown"` for the missing field
   - `get_config_value(config, missing_key)` → return `"default"` as fallback
   - `get_first_element([])` → return `None`
3. The `process_records()` function must handle mixed valid/invalid records gracefully.
4. Do **not** change function signatures — only add defensive handling.

## Output

Save the fixed file as `workspace/fixed.py`.
