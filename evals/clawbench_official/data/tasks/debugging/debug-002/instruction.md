# Task: Fix Logic Bugs in Calculator

You are given a Python file at `workspace/calculator.py` that has **three logic bugs**. The code runs without crashing, but produces incorrect results.

## Bugs to Fix

1. **Off-by-one error** — `factorial(n)` uses `range(1, n)` instead of `range(1, n + 1)`, so it misses multiplying by `n`
2. **Wrong comparison operator** — `is_eligible(age, min_age)` uses `>` instead of `>=`, so exact boundary values are rejected
3. **Integer division** — `divide(a, b)` uses `//` (floor division) instead of `/` (true division), losing precision

## Requirements

1. Read `workspace/calculator.py` and identify all three logic bugs.
2. Fix each bug so the functions return correct results.
3. Do **not** change function signatures or add new functions.
4. Expected correct behavior:
   - `factorial(5)` → `120` (not `24`)
   - `factorial(0)` → `1`
   - `is_eligible(18, 18)` → `True` (not `False`)
   - `divide(7, 2)` → `3.5` (not `3`)
   - `compute_stats([80, 90, 70])` should use the corrected helper functions

## Output

Save the fixed file as `workspace/fixed.py`.
