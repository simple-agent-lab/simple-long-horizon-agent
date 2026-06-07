# Task: Fix Python Syntax Errors

You are given a Python file at `workspace/buggy.py` that contains **three syntax errors** preventing it from running.

## Bugs to Fix

1. **Missing colon** — an `if` statement is missing its trailing `:`
2. **Bad indentation** — a `return` statement has incorrect indentation
3. **Unclosed parenthesis** — a parenthesized expression is never closed

## Requirements

1. Read `workspace/buggy.py` and identify all three syntax errors.
2. Fix each error so the file is valid Python.
3. The fixed code must parse without errors (`python -c "import ast; ast.parse(open('file').read())"` must succeed).
4. The fixed code must run correctly and produce the expected output:
   - `greet("World")` → `"Hello, World!"`
   - `greet("Alice")` → `"Hi, Alice!"`
   - `calculate_sum([1, 2, 3, 4, 5])` → `15`
   - `format_output([1, 2, 3, 4, 5])` → `"Output: 1, 2, 3, 4, 5"`
5. Do **not** change the logic or function signatures — only fix the syntax.

## Output

Save the fixed file as `workspace/fixed.py`.
