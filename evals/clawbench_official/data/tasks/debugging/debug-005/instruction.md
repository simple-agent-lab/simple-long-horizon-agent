# Task: Performance Bug Fix — O(n²) to O(n)

You are given a Python file at `workspace/slow_search.py` that contains four search/collection utility functions. All four use **O(n²) nested loops** where an O(n) hash-set-based approach would suffice.

## Functions to Optimize

1. **`find_duplicates(items)`** — Finds all values that appear more than once. Currently uses nested `for i / for j` loops.
2. **`find_common_elements(list_a, list_b)`** — Finds elements in both lists. Currently uses nested iteration.
3. **`count_unique(items)`** — Counts distinct elements. Currently uses a manual linear search for uniqueness.
4. **`has_pair_with_sum(numbers, target)`** — Checks if any two numbers sum to `target`. Currently uses nested `for i / for j` loops.

## Requirements

1. Rewrite each function to use **O(n) time complexity** with `set` operations.
2. **Preserve correctness** — every function must produce identical results to the original for all inputs.
3. `find_duplicates` and `find_common_elements` must still return **sorted lists** with no duplicates.
4. Do **not** change function signatures.
5. The optimized version must be **significantly faster** than the original on large inputs (10,000+ elements).

## Output

Save the optimized file as `workspace/fixed.py`.
