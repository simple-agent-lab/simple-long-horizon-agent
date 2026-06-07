#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

cat > "$WORKSPACE/fixed.py" << 'PYEOF'
"""Search utilities for finding common elements and duplicates.

Optimized versions using O(n) hash-set-based algorithms.
"""


def find_duplicates(items):
    """Find all duplicate values in a list.

    Returns a sorted list of values that appear more than once.
    """
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return sorted(duplicates)


def find_common_elements(list_a, list_b):
    """Find elements that exist in both lists.

    Returns a sorted list of common elements (no duplicates).
    """
    set_a = set(list_a)
    common = set()
    for item in list_b:
        if item in set_a:
            common.add(item)
    return sorted(common)


def count_unique(items):
    """Count the number of unique elements in a list."""
    return len(set(items))


def has_pair_with_sum(numbers, target):
    """Check if any two numbers in the list add up to the target.

    Returns True if such a pair exists, False otherwise.
    """
    seen = set()
    for num in numbers:
        if target - num in seen:
            return True
        seen.add(num)
    return False
PYEOF

echo "Optimized file written to $WORKSPACE/fixed.py"
