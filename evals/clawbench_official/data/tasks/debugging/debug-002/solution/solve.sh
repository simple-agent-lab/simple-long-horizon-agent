#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

python3 -c "
with open('$WORKSPACE/calculator.py', 'r') as f:
    code = f.read()

# Fix 1: Off-by-one in factorial — range(1, n) -> range(1, n + 1)
code = code.replace('range(1, n)', 'range(1, n + 1)')

# Fix 2: Wrong comparison — age > min_age -> age >= min_age
code = code.replace('return age > min_age', 'return age >= min_age')

# Fix 3: Integer division — a // b -> a / b
code = code.replace('return a // b', 'return a / b')

with open('$WORKSPACE/fixed.py', 'w') as f:
    f.write(code)
"

echo "Fixed file written to $WORKSPACE/fixed.py"
