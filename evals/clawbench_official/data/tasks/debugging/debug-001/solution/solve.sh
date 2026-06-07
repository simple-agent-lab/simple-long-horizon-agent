#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

python3 -c "
import re

with open('$WORKSPACE/buggy.py', 'r') as f:
    code = f.read()

# Fix 1: Add missing colon after if name == \"World\"
code = code.replace('if name == \"World\"', 'if name == \"World\":')

# Fix 2: Fix bad indentation of 'return total' (2-space indent -> 4-space indent)
code = code.replace('  return total', '    return total')

# Fix 3: Close the unclosed parenthesis in format_output
code = code.replace(
    '        \", \".join(str(v) for v in values)\n    return',
    '        \", \".join(str(v) for v in values)\n    )\n    return'
)

with open('$WORKSPACE/fixed.py', 'w') as f:
    f.write(code)
"

echo "Fixed file written to $WORKSPACE/fixed.py"
