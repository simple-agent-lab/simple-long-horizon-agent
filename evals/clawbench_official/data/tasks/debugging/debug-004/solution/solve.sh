#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

python3 -c "
with open('$WORKSPACE/utils.py', 'r') as f:
    code = f.read()

# Fix: replace all occurrences of get_settings with get_config
code = code.replace('get_settings', 'get_config')

with open('$WORKSPACE/utils.py', 'w') as f:
    f.write(code)
"

echo "Fixed utils.py — replaced get_settings with get_config"
