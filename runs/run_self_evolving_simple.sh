#!/usr/bin/env bash
# Run the config-backed simple self-evolving SWE-bench recipe.

set -euo pipefail
cd "$(dirname "$0")/.."
source runs/_python.sh

if command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run --extra swebench python)
fi

"${PYTHON[@]}" recipes/simple/evolve.py "$@"
