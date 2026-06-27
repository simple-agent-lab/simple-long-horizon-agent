#!/usr/bin/env bash
# Run the deterministic bash-use agent demo on the canonical runtime.

set -e
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
export PYTHONDONTWRITEBYTECODE=1
source "$(dirname "$0")/../lib/_python.sh"

"${PYTHON[@]}" scripts/run_bash_agent_demo.py
