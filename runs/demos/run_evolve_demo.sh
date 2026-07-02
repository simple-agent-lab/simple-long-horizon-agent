#!/usr/bin/env bash
# Run the deterministic evolution-harness demos (no model, no network):
# string hill-climb, then EVOLVE-BLOCK code evolution, then prompt evolution.

set -e
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
export PYTHONDONTWRITEBYTECODE=1
source "$(dirname "$0")/../lib/_python.sh"

"${PYTHON[@]}" scripts/run_evolve_demo.py
"${PYTHON[@]}" scripts/run_code_evolution_demo.py
"${PYTHON[@]}" scripts/run_prompt_evolution_demo.py
