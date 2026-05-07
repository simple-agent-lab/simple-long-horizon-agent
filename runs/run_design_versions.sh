#!/usr/bin/env bash
# Run the three architecture-version sketches.

set -e
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

printf '\n01_functional_loop\n'
printf '==================\n'
python3 examples/design_versions/01_functional_loop/demo.py

printf '\n02_balanced_runtime\n'
printf '===================\n'
python3 examples/design_versions/02_balanced_runtime/demo.py

printf '\n03_event_runtime\n'
printf '================\n'
python3 examples/design_versions/03_event_runtime/demo.py
