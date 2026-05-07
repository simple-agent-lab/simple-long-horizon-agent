#!/usr/bin/env bash
# Run the smallest self-evolution candidate-comparison probe.

set -e
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

python3 examples/design_versions/02_balanced_runtime/evolution_probe.py
