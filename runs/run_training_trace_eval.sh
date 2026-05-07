#!/usr/bin/env bash
# Collect trajectories, evaluate them, then export training examples.

set -e
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

python3 scripts/collect_design_version_trajectories.py
python3 evals/evaluate_design_version_traces.py
python3 scripts/export_training_examples.py
