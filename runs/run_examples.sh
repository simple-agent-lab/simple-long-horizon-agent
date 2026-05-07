#!/usr/bin/env bash
# Run the recipe demo on the canonical balanced runtime.
# This is the reference smoke run for src/simple_agent_lab/core.py.

set -e
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

python3 scripts/run_tiny_demo.py --recipe all
