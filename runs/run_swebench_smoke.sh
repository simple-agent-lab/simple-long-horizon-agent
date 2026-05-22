#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source runs/_python.sh

"${PYTHON[@]}" -m unittest \
  tests.unit.test_swebench_patch_extract \
  tests.unit.test_swebench_containerized_agent \
  tests.unit.test_swebench_evaluate_predictions
