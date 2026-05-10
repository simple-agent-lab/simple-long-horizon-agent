#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source runs/_python.sh

"${PYTHON[@]}" evals/swebench/prepare_workspace.py --skip-clone --force
"${PYTHON[@]}" evals/swebench/collect_trajectories.py \
  --workspace evals/out/swebench_workspaces/sympy__sympy-20590/repo \
  --allow-empty-patch
"${PYTHON[@]}" evals/swebench/evaluate_predictions.py --allow-missing-reports
