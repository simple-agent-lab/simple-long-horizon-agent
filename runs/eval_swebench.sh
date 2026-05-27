#!/usr/bin/env bash
# Evaluate SWE-bench Verified or Pro predictions.
#
# Verified (requires swebench installed + Docker):
#   bash runs/eval_swebench.sh --run-official \
#     --predictions evals/out/swebench/verified/predictions/swebench_predictions.jsonl
#
# Pro (uses official SWE-bench_Pro-os harness):
#   bash runs/eval_swebench.sh --pro --run-official \
#     --predictions evals/out/swebench/pro/predictions/swebench_pro_predictions.jsonl \
#     --instances evals/out/swebench/pro/instances/all-test.jsonl
#
# Normalize existing results (no Docker needed):
#   bash runs/eval_swebench.sh --pro \
#     --predictions evals/out/swebench/pro/predictions/swebench_pro_predictions.jsonl \
#     --instance-results-jsonl evals/out/swebench/pro/eval_results/instance_results.jsonl

set -euo pipefail
cd "$(dirname "$0")/.."
source runs/_python.sh

exec "${PYTHON[@]}" evals/swebench/evaluate_predictions.py "$@"
