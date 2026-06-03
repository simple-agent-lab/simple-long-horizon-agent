#!/usr/bin/env bash
# Evaluate SWE-bench Verified, Multilingual, or Pro predictions.
#
# Verified (requires swebench installed + Docker):
#   bash runs/eval_swebench.sh --run-official \
#     --predictions evals/out/swebench_predictions.jsonl
#
# Pro (uses official SWE-bench_Pro-os harness):
#   bash runs/eval_swebench.sh --pro --run-official \
#     --predictions evals/out/swebench_pro/swebench_pro_predictions.jsonl \
#     --instances evals/out/swebench_pro/instance_all-test.jsonl
#
# Multilingual (uses the standard SWE-bench harness):
#   bash runs/eval_swebench.sh --multilingual --run-official \
#     --predictions evals/out/swebench_multilingual/swebench_multilingual_predictions.jsonl
#
# Normalize existing results (no Docker needed):
#   bash runs/eval_swebench.sh --pro \
#     --predictions evals/out/swebench_pro/swebench_pro_predictions.jsonl \
#     --instance-results-jsonl evals/out/swebench_pro_eval/instance_results.jsonl

set -euo pipefail
cd "$(dirname "$0")/.."
source runs/_python.sh

exec "${PYTHON[@]}" evals/swebench/evaluate_predictions.py "$@"
