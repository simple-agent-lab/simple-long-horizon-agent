#!/usr/bin/env bash
# Run the agent on SWE-bench Verified through the Suite framework (ADR generic-containerized-eval-framework).
#
# Usage:
#   bash runs/run_swebench_verified.sh                          # default: sympy__sympy-23824
#   bash runs/run_swebench_verified.sh django__django-16379     # one custom instance
#   bash runs/run_swebench_verified.sh --all --parallel 4       # full split, 4 at a time
#
# Requires Docker, `uv sync --extra swebench`, and a .env with provider credentials.
# Downloading uncached dataset records uses `datasets`.
#
# The shared driver lives in runs/_swebench_run.sh; this launcher only sets the
# Verified dataset constants.

set -euo pipefail
cd "$(dirname "$0")/.."
source runs/_python.sh
source runs/_swebench_uv.sh
source runs/_swebench_run.sh

SWEBENCH_DATASET="princeton-nlp/SWE-bench_Verified"
SWEBENCH_DEFAULT_INSTANCE_ID="sympy__sympy-23824"
SWEBENCH_RUN_ROOT="evals/out/swebench"
SWEBENCH_MODEL_NAME="simple-agent-lab-verified"
SWEBENCH_MAX_TURNS=150
SWEBENCH_RUN_ID_PREFIX="verified"
SWEBENCH_LABEL="SWE-bench Verified"

swebench_main "$@"
