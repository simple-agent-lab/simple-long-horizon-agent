#!/usr/bin/env bash
# Run the agent on SWE-bench Multilingual through the Suite framework (ADR generic-containerized-eval-framework).
#
# Usage:
#   bash runs/run_swebench_multilingual.sh                          # default: first test instance
#   bash runs/run_swebench_multilingual.sh owner__repo-123          # one custom instance
#   bash runs/run_swebench_multilingual.sh --all --parallel 4       # full split, 4 at a time
#
# Requires Docker, `uv sync --extra swebench`, and a .env with provider credentials.
# Downloading uncached dataset records uses `datasets`.
#
# The shared driver lives in runs/_swebench_run.sh; this launcher only sets the
# Multilingual dataset constants.

set -euo pipefail
cd "$(dirname "$0")/.."
source runs/_python.sh
source runs/_swebench_uv.sh
source runs/_swebench_run.sh

SWEBENCH_DATASET="SWE-bench/SWE-bench_Multilingual"
# Empty: the driver resolves the first instance of the split on demand.
SWEBENCH_DEFAULT_INSTANCE_ID="${SWE_BENCH_MULTILINGUAL_DEFAULT_INSTANCE_ID:-}"
SWEBENCH_RUN_ROOT="evals/out/swebench_multilingual"
SWEBENCH_MODEL_NAME="simple-agent-lab-multilingual"
SWEBENCH_MAX_TURNS=150
SWEBENCH_RUN_ID_PREFIX="multilingual"
SWEBENCH_LABEL="SWE-bench Multilingual"

swebench_main "$@"
