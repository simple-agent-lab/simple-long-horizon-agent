#!/usr/bin/env bash
# Run the context-compression eval suite.
#
#   bash runs/run_compression_eval.sh            # offline only (no model)
#   bash runs/run_compression_eval.sh --live     # + live token/fidelity checks
#
# Extra args (e.g. --live, --run-id NAME) pass straight through to the runner.
# Live checks read OPENAI_MODEL / OPENAI_AUTH_TOKEN / OPENAI_BASE_URL / API_KIND
# from the environment (or a sourced .env).

set -e
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src:."
export PYTHONDONTWRITEBYTECODE=1
source "$(dirname "$0")/_python.sh"

"${PYTHON[@]}" -m evals.compression.run_eval "$@"
