#!/usr/bin/env bash
# Run the agent on SWE-bench Pro through the Suite framework (ADR generic-containerized-eval-framework).
#
# Usage:
#   bash runs/run_swebench_pro.sh                                            # default instance
#   bash runs/run_swebench_pro.sh instance_navidrome__navidrome-8e640bb8...  # one custom instance
#   bash runs/run_swebench_pro.sh --all --parallel 4                         # full split, 4 at a time
#
# Requires Docker, `uv sync --extra swebench`, and a .env with provider credentials.
# Downloading uncached dataset records uses `datasets`.
#
# The shared driver lives in runs/_swebench_run.sh; this launcher only sets the
# Pro dataset constants.

set -euo pipefail
cd "$(dirname "$0")/.."
source runs/_python.sh
source runs/_swebench_uv.sh
source runs/_swebench_run.sh

SWEBENCH_DATASET="ScaleAI/SWE-bench_Pro"
SWEBENCH_DEFAULT_INSTANCE_ID="instance_navidrome__navidrome-8e640bb8580affb7e0ea6225c0bbe240186b6b08"
SWEBENCH_RUN_ROOT="evals/out/swebench_pro"
SWEBENCH_MODEL_NAME="simple-agent-lab-pro"
SWEBENCH_MAX_TURNS=40
SWEBENCH_RUN_ID_PREFIX="pro"
SWEBENCH_LABEL="SWE-bench Pro"

swebench_main "$@"
