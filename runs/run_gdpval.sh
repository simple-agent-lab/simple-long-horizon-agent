#!/usr/bin/env bash
# Run GDPVal solver instances through Simple Agent Lab.
#
# Usage:
#   bash runs/run_gdpval.sh [path/to/gdpval.jsonl] [task-id]
#   JUDGE=1 bash runs/run_gdpval.sh [path/to/gdpval.jsonl] [task-id]
set -euo pipefail

cd "$(dirname "$0")/.."

source runs/_python.sh

INPUT="${1:-}"
TASK_ID="${2:-}"
RUN_ID="${RUN_ID:-gdpval-$(date +%Y%m%d-%H%M%S)}"

ARGS=(
  runs/run_gdpval.py
  --run-id "$RUN_ID"
)

if [ -n "$INPUT" ]; then
  ARGS+=("$INPUT")
fi

if [ -n "$TASK_ID" ]; then
  ARGS+=(--task-ids "$TASK_ID")
fi

if [ "${JUDGE:-}" = "1" ]; then
  ARGS+=(--judge)
fi

"${PYTHON[@]}" "${ARGS[@]}"
