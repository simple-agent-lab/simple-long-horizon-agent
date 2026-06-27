#!/usr/bin/env bash
# Pull the ProgramBench Docker image(s) for one instance.
#
# ProgramBench images live under the `programbench/` Docker org with the name
# rule `programbench/<instance-id with __ -> _1776_>`. Inference uses the
# `:task_cleanroom` tag; the official scorer (evaluate_submissions.py) uses
# `:task`. Image names are x86_64 Linux.
#
# Usage:
#   bash runs/programbench/setup_programbench.sh <instance-id> [--scoring]
#
# Examples:
#   bash runs/programbench/setup_programbench.sh abishekvashok__cmatrix.5c082c6
#   bash runs/programbench/setup_programbench.sh sitkevij__hex.61ae69b --scoring
set -euo pipefail

cd "$(dirname "$0")/../.."

INSTANCE_ID="${1:?Usage: $0 <instance-id> [--scoring]}"
WANT_SCORING=0
if [ "${2:-}" = "--scoring" ]; then
  WANT_SCORING=1
fi

# --- Resolve DOCKER_HOST when callers haven't set one ---
if [ -z "${DOCKER_HOST:-}" ]; then
  for SOCK in \
    "$HOME/.docker/run/docker.sock" \
    "$HOME/.colima/default/docker.sock"; do
    if [ -S "$SOCK" ]; then
      export DOCKER_HOST="unix://$SOCK"
      break
    fi
  done
fi

if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker is not running. Start Docker or Colima first." >&2
  exit 1
fi

# Image name rule: programbench/<instance-id with __ replaced by _1776_>.
IMAGE_BASE="programbench/${INSTANCE_ID//__/_1776_}"

echo "==> Pulling inference image ${IMAGE_BASE}:task_cleanroom"
docker pull "${IMAGE_BASE}:task_cleanroom"

if [ "$WANT_SCORING" -eq 1 ]; then
  echo "==> Pulling scoring image ${IMAGE_BASE}:task"
  docker pull "${IMAGE_BASE}:task" \
    || echo "warn: ':task' pull failed (only needed for evaluate_submissions.py)" >&2
fi

echo "Done. Run the agent with:"
echo "  bash runs/programbench/run_programbench.sh ${INSTANCE_ID}"
