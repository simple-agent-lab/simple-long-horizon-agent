#!/usr/bin/env bash
# Run one ProgramBench instance through the generic Suite framework.
#
# The agent is launched via run_suite_instance(ProgrambenchSuite,
# LocalDockerBackend, LocalDirStore). The container is online (so the agent's
# model calls work) but each agent bash command runs in sealed user + network
# namespaces (`programbench-reverse-engineering-adapter`). For batch /
# parallel runs, use runs/run_programbench.sh.
#
# Usage:
#   bash runs/run_programbench_suite.sh <instance-id> [max-turns] [run-id] [flags...]
#
# Examples:
#   bash runs/run_programbench_suite.sh abishekvashok__cmatrix.5c082c6
#   bash runs/run_programbench_suite.sh sitkevij__hex.61ae69b 200 my-experiment
#   bash runs/run_programbench_suite.sh abishekvashok__cmatrix.5c082c6 --dynamic-workflow
#
# Prerequisites:
#   - Docker running (colima / Docker Desktop / Docker Engine)
#   - ProgramBench image pulled (see runs/setup_programbench.sh)
#   - `uv sync --extra programbench`
#   - .env with OPENAI_MODEL, OPENAI_AUTH_TOKEN, and optionally OPENAI_BASE_URL
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

source runs/_python.sh
source runs/_swebench_uv.sh

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <instance-id> [max-turns] [run-id] [flags...]" >&2
  exit 2
fi
INSTANCE_ID="$1"
shift
MAX_TURNS=150
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  MAX_TURNS="$1"
  shift
fi
RUN_ID="programbench-$(date +%Y%m%d-%H%M%S)"
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  RUN_ID="$1"
  shift
fi
EXTRA_ARGS=("$@")

WHEELHOUSE="evals/out/programbench/wheelhouse/cp311-manylinux"

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

# --- Verify Docker ---
if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker is not running. Start Docker or Colima first." >&2
  exit 1
fi

# --- Verify .env ---
if [ ! -f .env ]; then
  echo "Error: .env not found. Create it with OPENAI_MODEL and OPENAI_AUTH_TOKEN." >&2
  exit 1
fi

# --- Prepare wheelhouse if missing ---
# The Python entry rebuilds the current project wheel on every run; this only
# does the one-time full populate of third-party wheels when the dir is empty.
if [ ! -d "$WHEELHOUSE" ] || [ -z "$(ls -A "$WHEELHOUSE" 2>/dev/null)" ]; then
  echo "==> Preparing wheelhouse..."
  "${PYTHON[@]}" - <<'PY'
from pathlib import Path
from evals.programbench.harness import prepare_wheelhouse
prepare_wheelhouse(Path("evals/out/programbench/wheelhouse/cp311-manylinux"))
PY
fi

# --- Ensure a Linux uv for the container (sets SWEBENCH_UV_BIN) ---
# ProgramBench images are language toolchain images (c/rust/go/...) that need not
# ship Python 3.11; uv builds the agent venv the cp311 wheels target.
swebench_ensure_linux_uv

# --- Run the agent through the Suite framework ---
"${PYTHON[@]}" runs/run_programbench_suite.py "$INSTANCE_ID" \
  --max-turns "$MAX_TURNS" \
  --run-id "$RUN_ID" \
  --uv-binary "$SWEBENCH_UV_BIN" \
  --network-mode host \
  --force \
  "${EXTRA_ARGS[@]}"

echo ""
echo "Score this run with:"
echo "  ${PYTHON[*]} evals/programbench/evaluate_submissions.py --run-id $RUN_ID"
