#!/usr/bin/env bash
# Run one SWE-bench instance through the generic Suite framework (ADR 0017).
#
# The agent is launched via run_suite_instance(SwebenchSuite, LocalDockerBackend,
# LocalDirStore), the same primitive every suite uses. For batch / parallel runs
# over a whole split, use runs/run_swebench_verified.sh or run_swebench_pro.sh.
#
# Usage:
#   bash runs/run_swebench_suite.sh <instance-id> [max-turns] [run-id] [agent-flavor]
#
# Examples:
#   bash runs/run_swebench_suite.sh django__django-12113
#   bash runs/run_swebench_suite.sh django__django-12113 20 my-experiment bash_task
#   MCP_CONFIG=evals/swebench/mcp.example.json bash runs/run_swebench_suite.sh sympy__sympy-23824 3 mcp-smoke bash
#
# Prerequisites:
#   - Docker running (colima / Docker Desktop / Docker Engine)
#   - SWE-bench instance image already built (see runs/setup_swebench_docker.sh)
#   - .env with OPENAI_MODEL, OPENAI_AUTH_TOKEN, and optionally OPENAI_BASE_URL
#   - Optional: MCP_CONFIG pointing to a JSON MCP server config
#   - Wheelhouse prepared (the script does this automatically if missing)
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

source runs/_python.sh
source runs/_swebench_uv.sh

INSTANCE_ID="${1:?Usage: $0 <instance-id> [max-turns] [run-id] [agent-flavor]}"
MAX_TURNS="${2:-75}"
RUN_ID="${3:-swebench-$(date +%Y%m%d-%H%M%S)}"
AGENT_FLAVOR="${4:-${AGENT_FLAVOR:-bash}}"

INSTANCE_JSONL="evals/out/swebench/instance_${INSTANCE_ID}.jsonl"
WHEELHOUSE="evals/out/swebench/wheelhouse/cp311-manylinux"

# --- Resolve DOCKER_HOST when callers haven't set one ---
# The Python `docker` SDK defaults to /var/run/docker.sock, which is absent on
# Docker Desktop (macOS) and Colima. Probe known locations in order.
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
  echo "  colima start --cpu 4 --memory 8 --arch aarch64 --vm-type vz --vz-rosetta" >&2
  exit 1
fi

# --- Verify instance JSONL exists ---
if [ ! -f "$INSTANCE_JSONL" ]; then
  echo "Instance file not found: $INSTANCE_JSONL" >&2
  echo "Fetch it first. Example:" >&2
  echo "  bash runs/setup_swebench_docker.sh $INSTANCE_ID" >&2
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
from evals.swebench.harness import prepare_wheelhouse
prepare_wheelhouse(Path("evals/out/swebench/wheelhouse/cp311-manylinux"))
PY
fi

# --- Ensure a Linux uv for the container (sets SWEBENCH_UV_BIN) ---
swebench_ensure_linux_uv

# --- Run the agent through the Suite framework ---
MCP_ARGS=()
if [ -n "${MCP_CONFIG:-}" ]; then
  MCP_ARGS+=(--mcp-config "$MCP_CONFIG")
fi

"${PYTHON[@]}" runs/run_swebench_suite.py "$INSTANCE_ID" \
  --max-turns "$MAX_TURNS" \
  --run-id "$RUN_ID" \
  --agent-flavor "$AGENT_FLAVOR" \
  --uv-binary "$SWEBENCH_UV_BIN" \
  --network-mode host \
  "${MCP_ARGS[@]}" \
  --force
