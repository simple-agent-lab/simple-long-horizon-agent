#!/usr/bin/env bash
# Run one SWE-bench instance in a Docker container.
#
# Usage:
#   bash runs/run_swebench_container.sh <instance-id> [max-turns] [run-id] [agent-flavor]
#
# Examples:
#   bash runs/run_swebench_container.sh django__django-12113
#   bash runs/run_swebench_container.sh django__django-12113 20 my-experiment bash_task
#
# Agent flavors (env AGENT_FLAVOR also accepted, CLI arg wins):
#   - bash      : parent agent has only the bash tool (default, current baseline).
#   - bash_task : parent has bash + task; task dispatches a bash worker
#                 ("explorer") sub-agent for context-isolating exploration.
#
# Prerequisites:
#   - Docker running (colima / Docker Desktop / Docker Engine)
#   - SWE-bench instance image already built (see runs/setup_swebench_docker.sh)
#   - .env with OPENAI_MODEL, OPENAI_AUTH_TOKEN, and optionally OPENAI_BASE_URL
#   - Wheelhouse prepared (the script does this automatically if missing)
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

source runs/_python.sh

INSTANCE_ID="${1:?Usage: $0 <instance-id> [max-turns] [run-id] [agent-flavor]}"
MAX_TURNS="${2:-75}"
RUN_ID="${3:-swebench-$(date +%Y%m%d-%H%M%S)}"
AGENT_FLAVOR="${4:-${AGENT_FLAVOR:-bash}}"

INSTANCE_JSONL="evals/out/instance_${INSTANCE_ID}.jsonl"
WHEELHOUSE="evals/out/wheelhouse/cp311-manylinux"

# --- Resolve DOCKER_HOST when callers haven't set one ---
# The Python `docker` SDK defaults to /var/run/docker.sock, which is absent on
# Docker Desktop (macOS) and Colima. Probe known locations in order so the
# launcher works headless under either runtime.
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
  echo "  curl -s 'https://datasets-server.huggingface.co/rows?dataset=princeton-nlp/SWE-bench_Lite&config=default&split=test&offset=0&length=300' \\" >&2
  echo "    | python3 -c \"import json,sys; [print(json.dumps(r['row'])) for r in json.load(sys.stdin)['rows'] if r['row']['instance_id']=='$INSTANCE_ID']\" \\" >&2
  echo "    > $INSTANCE_JSONL" >&2
  exit 1
fi

# --- Verify .env ---
if [ ! -f .env ]; then
  echo "Error: .env not found. Create it with OPENAI_MODEL and OPENAI_AUTH_TOKEN." >&2
  exit 1
fi

# --- Prepare wheelhouse if missing ---
# containerized_agent.py refreshes the current simple-agent-lab wheel on every
# run so a populated wheelhouse cannot silently use stale repo code.
if [ ! -d "$WHEELHOUSE" ] || [ -z "$(ls -A "$WHEELHOUSE" 2>/dev/null)" ]; then
  echo "==> Preparing wheelhouse..."
  "${PYTHON[@]}" - <<'PY'
from pathlib import Path
from evals.swebench.containerized_agent import prepare_wheelhouse
prepare_wheelhouse(Path("evals/out/wheelhouse/cp311-manylinux"))
PY
fi

# --- Resolve image namespace ---
# build_instance_images uses namespace=None which omits the prefix.
# containerized_agent.py defaults to --namespace swebench which prepends "swebench/".
# We detect which naming convention the local images use.
PLAIN_IMAGE="sweb.eval.x86_64.${INSTANCE_ID//__/__}:latest"
if docker image inspect "$PLAIN_IMAGE" >/dev/null 2>&1; then
  NAMESPACE_ARG=""
else
  NAMESPACE_ARG="--namespace swebench"
fi

# --- Run containerized agent ---
echo "==> Running SWE-bench agent"
echo "    instance:    $INSTANCE_ID"
echo "    max-turns:   $MAX_TURNS"
echo "    run-id:      $RUN_ID"
echo "    agent:       $AGENT_FLAVOR"
echo ""

"${PYTHON[@]}" evals/swebench/containerized_agent.py \
  --instance-json "$INSTANCE_JSONL" \
  --instance-id "$INSTANCE_ID" \
  --dataset-name princeton-nlp/SWE-bench_Lite \
  --split test \
  --model-name simple-agent-lab-containerized \
  --provider openai \
  --dotenv .env \
  --max-turns "$MAX_TURNS" \
  --agent-flavor "$AGENT_FLAVOR" \
  --run-id "$RUN_ID" \
  --network-mode host \
  $NAMESPACE_ARG \
  --force
