#!/usr/bin/env bash
# Run ProgramBench instances in batch through the Suite framework.
#
# Usage:
#   bash runs/programbench/run_programbench.sh                            # default instance
#   bash runs/programbench/run_programbench.sh sitkevij__hex.61ae69b      # one instance
#   bash runs/programbench/run_programbench.sh --all --parallel 4         # whole task set
#   bash runs/programbench/run_programbench.sh --filter 'sitkevij.*'      # regex on instance_id
#   bash runs/programbench/run_programbench.sh --slice 0:5                # first 5 instances
#
# Requires Docker, `uv sync --extra programbench`, and a .env with provider
# credentials. Score afterwards with evals/programbench/evaluate_submissions.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
source runs/lib/_python.sh
source runs/lib/_swebench_uv.sh

DEFAULT_INSTANCE_ID="abishekvashok__cmatrix.5c082c6"
RUN_ROOT="evals/out/programbench"
MAX_TURNS=1000
RUN_ID="programbench-$(date +%Y%m%d-%H%M%S)"
RUN_ALL=0
PARALLEL=1
FILTER=""
SLICE=""
POSITIONAL=()

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --all)
      RUN_ALL=1
      shift
      ;;
    --parallel)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --parallel requires a positive integer." >&2
        exit 2
      fi
      PARALLEL="$2"
      shift 2
      ;;
    --filter)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --filter requires a regex." >&2
        exit 2
      fi
      FILTER="$2"
      shift 2
      ;;
    --slice)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --slice requires a spec (e.g. 0:5)." >&2
        exit 2
      fi
      SLICE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if ! [[ "$PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --parallel must be a positive integer; got ${PARALLEL@Q}." >&2
  exit 2
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
if [ ! -f .env ]; then
  echo "Error: .env not found. Create it with OPENAI_MODEL and OPENAI_AUTH_TOKEN." >&2
  exit 1
fi

# --- Resolve the instance id list from the installed programbench task set ---
INSTANCE_IDS=()
if [ "$RUN_ALL" -eq 1 ] || [ -n "$FILTER" ] || [ -n "$SLICE" ]; then
  mapfile -t INSTANCE_IDS < <(
    FILTER="$FILTER" SLICE="$SLICE" "${PYTHON[@]}" - <<'PY'
import os
from evals.programbench.harness import load_instances
for inst in load_instances(
    filter_spec=os.environ.get("FILTER", ""),
    slice_spec=os.environ.get("SLICE", ""),
):
    print(inst["instance_id"])
PY
  )
elif [ "${#POSITIONAL[@]}" -gt 0 ]; then
  INSTANCE_IDS=("${POSITIONAL[@]}")
else
  INSTANCE_IDS=("$DEFAULT_INSTANCE_ID")
fi

if [ "${#INSTANCE_IDS[@]}" -eq 0 ]; then
  echo "No instances selected." >&2
  exit 1
fi

echo "=== ProgramBench batch ==="
echo "Run ID: $RUN_ID"
echo "Instances: ${#INSTANCE_IDS[@]}"
echo "Parallel: $PARALLEL"
echo ""

# --- Prepare wheelhouse + Linux uv once before launching the batch ---
echo "Preparing wheelhouse and Linux uv once before launching batch..."
swebench_ensure_linux_uv
"${PYTHON[@]}" - <<'PY'
from evals.programbench.harness import DEFAULT_WHEELHOUSE, prepare_wheelhouse
prepare_wheelhouse(DEFAULT_WHEELHOUSE)
PY

run_one() {
  local instance_id="$1"
  "${PYTHON[@]}" runs/run_bench.py programbench "$instance_id" \
    --max-turns "$MAX_TURNS" \
    --run-id "$RUN_ID" \
    --run-root "$RUN_ROOT" \
    --uv-binary "$SWEBENCH_UV_BIN" \
    --network-mode host \
    --force
}

running_jobs() {
  jobs -pr | wc -l | tr -d ' '
}

FAIL=0
for instance_id in "${INSTANCE_IDS[@]}"; do
  while [ "$(running_jobs)" -ge "$PARALLEL" ]; do
    wait -n || FAIL=$((FAIL + 1))
  done
  log="${RUN_ROOT}/${RUN_ID}/${instance_id}.log"
  mkdir -p "$(dirname "$log")"
  echo "Starting: ${instance_id}"
  run_one "$instance_id" > "$log" 2>&1 &
done

while [ "$(running_jobs)" -gt 0 ]; do
  wait -n || FAIL=$((FAIL + 1))
done

echo ""
echo "Outputs: ${RUN_ROOT}/${RUN_ID}/"
echo "Score with:"
echo "  ${PYTHON[*]} evals/programbench/evaluate_submissions.py --run-id $RUN_ID --workers $PARALLEL"
if [ "$FAIL" -gt 0 ]; then
  echo "Failed runs: $FAIL" >&2
  exit 1
fi
