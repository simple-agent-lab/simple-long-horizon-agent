#!/usr/bin/env bash
# Run the simple self-evolving SWE-bench recipe (recipes/simple/evolve.py).
#
# Usage:
#   bash runs/run_self_evolving_simple.sh --run-id simple-smoke --train-dataset train.jsonl --test-dataset test.jsonl
#   bash runs/run_self_evolving_simple.sh --run-id simple-real --train-dataset train.jsonl --test-dataset test.jsonl --execute
#
# The ergonomics showcase: a short recipe that runs the evolution kernel
# sequentially (Experiment.run, best parent selection, one container at a time).
# This wrapper just prepares Docker + the container wheelhouse, then runs it.
# Without --execute it prints the plan and validates inputs without Docker.

set -euo pipefail
cd "$(dirname "$0")/.."
source runs/_python.sh
source runs/_swebench_uv.sh
source runs/_docker.sh

# The recipe needs the `docker`/`datasets`/`swebench` SDKs from the project's
# `swebench` optional extra. Pin every `uv run` here to that extra.
if command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run --extra swebench python)
fi

RUN_ID=""
OUTPUT_ROOT="evals/out/self_evolving/simple"
TRAIN_DATASET=""
TEST_DATASET=""
ROUNDS=4
MAX_TURNS=75
DOTENV=".env"
WHEELHOUSE="evals/out/swebench/wheelhouse/cp311-manylinux"
EXECUTE=0

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --run-id ID                         Required reproducible run id.
  --output-root PATH                  Default: evals/out/self_evolving/simple.
  --train-dataset PATH                Train/evolution SWE-bench JSONL file.
  --test-dataset PATH                 Held-out scoring SWE-bench JSONL file.
  --rounds N                          Sequential evolution rounds. Default: 4.
  --max-turns N                       Per-instance agent turn budget. Default: 75.
  --dotenv PATH                       Provider env file. Default: .env.
  --wheelhouse PATH                   Container wheelhouse. Default: evals/out/swebench/wheelhouse/cp311-manylinux.
  --execute                           Run real model + Docker evolution.
  -h, --help                          Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --train-dataset) TRAIN_DATASET="$2"; shift 2 ;;
    --test-dataset) TEST_DATASET="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --max-turns) MAX_TURNS="$2"; shift 2 ;;
    --dotenv) DOTENV="$2"; shift 2 ;;
    --wheelhouse) WHEELHOUSE="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) echo "ERROR: unexpected positional argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$RUN_ID" ]; then
  echo "ERROR: --run-id is required." >&2
  usage >&2
  exit 2
fi

if [ -z "$TRAIN_DATASET" ] || [ -z "$TEST_DATASET" ]; then
  echo "ERROR: --train-dataset and --test-dataset are required." >&2
  usage >&2
  exit 2
fi

for pair in "rounds:$ROUNDS" "max-turns:$MAX_TURNS"; do
  name="${pair%%:*}"
  value="${pair#*:}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --${name} must be a positive integer; got '${value}'." >&2
    exit 2
  fi
done

docker_resolve_host

if [ "$EXECUTE" -eq 1 ]; then
  if ! docker_ensure_running; then
    echo "ERROR: Docker daemon is not reachable and could not be started automatically." >&2
    echo "Start Docker Desktop or Colima manually, then re-run (or set SAL_DOCKER_AUTOSTART=0 to skip auto-start)." >&2
    exit 1
  fi
  if [ ! -d "$WHEELHOUSE" ] || [ -z "$(ls -A "$WHEELHOUSE" 2>/dev/null)" ]; then
    echo "==> Preparing wheelhouse..."
    WHEELHOUSE="$WHEELHOUSE" "${PYTHON[@]}" - <<'PY'
import os
from pathlib import Path
from evals.swebench.harness import prepare_wheelhouse
prepare_wheelhouse(Path(os.environ["WHEELHOUSE"]))
PY
  fi
  # Always rebuild the project wheel so containers run the current source (the
  # evolving recipe mounts this wheelhouse directly; a stale wheel omits new
  # modules and the container crashes on import before producing result.json).
  echo "==> Refreshing project wheel..."
  WHEELHOUSE="$WHEELHOUSE" "${PYTHON[@]}" - <<'PY'
import os
from pathlib import Path
from evals.swebench.harness import prepare_project_wheel
prepare_project_wheel(Path(os.environ["WHEELHOUSE"]))
PY
  swebench_ensure_linux_uv
fi

ARGS=(
  recipes/simple/evolve.py
  --run-id "$RUN_ID"
  --output-root "$OUTPUT_ROOT"
  --train-dataset "$TRAIN_DATASET"
  --test-dataset "$TEST_DATASET"
  --rounds "$ROUNDS"
  --max-turns "$MAX_TURNS"
  --dotenv "$DOTENV"
  --wheelhouse "$WHEELHOUSE"
)

if [ "$EXECUTE" -eq 1 ]; then ARGS+=(--execute); fi

"${PYTHON[@]}" "${ARGS[@]}"
