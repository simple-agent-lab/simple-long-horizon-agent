#!/usr/bin/env bash
# Run the real model + Docker SWE-bench self-evolving workflow.
#
# Usage:
#   bash runs/run_self_evolving_swebench.sh --run-id real-smoke --train-dataset train.jsonl --test-dataset test.jsonl
#   bash runs/run_self_evolving_swebench.sh --run-id real-smoke --train-dataset train.jsonl --test-dataset test.jsonl --execute
#
# This runner wires the evolution kernel, SAL meta-strategy, SWE-bench Docker
# rollout, train-only promotion, and held-out reporting. Without --execute it
# prints the plan and validates dataset inputs without starting Docker.

set -euo pipefail
cd "$(dirname "$0")/.."
source runs/_python.sh
source runs/_swebench_uv.sh

# SWE-bench self-evolution needs the `docker`/`datasets`/`swebench` SDKs, which
# live in the project's `swebench` optional extra. Pin every `uv run` here to
# that extra so a clean (or auto-synced) venv always has them.
if command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run --extra swebench python)
fi

RUN_ID=""
OUTPUT_ROOT="evals/out/self_evolving/swebench"
DATASET_NAME="princeton-nlp/SWE-bench_Verified"
TRAIN_DATASET=""
TEST_DATASET=""
GENERATIONS=""
ROUNDS=""
BRANCHES=3
META_CONCURRENCY=0
PARENT_SELECTION="score_child_prop"
PARALLEL="auto"
MODEL_NAME="${OPENAI_MODEL:-hyperagents-swebench}"
API_KIND="openai-chat"
MAX_TURNS=75
DOTENV=".env"
WHEELHOUSE="evals/out/swebench/wheelhouse/cp311-manylinux"
EXECUTE=0
RESET=0
MONITOR=0

usage() {
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --run-id ID                         Required reproducible run id.
  --output-root PATH                  Default: evals/out/self_evolving/swebench.
  --dataset-name NAME                 Default: princeton-nlp/SWE-bench_Verified.
  --train-dataset PATH                Train/evolution SWE-bench JSONL file.
  --test-dataset PATH                 Held-out official-scoring SWE-bench JSONL file.
  --rounds N                          Sequential evolution rounds. Default: 4.
  --branches N                        Candidates evaluated concurrently per round. Default: 3.
  --meta-concurrency N                Concurrent meta-agent LLM calls per round. Default: branches.
  --generations N                     Deprecated alias for --rounds (candidates = rounds x branches).
  --parent-selection METHOD           latest|best|score_prop|score_child_prop. Default: score_child_prop.
  --parallel N|auto                   Global Docker worker cap. Default: auto (sized to the Docker VM memory/CPU).
  --model-name NAME                   OPENAI_MODEL value to write into provider.json.
  --api-kind KIND                     openai-chat|openai-responses. Default: openai-chat.
  --dotenv PATH                       Provider env file. Default: .env.
  --wheelhouse PATH                   Container wheelhouse. Default: evals/out/swebench/wheelhouse/cp311-manylinux.
  --max-turns N                       Per-instance agent turn budget. Default: 75.
  --execute                           Run real model + Docker evolution.
  --reset                             Remove the run root before starting.
  --monitor                           Print report for an existing run.
  -h, --help                          Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME="$2"; shift 2 ;;
    --train-dataset) TRAIN_DATASET="$2"; shift 2 ;;
    --test-dataset) TEST_DATASET="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --branches) BRANCHES="$2"; shift 2 ;;
    --meta-concurrency) META_CONCURRENCY="$2"; shift 2 ;;
    --generations) GENERATIONS="$2"; shift 2 ;;
    --parent-selection) PARENT_SELECTION="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --api-kind) API_KIND="$2"; shift 2 ;;
    --dotenv) DOTENV="$2"; shift 2 ;;
    --wheelhouse) WHEELHOUSE="$2"; shift 2 ;;
    --max-turns) MAX_TURNS="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    --reset) RESET=1; shift ;;
    --monitor) MONITOR=1; shift ;;
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

for pair in "max-turns:$MAX_TURNS" "branches:$BRANCHES"; do
  name="${pair%%:*}"
  value="${pair#*:}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --${name} must be a positive integer; got '${value}'." >&2
    exit 2
  fi
done

for pair in "rounds:$ROUNDS" "generations:$GENERATIONS"; do
  name="${pair%%:*}"
  value="${pair#*:}"
  if [ -n "$value" ] && ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --${name} must be a positive integer; got '${value}'." >&2
    exit 2
  fi
done

if [ -n "$META_CONCURRENCY" ] && ! [[ "$META_CONCURRENCY" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --meta-concurrency must be a non-negative integer; got '${META_CONCURRENCY}'." >&2
  exit 2
fi

if [ "$PARALLEL" != "auto" ] && ! [[ "$PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --parallel must be a positive integer or 'auto'; got '${PARALLEL}'." >&2
  exit 2
fi

case "$PARENT_SELECTION" in
  latest|best|score_prop|score_child_prop) ;;
  *) echo "ERROR: --parent-selection must be latest, best, score_prop, or score_child_prop." >&2; exit 2 ;;
esac

case "$API_KIND" in
  openai-chat|openai-responses) ;;
  *) echo "ERROR: --api-kind must be openai-chat or openai-responses." >&2; exit 2 ;;
esac

if [ -z "${DOCKER_HOST:-}" ]; then
  ACTIVE_DOCKER_HOST="$(docker context inspect --format '{{.Endpoints.docker.Host}}' 2>/dev/null || true)"
  if [ -n "$ACTIVE_DOCKER_HOST" ] && [ "$ACTIVE_DOCKER_HOST" != "<no value>" ]; then
    export DOCKER_HOST="$ACTIVE_DOCKER_HOST"
  else
    for SOCK in "$HOME/.docker/run/docker.sock" "$HOME/.colima/default/docker.sock"; do
      if [ -S "$SOCK" ]; then
        export DOCKER_HOST="unix://$SOCK"
        break
      fi
    done
  fi
fi

if [ "$EXECUTE" -eq 1 ]; then
  if [ ! -d "$WHEELHOUSE" ] || [ -z "$(ls -A "$WHEELHOUSE" 2>/dev/null)" ]; then
    echo "==> Preparing wheelhouse..."
    WHEELHOUSE="$WHEELHOUSE" "${PYTHON[@]}" - <<'PY'
import os
from pathlib import Path
from evals.swebench.harness import prepare_wheelhouse
prepare_wheelhouse(Path(os.environ["WHEELHOUSE"]))
PY
  fi
  # Always rebuild the project wheel so containers run the current source, not a
  # stale cached wheel. The evolving recipe mounts this wheelhouse directly, so a
  # stale wheel silently omits new modules (e.g. the evolving container) and the
  # container crashes on import before producing result.json.
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
  scripts/run_self_evolving_swebench.py
  --run-id "$RUN_ID"
  --output-root "$OUTPUT_ROOT"
  --dataset-name "$DATASET_NAME"
  --train-dataset "$TRAIN_DATASET"
  --test-dataset "$TEST_DATASET"
  --branches "$BRANCHES"
  --meta-concurrency "$META_CONCURRENCY"
  --parent-selection "$PARENT_SELECTION"
  --parallel "$PARALLEL"
  --model-name "$MODEL_NAME"
  --api-kind "$API_KIND"
  --dotenv "$DOTENV"
  --wheelhouse "$WHEELHOUSE"
  --uv-binary "${SWEBENCH_UV_BIN:-}"
  --max-turns "$MAX_TURNS"
)

if [ -n "$ROUNDS" ]; then ARGS+=(--rounds "$ROUNDS"); fi
if [ -n "$GENERATIONS" ]; then ARGS+=(--generations "$GENERATIONS"); fi

if [ "$EXECUTE" -eq 1 ]; then ARGS+=(--execute); fi
if [ "$RESET" -eq 1 ]; then ARGS+=(--reset); fi
if [ "$MONITOR" -eq 1 ]; then ARGS+=(--monitor); fi

"${PYTHON[@]}" "${ARGS[@]}"
