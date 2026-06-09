#!/usr/bin/env bash
# Run GDPVal solver + GSB judge with the current Responses API endpoints.
#
# Secrets are intentionally not hard-coded. Either export them before running:
#
#   export OPENAI_AUTH_TOKEN=...
#   export OPENAI_BASE_URL=...
#   export JUDGE_OPENAI_AUTH_TOKEN=...
#   export JUDGE_OPENAI_BASE_URL=...
#   bash runs/run_gdpval_full_azure_judge.sh
#
# JINA_API_KEY is optional for WebFetch. SERPER_API_KEY is required when
# ENABLE_WEB_TOOLS is enabled, because WebSearch uses Serper.
#
# Or put them in a local ignored env file and pass ENV_FILE:
#
#   ENV_FILE=evals/out/gdpval/.runtime_env_gdpval_responses_judge.env \
#     bash runs/run_gdpval_full_azure_judge.sh
#
# Useful overrides:
#
#   LIMIT=10 bash runs/run_gdpval_full_azure_judge.sh
#   DETACH=0 LIMIT=10 bash runs/run_gdpval_full_azure_judge.sh
#   RUN_ID=my-gdpval-run bash runs/run_gdpval_full_azure_judge.sh
#   TASK_IDS="task-id-1 task-id-2" LIMIT=2 bash runs/run_gdpval_full_azure_judge.sh
#   INPUT=/path/to/gdpval.jsonl bash runs/run_gdpval_full_azure_judge.sh
#   ENABLE_WEB_TOOLS=1 SERPER_API_KEY=... LIMIT=10 bash runs/run_gdpval_full_azure_judge.sh
#   DRY_RUN=1 LIMIT=10 bash runs/run_gdpval_full_azure_judge.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -n "${ENV_FILE:-}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

RUN_ID="${RUN_ID:-gdpval-full-$(date +%Y%m%d-%H%M%S)-responses-judge}"

# Dataset/input selection. Leave INPUT empty for the default Hugging Face source.
INPUT="${INPUT:-${1:-}}"
HF_DATASET="${HF_DATASET:-openai/gdpval}"
HF_SPLIT="${HF_SPLIT:-train}"
HF_CACHE_DIR="${HF_CACHE_DIR:-evals/out/gdpval/hf_cache}"
RUN_ROOT="${RUN_ROOT:-evals/out/gdpval}"

# Solver endpoint.
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.4-2026-03-05}"
# Base URL is the SDK base; the Responses client appends /responses.
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL for the solver endpoint.}"
export OPENAI_BASE_URL
export OPENAI_REASONING_EFFORT="${OPENAI_REASONING_EFFORT:-high}"
export OPENAI_SESSION_ID="${OPENAI_SESSION_ID:-$RUN_ID}"
: "${OPENAI_AUTH_TOKEN:?Set OPENAI_AUTH_TOKEN for the solver endpoint.}"
export OPENAI_AUTH_TOKEN

# Judge endpoint. Use the Responses adapter so encrypted reasoning history is
# requested and replayed across judge turns.
export JUDGE_OPENAI_MODEL="${JUDGE_OPENAI_MODEL:-gpt-5.2-2025-12-11}"
: "${JUDGE_OPENAI_BASE_URL:?Set JUDGE_OPENAI_BASE_URL for the judge endpoint.}"
export JUDGE_OPENAI_BASE_URL
export JUDGE_OPENAI_REASONING_EFFORT="${JUDGE_OPENAI_REASONING_EFFORT:-high}"
export JUDGE_OPENAI_SESSION_ID="${JUDGE_OPENAI_SESSION_ID:-$RUN_ID-judge}"
: "${JUDGE_OPENAI_AUTH_TOKEN:?Set JUDGE_OPENAI_AUTH_TOKEN for the judge endpoint.}"
export JUDGE_OPENAI_AUTH_TOKEN
unset JUDGE_AZURE_OPENAI_ENDPOINT
unset JUDGE_AZURE_OPENAI_API_VERSION
unset JUDGE_AZURE_OPENAI_LOGID

IMAGE="${IMAGE:-hub.byted.org/boyuan/gdpval-agent-base:v20260604.8}"
PULL="${PULL:-never}"
NETWORK_MODE="${NETWORK_MODE:-host}"
CONCURRENCY="${CONCURRENCY:-10}"
MAX_TURNS="${MAX_TURNS:-100}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
JUDGE_MODE="${JUDGE_MODE:-gsb}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-50}"
JUDGE_MAX_TURNS="${JUDGE_MAX_TURNS:-100}"
JUDGE_MAX_ATTEMPTS="${JUDGE_MAX_ATTEMPTS:-1}"
JUDGE_TOOL_MODE="${JUDGE_TOOL_MODE:-hybrid}"
JUDGE_SEMANTIC_MAX_ATTEMPTS="${JUDGE_SEMANTIC_MAX_ATTEMPTS:-2}"
GDPVAL_GSB_DIRECTION_ATTEMPTS="${GDPVAL_GSB_DIRECTION_ATTEMPTS:-1}"
export GDPVAL_GSB_DIRECTION_ATTEMPTS
ENABLE_WEB_TOOLS="${ENABLE_WEB_TOOLS:-0}"
DETACH="${DETACH:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$ENABLE_WEB_TOOLS" != "0" ]; then
  : "${SERPER_API_KEY:?Set SERPER_API_KEY for solver WebSearch, or set ENABLE_WEB_TOOLS=0.}"
  export SERPER_API_KEY
  if [ -n "${SERPER_ENDPOINT:-}" ]; then
    export SERPER_ENDPOINT
  fi
  if [ -n "${JINA_API_KEY:-}" ]; then
    export JINA_API_KEY
  fi
  if [ -n "${JINA_ENDPOINT:-}" ]; then
    export JINA_ENDPOINT
  fi
fi

ARGS=(
  uv run --with docker --with datasets
  python runs/run_gdpval.py
)

if [ -n "$INPUT" ]; then
  ARGS+=("$INPUT")
fi

ARGS+=(
  --run-id "$RUN_ID"
  --run-root "$RUN_ROOT"
  --hf-dataset "$HF_DATASET"
  --hf-split "$HF_SPLIT"
  --hf-cache-dir "$HF_CACHE_DIR"
  --backend local-docker
  --image "$IMAGE"
  --pull "$PULL"
  --network-mode "$NETWORK_MODE"
  --provider openai
  --api-kind openai-responses
  --max-turns "$MAX_TURNS"
  --max-attempts "$MAX_ATTEMPTS"
  --concurrency "$CONCURRENCY"
  --judge
  --judge-mode "$JUDGE_MODE"
  --judge-tool-mode "$JUDGE_TOOL_MODE"
  --judge-provider openai
  --judge-api-kind openai-responses
  --judge-max-turns "$JUDGE_MAX_TURNS"
  --judge-max-attempts "$JUDGE_MAX_ATTEMPTS"
  --judge-concurrency "$JUDGE_CONCURRENCY"
  --judge-semantic-max-attempts "$JUDGE_SEMANTIC_MAX_ATTEMPTS"
)

if [ "$ENABLE_WEB_TOOLS" = "0" ]; then
  ARGS+=(--disable-web-tools)
fi

if [ -n "${LIMIT:-}" ]; then
  ARGS+=(--limit "$LIMIT")
fi

if [ -n "${TASK_IDS:-}" ]; then
  read -r -a TASK_ID_ARGS <<< "$TASK_IDS"
  ARGS+=(--task-ids "${TASK_ID_ARGS[@]}")
fi

if [ -n "${REFERENCE_ROOT:-}" ]; then
  ARGS+=(--reference-root "$REFERENCE_ROOT")
fi

if [ -n "${DELIVERABLE_ROOT:-}" ]; then
  ARGS+=(--deliverable-root "$DELIVERABLE_ROOT")
fi

if [ -n "${JUDGE_RUN_ID:-}" ]; then
  ARGS+=(--judge-run-id "$JUDGE_RUN_ID")
fi

if [ -n "${WHEELHOUSE:-}" ]; then
  ARGS+=(--wheelhouse "$WHEELHOUSE")
fi

if [ -n "${WHEELHOUSE_MOUNT:-}" ]; then
  ARGS+=(--wheelhouse-mount "$WHEELHOUSE_MOUNT")
fi

if [ -n "${UV_BINARY:-}" ]; then
  ARGS+=(--uv-binary "$UV_BINARY")
fi

if [ -n "${PLATFORM:-}" ]; then
  ARGS+=(--platform "$PLATFORM")
fi

if [ "${PREPARE_WHEELHOUSE:-0}" = "1" ]; then
  ARGS+=(--prepare-wheelhouse)
fi

if [ "${KEEP_CONTAINER:-0}" = "1" ]; then
  ARGS+=(--keep-container)
fi

if [ "${INCLUDE_KNOWN_BAD_TASKS:-0}" = "1" ]; then
  ARGS+=(--include-known-bad-tasks)
fi

if [ "${INCLUDE_EMPTY_DELIVERABLES:-0}" = "1" ]; then
  ARGS+=(--include-empty-deliverables)
fi

echo "GDPVal run id: $RUN_ID"
echo "input:        ${INPUT:-huggingface:$HF_DATASET/$HF_SPLIT}"
echo "run root:     $RUN_ROOT"
echo "solver model: $OPENAI_MODEL"
echo "solver session: $OPENAI_SESSION_ID"
echo "judge model:  $JUDGE_OPENAI_MODEL"
echo "judge api:    openai-responses"
echo "judge mode:   $JUDGE_MODE"
echo "judge tools:  $JUDGE_TOOL_MODE"
echo "web tools:    $ENABLE_WEB_TOOLS"
echo "image:        $IMAGE"
echo "turns:        solver=$MAX_TURNS judge=$JUDGE_MAX_TURNS"
echo "attempts:     solver=$MAX_ATTEMPTS judge=$JUDGE_MAX_ATTEMPTS semantic=$JUDGE_SEMANTIC_MAX_ATTEMPTS gsb_dirs=$GDPVAL_GSB_DIRECTION_ATTEMPTS"
echo "concurrency:  solver=$CONCURRENCY judge=$JUDGE_CONCURRENCY"

printf -v COMMAND "%q " "${ARGS[@]}"
if [ "$DRY_RUN" = "1" ]; then
  echo "command:      $COMMAND"
  exit 0
fi

if [ "$DETACH" = "1" ]; then
  mkdir -p evals/out/gdpval
  LOG="evals/out/gdpval/${RUN_ID}.tmux.log"
  SOCKET="sal_${RUN_ID}"
  tmux -L "$SOCKET" new-session -d -s "$RUN_ID" -c "$PWD" \
    "$COMMAND >> $(printf "%q" "$LOG") 2>&1"
  echo "detached:     tmux -L $SOCKET attach -t $RUN_ID"
  echo "log:          $LOG"
else
  "${ARGS[@]}"
fi
