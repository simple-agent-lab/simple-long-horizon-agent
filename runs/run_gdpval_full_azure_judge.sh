#!/usr/bin/env bash
# Run GDPVal solver + GSB judge with the AzureOpenAI judge endpoint.
#
# Secrets are intentionally not hard-coded. Either export them before running:
#
#   export OPENAI_AUTH_TOKEN=...
#   export JUDGE_OPENAI_AUTH_TOKEN=...
#   bash runs/run_gdpval_full_azure_judge.sh
#
# Or put them in a local ignored env file and pass ENV_FILE:
#
#   ENV_FILE=evals/out/gdpval/.runtime_env_gdpval_azure_judge.env \
#     bash runs/run_gdpval_full_azure_judge.sh
#
# Useful overrides:
#
#   LIMIT=10 bash runs/run_gdpval_full_azure_judge.sh
#   DETACH=0 LIMIT=10 bash runs/run_gdpval_full_azure_judge.sh
#   RUN_ID=my-gdpval-run bash runs/run_gdpval_full_azure_judge.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -n "${ENV_FILE:-}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

RUN_ID="${RUN_ID:-gdpval-full-$(date +%Y%m%d-%H%M%S)-azure-judge}"

# Solver endpoint.
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.5-2026-04-24}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://aidp.bytedance.net/api/modelhub/online}"
export OPENAI_REASONING_EFFORT="${OPENAI_REASONING_EFFORT:-high}"
export OPENAI_SESSION_ID="${OPENAI_SESSION_ID:-$RUN_ID}"
: "${OPENAI_AUTH_TOKEN:?Set OPENAI_AUTH_TOKEN for the solver endpoint.}"
export OPENAI_AUTH_TOKEN

# Judge endpoint. Setting AZURE_* makes the openai-chat adapter use AzureOpenAI.
export JUDGE_OPENAI_MODEL="${JUDGE_OPENAI_MODEL:-gpt-5-2025-08-07}"
export JUDGE_AZURE_OPENAI_ENDPOINT="${JUDGE_AZURE_OPENAI_ENDPOINT:-https://aidp.bytedance.net/api/modelhub/online/v2/crawl}"
export JUDGE_AZURE_OPENAI_API_VERSION="${JUDGE_AZURE_OPENAI_API_VERSION:-2024-03-01-preview}"
: "${JUDGE_OPENAI_AUTH_TOKEN:?Set JUDGE_OPENAI_AUTH_TOKEN for the judge endpoint.}"
export JUDGE_OPENAI_AUTH_TOKEN
unset JUDGE_OPENAI_BASE_URL

IMAGE="${IMAGE:-hub.byted.org/boyuan/gdpval-agent-base:v20260604.8}"
PULL="${PULL:-never}"
CONCURRENCY="${CONCURRENCY:-10}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-$CONCURRENCY}"
JUDGE_TOOL_MODE="${JUDGE_TOOL_MODE:-hybrid}"
JUDGE_SEMANTIC_MAX_ATTEMPTS="${JUDGE_SEMANTIC_MAX_ATTEMPTS:-2}"
DETACH="${DETACH:-1}"

ARGS=(
  uv run --with docker --with datasets
  python runs/run_gdpval.py
  --run-id "$RUN_ID"
  --backend local-docker
  --image "$IMAGE"
  --pull "$PULL"
  --provider openai
  --api-kind openai-responses
  --concurrency "$CONCURRENCY"
  --judge
  --judge-mode gsb
  --judge-tool-mode "$JUDGE_TOOL_MODE"
  --judge-provider openai
  --judge-api-kind openai-chat
  --judge-concurrency "$JUDGE_CONCURRENCY"
  --judge-semantic-max-attempts "$JUDGE_SEMANTIC_MAX_ATTEMPTS"
)

if [ -n "${LIMIT:-}" ]; then
  ARGS+=(--limit "$LIMIT")
fi

if [ -n "${TASK_IDS:-}" ]; then
  read -r -a TASK_ID_ARGS <<< "$TASK_IDS"
  ARGS+=(--task-ids "${TASK_ID_ARGS[@]}")
fi

echo "GDPVal run id: $RUN_ID"
echo "solver model: $OPENAI_MODEL"
echo "solver session: $OPENAI_SESSION_ID"
echo "judge model:  $JUDGE_OPENAI_MODEL"
echo "judge client: AzureOpenAI"
echo "image:        $IMAGE"
echo "concurrency:  solver=$CONCURRENCY judge=$JUDGE_CONCURRENCY"

if [ "$DETACH" = "1" ]; then
  mkdir -p evals/out/gdpval
  LOG="evals/out/gdpval/${RUN_ID}.tmux.log"
  SOCKET="sal_${RUN_ID}"
  printf -v COMMAND "%q " "${ARGS[@]}"
  tmux -L "$SOCKET" new-session -d -s "$RUN_ID" -c "$PWD" \
    "$COMMAND >> $(printf "%q" "$LOG") 2>&1"
  echo "detached:     tmux -L $SOCKET attach -t $RUN_ID"
  echo "log:          $LOG"
else
  "${ARGS[@]}"
fi
