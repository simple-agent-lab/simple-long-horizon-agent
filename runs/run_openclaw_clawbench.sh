#!/bin/bash
# OpenClaw ClawBench 评测 (303 tasks, Pytest verifier)
# 用法: ./run_openclaw_clawbench.sh [sample数量]

cd "$(dirname "$0")/.."
if [ -f .env ]; then
  set -a; source .env; set +a
fi
uv run python -m evals.openclaw \
  --bench clawbench \
  --model gpt-4.1-2025-04-14 \
  --api-url "https://search.bytedance.net/gpt/openapi/online/v2/crawl" \
  --api-key-env GPT_API_KEY \
  --provider azure_openai \
  --max-turns 15 \
  --timeout 300 \
  ${1:+--sample $1} \
  2>&1 | tee docs/exp1_openclaw/assets/logs/run_clawbench_gpt41.log
