#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

default_timestamp="$(date +%Y%m%d-%H%M%S)"
run_timestamp="${RUN_TIMESTAMP:-$default_timestamp}"
job_name="${HARBOR_JOB_NAME:-sal-tb21-bash-$run_timestamp}"

export REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}astral.sh,releases.astral.sh"
export no_proxy="${no_proxy:+${no_proxy},}astral.sh,releases.astral.sh"

extra_args=()
if [[ "${HARBOR_DRY_RUN:-0}" == "1" ]]; then
  extra_args+=(--dry-run)
fi

uv run python -m runs.run_bench harbor \
  --dataset terminal-bench/terminal-bench-2-1 \
  --job-name "$job_name" \
  --n-concurrent 10 \
  --max-turns 150 \
  --timeout-multiplier 3 \
  --agent-setup-timeout-multiplier 3 \
  --agent-kwarg install_timeout_sec=900 \
  --agent-env SAL_BASH_MAX_TIMEOUT_SECONDS=300 \
  --setup-proxy-from-env \
  --agent-flavor bash \
  --debug \
  --json \
  "${extra_args[@]}"
