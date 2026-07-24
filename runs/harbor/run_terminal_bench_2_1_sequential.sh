#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

default_timestamp="$(date +%Y%m%d-%H%M%S)"
export RUN_TIMESTAMP="${RUN_TIMESTAMP:-$default_timestamp}"

bash_job_name="${BASH_JOB_NAME:-sal-tb21-bash-$RUN_TIMESTAMP}"
bash_task_job_name="${BASH_TASK_JOB_NAME:-sal-tb21-bash-task-$RUN_TIMESTAMP}"

echo "Starting bash experiment: $bash_job_name"
HARBOR_JOB_NAME="$bash_job_name" \
  bash runs/harbor/run_terminal_bench_2_1_bash.sh

echo "Bash experiment finished; starting bash_task experiment: $bash_task_job_name"
HARBOR_JOB_NAME="$bash_task_job_name" \
  bash runs/harbor/run_terminal_bench_2_1_bash_task.sh
