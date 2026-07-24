#!/usr/bin/env bash
# Run the deterministic bash-use agent demo on the canonical runtime.

set -e
cd "$(dirname "$0")/../.."
export PYTHONDONTWRITEBYTECODE=1
source runs/lib/_python.sh

"${PYTHON[@]}" -m scripts.run_bash_agent_demo
