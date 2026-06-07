#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="${1:-$TASK_DIR/workspace}"
mkdir -p "$WORKSPACE"
cp "$TASK_DIR/environment/data/main.py" "$WORKSPACE/main.py"
cp "$TASK_DIR/environment/data/utils.py" "$WORKSPACE/utils.py"
cp "$TASK_DIR/environment/data/config.py" "$WORKSPACE/config.py"
echo "Workspace ready with main.py, utils.py, config.py"
