#!/usr/bin/env bash
# Serve the Observatory trace viewer with live scanning of evals/out/.
#
# Usage:
#   bash runs/demos/run_trace_viewer.sh                       # http://127.0.0.1:8765
#   bash runs/demos/run_trace_viewer.sh --port 9000           # override port
#   bash runs/demos/run_trace_viewer.sh --dir evals/out/swebench_container_runs
#
# The server is stdlib-only Python (no dependencies). It scans the target
# directory recursively, classifies each .jsonl / .json file (trajectory,
# eval_result, prediction, instance_metadata, other), and exposes a polling
# API the viewer uses to surface new traces as evals finish.

set -e
export PYTHONDONTWRITEBYTECODE=1
source "$(dirname "$0")/../lib/_python.sh"

"${PYTHON[@]}" "$(dirname "$0")/../studio/trace-viewer/serve.py" "$@"
