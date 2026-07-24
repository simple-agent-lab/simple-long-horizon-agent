#!/usr/bin/env bash
# Thin compatibility wrapper; batch behavior lives in runs/_benches/swebench.py.

set -euo pipefail
cd "$(dirname "$0")/../.."
exec uv run --extra swebench python -m runs.run_bench batch swebench "$@"
