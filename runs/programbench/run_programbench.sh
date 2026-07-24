#!/usr/bin/env bash
# Thin compatibility wrapper; batch behavior lives in runs/_benches/programbench.py.

set -euo pipefail
cd "$(dirname "$0")/../.."
exec uv run --extra programbench python -m runs.run_bench batch programbench "$@"
