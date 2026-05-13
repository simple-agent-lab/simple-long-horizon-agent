#!/usr/bin/env bash
# Local mirror of .github/workflows/ci.yml — run the same checks GitHub runs.
#
# Usage:
#   bash runs/run_ci.sh

set -e
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
source "$(dirname "$0")/_python.sh"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required for CI parity. Install: https://docs.astral.sh/uv/" >&2
  exit 1
fi

uv sync --group dev

printf '\n=== ruff format --check . ===\n'
uv run ruff format --check .

printf '\n=== ty check src ===\n'
uv run ty check src

printf '\n=== unittest discover -s tests/unit ===\n'
uv run python -m unittest discover -s tests/unit

printf '\nAll CI checks passed.\n'
