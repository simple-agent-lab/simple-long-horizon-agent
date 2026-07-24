#!/usr/bin/env bash
# Local mirror of .github/workflows/ci.yml — run the same checks GitHub runs.
#
# Usage:
#   bash runs/dev/run_ci.sh

set -e

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required for CI parity. Install: https://docs.astral.sh/uv/" >&2
  exit 1
fi

# `--extra mcp` because the MCP integration lives under `src/` and is both
# type-checked (`ty check src`) and unit-tested; the extra keeps the `mcp`
# SDK present so those checks cover it instead of silently skipping.
uv sync --group dev --extra mcp

printf '\n=== ruff format --check . ===\n'
uv run ruff format --check .

printf '\n=== ruff check . ===\n'
uv run ruff check .

printf '\n=== docs lint ===\n'
uv run python scripts/lint_docs.py

printf '\n=== generated docs ===\n'
uv run python docs/decisions/build_index.py --check
uv run python scripts/build_config_reference.py --check

printf '\n=== architecture and environment lint ===\n'
uv run python scripts/arch_lint.py
uv run python scripts/env_lint.py

printf '\n=== ty check src ===\n'
uv run ty check src

printf '\n=== unittest discover -s tests/unit ===\n'
uv run python -m unittest discover -s tests/unit

printf '\n=== run bash agent demo ===\n'
bash runs/demos/run_bash_agent_demo.sh

printf '\nAll CI checks passed.\n'
