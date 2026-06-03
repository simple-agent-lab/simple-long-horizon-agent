#!/usr/bin/env bash
#
# Launch the simple-agent-lab terminal UI (and, through it, the gateway).
#
# Architecture note: the TS UI process OWNS the gateway — it spawns
# `uv run python -m simple_agent_lab.tui_gateway.entry` over stdio JSON-RPC.
# So there is nothing to start separately; launching the UI brings the gateway
# up with it (and tears it down on exit). This script just makes that one
# command convenient: it installs the Node deps on first run and forwards any
# flags (e.g. --provider openai) straight to the UI.
#
# Usage:
#   scripts/run_tui.sh                     # fake provider (no credentials)
#   scripts/run_tui.sh --provider openai   # live model; reads .env from repo root
#   scripts/run_tui.sh --provider openai --max-turns 20
#
# Env:
#   SAL_PYTHON   explicit python interpreter for the gateway (default: uv run python)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_DIR="$ROOT/studio/tui"

# --- preflight ---------------------------------------------------------------
if ! command -v node >/dev/null 2>&1; then
  echo "error: 'node' not found. Install Node >= 22.19 (pi-tui requires it)." >&2
  exit 1
fi
node_major="$(node -p 'process.versions.node.split(".")[0]')"
if [ "$node_major" -lt 22 ]; then
  echo "error: Node $node_major detected; pi-tui needs Node >= 22.19." >&2
  exit 1
fi
# The UI defaults to spawning the gateway via `uv run`; warn if neither uv nor
# an explicit interpreter is available.
if [ -z "${SAL_PYTHON:-}" ] && ! command -v uv >/dev/null 2>&1; then
  echo "warning: 'uv' not found and SAL_PYTHON unset — the gateway may fail to start." >&2
  echo "         install uv, or set SAL_PYTHON=/path/to/python with the project installed." >&2
fi

# --- deps --------------------------------------------------------------------
if [ ! -d "$UI_DIR/node_modules" ]; then
  echo ">>> installing UI dependencies (first run)…" >&2
  (cd "$UI_DIR" && npm install --no-fund --no-audit)
fi

# --- launch ------------------------------------------------------------------
# `npm run dev` uses tsx to run the TS directly (no build step). The `--` passes
# the remaining flags through to entry.ts.
echo ">>> starting UI (it will spawn the gateway)…" >&2
cd "$UI_DIR"
exec npm run --silent dev -- "$@"
