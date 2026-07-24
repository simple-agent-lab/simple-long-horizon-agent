#!/usr/bin/env bash
# Run the multimodal MCP agent demo: an agent calls a tool on a local MCP
# server (stdio) that returns a PNG image, and the image flows back through
# the runtime tool boundary as an ImageBlock.
#
# Needs the optional `mcp` extra. With uv this is handled via `--extra mcp`;
# without uv, install it first: pip install "simple-agent-lab[mcp]".

set -e
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
export PYTHONDONTWRITEBYTECODE=1

if command -v uv >/dev/null 2>&1; then
  uv run --extra mcp python scripts/run_mcp_agent_demo.py "$@"
else
  python3 scripts/run_mcp_agent_demo.py "$@"
fi
