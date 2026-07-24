#!/usr/bin/env bash
# Run the multimodal MCP agent demo: an agent calls a tool on a local MCP
# server (stdio) that returns a PNG image, and the image flows back through
# the runtime tool boundary as an ImageBlock.
#
# Needs the optional `mcp` extra, installed for this run by uv.

set -e
cd "$(dirname "$0")/../.."
export PYTHONDONTWRITEBYTECODE=1

exec uv run --extra mcp python -m scripts.run_mcp_agent_demo "$@"
