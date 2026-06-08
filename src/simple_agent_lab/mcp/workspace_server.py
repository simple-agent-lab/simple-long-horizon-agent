"""Small stdio MCP server for workspace inspection tests.

The server is intentionally read-only and dependency-light beyond the optional
`mcp` package. It runs in whichever workspace is set as the process cwd.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

server = FastMCP("workspace", log_level="WARNING")


@server.tool(description="List files in the current workspace directory.")
def list_files(relative_path: str = ".", max_entries: int = 50) -> str:
    """Return a compact newline-separated directory listing."""

    root = Path.cwd().resolve()
    target = (root / relative_path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return f"Refusing to list path outside workspace: {relative_path}"
    if not target.exists():
        return f"Path does not exist: {relative_path}"
    if target.is_file():
        return str(target.relative_to(root))

    entries = sorted(target.iterdir(), key=lambda path: (not path.is_dir(), path.name))
    limited = entries[: max(1, int(max_entries))]
    lines = [
        f"{entry.relative_to(root)}{'/' if entry.is_dir() else ''}" for entry in limited
    ]
    if len(entries) > len(limited):
        lines.append(f"... {len(entries) - len(limited)} more")
    return "\n".join(lines) if lines else "(empty directory)"


if __name__ == "__main__":
    server.run()
