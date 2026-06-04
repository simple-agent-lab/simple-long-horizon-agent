"""GDPVal judge MCP tool registry and run-scoped connection helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast

from simple_agent_lab.llm.provider import Provider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.tools import AgentTool

from .tools import make_gdpval_tools

JudgeToolMode = Literal["local", "mcp", "hybrid"]

JUDGE_TOOL_MODES: tuple[JudgeToolMode, ...] = ("local", "mcp", "hybrid")
DEFAULT_JUDGE_TOOL_MODE: JudgeToolMode = "hybrid"
MCP_WARNING_FILE = "_gdpval_mcp_warnings.json"
GDPVAL_JUDGE_MCP_TOOL_ALLOW_TERMS: tuple[str, ...] = (
    "read",
    "list",
    "get",
    "extract",
    "search",
    "inspect",
    "parse",
    "profile",
    "query",
    "analyze",
    "describe",
    "info",
    "metadata",
    "text",
    "content",
    "sheet",
    "workbook",
    "document",
    "paragraph",
    "table",
    "slide",
    "pdf",
)
GDPVAL_JUDGE_MCP_TOOL_DENY_TERMS: tuple[str, ...] = (
    "write",
    "create",
    "add",
    "insert",
    "update",
    "delete",
    "remove",
    "replace",
    "edit",
    "format",
    "set",
    "merge",
    "split",
    "copy",
    "move",
    "upload",
    "download",
    "save",
    "export",
    "import",
    "render",
)
GDPVAL_PPT_JUDGE_MCP_EXTRA_ALLOW_TERMS: tuple[str, ...] = (
    # PPT MCP tools are stateful: some read operations first require loading a
    # .pptx file into the MCP session, and visual checks may need rendering.
    "open",
    "load",
    "import",
    "render",
    "export",
    "image",
    "thumbnail",
)
GDPVAL_PPT_JUDGE_MCP_HARD_DENY_TERMS: tuple[str, ...] = (
    "write",
    "create",
    "add",
    "insert",
    "update",
    "delete",
    "remove",
    "replace",
    "edit",
    "format",
    "save",
)
_PPT_MCP_SERVER_NAMES = {"ppt", "ppt_mcp_server"}


def normalize_judge_tool_mode(value: Any) -> JudgeToolMode:
    text = str(value or DEFAULT_JUDGE_TOOL_MODE).strip().lower()
    if text not in JUDGE_TOOL_MODES:
        choices = ", ".join(JUDGE_TOOL_MODES)
        raise ValueError(
            f"unknown GDPVal judge tool mode {value!r}; expected {choices}"
        )
    return cast(JudgeToolMode, text)


def is_gdpval_judge_read_only_mcp_tool_name(
    name: str,
    *,
    server_name: str = "",
) -> bool:
    """Return whether a raw MCP tool name is safe for GDPVal judge reads."""

    tool_name = str(name or "").lower()
    if not tool_name:
        return False
    normalized_server = str(server_name or "").lower()
    if normalized_server in _PPT_MCP_SERVER_NAMES:
        if any(term in tool_name for term in GDPVAL_PPT_JUDGE_MCP_HARD_DENY_TERMS):
            return False
        if any(term in tool_name for term in GDPVAL_JUDGE_MCP_TOOL_ALLOW_TERMS):
            return True
        return any(term in tool_name for term in GDPVAL_PPT_JUDGE_MCP_EXTRA_ALLOW_TERMS)
    if any(term in tool_name for term in GDPVAL_JUDGE_MCP_TOOL_DENY_TERMS):
        return False
    return any(term in tool_name for term in GDPVAL_JUDGE_MCP_TOOL_ALLOW_TERMS)


def gdpval_mcp_server_configs(*, workdir: str | Path, reference_dir: str | Path):
    """Return local stdio MCP configs for one GDPVal judge workspace."""

    from simple_agent_lab.mcp import MCPServerConfig

    workspace = Path(workdir).resolve()
    references = Path(reference_dir).resolve()
    return (
        MCPServerConfig.stdio(
            "filesystem",
            "mcp-server-filesystem",
            str(workspace),
            str(references),
            cwd=str(workspace),
            init_timeout=20.0,
            call_timeout=60.0,
        ),
        MCPServerConfig.stdio(
            "pdf",
            "pdf-reader-mcp",
            cwd=str(references),
            init_timeout=20.0,
            call_timeout=90.0,
        ),
        MCPServerConfig.stdio(
            "excel",
            "excel-mcp-server",
            "stdio",
            cwd=str(references),
            init_timeout=20.0,
            call_timeout=90.0,
        ),
        MCPServerConfig.stdio(
            "word",
            "word_mcp_server",
            cwd=str(references),
            init_timeout=20.0,
            call_timeout=90.0,
        ),
        MCPServerConfig.stdio(
            "ppt",
            "ppt_mcp_server",
            cwd=str(references),
            init_timeout=20.0,
            call_timeout=90.0,
        ),
    )


@contextmanager
def open_gdpval_judge_tools(
    *,
    workdir: str | Path,
    reference_dir: str | Path,
    mode: str | None = None,
) -> Iterator[tuple[AgentTool, ...]]:
    """Open GDPVal judge tools, keeping MCP connections alive for the run."""

    workspace = Path(workdir).resolve()
    references = Path(reference_dir).resolve()
    tool_mode = normalize_judge_tool_mode(mode)
    local_tools = (
        make_gdpval_tools(workdir=workspace, reference_dir=references)
        if tool_mode in {"local", "hybrid"}
        else ()
    )
    connections = []
    mcp_tools: list[AgentTool] = []
    warnings: list[dict[str, str]] = []
    try:
        if tool_mode in {"mcp", "hybrid"}:
            try:
                from simple_agent_lab.mcp import connect_mcp
            except ModuleNotFoundError as exc:
                _handle_mcp_startup_error(
                    tool_mode=tool_mode,
                    warnings=warnings,
                    server="mcp",
                    exc=exc,
                )
            else:
                for config in gdpval_mcp_server_configs(
                    workdir=workspace, reference_dir=references
                ):
                    try:
                        connection = connect_mcp(config)
                    except Exception as exc:  # noqa: BLE001 - fallback in hybrid mode
                        _handle_mcp_startup_error(
                            tool_mode=tool_mode,
                            warnings=warnings,
                            server=config.name,
                            exc=exc,
                        )
                        continue
                    connections.append(connection)
                    selected_tools = _filtered_mcp_agent_tools(
                        connection,
                        server_name=config.name,
                        call_timeout=config.call_timeout,
                    )
                    if not selected_tools:
                        warnings.append(
                            {
                                "server": config.name,
                                "error_type": "MCPToolFilter",
                                "error": (
                                    "server started but no GDPVal read-only "
                                    "MCP tools matched the allowlist"
                                ),
                            }
                        )
                    mcp_tools.extend(selected_tools)
        if warnings:
            _write_mcp_warnings(workspace, warnings)
        tools = tuple(local_tools) + tuple(mcp_tools)
        if tool_mode == "mcp" and not mcp_tools:
            raise RuntimeError("GDPVal judge MCP mode selected no MCP tools")
        yield tools
    finally:
        for connection in reversed(connections):
            connection.close()


@contextmanager
def gdpval_judge_agent_context(
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None,
    instance: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None,
    name: str,
    role: str,
    system_prompt: str,
) -> Iterator:
    """Yield a judge Agent with local and optional MCP tools bound for one run."""

    workdir = Path(cwd)
    reference_dir = Path(
        str((context or {}).get("input_dir") or workdir.parent / "judge_inputs")
    )
    mode = normalize_judge_tool_mode((instance or {}).get("judge_tool_mode"))
    with open_gdpval_judge_tools(
        workdir=workdir,
        reference_dir=reference_dir,
        mode=mode,
    ) as tools:
        yield make_llm_agent(
            name=name,
            provider=provider,
            role=role,
            tools=tools,
            system_prompt=system_prompt,
            target="user",
            request_extra=request_extra,
        )


def _handle_mcp_startup_error(
    *,
    tool_mode: JudgeToolMode,
    warnings: list[dict[str, str]],
    server: str,
    exc: BaseException,
) -> None:
    if tool_mode == "mcp":
        raise RuntimeError(
            f"GDPVal judge MCP server {server!r} failed to start: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    warnings.append(
        {
            "server": server,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    )


def _filtered_mcp_agent_tools(
    connection: Any,
    *,
    server_name: str,
    call_timeout: float,
) -> list[AgentTool]:
    """Wrap only read/inspection MCP tools for GDPVal judge use."""

    from simple_agent_lab.mcp import mcp_tool_to_agent_tool

    return [
        mcp_tool_to_agent_tool(
            connection,
            tool,
            name_prefix=f"{connection.name}_",
            call_timeout=call_timeout,
        )
        for tool in connection.tools
        if is_gdpval_judge_read_only_mcp_tool_name(
            tool.name,
            server_name=server_name,
        )
    ]


def _write_mcp_warnings(workspace: Path, warnings: list[dict[str, str]]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / MCP_WARNING_FILE).write_text(
        json.dumps({"warnings": warnings}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
