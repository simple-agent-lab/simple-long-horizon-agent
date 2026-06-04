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
GDPVAL_MCP_READ_TOOL_NAMES: Mapping[str, frozenset[str]] = {
    "filesystem": frozenset(
        {
            "read_file",
            "read_text_file",
            "read_media_file",
            "read_multiple_files",
            "list_directory",
            "list_directory_with_sizes",
            "directory_tree",
            "search_files",
            "get_file_info",
            "list_allowed_directories",
        }
    ),
    "pdf": frozenset({"read_pdf"}),
    "excel": frozenset(
        {
            "read_data_from_excel",
            "get_workbook_metadata",
            "validate_excel_range",
            "validate_formula_syntax",
            "get_merged_cells",
            "get_data_validation_info",
        }
    ),
    "word": frozenset(
        {
            "get_document_info",
            "get_document_text",
            "get_document_outline",
            "list_available_documents",
            "get_document_xml",
            "get_paragraph_text_from_document",
            "find_text_in_document",
            "get_all_comments",
            "get_comments_by_author",
            "get_comments_for_paragraph",
            "validate_document_footnotes",
        }
    ),
    "ppt": frozenset(
        {
            "open_presentation",
            "get_presentation_info",
            "get_template_file_info",
            "get_slide_info",
            "extract_slide_text",
            "extract_presentation_text",
        }
    ),
}


def normalize_judge_tool_mode(value: Any) -> JudgeToolMode:
    text = str(value or DEFAULT_JUDGE_TOOL_MODE).strip().lower()
    if text not in JUDGE_TOOL_MODES:
        choices = ", ".join(JUDGE_TOOL_MODES)
        raise ValueError(
            f"unknown GDPVal judge tool mode {value!r}; expected {choices}"
        )
    return cast(JudgeToolMode, text)


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

    allowed = GDPVAL_MCP_READ_TOOL_NAMES.get(server_name, frozenset())
    return [
        mcp_tool_to_agent_tool(
            connection,
            tool,
            name_prefix=f"{connection.name}_",
            call_timeout=call_timeout,
        )
        for tool in connection.tools
        if tool.name in allowed
    ]


def _write_mcp_warnings(workspace: Path, warnings: list[dict[str, str]]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / MCP_WARNING_FILE).write_text(
        json.dumps({"warnings": warnings}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
