"""GDPVal judge MCP tool registry and run-scoped connection helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast

from simple_agent_lab.llm.provider import Provider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.messages import ImageBlock, TextBlock
from simple_agent_lab.tools import AgentTool
from simple_agent_lab.tools import ToolResult

from .judge_excel_tools import make_judge_excel_tools
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
_LOCAL_WRITE_TOOL_NAMES = {"write_file", "edit_file", "multi_edit_file", "TodoWrite"}
_MAX_JUDGE_TOOL_TEXT_CHARS = 200_000
_MAX_JUDGE_TOOL_FILE_TEXT_CHARS = 400_000
_BINARY_JSON_KEYS = {
    "base64",
    "blob",
    "data",
    "image",
    "image_data",
    "imageData",
    "bytes",
}


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
    include_local_write_tools: bool = True,
    include_excel_helpers: bool = False,
) -> Iterator[tuple[AgentTool, ...]]:
    """Open GDPVal judge tools, keeping MCP connections alive for the run."""

    workspace = Path(workdir).resolve()
    references = Path(reference_dir).resolve()
    tool_mode = normalize_judge_tool_mode(mode)
    local_tools: tuple[AgentTool, ...] = ()
    if tool_mode in {"local", "hybrid"}:
        selected_local_tools = list(
            make_gdpval_tools(workdir=workspace, reference_dir=references)
        )
        if not include_local_write_tools:
            selected_local_tools = [
                tool
                for tool in selected_local_tools
                if tool.name not in _LOCAL_WRITE_TOOL_NAMES
            ]
        if include_excel_helpers:
            selected_local_tools.extend(
                make_judge_excel_tools(workdir=workspace, reference_dir=references)
            )
        local_tools = tuple(selected_local_tools)
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
    include_local_write_tools: bool = True,
    include_excel_helpers: bool = False,
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
        include_local_write_tools=include_local_write_tools,
        include_excel_helpers=include_excel_helpers,
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

    tools: list[AgentTool] = []
    for tool in connection.tools:
        if not is_gdpval_judge_read_only_mcp_tool_name(
            tool.name,
            server_name=server_name,
        ):
            continue
        agent_tool = mcp_tool_to_agent_tool(
            connection,
            tool,
            name_prefix=f"{connection.name}_",
            call_timeout=call_timeout,
        )
        tools.append(_sanitize_judge_tool_output(agent_tool))
    return tools


def _sanitize_judge_tool_output(tool: AgentTool) -> AgentTool:
    def execute(call_id: str, args: dict[str, Any], abort, on_update) -> ToolResult:
        result = tool.execute(call_id, args, abort, on_update)
        content = []
        for block in result.content:
            if isinstance(block, TextBlock):
                content.append(TextBlock(_preprocess_judge_tool_text(block.text)))
            elif isinstance(block, ImageBlock):
                content.append(block)
            else:
                content.append(block)
        return ToolResult(
            content=tuple(content),
            details=result.details,
            is_error=result.is_error,
            terminate=result.terminate,
        )

    return AgentTool(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        execute=execute,
        execution_mode=tool.execution_mode,
        timeout_seconds=tool.timeout_seconds,
    )


def _preprocess_judge_tool_text(text: str) -> str:
    if not text:
        return text
    cleaned = _strip_large_binary_json(text)
    max_chars = (
        _MAX_JUDGE_TOOL_FILE_TEXT_CHARS
        if _looks_like_file_payload(cleaned)
        else _MAX_JUDGE_TOOL_TEXT_CHARS
    )
    if len(cleaned) <= max_chars:
        return cleaned
    return _truncate_middle(
        cleaned,
        max_chars=max_chars,
        marker=(
            "\n...[GDPVal judge tool output truncated: "
            f"original_chars={len(cleaned)}]...\n"
        ),
    )


def _strip_large_binary_json(text: str) -> str:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return text
    if not any(f'"{key}"' in stripped for key in _BINARY_JSON_KEYS):
        return text
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return text
    sanitized, changed = _sanitize_binary_json_value(parsed)
    if not changed:
        return text
    return json.dumps(sanitized, ensure_ascii=False, default=str)


def _sanitize_binary_json_value(value: Any) -> tuple[Any, bool]:
    if isinstance(value, Mapping):
        changed = False
        output: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key)
            if (
                normalized in _BINARY_JSON_KEYS
                and isinstance(item, str)
                and len(item) > 1024
            ):
                output[normalized] = (
                    f"[stripped binary/base64 payload: chars={len(item)}]"
                )
                changed = True
                continue
            output_item, item_changed = _sanitize_binary_json_value(item)
            output[normalized] = output_item
            changed = changed or item_changed
        return output, changed
    if isinstance(value, list):
        changed = False
        output_list = []
        for item in value:
            output_item, item_changed = _sanitize_binary_json_value(item)
            output_list.append(output_item)
            changed = changed or item_changed
        return output_list, changed
    return value, False


def _looks_like_file_payload(text: str) -> bool:
    lowered = text[:2000].lower()
    return any(marker in lowered for marker in ("filepath", "filename", "file:"))


def _truncate_middle(text: str, *, max_chars: int, marker: str) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}{marker}{text[-tail:]}"


def _write_mcp_warnings(workspace: Path, warnings: list[dict[str, str]]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / MCP_WARNING_FILE).write_text(
        json.dumps({"warnings": warnings}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
