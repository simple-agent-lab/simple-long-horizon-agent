"""GDPVal judge MCP tool registry and run-scoped connection helpers."""

from __future__ import annotations

import copy
import html
import json
import re
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast

from simple_agent_lab.llm.provider import Provider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.messages import ImageBlock, TextBlock
from simple_agent_lab.tools import AgentTool
from simple_agent_lab.tools import ToolResult
from simple_agent_lab.tools import tool_result_text

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
_MAX_TEXT_ITEM_CHARS = 160_000
_MAX_GENERIC_FILE_CHARS = 60_000
_MAX_NOTEBOOK_FILE_CHARS = 120_000
_MAX_NOTEBOOK_CELL_SOURCE_CHARS = 8_000
_MAX_NOTEBOOK_CELL_OUTPUT_CHARS = 2_000
_DETERMINISTIC_TABULAR_COMPACT_PREFIX = "[Deterministic tabular tool output compacted:"
_LONG_BASE64_RE = re.compile(
    r"(?<![A-Za-z0-9+/])([A-Za-z0-9+/]{4096,}={0,2})(?![A-Za-z0-9+/])"
)
_FILE_TEXT_HEADER_RE = re.compile(
    r"(?m)^(?P<path>[^\n:]+\.(?:ipynb|py|pyi|js|jsx|ts|tsx|sol|json|md|txt|csv|yaml|yml|toml|html|css|scss|sh|sql|xml|overpassql|geojson)):\n"
)
_JUDGE_PATH_KEYS = {
    "filepath",
    "file_path",
    "filename",
    "file",
    "path",
    "workbook_path",
    "document_path",
    "uri",
}
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
    include_local_workspace_tools: bool = True,
    include_excel_helpers: bool = False,
    path_label_roles: Mapping[str, str] | None = None,
) -> Iterator[tuple[AgentTool, ...]]:
    """Open GDPVal judge tools, keeping MCP connections alive for the run."""

    workspace = Path(workdir).resolve()
    references = Path(reference_dir).resolve()
    tool_mode = normalize_judge_tool_mode(mode)
    local_tools: tuple[AgentTool, ...] = ()
    if tool_mode in {"local", "hybrid"} and include_local_workspace_tools:
        selected_local_tools = list(
            make_gdpval_tools(workdir=workspace, reference_dir=references)
        )
        if not include_local_write_tools:
            selected_local_tools = [
                tool
                for tool in selected_local_tools
                if tool.name not in _LOCAL_WRITE_TOOL_NAMES
            ]
        local_tools = tuple(selected_local_tools)
    if include_excel_helpers:
        local_tools = (
            *local_tools,
            *make_judge_excel_tools(workdir=workspace, reference_dir=references),
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
                        workspace=workspace,
                        reference_dir=references,
                        path_label_roles=path_label_roles,
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
    include_local_workspace_tools: bool = True,
    include_excel_helpers: bool = False,
    path_label_roles: Mapping[str, str] | None = None,
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
        include_local_workspace_tools=include_local_workspace_tools,
        include_excel_helpers=include_excel_helpers,
        path_label_roles=path_label_roles,
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
    workspace: Path,
    reference_dir: Path,
    path_label_roles: Mapping[str, str] | None,
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
        tools.append(
            _sanitize_judge_tool_output(
                agent_tool,
                workspace=workspace,
                reference_dir=reference_dir,
                path_label_roles=path_label_roles,
            )
        )
    return tools


def _sanitize_judge_tool_output(
    tool: AgentTool,
    *,
    workspace: Path,
    reference_dir: Path,
    path_label_roles: Mapping[str, str] | None = None,
) -> AgentTool:
    def execute(call_id: str, args: dict[str, Any], abort, on_update) -> ToolResult:
        repaired_args = _repair_judge_tool_file_paths(
            args,
            workspace=workspace,
            reference_dir=reference_dir,
            path_label_roles=path_label_roles,
        )
        result = tool.execute(call_id, repaired_args, abort, on_update)
        result = _maybe_apply_word_fallback(
            tool.name,
            repaired_args,
            result,
            workspace=workspace,
            reference_dir=reference_dir,
            path_label_roles=path_label_roles,
        )
        content = []
        for block in result.content:
            if isinstance(block, TextBlock):
                content.append(
                    TextBlock(
                        _preprocess_judge_tool_text(
                            block.text,
                            tool_name=tool.name,
                            args=repaired_args,
                        )
                    )
                )
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


def _preprocess_judge_tool_text(
    text: str,
    *,
    tool_name: str | None = None,
    args: Mapping[str, Any] | None = None,
) -> str:
    if not text:
        return text
    cleaned = _preprocess_judge_mcp_json_content(text)
    cleaned = _strip_large_binary_json(cleaned)
    compacted = _maybe_compact_large_excel_cells_payload(
        cleaned,
        tool_name=tool_name,
        params=args,
        min_chars=_MAX_JUDGE_TOOL_FILE_TEXT_CHARS,
    )
    if compacted is not None:
        return compacted
    cleaned, _ = _compact_large_text_payload(cleaned, max_chars=_MAX_TEXT_ITEM_CHARS)
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


def _repair_judge_tool_file_paths(
    args: dict[str, Any],
    *,
    workspace: Path,
    reference_dir: Path,
    path_label_roles: Mapping[str, str] | None,
) -> dict[str, Any]:
    if not isinstance(args, dict):
        return args
    repaired_args = copy.deepcopy(args)
    for location, original in _iter_judge_file_path_locations(repaired_args):
        resolved = _resolve_judge_tool_file_path(
            original,
            workspace=workspace,
            reference_dir=reference_dir,
            path_label_roles=path_label_roles,
        )
        if resolved is not None and str(resolved) != original:
            _set_nested_value(repaired_args, location, str(resolved))
    return repaired_args


def _iter_judge_file_path_locations(
    value: Any,
    path: tuple[Any, ...] = (),
) -> list[tuple[tuple[Any, ...], str]]:
    locations: list[tuple[tuple[Any, ...], str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _JUDGE_PATH_KEYS and _looks_like_judge_file_path(child):
                locations.append(((*path, key), str(child).strip()))
            elif isinstance(child, Mapping | list):
                locations.extend(_iter_judge_file_path_locations(child, (*path, key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if _looks_like_judge_file_path(child):
                locations.append(((*path, index), str(child).strip()))
            elif isinstance(child, Mapping | list):
                locations.extend(_iter_judge_file_path_locations(child, (*path, index)))
    return locations


def _looks_like_judge_file_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or "\x00" in text or "\n" in text or "\r" in text:
        return False
    if text.startswith(("http://", "https://", "data:")):
        return False
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", text):
        return text.startswith("file://")
    if text in {".", "./", "workdir", "./workdir"}:
        return True
    normalized = text.replace("\\", "/")
    if ".." in normalized.split("/"):
        return False
    basename = Path(normalized.rstrip("/")).name
    return bool(text.startswith("/") or "/" in normalized or "." in basename)


def _set_nested_value(value: Any, path: tuple[Any, ...], replacement: str) -> None:
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement


def _resolve_judge_tool_file_path(
    value: Any,
    *,
    workspace: Path,
    reference_dir: Path,
    path_label_roles: Mapping[str, str] | None,
) -> Path | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.startswith("file://"):
        text = text[7:]
    if text in {".", "./", "workdir", "./workdir"}:
        return workspace

    label = ""
    relative_text = text
    label_match = re.match(r"^([AB])[/\\](.+)$", text)
    if label_match:
        label = label_match.group(1).upper()
        relative_text = label_match.group(2)

    direct_candidates = _candidate_paths_for_text(
        relative_text,
        workspace=workspace,
        reference_dir=reference_dir,
        label=label,
        path_label_roles=path_label_roles,
    )
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate.resolve()

    basename = Path(relative_text.replace("\\", "/").rstrip("/")).name
    if not basename:
        return None
    matches = _find_matching_judge_paths(
        basename,
        workspace=workspace,
        reference_dir=reference_dir,
    )
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda path: _judge_path_priority(
            path,
            label=label,
            workspace=workspace,
            reference_dir=reference_dir,
            path_label_roles=path_label_roles,
        ),
    )[0].resolve()


def _candidate_paths_for_text(
    text: str,
    *,
    workspace: Path,
    reference_dir: Path,
    label: str,
    path_label_roles: Mapping[str, str] | None,
) -> list[Path]:
    path = Path(text)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if label:
            role = _role_for_label(label, path_label_roles)
            if role:
                candidates.append(reference_dir / role / path)
        candidates.extend([workspace / path, reference_dir / path])
    return _dedupe_paths(candidates)


def _find_matching_judge_paths(
    basename: str,
    *,
    workspace: Path,
    reference_dir: Path,
) -> list[Path]:
    roots = _dedupe_paths(
        [
            workspace,
            reference_dir,
            reference_dir / "candidate",
            reference_dir / "gold",
            reference_dir / "reference",
            reference_dir / "__zip_extracts",
        ]
    )
    matches: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        try:
            iterator = root.rglob(basename) if root.is_dir() else iter(())
            for candidate in iterator:
                resolved = candidate.resolve()
                if resolved.is_file() and resolved not in seen:
                    seen.add(resolved)
                    matches.append(resolved)
        except OSError:
            continue
    return matches


def _judge_path_priority(
    path: Path,
    *,
    label: str,
    workspace: Path,
    reference_dir: Path,
    path_label_roles: Mapping[str, str] | None,
) -> tuple[int, int, int, int, str]:
    role = _role_for_label(label, path_label_roles)
    role_root = reference_dir / role if role else None
    role_rank = 0 if role_root is not None and _inside(path, role_root) else 1
    workspace_rank = 0 if _inside(path, workspace) else 1
    reference_rank = 0 if _inside(path, reference_dir) else 1
    return (role_rank, reference_rank, workspace_rank, len(str(path)), str(path))


def _role_for_label(
    label: str,
    path_label_roles: Mapping[str, str] | None,
) -> str:
    if not label or not path_label_roles:
        return ""
    role = str(path_label_roles.get(label.upper()) or "").strip().lower()
    return role if role in {"candidate", "gold", "reference"} else ""


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    output: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        output.append(resolved)
    return output


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _maybe_apply_word_fallback(
    tool_name: str,
    args: Mapping[str, Any],
    result: ToolResult,
    *,
    workspace: Path,
    reference_dir: Path,
    path_label_roles: Mapping[str, str] | None,
) -> ToolResult:
    output = tool_result_text(result)
    if "attributes construct error" not in output:
        return result
    if not _is_word_fallback_tool(tool_name):
        return result
    path_value = _extract_judge_path(args)
    resolved = _resolve_judge_tool_file_path(
        path_value,
        workspace=workspace,
        reference_dir=reference_dir,
        path_label_roles=path_label_roles,
    )
    if resolved is None or resolved.suffix.lower() not in {".docx", ".doc"}:
        return result
    mode = "info" if "info" in _base_tool_name(tool_name) else "text"
    fallback = _extract_docx_fallback(resolved, mode=mode)
    if fallback is None:
        return result
    if mode == "text":
        text = str(fallback.get("text") or "")
        message = (
            "Word MCP extraction failed with attributes construct error. "
            "Fallback DOCX extraction succeeded using zip/XML text extraction.\n\n"
            + text
        )
    else:
        message = json.dumps(
            {
                "note": (
                    "Word MCP document properties failed with attributes construct "
                    "error. Fallback DOCX property/text statistics extraction "
                    "succeeded."
                ),
                **fallback,
            },
            ensure_ascii=False,
            default=str,
        )
    payload = {
        "content": [{"type": "text", "text": message}],
        "structuredContent": {"mcp_error": output, "fallback": fallback},
    }
    return ToolResult(
        content=(TextBlock(json.dumps(payload, ensure_ascii=False, default=str)),),
        details={"word_fallback": fallback, "mcp_error": output},
        is_error=False,
        terminate=result.terminate,
    )


def _is_word_fallback_tool(tool_name: str) -> bool:
    base_name = _base_tool_name(tool_name)
    return "document" in base_name and (
        "text" in base_name or "info" in base_name or "propert" in base_name
    )


def _base_tool_name(tool_name: str) -> str:
    normalized = str(tool_name or "").lower()
    if "__" in normalized:
        normalized = normalized.split("__", 1)[1]
    for prefix in ("word_", "filesystem_", "pdf_", "excel_", "ppt_", "powerpoint_"):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


def _extract_judge_path(args: Mapping[str, Any]) -> str:
    for key in (
        "filepath",
        "file_path",
        "filename",
        "file",
        "path",
        "workbook_path",
        "document_path",
    ):
        value = args.get(key)
        if value not in (None, ""):
            return str(value)
    for key in ("source", "input", "target"):
        nested = args.get(key)
        if isinstance(nested, Mapping):
            nested_path = _extract_judge_path(nested)
            if nested_path:
                return nested_path
    sources = args.get("sources")
    if isinstance(sources, list) and sources:
        first = sources[0]
        if isinstance(first, Mapping):
            return _extract_judge_path(first)
        return str(first)
    return ""


def _extract_docx_fallback(path: Path, *, mode: str) -> dict[str, Any] | None:
    try:
        with zipfile.ZipFile(path) as archive:
            parts = ["word/document.xml"]
            parts.extend(
                name
                for name in archive.namelist()
                if re.match(r"word/(header|footer)\d+\.xml$", name)
            )
            paragraphs: list[str] = []
            for part in parts:
                paragraphs.extend(
                    _extract_docx_paragraphs(_read_zip_text(archive, part))
                )
            text = "\n".join(paragraphs)
            properties = _extract_docx_core_props(
                _read_zip_text(archive, "docProps/core.xml")
            )
    except Exception:
        return None
    result: dict[str, Any] = {
        "ok": True,
        "method": "zip_regex_docx_fallback",
        "path": str(path),
        "mode": mode,
        "paragraph_count": len(paragraphs),
        "word_count": len(re.findall(r"\S+", text)),
        "char_count": len(text),
        "properties": properties,
    }
    if mode == "text":
        result["text"] = text
    return result


def _read_zip_text(archive: zipfile.ZipFile, name: str) -> str:
    try:
        return archive.read(name).decode("utf-8", errors="replace")
    except KeyError:
        return ""


def _extract_docx_paragraphs(xml: str) -> list[str]:
    paragraphs: list[str] = []
    for paragraph_xml in re.split(r"</w:p>", xml):
        texts = re.findall(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", paragraph_xml, flags=re.S)
        text = "".join(_clean_xml_text(item) for item in texts).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def _extract_docx_core_props(xml: str) -> dict[str, str]:
    props: dict[str, str] = {}
    patterns = {
        "title": r"<dc:title[^>]*>(.*?)</dc:title>",
        "creator": r"<dc:creator[^>]*>(.*?)</dc:creator>",
        "description": r"<dc:description[^>]*>(.*?)</dc:description>",
        "subject": r"<dc:subject[^>]*>(.*?)</dc:subject>",
        "created": r"<dcterms:created[^>]*>(.*?)</dcterms:created>",
        "modified": r"<dcterms:modified[^>]*>(.*?)</dcterms:modified>",
        "last_modified_by": r"<cp:lastModifiedBy[^>]*>(.*?)</cp:lastModifiedBy>",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, xml, flags=re.S)
        if match:
            props[key] = _clean_xml_text(match.group(1)).strip()
    return props


def _clean_xml_text(value: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(value or ""))


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


def _preprocess_judge_mcp_json_content(
    text: str,
    *,
    max_chars: int = _MAX_TEXT_ITEM_CHARS,
) -> str:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{" or '"content"' not in stripped:
        return text
    try:
        payload = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return text
    if not isinstance(payload, Mapping):
        return text
    content = payload.get("content")
    if not isinstance(content, list):
        return text

    changed = False
    processed: list[Any] = []
    for item in content:
        if not isinstance(item, Mapping):
            processed.append(item)
            continue
        item_type = str(item.get("type") or "")
        if item_type == "text" and isinstance(item.get("text"), str):
            raw_item_text = str(item.get("text") or "")
            new_text, compacted = _compact_large_text_payload(
                raw_item_text,
                max_chars=max_chars,
            )
            if compacted:
                new_text, _ = _cap_deterministic_tool_output(
                    new_text,
                    max_chars=max_chars,
                    reason="large_text_item_after_compaction",
                )
                processed.append({**item, "text": new_text})
                changed = True
            else:
                processed.append(item)
            continue
        if item_type == "image" and isinstance(item.get("data"), str):
            mime_type = str(
                item.get("mimeType") or item.get("mime_type") or "image/png"
            )
            byte_size = _approx_base64_bytes(str(item.get("data") or ""))
            processed.append(
                {
                    "type": "text",
                    "text": (
                        f"[Image ({mime_type}, ~{byte_size // 1024}KB) - "
                        "base64 removed from judge tool message; use document "
                        "text-extraction tools or targeted page/image reads for "
                        "visual verification.]"
                    ),
                }
            )
            changed = True
            continue
        if item_type == "resource" and isinstance(item.get("resource"), Mapping):
            resource = item["resource"]
            if isinstance(resource.get("blob"), str):
                mime_type = str(resource.get("mimeType") or "application/octet-stream")
                byte_size = _approx_base64_bytes(str(resource.get("blob") or ""))
                processed.append(
                    {
                        "type": "text",
                        "text": (
                            f"[Binary resource ({mime_type}, ~{byte_size // 1024}KB) "
                            "- data omitted]"
                        ),
                    }
                )
                changed = True
                continue
        processed.append(item)

    if not changed:
        return text
    output = dict(payload)
    output["content"] = processed
    return json.dumps(output, ensure_ascii=False, default=str)


def _cap_deterministic_tool_output(
    text: str,
    *,
    max_chars: int,
    reason: str,
) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    capped = _truncate_middle(
        text,
        max_chars=max_chars,
        marker=(
            f"\n...[deterministically capped {reason}: original_chars={len(text)}]...\n"
        ),
    )
    return (
        "[Tool output deterministically capped: "
        f"reason={reason}, original_chars={len(text)}, "
        f"capped_chars={len(capped)}]\n\n{capped}"
    ), True


def _approx_base64_bytes(value: str) -> int:
    if not value:
        return 0
    padding = value.count("=")
    return max(0, (len(value) * 3) // 4 - padding)


def _strip_long_base64_runs(text: str) -> tuple[str, int]:
    stripped = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal stripped
        stripped += 1
        return f"[base64/blob omitted: {len(match.group(1))} chars]"

    return _LONG_BASE64_RE.sub(replace, text), stripped


def _compact_large_text_payload(
    text: str,
    *,
    max_chars: int,
) -> tuple[str, bool]:
    if len(text) <= max_chars and not _LONG_BASE64_RE.search(text):
        return text, False
    parts = _split_file_text_payload(text)
    per_file_limit = max(
        20_000,
        min(_MAX_GENERIC_FILE_CHARS, max_chars // max(len(parts), 1)),
    )
    compacted_parts: list[str] = []
    changed = False
    for path, body in parts:
        compacted_body, part_changed = _compact_file_text(
            path,
            body,
            per_file_max_chars=per_file_limit,
        )
        changed = changed or part_changed
        compacted_parts.append(f"{path}:\n{compacted_body}" if path else compacted_body)
    compacted = "\n\n".join(compacted_parts)
    if len(compacted) > max_chars:
        compacted = _truncate_middle(
            compacted,
            max_chars=max_chars,
            marker=(
                "\n...[middle omitted from compacted tool text: "
                f"original_chars={len(text)}]...\n"
            ),
        )
        changed = True
    if changed:
        compacted = (
            "[Large text tool output compacted deterministically: "
            f"original_chars={len(text)}, compacted_chars={len(compacted)}]\n\n"
            + compacted
        )
    return compacted, changed


def _split_file_text_payload(text: str) -> list[tuple[str | None, str]]:
    matches = list(_FILE_TEXT_HEADER_RE.finditer(text))
    if not matches:
        return [(None, text)]
    parts: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        parts.append((None, text[: matches[0].start()].strip()))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        parts.append((match.group("path"), text[start:end]))
    return [(path, body) for path, body in parts if body]


def _compact_file_text(
    path: str | None,
    text: str,
    *,
    per_file_max_chars: int,
) -> tuple[str, bool]:
    original = text
    suffix = (path or "").lower().rsplit(".", 1)[-1]
    if suffix == "ipynb" or ('"nbformat"' in text and '"cells"' in text):
        try:
            notebook = json.loads(text.strip())
            if isinstance(notebook, Mapping):
                return _compact_notebook_json(path, notebook, len(original)), True
        except json.JSONDecodeError:
            pass
    text, base64_count = _strip_long_base64_runs(text)
    changed = base64_count > 0
    if len(text) > per_file_max_chars:
        text = _truncate_middle(
            text,
            max_chars=per_file_max_chars,
            marker=(
                "\n...[middle omitted from large file: "
                f"original_chars={len(original)}]...\n"
            ),
        )
        changed = True
    return text, changed


def _compact_notebook_json(
    path: str | None,
    notebook: Mapping[str, Any],
    original_chars: int,
) -> str:
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError("not a notebook")
    lines = [
        (
            f"[Notebook compacted: original_chars={original_chars}, "
            f"cells={len(cells)}; cell metadata and binary outputs removed]"
        )
    ]
    if path:
        lines.insert(0, f"{path}:")
    omitted_cells = 0
    omitted_outputs = 0
    for index, cell in enumerate(cells, start=1):
        if not isinstance(cell, Mapping):
            continue
        cell_map = cast(Mapping[str, Any], cell)
        cell_type = str(cell_map.get("cell_type") or "unknown")
        source = _join_cell_source(cell_map.get("source"))
        source, source_blobs = _strip_long_base64_runs(source)
        omitted_outputs += source_blobs
        source = _truncate_middle(
            source.strip(),
            max_chars=_MAX_NOTEBOOK_CELL_SOURCE_CHARS,
            marker="\n...[notebook cell source omitted]...\n",
        )
        cell_lines = [f"\n## Cell {index} [{cell_type}]"]
        if source:
            if cell_type == "code":
                cell_lines.extend(["```python", source, "```"])
            else:
                cell_lines.append(source)
        if cell_type == "code":
            output_parts: list[str] = []
            outputs = cell_map.get("outputs") or []
            if isinstance(outputs, list):
                for output in outputs[:3]:
                    if not isinstance(output, Mapping):
                        continue
                    output_text, omitted = _extract_notebook_text_output(output)
                    omitted_outputs += omitted
                    if output_text:
                        output_parts.append(
                            _truncate_middle(
                                output_text.strip(),
                                max_chars=_MAX_NOTEBOOK_CELL_OUTPUT_CHARS,
                                marker="\n...[notebook cell output omitted]...\n",
                            )
                        )
                if len(outputs) > 3:
                    omitted_outputs += len(outputs) - 3
            if output_parts:
                cell_lines.append("[Selected text output]")
                cell_lines.extend(["```text", "\n\n".join(output_parts), "```"])
        candidate = "\n".join([*lines, *cell_lines])
        if len(candidate) > _MAX_NOTEBOOK_FILE_CHARS:
            omitted_cells = len(cells) - index + 1
            break
        lines.extend(cell_lines)
    if omitted_cells:
        lines.append(f"\n[Notebook cells omitted due to size cap: {omitted_cells}]")
    if omitted_outputs:
        lines.append(f"\n[Notebook outputs/binary blobs omitted: {omitted_outputs}]")
    return _truncate_middle(
        "\n".join(lines),
        max_chars=_MAX_NOTEBOOK_FILE_CHARS,
        marker="\n...[notebook compact omitted]...\n",
    )


def _extract_notebook_text_output(output: Mapping[str, Any]) -> tuple[str, int]:
    omitted_binary = 0
    parts: list[str] = []
    output_type = output.get("output_type", "output")
    if output_type == "stream":
        text = _join_cell_source(output.get("text"))
        if text:
            parts.append(text)
    data = output.get("data")
    if isinstance(data, Mapping):
        for mime in ("text/plain", "text/markdown", "text/html"):
            if mime in data:
                text = _join_cell_source(data.get(mime))
                if text:
                    parts.append(text)
                    break
        omitted_binary += sum(
            1
            for mime in data
            if str(mime).startswith("image/")
            or str(mime) in {"application/pdf", "application/octet-stream"}
        )
    if output.get("ename") or output.get("evalue"):
        parts.append(
            f"{output.get('ename', '')}: {output.get('evalue', '')}".strip(": ")
        )
    text = "\n".join(part for part in parts if part)
    text, base64_omitted = _strip_long_base64_runs(text)
    return text, omitted_binary + base64_omitted


def _join_cell_source(value: Any) -> str:
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    if value is None:
        return ""
    return str(value)


def _looks_like_file_payload(text: str) -> bool:
    lowered = text[:2000].lower()
    return any(marker in lowered for marker in ("filepath", "filename", "file:"))


def _maybe_compact_large_excel_cells_payload(
    text: str,
    *,
    tool_name: str | None = None,
    params: Mapping[str, Any] | None = None,
    min_chars: int,
) -> str | None:
    if len(text) <= min_chars:
        return None
    if text.lstrip().startswith(_DETERMINISTIC_TABULAR_COMPACT_PREFIX):
        return None
    if '"cells"' not in text and '\\"cells\\"' not in text:
        return None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    payload = _find_excel_cells_payload(parsed)
    if payload is None:
        return None
    return _compact_excel_cells_payload(
        payload,
        tool_name=tool_name,
        params=params,
        original_chars=len(text),
    )


def _find_excel_cells_payload(obj: Any, max_depth: int = 8) -> Mapping[str, Any] | None:
    stack: list[tuple[Any, int]] = [(obj, 0)]
    seen_ids: set[int] = set()
    while stack:
        current, depth = stack.pop()
        if id(current) in seen_ids:
            continue
        seen_ids.add(id(current))
        if isinstance(current, Mapping):
            cells = current.get("cells")
            if (
                isinstance(cells, list)
                and cells
                and all(
                    isinstance(cell, Mapping) for cell in cells[: min(10, len(cells))]
                )
            ):
                return current
            if depth >= max_depth:
                continue
            stack.extend(
                (parsed, depth + 1) for parsed in _extract_text_item_jsons(current)
            )
            for value in current.values():
                if isinstance(value, Mapping | list):
                    stack.append((value, depth + 1))
                elif isinstance(value, str) and (
                    '"cells"' in value or '\\"cells\\"' in value
                ):
                    try:
                        stack.append((json.loads(value), depth + 1))
                    except json.JSONDecodeError:
                        continue
        elif isinstance(current, list) and depth < max_depth:
            for value in current:
                if isinstance(value, Mapping | list):
                    stack.append((value, depth + 1))
                elif isinstance(value, str) and (
                    '"cells"' in value or '\\"cells\\"' in value
                ):
                    try:
                        stack.append((json.loads(value), depth + 1))
                    except json.JSONDecodeError:
                        continue
        elif (
            isinstance(current, str)
            and depth < max_depth
            and ('"cells"' in current or '\\"cells\\"' in current)
        ):
            try:
                stack.append((json.loads(current), depth + 1))
            except json.JSONDecodeError:
                continue
    return None


def _extract_text_item_jsons(obj: Mapping[str, Any]) -> list[Any]:
    content = obj.get("content")
    if not isinstance(content, list):
        return []
    extracted = []
    for item in content:
        if not isinstance(item, Mapping):
            continue
        text = item.get("text")
        if not isinstance(text, str) or '"cells"' not in text:
            continue
        try:
            extracted.append(json.loads(text))
        except json.JSONDecodeError:
            continue
    return extracted


def _compact_excel_cells_payload(
    payload: Mapping[str, Any],
    *,
    tool_name: str | None,
    params: Mapping[str, Any] | None,
    original_chars: int,
) -> str | None:
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        return None
    rows: dict[int, dict[int, Any]] = {}
    formula_count = 0
    non_empty_cells = 0
    positioned_cells = 0
    for raw_cell in cells:
        if not isinstance(raw_cell, Mapping):
            continue
        row_idx, col_idx = _excel_cell_position(raw_cell)
        if row_idx is None or col_idx is None:
            continue
        value = _excel_cell_value(raw_cell)
        if value not in (None, ""):
            non_empty_cells += 1
        if raw_cell.get("formula") not in (None, ""):
            formula_count += 1
        rows.setdefault(row_idx, {})[col_idx] = value
        positioned_cells += 1
    if not rows:
        return None

    all_rows = sorted(rows)
    all_cols = sorted({col for row_cells in rows.values() for col in row_cells})
    if not all_cols:
        return None
    if len(all_cols) <= 40:
        sampled_cols = all_cols
        omitted_cols = 0
    else:
        sampled_cols = [*all_cols[:30], *all_cols[-10:]]
        omitted_cols = len(all_cols) - len(sampled_cols)

    first_rows = all_rows[:25]
    last_rows = [row for row in all_rows[-10:] if row not in set(first_rows)]
    header_row = min(all_rows)
    filepath = _first_present_mapping(
        payload,
        params,
        keys=("filepath", "file_path", "path", "workbook"),
    )
    sheet_name = _first_present_mapping(
        payload,
        params,
        keys=("sheet_name", "sheet", "worksheet"),
    )
    cell_range = _first_present_mapping(payload, params, keys=("range", "cell_range"))
    if not cell_range and params:
        start_cell = params.get("start_cell") or params.get("start")
        end_cell = params.get("end_cell") or params.get("end")
        if start_cell or end_cell:
            cell_range = f"{start_cell or ''}:{end_cell or ''}"

    lines = [
        (
            f"{_DETERMINISTIC_TABULAR_COMPACT_PREFIX} "
            f"reason=large_excel_cells_output, original_chars={original_chars}, "
            f"cells_reported={len(cells)}, positioned_cells={positioned_cells}]"
        ),
        (
            "Full per-cell metadata was too large for prompt history and was "
            "replaced with this deterministic outline."
        ),
    ]
    if tool_name:
        lines.append(f"tool_name={tool_name}")
    if filepath:
        lines.append(f"file={filepath}")
    if sheet_name:
        lines.append(f"sheet={sheet_name}")
    if cell_range:
        lines.append(f"range={cell_range}")
    row_span = f"{all_rows[0]}..{all_rows[-1]}"
    col_span = (
        f"{_excel_index_to_column(all_cols[0])}..{_excel_index_to_column(all_cols[-1])}"
    )
    lines.extend(
        [
            (
                f"detected_rows={len(all_rows)} ({row_span}); "
                f"detected_columns={len(all_cols)} ({col_span}); "
                f"non_empty_cells={non_empty_cells}; formula_cells={formula_count}"
            ),
            "sampled_columns="
            + ", ".join(f"{_excel_index_to_column(col)}({col})" for col in sampled_cols)
            + (f"; omitted_columns={omitted_cols}" if omitted_cols else ""),
            "",
            f"header_or_first_row: {_format_compact_excel_row(header_row, rows[header_row], sampled_cols)}",
            "",
            "first_rows:",
        ]
    )
    lines.extend(
        _format_compact_excel_row(row, rows[row], sampled_cols) for row in first_rows
    )
    if last_rows:
        lines.extend(["", "last_rows:"])
        lines.extend(
            _format_compact_excel_row(row, rows[row], sampled_cols) for row in last_rows
        )
    omitted_rows = len(all_rows) - len(first_rows) - len(last_rows)
    if omitted_rows > 0:
        lines.append(f"\nrows_omitted_from_prompt={omitted_rows}")
    lines.append(
        "For exact verification, call excel_profile_sheet, "
        "read_data_from_excel_compact, excel_filter_rows, or excel_aggregate "
        "on the needed bounded range/filter."
    )
    return "\n".join(lines)


def _excel_cell_position(cell: Mapping[str, Any]) -> tuple[int | None, int | None]:
    row = cell.get("row") or cell.get("row_index") or cell.get("rowIndex")
    col = (
        cell.get("column")
        or cell.get("col")
        or cell.get("column_index")
        or cell.get("columnIndex")
    )
    try:
        row_int = int(row) if row is not None else None
    except (TypeError, ValueError):
        row_int = None
    col_int = _excel_column_to_index(col)
    address = cell.get("address") or cell.get("cell") or cell.get("coordinate")
    if isinstance(address, str):
        match = re.match(r"^\$?([A-Za-z]+)\$?(\d+)$", address.strip())
        if match:
            col_int = col_int or _excel_column_to_index(match.group(1))
            row_int = row_int or int(match.group(2))
    return row_int, col_int


def _excel_column_to_index(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    text = str(value).strip()
    if text.isdigit():
        col = int(text)
        return col if col > 0 else None
    letters = "".join(ch for ch in text.upper() if "A" <= ch <= "Z")
    if not letters:
        return None
    col = 0
    for ch in letters:
        col = col * 26 + ord(ch) - ord("A") + 1
    return col


def _excel_index_to_column(index: int) -> str:
    if index <= 0:
        return str(index)
    letters = ""
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


def _excel_cell_value(cell: Mapping[str, Any]) -> Any:
    for key in (
        "formatted_value",
        "formattedValue",
        "display_value",
        "displayValue",
        "value",
        "text",
        "raw_value",
        "rawValue",
    ):
        if key in cell and cell[key] is not None:
            return cell[key]
    formula = cell.get("formula")
    if formula is not None:
        return formula
    return ""


def _short_excel_cell_value(value: Any, max_chars: int = 180) -> str:
    if value is None:
        return ""
    text = repr(value) if isinstance(value, float) else str(value)
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    return _truncate_middle(text, max_chars=max_chars, marker="...[omitted]...")


def _format_compact_excel_row(
    row_index: int,
    row_cells: Mapping[int, Any],
    columns: list[int],
) -> str:
    values = [_short_excel_cell_value(row_cells.get(col, "")) for col in columns]
    return f"row {row_index}: " + json.dumps(values, ensure_ascii=False)


def _first_present_mapping(
    *mappings: Mapping[str, Any] | None,
    keys: tuple[str, ...],
) -> Any:
    for mapping in mappings:
        if not isinstance(mapping, Mapping):
            continue
        for key in keys:
            value = mapping.get(key)
            if value not in (None, ""):
                return value
    return None


def _truncate_middle(text: str, *, max_chars: int, marker: str) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= len(marker):
        return text[:max_chars]
    remaining = max_chars - len(marker)
    head = remaining // 2
    tail = remaining - head
    tail_text = text[-tail:] if tail else ""
    return f"{text[:head]}{marker}{tail_text}"


def _write_mcp_warnings(workspace: Path, warnings: list[dict[str, str]]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / MCP_WARNING_FILE).write_text(
        json.dumps({"warnings": warnings}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
