"""Proxy tools: search a registry, then invoke real tools by name."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from simple_agent_lab.tools import (
    AbortFlag,
    AgentTool,
    ToolResult,
    ToolUpdateFn,
    coerce_int,
    text_result,
)

from .registry import ToolRegistry
from .retrievers import BM25ToolRetriever, SearchResult


def make_search_tools_tool(
    registry: ToolRegistry,
    retriever: BM25ToolRetriever | None = None,
    *,
    default_k: int = 8,
    max_k: int = 20,
    tool_name: str = "search_tools",
) -> AgentTool:
    """Return a tool that retrieves candidate tools from `registry`."""

    resolved = retriever or BM25ToolRetriever(registry)

    def execute(
        call_id: str,
        args: dict[str, Any],
        abort: AbortFlag,
        on_update: ToolUpdateFn | None,
    ) -> ToolResult:
        del call_id, abort, on_update
        query = str(args.get("query", "")).strip()
        if not query:
            return text_result("`query` is required.", is_error=True)
        try:
            k = coerce_int("k", args.get("k", default_k), minimum=1)
        except ValueError as exc:
            return text_result(f"Invalid search_tools argument: {exc}", is_error=True)
        k = min(k, max_k)
        results = resolved.search(query, k=k)
        payload = [_result_payload(result) for result in results]
        if not payload:
            return text_result(
                json.dumps({"query": query, "tools": []}, sort_keys=True),
                details={"query": query, "candidates": []},
            )
        return text_result(
            json.dumps({"query": query, "tools": payload}, sort_keys=True),
            details={"query": query, "candidates": payload},
        )

    return AgentTool(
        name=tool_name,
        description=(
            "Search the local tool registry. Use this before invoke_tool when "
            "you are not sure which concrete tool should handle the task."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of the needed tool.",
                },
                "k": {
                    "type": "integer",
                    "minimum": 1,
                    "description": f"Number of candidates to return (default {default_k}).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        execute=execute,
        execution_mode="sequential",
    )


def make_invoke_tool(
    registry: ToolRegistry,
    *,
    tool_name: str = "invoke_tool",
) -> AgentTool:
    """Return a proxy that validates and executes a registered tool."""

    def execute(
        call_id: str,
        args: dict[str, Any],
        abort: AbortFlag,
        on_update: ToolUpdateFn | None,
    ) -> ToolResult:
        name = str(args.get("tool_name", "")).strip()
        if not name:
            return text_result("`tool_name` is required.", is_error=True)
        raw_arguments = args.get("arguments", {})
        if not isinstance(raw_arguments, Mapping):
            return text_result("`arguments` must be an object.", is_error=True)
        try:
            record = registry.get(name)
        except KeyError as exc:
            return text_result(str(exc), is_error=True)
        arguments = dict(raw_arguments)
        validation_error = _validate_arguments(arguments, record.tool.parameters)
        if validation_error:
            return text_result(validation_error, is_error=True)
        result = record.tool.execute(call_id, arguments, abort, on_update)
        details = {
            "tool_id": record.tool_id,
            "tool_name": record.tool.name,
            "arguments": arguments,
            "result_details": result.details,
        }
        return ToolResult(
            content=result.content,
            details=details,
            is_error=result.is_error,
            terminate=result.terminate,
        )

    return AgentTool(
        name=tool_name,
        description=(
            "Invoke one concrete tool returned by search_tools. Pass the exact "
            "tool name or id and an arguments object matching that tool's schema."
        ),
        parameters={
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Concrete tool name or namespaced tool id.",
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments for the concrete tool.",
                },
            },
            "required": ["tool_name", "arguments"],
            "additionalProperties": False,
        },
        execute=execute,
        execution_mode="sequential",
    )


def _result_payload(result: SearchResult) -> dict[str, Any]:
    tool = result.record.tool
    return {
        "rank": result.rank,
        "score": round(result.score, 4),
        "tool_id": result.record.tool_id,
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "tags": list(result.record.tags),
        "risk": result.record.risk,
    }


def _validate_arguments(args: Mapping[str, Any], schema: Mapping[str, Any]) -> str:
    required = schema.get("required") or ()
    for name in required:
        if name not in args:
            return f"Missing required argument {name!r}."
    if schema.get("additionalProperties") is False:
        properties = schema.get("properties") or {}
        extra = sorted(set(args) - set(properties))
        if extra:
            return f"Unexpected argument(s): {', '.join(extra)}."
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        for name, spec in properties.items():
            if name not in args or not isinstance(spec, Mapping):
                continue
            expected = spec.get("type")
            if expected and not _matches_type(args[name], str(expected)):
                return f"Argument {name!r} must be {expected}, got {args[name]!r}."
    return ""


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    return True
