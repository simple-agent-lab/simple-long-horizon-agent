"""Tool registry for retrieval experiments.

The runtime's native `AgentTool` is already the executable unit. This layer adds
search metadata around those tools without changing how they run.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from simple_agent_lab.tools import AgentTool


@dataclass(frozen=True)
class ToolRecord:
    """One searchable, executable tool."""

    tool: AgentTool
    namespace: str = "local"
    tags: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    when_to_use: str = ""
    risk: str = "low"

    @property
    def tool_id(self) -> str:
        return (
            f"{self.namespace}.{self.tool.name}" if self.namespace else self.tool.name
        )


class ToolRegistry:
    """Immutable name/id lookup over searchable tools."""

    def __init__(self, records: Iterable[ToolRecord | AgentTool]) -> None:
        normalized = tuple(
            record if isinstance(record, ToolRecord) else ToolRecord(record)
            for record in records
        )
        by_id: dict[str, ToolRecord] = {}
        by_name: dict[str, ToolRecord] = {}
        for record in normalized:
            if record.tool_id in by_id:
                raise ValueError(f"duplicate tool id: {record.tool_id}")
            if record.tool.name in by_name:
                raise ValueError(
                    f"duplicate tool name: {record.tool.name!r}; use unique names "
                    "or wrap them in namespace-specific proxy names"
                )
            by_id[record.tool_id] = record
            by_name[record.tool.name] = record
        self.records = normalized
        self._by_id = by_id
        self._by_name = by_name

    def get(self, name_or_id: str) -> ToolRecord:
        if name_or_id in self._by_id:
            return self._by_id[name_or_id]
        if name_or_id in self._by_name:
            return self._by_name[name_or_id]
        known = ", ".join(sorted(self._by_name)[:10])
        suffix = "..." if len(self._by_name) > 10 else ""
        raise KeyError(f"unknown tool {name_or_id!r}; known tools: {known}{suffix}")

    def tools(self) -> tuple[AgentTool, ...]:
        return tuple(record.tool for record in self.records)


def tool_document(record: ToolRecord) -> str:
    """Render the searchable text for one tool."""

    tool = record.tool
    schema_text = _schema_text(tool.parameters)
    examples = "\n".join(f"example: {example}" for example in record.examples)
    tags = ", ".join(record.tags)
    return "\n".join(
        part
        for part in (
            f"id: {record.tool_id}",
            f"name: {tool.name}",
            f"namespace: {record.namespace}",
            f"description: {tool.description}",
            f"when_to_use: {record.when_to_use}",
            f"tags: {tags}",
            examples,
            f"parameters: {schema_text}",
        )
        if part.strip()
    )


def _schema_text(schema: Mapping[str, Any]) -> str:
    """Compact schema text for retrieval, not provider serialization."""

    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return json.dumps(schema, sort_keys=True)
    required = set(schema.get("required") or ())
    bits: list[str] = []
    for name, spec in properties.items():
        if not isinstance(spec, Mapping):
            bits.append(str(name))
            continue
        marker = "required" if name in required else "optional"
        desc = str(spec.get("description") or "")
        typ = str(spec.get("type") or "value")
        bits.append(f"{name} ({typ}, {marker}): {desc}")
    return "; ".join(bits)
