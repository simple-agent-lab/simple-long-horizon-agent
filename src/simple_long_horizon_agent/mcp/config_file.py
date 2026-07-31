"""JSON file loading for MCP server configs.

The eval runners stage MCP settings as JSON so provider credentials can stay in
`.env` while server definitions remain explicit, local, and easy to inspect.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from .config import MCPServerConfig

_SERVER_FIELDS = {
    "name",
    "transport",
    "command",
    "args",
    "env",
    "cwd",
    "url",
    "headers",
    "init_timeout",
    "call_timeout",
}


def load_mcp_server_configs(path: str | Path) -> tuple[MCPServerConfig, ...]:
    """Read a JSON MCP config file and return validated server configs."""

    source = str(path)
    return mcp_server_configs_from_json(
        Path(path).read_text(encoding="utf-8"), source=source
    )


def mcp_server_configs_from_json(
    text: str, *, source: str = "MCP config"
) -> tuple[MCPServerConfig, ...]:
    """Parse a JSON object with a `servers` list into `MCPServerConfig` values."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: expected a JSON object")
    raw_servers = payload.get("servers")
    if not isinstance(raw_servers, list):
        raise ValueError(f"{source}: expected `servers` to be a list")

    configs: list[MCPServerConfig] = []
    for index, raw_server in enumerate(raw_servers):
        where = f"{source}: servers[{index}]"
        if not isinstance(raw_server, dict):
            raise ValueError(f"{where}: expected an object")
        configs.append(_server_config(cast(dict[str, Any], raw_server), source=where))
    return tuple(configs)


def _server_config(raw: dict[str, Any], *, source: str) -> MCPServerConfig:
    unknown = sorted(set(raw) - _SERVER_FIELDS)
    if unknown:
        raise ValueError(f"{source}: unknown field(s): {', '.join(unknown)}")

    name = _string(raw, "name", source=source, required=True)
    transport = _string(raw, "transport", source=source, default="stdio")
    init_timeout = _number(raw, "init_timeout", source=source, default=30.0)
    call_timeout = _number(raw, "call_timeout", source=source, default=60.0)

    if transport == "stdio":
        command = _string(raw, "command", source=source, required=True)
        args = _strings(raw, "args", source=source)
        return MCPServerConfig.stdio(
            name,
            command,
            *args,
            env=_mapping(raw, "env", source=source),
            cwd=_optional_string(raw, "cwd", source=source),
            init_timeout=init_timeout,
            call_timeout=call_timeout,
        )
    if transport == "http":
        return MCPServerConfig.http(
            name,
            _string(raw, "url", source=source, required=True),
            headers=_mapping(raw, "headers", source=source) or {},
            init_timeout=init_timeout,
            call_timeout=call_timeout,
        )
    raise ValueError(f"{source}: unsupported transport {transport!r}")


def _string(
    raw: dict[str, Any],
    key: str,
    *,
    source: str,
    required: bool = False,
    default: str = "",
) -> str:
    value = raw.get(key, default)
    if required and key not in raw:
        raise ValueError(f"{source}: missing required field `{key}`")
    if not isinstance(value, str):
        raise ValueError(f"{source}: `{key}` must be a string")
    return value


def _optional_string(raw: dict[str, Any], key: str, *, source: str) -> str | None:
    if key not in raw or raw[key] is None:
        return None
    return _string(raw, key, source=source)


def _strings(raw: dict[str, Any], key: str, *, source: str) -> tuple[str, ...]:
    if key not in raw:
        return ()
    value = raw[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{source}: `{key}` must be a list of strings")
    return tuple(value)


def _mapping(raw: dict[str, Any], key: str, *, source: str) -> dict[str, str] | None:
    if key not in raw or raw[key] is None:
        return None
    value = raw[key]
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise ValueError(f"{source}: `{key}` must be an object of string values")
    return dict(value)


def _number(raw: dict[str, Any], key: str, *, source: str, default: float) -> float:
    if key not in raw:
        return default
    value = raw[key]
    if not isinstance(value, int | float):
        raise ValueError(f"{source}: `{key}` must be a number")
    return float(value)
