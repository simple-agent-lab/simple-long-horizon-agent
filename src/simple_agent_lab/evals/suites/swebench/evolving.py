"""Evolving SWE-bench container module: builds the agent from the Version's package.

Same container-module contract as ``container.py``; re-exports its task/prepare/
extract/evaluate hooks and adds a ``build_agent`` hook that runs the staged
evolvable agent package (``AGENT_PACKAGE_KEY``). Any failure falls back to the
baked-in bash agent, so a broken evolved program can never break the harness.
The recipe points its Suite at this module via ``container_module``.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from simple_agent_lab.core import Agent
from simple_agent_lab.evals.in_container import build_agent as _base_build_agent
from simple_agent_lab.evals.protocols import AGENT_PACKAGE_KEY
from simple_agent_lab.llm import Provider

from . import agent_package
from .container import (
    AGENT_SYSTEM_PROMPT,
    agent_spec,
    apply_oracle,
    build_task,
    evaluate,
    extract_result,
    prepare,
)

__all__ = [
    "AGENT_SYSTEM_PROMPT",
    "agent_spec",
    "apply_oracle",
    "build_agent",
    "build_task",
    "evaluate",
    "extract_result",
    "prepare",
]

_PACKAGE_CACHE: list[Callable[..., Any] | None] = []


def _staged_agent_builder() -> Callable[..., Any] | None:
    if _PACKAGE_CACHE:
        return _PACKAGE_CACHE[0]
    builder: Callable[..., Any] | None = None
    try:
        from simple_agent_lab.evals.stores import container_store_from_env

        store = container_store_from_env()
        files = json.loads(store.get(AGENT_PACKAGE_KEY).decode("utf-8"))
        if isinstance(files, Mapping):
            root = Path(tempfile.mkdtemp(prefix="sal_agent_pkg_"))
            builder = agent_package.load_agent_package(dict(files), root=root)
    except Exception:
        builder = None
    _PACKAGE_CACHE.append(builder)
    return builder


def build_agent(
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Container-module build_agent hook (preferred by the generic runner).

    Uses the staged evolvable package when present and valid; otherwise builds
    the baked-in bash agent from ``agent_spec()``.
    """

    builder = _staged_agent_builder()
    if builder is not None:
        try:
            return builder(
                provider=provider, cwd=cwd, base_system_prompt=AGENT_SYSTEM_PROMPT
            )
        except Exception:
            pass
    return _base_build_agent(
        spec=agent_spec(), provider=provider, cwd=cwd, request_extra=request_extra
    )
