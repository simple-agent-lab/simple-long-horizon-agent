"""Typed config that SELECTS components; it never hides them. Every name maps
to a factory you can grep, and ``Experiment(...)`` can bypass config entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Use:
    """A component reference: a registry name plus the kwargs its factory takes."""

    name: str
    args: Mapping[str, Any] = field(default_factory=dict)

    def __init__(self, name: str, **args: Any) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "args", dict(args))


@dataclass(frozen=True)
class Config:
    """Names the four components + run settings. Built in Python (a YAML loader
    is a trivial later add)."""

    workspace: str | Path
    rollout: Use
    reward: Use = field(default_factory=lambda: Use("result_key"))
    strategy: Use | None = None  # Plan 2: required for agent-driven runs
    criterion: Use = field(default_factory=lambda: Use("improve", dim="reward"))
    slice_id: str = "custom"
    instances: tuple[Mapping[str, Any], ...] = ()
    seed: Mapping[str, str] = field(default_factory=dict)
    auto_promote: bool = True
