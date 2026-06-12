"""Shallow {name -> factory} dicts per component category. No entry-point
scanning, no metaclasses. Register a custom component by assigning into the dict
(or with the 3-line ``register`` helper below).
"""

from __future__ import annotations

from typing import Any, Callable

from simple_agent_lab.evolution.components import criterion as _criterion
from simple_agent_lab.evolution.components import reward as _reward
from simple_agent_lab.evolution.config import Use

# Each factory takes the Use.args as kwargs and returns the component callable.
ROLLOUTS: dict[str, Callable[..., Any]] = {}
REWARDS: dict[str, Callable[..., Any]] = {
    # rewards are themselves the callable; wrap so build() can pass args uniformly
    "result_key": lambda: _reward.result_key,
    "cost_tokens": lambda: _reward.cost_tokens,
}
STRATEGIES: dict[str, Callable[..., Any]] = {}
CRITERIA: dict[str, Callable[..., Any]] = {
    "improve": _criterion.improve,
    "not_worse": _criterion.not_worse,
}

_TABLES = {
    "rollout": ROLLOUTS,
    "reward": REWARDS,
    "strategy": STRATEGIES,
    "criterion": CRITERIA,
}


def register(category: str, name: str, factory: Callable[..., Any]) -> None:
    _TABLES[category][name] = factory


def build(category: str, use: Use) -> Any:
    table = _TABLES[category]
    if use.name not in table:
        raise KeyError(
            f"unknown {category} {use.name!r}; registered: {sorted(table)}"
        )
    return table[use.name](**dict(use.args))
