"""Reward functions: how to score one run.

A reward is ``(Run) -> float | Mapping[str, float]``. Returning a float means
the single ``reward`` dimension; returning a dict declares multiple dimensions.
This is the ONLY scoring surface — there is no separate "measures" system.
"""

from __future__ import annotations

from simple_agent_lab.evolution.types import Run


def result_key(run: Run) -> float:
    """The default: the ``result.json`` reward key; a crashed run scores 0."""

    return run.reward if run.reward is not None else 0.0


def cost_tokens(run: Run) -> float:
    """Total input+output tokens summed across the run's usage events."""

    total = 0.0
    for event in run.events():
        usage = event.get("usage")
        if isinstance(usage, dict):
            total += float(usage.get("input_tokens") or 0)
            total += float(usage.get("output_tokens") or 0)
    return total
