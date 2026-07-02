"""Shared agent flavor names used by runners and benchmark harnesses."""

import os
from collections.abc import Mapping

AGENT_FLAVOR_ENV = "AGENT_FLAVOR"
DEFAULT_AGENT_FLAVOR = "bash"

# One-turn/multi-turn agents that the generic in-container runner can build
# directly from an AgentSpec.
SIMPLE_AGENT_FLAVORS = ("bash", "bash_task", "bash_task_read", "bash_skills")

# Workflow arms are selected with the same AGENT_FLAVOR knob, but suites that
# support them provide a custom build_agent facade.
WORKFLOW_AGENT_FLAVORS = ("loop", "pdr")

AGENT_FLAVORS = SIMPLE_AGENT_FLAVORS + WORKFLOW_AGENT_FLAVORS


def flavor_from_env(
    *,
    flavors: tuple[str, ...] = AGENT_FLAVORS,
    default: str = DEFAULT_AGENT_FLAVOR,
    env: Mapping[str, str] | None = None,
    label: str = "",
) -> str:
    """The selected flavor from ``AGENT_FLAVOR``, validated against a vocabulary.

    Suites with their own flavor vocabulary (e.g. OneMillion-Bench) pass their
    ``flavors``/``default`` and a ``label`` naming the suite in the error.
    """

    source = os.environ if env is None else env
    flavor = (source.get(AGENT_FLAVOR_ENV) or default).strip().lower() or default
    if flavor not in flavors:
        suffix = f" for {label}" if label else ""
        raise SystemExit(
            f"Unsupported {AGENT_FLAVOR_ENV}={flavor!r}{suffix}; "
            f"expected one of {flavors}."
        )
    return flavor
