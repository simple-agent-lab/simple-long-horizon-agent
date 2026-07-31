"""Shared agent flavor names used by runners and benchmark harnesses."""

import os

AGENT_FLAVOR_ENV = "AGENT_FLAVOR"
DEFAULT_AGENT_FLAVOR = "bash"

# One-turn/multi-turn agents that the generic in-container runner can build
# directly from an AgentSpec.
SIMPLE_AGENT_FLAVORS = ("bash", "bash_task", "bash_task_read", "bash_skills")

# Workflow arms are selected with the same AGENT_FLAVOR knob, but suites that
# support them provide a custom build_agent facade.
WORKFLOW_AGENT_FLAVORS = ("loop", "pdr")

AGENT_FLAVORS = SIMPLE_AGENT_FLAVORS + WORKFLOW_AGENT_FLAVORS


def flavor_from_env() -> str:
    """The selected flavor from ``AGENT_FLAVOR``, validated against the vocabulary."""

    flavor = (
        os.environ.get(AGENT_FLAVOR_ENV) or DEFAULT_AGENT_FLAVOR
    ).strip().lower() or DEFAULT_AGENT_FLAVOR
    if flavor not in AGENT_FLAVORS:
        raise SystemExit(
            f"Unsupported {AGENT_FLAVOR_ENV}={flavor!r}; "
            f"expected one of {AGENT_FLAVORS}."
        )
    return flavor
