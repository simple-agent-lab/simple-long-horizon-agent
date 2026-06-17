"""Evolution infra: a legible, modular substrate for self-evolving agents.

User surface (plain functions, the framework owns the machinery):

    exp = Experiment(workspace, rollout=..., reward=..., criterion=...)
    def my_strategy(ctx: Context) -> Proposal | None: ...
    exp.step(my_strategy);  exp.history();  exp.rollback()

Components are swappable (rollout / reward / strategy / criterion); the kernel
(store / log / loop) owns the guarantees. Design:
docs/design/20260612-evolution-infra-redesign.md
"""

from simple_agent_lab.evolution.config import Config, Use
from simple_agent_lab.evolution.experiment import Experiment
from simple_agent_lab.evolution.types import (
    Context,
    Decision,
    Proposal,
    Run,
    Slice,
    Verdict,
    Version,
)

__all__ = [
    "Experiment",
    "Config",
    "Use",
    "Context",
    "Proposal",
    "Decision",
    "Run",
    "Slice",
    "Verdict",
    "Version",
]
