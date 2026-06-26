"""Evolution infra: a legible, modular substrate for self-evolving agents.

User surface (plain functions, the framework owns the machinery):

    exp = Experiment(workspace, rollout=..., reward=..., criterion=...)
    def my_strategy(ctx: Context) -> Proposal | None: ...
    exp.step(my_strategy);  exp.history();  exp.rollback()

Components are swappable (rollout / reward / strategy / criterion); the kernel
(store / log / loop) owns the guarantees. See README.md in this package for the
framework guide.
"""

from simple_agent_lab.evals.instances import InstanceSet
from simple_agent_lab.evolution.experiment import Experiment
from simple_agent_lab.evolution.registry import Use
from simple_agent_lab.evolution.source_tree import (
    CANDIDATE_PACKAGE,
    CANDIDATE_SOURCE_CONTAINER_SRC,
    CANDIDATE_SRC,
    CANDIDATE_TREE,
    SOURCE_ROOT,
    candidate_source_artifacts,
    cheap_validate_source_tree,
    source_tree_agent_surface,
    source_tree_surface,
    validate_source_tree_edits,
)
from simple_agent_lab.evolution.surface import (
    AgentSurface,
    SurfaceComponent,
    ValidatedEdits,
)
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
    "InstanceSet",
    "Use",
    "SOURCE_ROOT",
    "CANDIDATE_TREE",
    "CANDIDATE_SRC",
    "CANDIDATE_PACKAGE",
    "CANDIDATE_SOURCE_CONTAINER_SRC",
    "source_tree_surface",
    "source_tree_agent_surface",
    "validate_source_tree_edits",
    "candidate_source_artifacts",
    "cheap_validate_source_tree",
    "AgentSurface",
    "SurfaceComponent",
    "ValidatedEdits",
    "Context",
    "Proposal",
    "Decision",
    "Run",
    "Slice",
    "Verdict",
    "Version",
]
