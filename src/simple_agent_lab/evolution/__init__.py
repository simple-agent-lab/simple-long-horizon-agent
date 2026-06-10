"""Evolution framework (library-first skeleton).

Three nouns, two verbs: a *bundle* (immutable directory = one version of
agent behavior), a *run* (one rollout's artifacts), and the *decision log*
(append-only gate verdicts). ``rollout`` produces runs from a bundle;
``update`` functions (cookbook code, not framework code) propose candidate
bundles; the ``gate`` compares and logs; promotion moves a pointer.

Design: docs/design/20260610-evolution-framework-spec.md. The containerized
rollout adapter lives in ``simple_agent_lab.evolution.rollout`` and is
imported explicitly (it pulls in the evals framework).
"""

from simple_agent_lab.evolution.bundle import (
    Manifest,
    bundle_hash,
    load_provider,
    promote,
    read_manifest,
    resolve,
    stage_bundle,
)
from simple_agent_lab.evolution.catalog import CatalogRow, build_catalog
from simple_agent_lab.evolution.decisions import (
    Decision,
    append_decision,
    hit_rate,
    read_decisions,
    seen_candidate,
)
from simple_agent_lab.evolution.gate import (
    COST_TOKENS,
    Criterion,
    EvalSlice,
    GateResult,
    Judgment,
    Measure,
    Measurement,
    REWARD,
    Rollout,
    gate,
    guarded,
    improve,
    minimize,
    not_worse,
)
from simple_agent_lab.evolution.agent import (
    EpisodeReport,
    EvolutionConfig,
    make_evolution_agent,
    make_evolution_tools,
    run_episode,
)

__all__ = [
    "Manifest",
    "bundle_hash",
    "load_provider",
    "promote",
    "read_manifest",
    "resolve",
    "stage_bundle",
    "CatalogRow",
    "build_catalog",
    "Decision",
    "append_decision",
    "hit_rate",
    "read_decisions",
    "seen_candidate",
    "COST_TOKENS",
    "Criterion",
    "EvalSlice",
    "GateResult",
    "Judgment",
    "Measure",
    "Measurement",
    "REWARD",
    "Rollout",
    "gate",
    "guarded",
    "improve",
    "minimize",
    "not_worse",
    "EpisodeReport",
    "EvolutionConfig",
    "make_evolution_agent",
    "make_evolution_tools",
    "run_episode",
]
