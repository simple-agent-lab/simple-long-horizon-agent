"""Evolution framework.

User surface (verl/slime-style — plain functions, the framework owns the
machinery; see ``lab.py`` for the quickstart):

    Lab          one experiment: workspace + how to run + how to score
    reward fn    (run_dir) -> float                 # optional, verl-style
    strategy fn  (EpisodeContext) -> Proposal|None  # optional; omit and use
                                                    # lab.evolve() for the
                                                    # agent-driven mode

Engine room (imported explicitly when you work on the framework itself):
content-addressed bundles, the gate (measures x criteria), the append-only
decision log, the run catalog, and the containerized rollout adapter in
``simple_agent_lab.evolution.rollout``. Design:
docs/design/20260610-evolution-framework-spec.md.
"""

from simple_agent_lab.evolution.lab import (
    EpisodeContext,
    Lab,
    Proposal,
    RewardFn,
    StepReport,
    StrategyFn,
)
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

__all__ = [
    # --- user surface ---
    "Lab",
    "EpisodeContext",
    "Proposal",
    "StepReport",
    "RewardFn",
    "StrategyFn",
    # --- engine room ---
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
]
