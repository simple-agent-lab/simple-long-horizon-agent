"""Evolution framework.

User surface (verl/slime-style — plain functions, the framework owns the
machinery; see ``lab.py`` for the quickstart):

    Lab          one experiment: workspace + how to run + how to score
    reward fn    (Run) -> float                     # optional, verl-style
    strategy fn  (EpisodeContext) -> Proposal|None  # optional; omit and use
                                                    # lab.evolve() for the
                                                    # agent-driven mode

Extension points take typed read-only views, never raw paths: ``Run`` (one
instance run: result / reward / events / dir) and ``Bundle`` (one agent
version: read / manifest / hash / dir). The directories stay the source of
truth; the views are how code learns the layout contract.

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
    default_reward,
)
from simple_agent_lab.evolution.bundle import (
    Bundle,
    Manifest,
    bundle_hash,
    load_provider,
    promote,
    read_manifest,
    resolve,
    stage_bundle,
)
from simple_agent_lab.evolution.catalog import Run, build_catalog, runs_for
from simple_agent_lab.evolution.decisions import (
    Decision,
    append_decision,
    hit_rate,
    read_decisions,
    seen_comparison,
)
from simple_agent_lab.evolution.gate import (
    COST_TOKENS,
    Criterion,
    EvalSlice,
    GateResult,
    Judgment,
    Measure,
    MeasureFrame,
    Measurement,
    REWARD,
    Rollout,
    gate,
    guarded,
    improve,
    minimize,
    not_worse,
    paired_improve,
)

__all__ = [
    # --- user surface ---
    "Lab",
    "EpisodeContext",
    "Proposal",
    "StepReport",
    "RewardFn",
    "StrategyFn",
    "default_reward",
    # --- typed views (what's inside a path) ---
    "Run",
    "Bundle",
    # --- engine room ---
    "Manifest",
    "bundle_hash",
    "load_provider",
    "promote",
    "read_manifest",
    "resolve",
    "stage_bundle",
    "build_catalog",
    "runs_for",
    "Decision",
    "append_decision",
    "hit_rate",
    "read_decisions",
    "seen_comparison",
    "COST_TOKENS",
    "Criterion",
    "EvalSlice",
    "GateResult",
    "Judgment",
    "Measure",
    "MeasureFrame",
    "Measurement",
    "REWARD",
    "Rollout",
    "gate",
    "guarded",
    "improve",
    "minimize",
    "not_worse",
    "paired_improve",
]
