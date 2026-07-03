"""Evolution harness: one readable loop over four callable seams.

The research field's methods — ShinkaEvolve, Darwin Gödel Machine and
Hyperagents, harness evolution, ADAS/AFlow, prompt evolution — all share one
outer loop and differ only in what they substitute into it. This package is
that loop, kept small enough to read in one sitting (the mini-swe-agent
posture, applied to agent-evolution research)::

    archive = Archive(path="runs/out/exp/archive.jsonl")
    result = run_evolution(
        seeds=[{"system_prompt": "You are a helpful agent."}],
        propose=...,    # ProposeFn:  (parents, rng) -> Proposal
        evaluate=...,   # EvaluateFn: (candidate)    -> Evaluation
        select=...,     # SelectFn:   (archive, rng) -> [parent, *inspirations]
        accept=...,     # AcceptFn:   (candidate, evaluation, archive) -> Decision
        budgets=EvolutionBudgets(max_candidates=100),
        archive=archive,
    )

What evolves is plain data (`Candidate.payload`, a JSON-able mapping); the
harness never interprets it. Every candidate — accepted, rejected, or failed
— lands in the append-only JSONL archive with lineage, evaluation, and the
decision reason, so a run is auditable, resumable (`Archive.load`), and
comparable across methods (ADR evolve-harness-with-four-callable-seams).

Helpers, each optional:

- `ComponentSpec` / `GenomeSpec` / `genome_propose` — the typed component
  layer: declare which components evolve (kind, proposer-facing docs,
  validation) and get a schema-aware proposer with legality checks; the
  standard `agent_genome` + `build_genome_agent` cover the "evolve the
  agent definition" experiment end to end (`genome`).
- `llm_propose` / `ask_from_provider` — LLM-driven mutation of named string
  payload fields (`propose`).
- `mix_operators` / `crossover_propose` — weighted operator sampling and an
  LLM crossover operator, labeled per child in the archive (`operators`).
- `evolve_blocks` / `replace_evolve_blocks` / `check_immutable_regions` —
  ShinkaEvolve-style `# EVOLVE-BLOCK-START/END` constrained code mutation
  (`code_blocks`).
- `agent_task_evaluator` — fitness = an `Agent` built from the candidate,
  run over a task list via the core runtime (`evaluators`).
- `select_best` / `select_uniform` / `select_weighted` / `select_islands`
  (lineage-derived islands with migration), `accept_correct` /
  `accept_improves_best` — selection pressure (`select`).
"""

from .archive import Archive, record_from_dict, record_to_dict
from .code_blocks import (
    EVOLVE_END,
    EVOLVE_START,
    check_immutable_regions,
    evolve_blocks,
    replace_evolve_blocks,
)
from .evaluators import agent_task_evaluator
from .genome import (
    INSTRUCTIONS_KEY,
    SYSTEM_PROMPT_KEY,
    TOOL_DESCRIPTIONS_KEY,
    ComponentKind,
    ComponentSpec,
    GenomeSpec,
    agent_genome,
    build_genome_agent,
    genome_propose,
    seed_agent_payload,
    validate_component,
    validate_payload,
)
from .loop import (
    EvolutionBudgets,
    EvolutionResult,
    EvolutionStatus,
    run_evolution,
)
from .operators import crossover_propose, mix_operators
from .propose import (
    AskFn,
    ask_from_provider,
    build_mutation_prompt,
    llm_propose,
    parse_fields,
    proposal_note,
    render_fields,
)
from .select import (
    accept_correct,
    accept_improves_best,
    select_best,
    select_islands,
    select_uniform,
    select_weighted,
)
from .types import (
    AcceptFn,
    Candidate,
    Decision,
    EvaluateFn,
    Evaluation,
    EvolutionRecord,
    Payload,
    Proposal,
    ProposeFn,
    SelectFn,
)

__all__ = [
    "AcceptFn",
    "Archive",
    "AskFn",
    "Candidate",
    "ComponentKind",
    "ComponentSpec",
    "Decision",
    "EVOLVE_END",
    "EVOLVE_START",
    "EvaluateFn",
    "Evaluation",
    "EvolutionBudgets",
    "EvolutionRecord",
    "EvolutionResult",
    "EvolutionStatus",
    "GenomeSpec",
    "INSTRUCTIONS_KEY",
    "Payload",
    "Proposal",
    "ProposeFn",
    "SYSTEM_PROMPT_KEY",
    "SelectFn",
    "TOOL_DESCRIPTIONS_KEY",
    "accept_correct",
    "accept_improves_best",
    "agent_genome",
    "agent_task_evaluator",
    "ask_from_provider",
    "build_genome_agent",
    "build_mutation_prompt",
    "check_immutable_regions",
    "crossover_propose",
    "evolve_blocks",
    "genome_propose",
    "llm_propose",
    "mix_operators",
    "parse_fields",
    "proposal_note",
    "record_from_dict",
    "record_to_dict",
    "render_fields",
    "replace_evolve_blocks",
    "run_evolution",
    "seed_agent_payload",
    "select_best",
    "select_islands",
    "select_uniform",
    "select_weighted",
    "validate_component",
    "validate_payload",
]
