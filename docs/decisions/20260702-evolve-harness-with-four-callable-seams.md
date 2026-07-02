---
title: "Ship An Evolution Harness With Four Callable Seams"
status: Accepted
date: 2026-07-02
slug: evolve-harness-with-four-callable-seams
---

# Ship An Evolution Harness With Four Callable Seams

## Context

ADR [treat-self-evolution-as-harness-capability](20260507-treat-self-evolution-as-harness-capability.md)
committed this repo to an explicit evolution loop
(`run -> trace -> evaluate -> propose candidate -> compare -> accept or
reject`) but did not fix its code shape. Since then the research field has
converged on one outer loop with different substitutions:

- ShinkaEvolve evolves a single program file (LLM mutations inside
  `EVOLVE-BLOCK` markers, archive + islands + novelty filtering).
- Agentic Harness Engineering (arXiv:2604.25850) evolves the harness files
  around a fixed coding agent (prompt, tools, middleware, memory).
- Hyperagents / Darwin Gödel Machine (arXiv:2603.19461) evolve agent code —
  including the code that performs the evolution — over a stepping-stone
  archive with performance-proportional parent selection.
- ADAS, AFlow, EvoAgent, and prompt-evolution methods evolve workflow graphs,
  agent configs, or prompt text.

Every one of these is: sample parents from an archive, propose a mutated
candidate (usually with an LLM), evaluate it for a fitness signal, decide
acceptance, record lineage, repeat. What differs is only the *payload* (what
evolves), the *proposer*, the *evaluator*, and the *selection pressure*.

Simple Agent Lab wants to be to agent-evolution research what mini-swe-agent
is to coding agents: a harness small enough to read in one sitting, where a
new method is a fork or a couple of substituted callables, not a plugin.

## Decision

Add `src/simple_agent_lab/evolve/`: a sequential, inspectable evolution
harness whose core is one plain function, `run_evolution`, over four callable
seams:

- `ProposeFn(parents, rng) -> Proposal` — how a new candidate payload is
  produced (LLM-driven or scripted).
- `EvaluateFn(candidate) -> Evaluation` — the feedback signal: scalar
  `fitness`, a `correct` flag, structured `metrics`, and free-text `feedback`
  that flows back into the next proposal prompt.
- `SelectFn(archive, rng) -> parents` — parent + inspiration sampling
  (greedy, uniform, and rank-power-law ship as defaults).
- `AcceptFn(candidate, evaluation, archive) -> Decision` — explicit
  accept/reject with a human-readable reason.

Supporting choices:

- **What evolves is plain data.** A `Candidate.payload` is a JSON-able
  mapping (prompt text, code, config, tool descriptions...). The harness
  never interprets it; only the user's proposer and evaluator do. This is
  what lets one loop cover prompt evolution, harness evolution, and program
  evolution.
- **The archive is an append-only JSONL file**, one `EvolutionRecord` per
  line: candidate (payload + lineage + operator + proposer note), evaluation,
  and the accept/reject decision with its reason. This satisfies the earlier
  ADR's record contract, makes every run diff-able and resumable
  (`Archive.load`), and avoids a database dependency.
- **Sequential first.** One candidate is proposed and evaluated at a time.
  Async proposal/evaluation overlap (ShinkaEvolve's big speedup) is a later,
  additive concern; it must not complicate the readable v1 loop.
- **Incorrect candidates never enter the population.** They are recorded
  (auditable) but not selectable as parents.
- **LLM mutation is a helper, not the core.** `llm_propose` renders parent
  payload fields as fenced blocks and parses the same shape back, over an
  injectable `ask: Callable[[str], str]` so tests and offline demos never
  need a provider. `EVOLVE-BLOCK` marker utilities (`code_blocks`) support
  ShinkaEvolve-style constrained code mutation as one payload convention
  among many.
- **Agent evaluation reuses the existing runtime.** `agent_task_evaluator`
  builds an `Agent` from a candidate, runs it over a task list via
  `workflow.base.run_agent`, and folds per-task scores into an `Evaluation` —
  no parallel agent loop.

## Consequences

- A new evolution method is expressed as substituted callables plus a run
  script, mirroring how `workflow/` treats orchestration patterns; methods
  become comparable because they share the archive format and loop.
- The archive JSONL doubles as the experiment log, the resume point, and the
  dataset for later analysis (or training), in line with the trace-first
  direction.
- Out of scope for v1, deliberately: islands/migration, novelty filtering via
  embeddings, bandit LLM ensembles, async pipelines, and model-weight
  training. Each can be layered on (a `SelectFn` owns islands; a `ProposeFn`
  owns novelty resampling) without changing record shapes.
- The scalar-`fitness` contract is a simplification; multi-objective work
  must encode its trade-off into `fitness` (metrics keep the raw values).

## Alternatives Considered

- Adopt ShinkaEvolve's shape directly (SQLite archive, single-file program
  payload, Hydra config). Rejected: heavier dependencies and a code-only
  payload contract; this repo's subjects are agents, not just programs.
- Make evolution a `workflow/` orchestrator. Rejected: workflows compose
  agents inside one task; evolution runs *many experiments over agent
  definitions* and owns persistence/lineage, which no workflow needs.
- A class-based framework (`Mutator`/`Evaluator` base classes). Rejected:
  plain callables match `core.GenerateFn` taste and keep forks trivial.
- Payload as files on disk (AHE-style harness directory). Deferred: a
  file-tree payload can be represented as a mapping of paths to contents
  today; a dedicated convention can come with a real use case.
