# Self-Evolving Agents

This guide explains the self-evolving part of Simple Agent Lab: what it is, how
the pieces fit, and how to run or write your own recipe. It is the doc to read
when "the agent improves itself" needs to become something you can inspect,
modify, and trust.

For the comprehensive framework README with direct Python use, YAML config
shape, output artifacts, and source layout, start with
[`src/simple_agent_lab/evolution/README.md`](../../src/simple_agent_lab/evolution/README.md).
This agent-native guide keeps the mental model and boundary rules close at hand.

## The one-sentence idea

A **strategy** proposes a change to the agent, the change is run on a frozen set
of tasks, a **criterion** judges it against the unchanged agent on the *same*
tasks, and the comparison is written to an append-only log. Repeat. The agent
that survives is the one the evidence selected.

## Mental model: substrate vs. recipe

There are two layers, and keeping them separate is the whole design.

- The **substrate** (`src/simple_agent_lab/evolution/`) is benchmark-agnostic.
  It owns the machinery and the guarantees: immutable versions, fair A/B
  comparison, the decision log, and promotion. You do not edit it to add a new
  benchmark.
- A **recipe** (`recipes/`) is a small, runnable example that plugs concrete
  policies into the substrate: *how* to run the agent (rollout), *how* to score
  a run (reward), *what* change to try (strategy), and *how* to judge it
  (criterion).

The substrate is the engine; a recipe is the experiment you run on it.

### What the substrate guarantees vs. what a recipe supplies

| The kernel owns (you get it for free) | The recipe supplies (you write it) |
| --- | --- |
| Immutable, content-addressed **versions** (`kernel/store.py`) | **rollout**: `(Version, Slice) -> Runs` — run the agent on a task set |
| **Fair A/B**: baseline and candidate run the *same* frozen `Slice` (`kernel/loop.py` `_check_pair`) | **reward**: `Run -> float | {dim: float}` — turn a run into a score |
| Append-only **decision log** (`kernel/log.py`) | **strategy**: `Context -> Proposal | None` — the change to try |
| **Promotion** is evidence-driven and log-then-promote | **criterion**: `(baseline, candidate) -> Verdict` — how to judge |
| Proposal bases compare against the exact version they branch from | the **slice** (instances), **seed** version, and any recipe-specific archive |

## The loop

One generation (`Experiment.step`, in `kernel/loop.py`) is four moves —
observe, propose, compare, record:

```text
   current version ──rollout──▶ base runs ──reward──▶ base scores
        │                                                  │
        ▼                                                  │
   strategy(Context)                                       │
        │                                                  │
        ▼                                                  ▼
   Proposal(edits) ─stage─▶ candidate ─rollout─▶ cand runs ─reward─▶ cand scores
                                                                        │
                                                                        ▼
                       criterion(base scores, cand scores) ────▶ Verdict
                                                                        │
                                                              append to decision log
                                                                        │
                                                          accepted? ──▶ promote
```

Two properties make this trustworthy and are why the kernel — not the recipe —
owns this sequence:

- **Fair comparison.** Baseline and candidate are rolled out on the *same*
  frozen `Slice`; the kernel refuses to compare different instance sets.
- **Log before promote.** The decision is written first, then the pointer
  moves. A crash can leave an accepted-but-unpromoted record, never a promoted
  version with no evidence.

## Two ways to drive the loop

**Sequential** — `Experiment.run(strategy, n=...)` calls `step` n times,
auto-promoting whenever the criterion accepts. The configured simple runner uses
the same step loop directly so it can add heldout checkpoints before, during, or
after training.

**Parallel, open-ended** —
`recipes.dgm.algorithm.open_ended.run_evolution(...)` runs several branches per
round, admits every *valid* child into the archive (worse-but-valid versions
stay as stepping stones), promotes the best valid child of the round, and lets
the strategy pick a parent from the whole archive. This is the DGM recipe; the
archive machinery is intentionally recipe-local.

Both share the exact same kernel guarantees; open-ended only changes *which*
parents are explored and *how many* candidates run at once.

## Key building blocks (real symbols)

- `Experiment(workspace, *, rollout, reward, criterion, strategy=None, slice_id, instances, seed)` —
  the slim wirer (`evolution/experiment.py`). `step` / `run` / `history` /
  `rollback` / `current`. No policy lives here.
- `Context` (`evolution/types.py`) — what a strategy sees: `runs`, `current`,
  `workspace`, `decisions`, `reward`, plus `.failures` and `.version(hash)`.
- `Proposal(edits, note, evidence, base, kind)` — a strategy's output. `edits`
  maps a path to full file content (or `None` to retire it); `base` is the
  parent hash to branch from (`""` = current).
- Criterion combinators (`evolution/components/criterion.py`): `improve(dim)`
  (accept on strict gain), `promote_not_worse(dim, tol=)` (simple recipe
  promotion), `not_worse(dim, tol=)` (a guard), `valid_when(dim)` (open-ended
  admission — accept any child that produced gradable runs), and
  `guarded(objective, guards)` ("optimize X subject to Y").
- `source_tree_agent_strategy(*, provider, repo_root, ...)`
  (`evolution/components/repo_strategy.py`) — the source-tree meta-strategy used
  by the simple recipe: it copies the repo, overlays the parent version's
  `src/simple_agent_lab/**` files, writes a compact `SELF_EVOLUTION_CONTEXT.md`
  briefing, lets a bash-capable meta-agent inspect and edit the temporary copy,
  and converts changed Python source files into a `Proposal`.
- `recipes.dgm.evolve.dgm_agentic_strategy(...)` — the DGM recipe's agentic
  self-improvement strategy: select a parent from the archive, materialize its
  `agent/` package, load that package as a SAL agent, run it on a
  self-improvement task, and turn `agent/` diffs into a child proposal.
- `recipes.ahe.strategy.ahe_agent_strategy(...)` — the AHE recipe's
  role-separated evolve-agent strategy: run analysis, stage the analysis and
  harness files into a workspace, let a SAL evolve agent edit `harness/`, and
  write `change_manifest.json`.
- `model_program_strategy(*, provider, prefix="agent/", system_prompt, parent_selection="current", parent_selector=None)`
  (`evolution/components/strategy.py`) — a lower-level model-driven wrapper
  strategy: an LLM rewrites whole files under a path prefix (Python is
  AST-validated), returning a `Proposal`. Benchmark-agnostic; recipes or tests
  inject the domain prompt. Any non-current parent-selection policy must be
  supplied by the recipe through `parent_selector`.
- `recipes.dgm.algorithm.archive.nodes(workspace)` and
  `recipes.dgm.algorithm.archive.select_parent(nodes, method=)` — the DGM parent-selection
  policies (`latest`, `best`, `score_prop`, `score_child_prop`, `random`)
  derived from the decision log.

## The three recipes

| Recipe | What it shows | Loop | Knobs |
| --- | --- | --- | --- |
| [`recipes/simple/`](../../recipes/simple/README.md) | Config-backed generic `algorithm: simple` train-slice evolution with optional heldout before/final reporting | Sequential `Experiment.step` loop | YAML config — train/heldout paths, rounds, evaluation flags, execution settings |
| [`recipes/ahe/`](../../recipes/ahe/README.md) | AHE-style analysis, evolve-agent workspace, component manifest, and ledgering on SWE-bench | Sequential `Experiment.step` loop | YAML config — train/heldout paths, rounds, evaluation flags, execution settings |
| [`recipes/dgm/`](../../recipes/dgm/README.md) | Config-backed DGM-style archive mechanics with parent-agent self-improvement and recipe-local held-out scoring | `recipes.dgm.algorithm.open_ended.run_evolution` (parallel) | YAML config plus CLI overrides — branches, parent selection, meta-concurrency, parallelism |

The simple recipe now evolves the framework source under `src/simple_agent_lab/**`
through the `source_tree` surface and `source_tree_agent` strategy. Each
candidate is staged as a source tree and graded on a train slice in a SWE-bench
Docker sandbox. The simple recipe runs through the generic config-backed runner
and supports dry-runs plus generic heldout before/final reports when
`instances.heldout` and `evaluation.*` are enabled. The AHE recipe adds a
model-backed analyzer, stages its artifacts for a role-separated SAL evolve
agent, and records a change manifest plus ledger. The DGM recipe is also
YAML-backed through `configs/dgm_swebench.yaml` and `recipes/dgm/config.py`, but
owns the open-ended archive policy, parent-agent self-improvement strategy, and
archive-specific held-out official scoring workflow. Start with `recipes/` for
prerequisites and quick-starts.

## Writing a self-evolving run

The generic v1 authoring model is: name objects in YAML, implement behavior in
Python, and register the Python factories before loading the config. Framework
objects do not read YAML directly.

For the simple path, choose these pieces:

- `Suite` — the benchmark or task runner. For SWE-bench, recipes use the existing
  `evals/swebench/suite.py` `SwebenchSuite`.
- `AgentSurface` — the editable agent shape: default files, valid editable
  components, and how a version is staged into each run. The simple recipe's
  `source_tree` surface seeds current `src/simple_agent_lab/**/*.py` files.
- editable components — the surface slices the strategy may change, such as
  `agent_program`, `prompts`, `tool_policy`, `memory_policy`, or `everything`.
- `InstanceSet` — the frozen train slice loaded from the JSONL path named in
  config.
- `evolution.algorithm` — currently `simple` in the generic builder. DGM's
  open-ended archive algorithm and DGM YAML schema stay recipe-local under
  `recipes/dgm/`.

The YAML config selects names and ordinary settings (`suite.name`,
`surface.name`, `instances.train.path`, execution options, strategy settings,
criterion, and rounds). Python supplies the behavior behind those names:
`AgentSurface` validation/staging, `rollout_from_suite`, suite host/container
halves, strategy factories, reward, and criterion. SWE-bench simple runs compose
through `recipes/simple/evolve.py` factory registration, `SwebenchSuite`,
`source_tree_agent_surface`, `source_tree_agent_strategy`, and
`rollout_from_suite`; DGM uses `recipes/dgm/config.py` for its recipe-local YAML
schema and `recipes/dgm/swebench.py` for its rollout and official-scoring
workflow.

## Write your own recipe

A recipe is just the four components plus a slice. The smallest possible shape:

```python
from pathlib import Path

from simple_agent_lab.evolution import Experiment
from simple_agent_lab.evolution.components.criterion import improve
from simple_agent_lab.evolution.components.repo_strategy import source_tree_agent_strategy

exp = Experiment(
    workspace,
    rollout=my_rollout,          # (Version, Slice) -> Runs   (run the agent)
    reward=my_reward,            # Run -> float               (score a run)
    criterion=improve("reward"), # judge candidate vs baseline
    slice_id="train",
    instances=train_records,     # the frozen task set
    seed=seed_files,             # the initial agent's files
)

strategy = source_tree_agent_strategy(provider=provider, repo_root=Path("."))
decisions = exp.run(strategy, n=4)   # sequential
```

To go open-ended (archive + parallel branches), treat that as a recipe policy:
copy or import the DGM recipe helpers under `recipes/dgm/` and pass the same
components to
`recipes.dgm.algorithm.open_ended.run_evolution(workspace, components, slice_, rounds=..., branches=...)`
with `criterion=valid_when("reward")` so worse-but-valid children stay as
stepping stones. See `recipes/dgm/evolve.py` for the fully wired version.

The benchmark-specific piece is the suite plus whatever rollout/reward support
the recipe chooses to add. For SWE-bench, the benchmark interface is
`SwebenchSuite`; the in-wheel
`simple_agent_lab.evals.suites.swebench.evolving` module can run against the
staged source tree for the simple recipe. DGM's extra run/scoring helpers live
in `recipes/dgm/swebench.py` because they are DGM workflow support, not universal
benchmark API. A new benchmark adds its own suite and, only if it needs
evolvable in-container code, a small container hook that consumes the recipe's
chosen artifact without touching the substrate.

## Where things live

```text
src/simple_agent_lab/evolution/    # substrate (benchmark-agnostic)
  kernel/        store, log, loop   # versions, decision log, the loop + guarantees
  components/    reward, criterion, rollout, strategy
  surface.py, source_tree.py        # editable surfaces + source-tree staging
  agent_package.py                  # lower-level wrapper-package test support
evals/swebench/suite.py             # SWE-bench benchmark interface
src/simple_agent_lab/evals/suites/swebench/evolving.py
                                    # SWE-bench container hook for staged candidates
recipes/                            # runnable recipes + ops scripts
  simple/evolve.py
  runtime.py
  dgm/config.py                      # DGM YAML schema and CLI overrides
  dgm/evolve.py
  dgm/swebench.py                    # DGM-specific SWE-bench run/scoring support
  dgm/algorithm/archive.py, open_ended.py, repo_edits.py
  dgm/ops/baseline.py, report.py
```

The boundary is enforced: the substrate never imports a benchmark or the
top-level host `evals/` tree, DGM's SWE-bench support never imports Docker
(`scripts/arch_lint.py`), and Docker/host probing lives only in the recipe layer
(`recipes/runtime.py`).

## Research-readiness bar

Treat a self-evolving run as research evidence only when the artifact set names
the exact config, train/test instance files, model/provider, image namespace/tag
policy, fallback count, missing-result count, decision log, train score deltas,
and held-out before/final scores. Accepted ties and DGM valid-but-regressed
archive children must be reported separately from strict improvements. A dry-run
or a train-only lineage is useful engineering evidence, but it is not a
performance claim.
