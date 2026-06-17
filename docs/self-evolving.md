# Self-Evolving Agents

This guide explains the self-evolving part of Simple Agent Lab: what it is, how
the pieces fit, and how to run or write your own recipe. It is the doc to read
when "the agent improves itself" needs to become something you can inspect,
modify, and trust.

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
| A derived **archive** of selectable versions (`archive.py`) | the **slice** (instances) and the **seed** version |

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
auto-promoting whenever the criterion accepts. This is the simple recipe.

**Parallel, open-ended** — `open_ended.run_evolution(...)` runs several branches
per round, admits every *valid* child into the archive (worse-but-valid versions
stay as stepping stones), promotes the best valid child of the round, and lets
the strategy pick a parent from the whole archive. This is the DGM recipe.

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
  (accept on strict gain), `not_worse(dim, tol=)` (a guard), `valid_when(dim)`
  (open-ended admission — accept any child that produced gradable runs), and
  `guarded(objective, guards)` ("optimize X subject to Y").
- `model_program_strategy(*, provider, prefix="agent/", system_prompt, parent_selection="best")`
  (`evolution/components/strategy.py`) — the model-driven meta-strategy: an LLM
  rewrites whole files under a path prefix (Python is AST-validated), returning a
  `Proposal`. Benchmark-agnostic; the recipe injects the domain prompt.
- `archive.nodes(workspace)` and `archive.select_parent(nodes, method=)` — the
  DGM/HyperAgents parent-selection policies (`latest`, `best`, `score_prop`,
  `score_child_prop`, `random`) derived from the decision log.

## The two recipes

| Recipe | What it shows | Loop | Knobs |
| --- | --- | --- | --- |
| [`recipes/simple/`](../recipes/simple/README.md) | How little code starts a real self-evolving run | `Experiment.run` (sequential) | minimal — rounds, max-turns |
| [`recipes/dgm/`](../recipes/dgm/README.md) | A faithful Darwin Gödel Machine reproduction | `open_ended.run_evolution` (parallel) | all of them — branches, parent selection, meta-concurrency, parallelism |

Both evolve the **whole agent program** under `agent/`: the model rewrites the
agent's own Python, each candidate is graded on a train slice in a SWE-bench
Docker sandbox, and the best valid agent is scored on a held-out test slice.
Start with `recipes/` for prerequisites and quick-starts.

## Write your own recipe

A recipe is just the four components plus a slice. The smallest possible shape:

```python
from simple_agent_lab.evolution import Experiment
from simple_agent_lab.evolution.components.criterion import improve
from simple_agent_lab.evolution.components.strategy import model_program_strategy

exp = Experiment(
    workspace,
    rollout=my_rollout,          # (Version, Slice) -> Runs   (run the agent)
    reward=my_reward,            # Run -> float               (score a run)
    criterion=improve("reward"), # judge candidate vs baseline
    slice_id="train",
    instances=train_records,     # the frozen task set
    seed=seed_files,             # the initial agent's files
)

strategy = model_program_strategy(provider=provider, prefix="agent/")
decisions = exp.run(strategy, n=4)   # sequential
```

To go open-ended (archive + parallel branches), pass the same components to
`open_ended.run_evolution(workspace, components, slice_, rounds=..., branches=...)`
with `criterion=valid_when("reward")` so worse-but-valid children stay as
stepping stones. See `recipes/dgm/evolve.py` for the fully wired version.

The only benchmark-specific piece is the rollout (and its reward). For
SWE-bench, that glue already exists in the adapter
`src/simple_agent_lab/evals/suites/swebench/evolving_rollout.py`; a new benchmark
adds its own adapter and recipe and never touches the substrate.

## Where things live

```text
src/simple_agent_lab/evolution/    # substrate (benchmark-agnostic)
  kernel/        store, log, loop   # versions, decision log, the loop + guarantees
  components/    reward, criterion, rollout, strategy
  archive.py, open_ended.py         # archive + parallel open-ended loop
src/simple_agent_lab/evals/suites/swebench/evolving_rollout.py   # SWE-bench adapter
recipes/                            # runnable recipes + ops scripts
```

The boundary is enforced: the substrate never imports a benchmark, the SWE-bench
adapter never imports Docker (`scripts/arch_lint.py`), and Docker/host probing
lives only in the recipe layer (`recipes/_shared.py`).
