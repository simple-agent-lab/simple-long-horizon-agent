---
title: "Evolution Infra: A Legible, Modular Substrate for Self-Evolving Agents"
status: Proposed
date: 2026-06-12
slug: evolution-infra-redesign
---

# Evolution Infra: A Legible, Modular Substrate for Self-Evolving Agents

## Thesis

The contribution we open-source is **infrastructure, not a method** — "VeRL for
self-evolving agents." VeRL is widely adopted not because it ships one clever
algorithm but because its *substrate* is clean and every part is swappable;
PPO/GRPO are just things it enables. We aim for the same: a small, legible
platform where a researcher can understand the whole loop in one sitting and
replace any module with their own idea.

This document supersedes the earlier, heavier draft (the
`evolution-framework-spec` / `self-evolution-mechanism` memos on the `pr-30`
branch, kept there as literature review and rationale). Those established the
guarantees and the research landscape; this redesign keeps the guarantees and
removes the front-loaded complexity.

## Mental model (the whole thing)

> A **version** of the agent → **run** it and **score** it → **propose** a
> change → **compare** old vs new → **record** the decision.

That is the entire system. Everything else is an implementation of one of those
steps.

**Five nouns** a reader must know:

| Noun | What it is |
|---|---|
| `Version` | immutable, content-addressed agent state: `{prompt, playbook, lessons, skills, model}`. `.hash` `.parent` `.read(file)` `.dir` |
| `Run` | one task instance executed: `.instance_id` `.reward` `.result` `.events()` `.dir` |
| `Slice` | the frozen instance set used for a fair A/B: `.id` `.instances` |
| `Proposal` | a strategy's output: `.edits {path: content \| None}` `.note` `.evidence` `.base?` |
| `Decision` | one logged comparison: `.baseline` `.candidate` `.scores` `.verdict` `.reason` `.runs` |

**Four swap points** (all plain callables):

```python
Rollout   = (Version, Slice)        -> Sequence[Run]
Reward    = (Run)                   -> float | Mapping[str, float]   # float = the "reward" dim
Strategy  = (Context)              -> Proposal | None
Criterion = (RunScores, RunScores) -> Verdict        # (baseline, candidate)
```

Three small supporting types complete the vocabulary: `Context` (what a strategy
sees), `RunScores = Mapping[instance_id, Mapping[dim, float]]`, and `Verdict`
(`accepted: bool`, `reason: str`, `deltas`). That is the *complete* surface — on
the order of a dozen names, not thirty.

## Architecture: a fixed kernel + a small registry

```text
evolution/
  kernel/                # THE GUARANTEES — plain code, not swappable
    store.py             #   immutable content-addressed versions + promote/rollback
    log.py               #   append-only decision log
    loop.py              #   the ~20-line driver that wires components & runs steps
  components/            # THE RESEARCH SURFACE — everything here is swappable
    rollout.py           #   how to run a version on a task slice
    reward.py            #   how to score a run
    strategy.py          #   what to change about the agent
    criterion.py         #   how to judge candidate vs baseline
  registry.py            # shallow {name -> factory} dicts; greppable, no magic
  config.py              # a typed dataclass that SELECTS components (never hides them)
  experiment.py          # the slim entry object: holds config, wires, runs
```

The split is the design's backbone: the **kernel owns the guarantees** and is
not swappable; the **components own all policy** and are entirely swappable; the
**experiment object only wires**. No policy (measure resolution, criterion
construction, agent loops) lives in the entry object.

## The kernel (the guarantees)

These rules never break. They are the reason a researcher can trust an
experiment's record and why an evolving agent cannot corrupt its own substrate.

1. **Versions are immutable and promoted whole.** A version is
   content-addressed; "current" is a pointer file. Context and weights co-adapt,
   so comparison, logging, and rollback always operate on a version hash, never
   on a single component.
2. **The kernel is not agentic and not evolvable.** `store`, `log`, and the loop
   driver are plain code no candidate can modify. This is the safety boundary: a
   strategy (human- or agent-written) can only *propose*; it can never promote,
   edit the log, or touch a promoted version.
3. **Nothing is deleted.** Rejected candidates stay in the store as stepping
   stones. Promotion and retention are separate concerns.
4. **A running agent never modifies itself.** A version is loaded at run start; a
   promotion takes effect on the *next* run. Rollback = move the pointer back.

### `store.py`

```python
def current(ws, *, namespace="") -> Version          # resolve the active pointer
def stage(ws, *, base, edits, note="") -> Version     # copy base, apply edits, store by hash
def promote(ws, version, *, namespace="") -> None      # atomic pointer rewrite (the ONLY mutation)
def version(ws, hash_) -> Version                      # any archived version, rejected ones included
```

`stage` applies `edits` (a `{path: content}` map; a `None` value is a tombstone
that retires an inherited file). Identical content re-staged returns the existing
version with its original provenance. The optional `namespace` argument is the
seam for future sandboxed (meta) evaluation; the default empty namespace is the
real pointer.

### `log.py`

```python
def append(ws, baseline, candidate, verdict) -> Decision   # one JSONL record per comparison
def read(ws, *, limit=None) -> Sequence[Decision]
def hit_rate(ws, *, kind=None, window=None) -> float       # for diagnostics / future scheduling
```

One append-only JSONL record per comparison, referencing run directories by
**workspace-relative** ref (the log outlives any one machine). Records carry the
baseline/candidate hashes, the per-dimension scores, the verdict, the
human-readable reason, and the slice identity.

### `loop.py`

The driver, readable in one sitting:

```python
def step(ws, c, slice_, *, auto_promote=True) -> Decision | None:
    current   = store.current(ws)
    base_runs = c.rollout(current, slice_)               # reused if already measured
    proposal  = c.strategy(Context(base_runs, current, log.read(ws), c.reward))
    if proposal is None:
        return None                                      # the strategy declined this step
    candidate = store.stage(ws, base=proposal.base or current,
                            edits=proposal.edits, note=proposal.note)
    cand_runs = c.rollout(candidate, slice_)
    verdict   = c.criterion(score(base_runs, c.reward), score(cand_runs, c.reward))
    decision  = log.append(ws, current, candidate, verdict)   # the ONLY decision record
    if verdict.accepted and auto_promote:
        store.promote(ws, candidate)
    return decision

def run(ws, c, slice_, *, n=1, auto_promote=True) -> list[Decision]:
    out = (step(ws, c, slice_, auto_promote=auto_promote) for _ in range(n))
    return [d for d in out if d is not None]

# score(runs, reward) -> RunScores: an internal helper that applies `reward`
# to each Run, yielding {instance_id: {dim: float}} for the criterion.
```

The "gate" is no longer a separate noun: it *is* this sequence (two rollouts +
apply criterion + record). Promotion is host-side and evidence-driven — the same
guarantee whether the strategy is a human function or an LLM agent.

## The components (the research surface)

Each component is one plain callable with one job. Shipped defaults are listed;
each is one function a researcher can replace.

- **`rollout`** — `(Version, Slice) -> Sequence[Run]`. Default `dataset_rollout`
  wraps the existing containerized eval harness (`run_dataset`). One `Run` per
  instance, always; a crashed instance has no `result.json` and scoring decides
  how to account for it. The slice is a *call argument*, so the same rollout
  serves the main comparison, a guard slice, or a held-out rotation without being
  rebuilt.
- **`reward`** — `(Run) -> float | Mapping[str, float]`. Default reads the
  `result.json` reward key. Returning a float means the single `reward`
  dimension; returning a dict declares multiple dimensions (e.g.
  `{"reward": 1.0, "cost_tokens": 78500}`). **This is also how multi-objective
  scoring enters — no separate "measures" system.**
- **`strategy`** — `(Context) -> Proposal | None`. This is what makes the system
  *self-evolving*: it diagnoses and decides *what* to change. `Context` exposes
  `runs`, `failures`, `current`, `decisions`, `reward`, and `version(hash)` so a
  strategy can branch from any archived version. The agent-driven mode is simply
  a strategy backed by an LLM agent (see Deferred), not a special code path.
- **`criterion`** — `(RunScores, RunScores) -> Verdict`. Judges baseline vs
  candidate. Aggregation lives *inside* the criterion (it receives per-run
  scores for both sides), so paired and multi-dimensional judgments are possible
  without extra machinery. Shipped: `improve("reward")` (mean climb),
  `not_worse(dim, tol)` (a guard), `guarded(objective, guards)` ("optimize X
  subject to Y"). A custom criterion is one function.

### Scoring without a measures algebra

The earlier draft had `RewardFn` *and* `Measure` *and* `MeasureFrame` *and*
`Measurement` *and* a criterion `.requires` auto-registration step. This redesign
collapses all of it into the natural pipeline:

```text
runs --reward--> per-run scores {instance_id: {dim: float}} --criterion--> verdict
```

`reward` produces named numbers; `criterion` consumes named numbers and does its
own aggregation. Multi-dimensional and paired statistics still work; five nouns
and the resolution machinery are gone.

## Config & registry (VeRL ergonomics, no magic)

Three levels of use, all resolving to the *same greppable functions*:

```python
# Level 1 — pass functions directly (most explicit)
exp = Experiment(workspace, rollout=my_rollout, reward=my_reward,
                 strategy=my_strategy, criterion=improve("reward"))

# Level 2 — name them in a typed Python config
cfg = Config(
    workspace="...",
    rollout   = Use("dataset", suite="swebench", n=20),
    reward    = Use("result_key"),
    strategy  = Use("reflect_failures"),
    criterion = Use("improve", dim="reward"),
)
exp = Experiment.from_config(cfg)
```

The registry is a **plain shallow dict per category** — no entry-point scanning,
no metaclass tricks:

```python
ROLLOUTS: dict[str, Callable] = {"dataset": dataset_rollout}
ROLLOUTS["mine"] = my_rollout          # register a custom one (or a 3-line @register)
```

`Config` is a typed dataclass, so the schema is discoverable in code; `Use(name,
**args)` is a name plus kwargs the factory receives. **Config selects; it never
hides** — every name maps to a function you can grep, and Level 1 bypasses the
registry entirely. This satisfies the repo's "no magic configuration" rule while
giving the config-driven feel ML researchers expect. (A YAML loader is a trivial
"Level 3" add when someone needs it; deferred per YAGNI.)

## On-disk layout

A workspace directory (default `.evolution/`, gitignored; location is a
parameter) holds all state as plain files — every concept inspectable with `ls`
and `cat`:

```text
.evolution/
  versions/<hash>/        # immutable version dirs: manifest.json + artifact files
                          #   (prompt.md, playbook.md, lessons.jsonl, skills/, provider.json)
  pointers/
    current.json          # {"hash": "...", "updated": "..."}
    shadow/               # reserved namespace dir for future sandboxed (meta) eval
  runs/<run_id>/<instance_id>/{input/, out/result.json, out/trajectory.jsonl}
  decisions.jsonl         # the append-only decision log
```

Versions are content-addressed (sha256 over the sorted file walk, excluding the
manifest). This is deliberately the storage model of git objects and OCI
layers: object-store keys are the hashes, sync is conflict-free, and only three
points ever need coordination (pointer update, log append, run-artifact
locality) — so a future distributed backend is a mechanical swap behind the
`store`/`log` functions, not a redesign.

## Deferred capabilities (seams left open, engines not built)

Because the contribution is the substrate, advanced capabilities are *things the
substrate enables later*, not core reading. Each leaves a cheap seam now.

| Capability | Seam built now (cheap) | Deferred (expensive) |
|---|---|---|
| **RL / SFT** (`trainer`) | run provenance that cannot be backfilled: trace records model/version, sampling params, and logprobs when available; `result.json` standard reward key; per-turn export already exists | the trainer = a "slow" `strategy` returning a `Version` whose `model` record points at a new checkpoint; gated like any other candidate |
| **Population / branching search** (`selector`) | `Proposal.base` + store keeps every version | a `selector(archive) -> base` component; until then base-choice lives in the strategy |
| **Meta** (self-improving improver) | `store`/`promote` accept an optional `namespace`; the loop is invocable in a sandbox | an `imp@k`-style criterion (scores episodes, not instances) + scheduling; ships as a one-page future-extension doc — **the same four components applied one level up** |

The agent-driven evolution mode (an LLM agent as the `strategy`, with tools that
map to `Context` reads and `Proposal` writes) is in scope as a component, but it
adds no new kernel concept: it is one more `strategy` implementation, gated by
the same loop.

## Reference-architecture validation target: HyperAgents / DGM

This substrate is meant to *host* published self-evolution methods as recipes,
not to bake any one of them in. HyperAgents (DGM's successor;
`.idea/hyperagents/`) is the strongest stress test, so we record the mapping
here as a standing validation target. The claim being validated: **reproducing
HyperAgents requires no kernel change — only component implementations on seams
this design already reserves.** (Per the
`treat-self-evolution-as-harness-capability` ADR, such open-ended
self-modification is an *advanced reference example*, not part of the core.)

| HyperAgents mechanic | Maps to | Status here |
|---|---|---|
| version = the agent's own source code (repo@commit + diff stack) | `Version` (a dir; `edits` carries any file, incl. `.py`) | substrate ready |
| archive of stepping stones; branch from any | store retains every version; `Proposal.base` | built |
| lineage / child-counts for parent selection | `Manifest.parent` + per-decision lineage in the log | recoverable |
| meta agent edits code → diff | a coding-agent `strategy` emitting `edits` | component (Plan 2, generalized) |
| eval runs the evolved *program* | a `code_rollout` (the version's code *is* the agent) | component (new `rollout` impl) |
| parent selection (`score_child_prop`) | the `selector` component | reserved seam |
| self-referential (meta edits meta/selector) | the meta layer (same four components, one level up) | reserved seam |
| staged eval (small slice → full) | a staged `criterion` + a second slice | expressible, not built |

**The three real gaps** (all component work, no redesign): (1) a `code_rollout`
that builds/runs the version's own program in a sandboxed container — this is
the one genuine impedance point, because our default rollout injects a bundle
into a *fixed* runtime rather than executing evolved code; (2) a coding-agent
`strategy` that edits arbitrary repo files; (3) the `selector` (and, for the
full "Hyper" claim, the meta layer). A concrete HyperAgents recipe ("Plan 3")
is therefore realistic future work, deliberately out of Plan 1.

## MVP scope and acceptance

**In:** `kernel/` (`store`, `log`, `loop`), the four `components/` with their
shipped defaults, `registry.py`, `config.py` (Levels 1-2), `experiment.py`, the
`dataset_rollout` adapter over the existing eval harness, two example strategies
(`reflect_failures`, `induce_skill`), and the LLM-agent strategy. Run against a
SWE-bench slice on the local Docker backend; unit-tested on the fake backend +
fake adapter.

**Out (seams reserved):** `trainer`, `selector`, meta / `imp@k` / namespaced
sandboxing, near-duplicate novelty detection (hash-equality only at first), and
the YAML config loader.

**Acceptance criteria:**

1. The decision log shows ≥2 different proposal kinds attempted (evidence a
   strategy is *deciding*, not an ETL script running).
2. ≥1 candidate is promoted through the loop and the improvement reproduces on
   the frozen slice.
3. Pointer rollback restores baseline behavior.
4. Rejected candidates remain retrievable from the store.

**Testing:** every kernel function is pure or filesystem-local → unit-testable
with `tmp_path`; an end-to-end episode runs on the fake backend with a scripted
fake adapter driving the LLM-agent strategy; one live smoke test mirrors the
existing eval smoke-script conventions.

## Migration from the current (pr-30) code

The pr-30 evolution package is treated as a toy and not carried forward. The
mapping for anyone porting useful pieces:

- `bundle.py` → `kernel/store.py` (rename `Bundle` → `Version`; same
  content-addressing).
- `decisions.py` → `kernel/log.py`; `catalog.py` → a read helper used by
  `Context`.
- `gate.py` → folded into `loop.py` + `components/criterion.py`. **Delete**
  `Measure`, `MeasureFrame`, `Measurement`, and `.requires` resolution.
- `lab.py` (god-object) → split into `kernel/loop.py` (the driver) +
  `experiment.py` (wiring only). The embedded `_resolve_measure`/`_measures_for`
  logic is removed; `evolve()` becomes an LLM-agent `strategy`.
- `rollout.py` → `components/rollout.py`, largely unchanged.
- Public exports shrink from ~30 names to ~12.

## Open questions

1. Lesson retrieval at injection time: top-k by tag is the MVP; when does
   similarity search earn its dependency?
2. Default guards on every promotion (e.g. a `tool_error_rate` or non-target
   regression guard): mandatory by default, or opt-in per proposal kind?
3. How much episode procedure lives in an evolvable strategy/playbook vs. fixed
   in the loop? Start maximally fixed; migrate into the strategy as evidence
   accumulates.
