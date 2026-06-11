---
title: "Evolution Framework Technical Spec (Plan A: Library-First)"
status: Proposed
date: 2026-06-10
slug: evolution-framework-spec
---

# Evolution Framework Technical Spec (Plan A: Library-First)

Concrete technical specification for the evolution framework selected in
[`20260610-self-evolution-mechanism.md`](20260610-self-evolution-mechanism.md)
(see that memo for the literature review, competitive positioning, and
alternatives considered). Plan A is the library-first form: no resident
services, all state on the filesystem, every concept inspectable with `ls`
and `cat`. It is the common kernel of the two later growth forms (service
deployment, population search) — their interfaces are fixed here even
though their implementations are out of scope.

The whole model in one paragraph: *an agent's behavior is a directory
(bundle); evolution proposes a new directory (update); evidence is running
both and comparing (gate); history is an append-only decision log. All
intelligence lives in the evolution agent and update functions; all
guarantees live in the substrate.*

Those nouns are the **engine room** — the audience of this spec. The
**user surface** is deliberately smaller: three plain functions in the
verl/slime integration style (§3.3). A researcher integrating a strategy
or a custom reward never names a bundle, a gate, or the decision log.

> **Implementation status (2026-06):** the skeleton is merged on this
> branch — substrate (`bundle` / `decisions` / `gate` / `catalog`), the
> `dataset_rollout` adapter (provider injection only), the evolution agent
> with its five tools (episodes traced to `episodes/`), and the `Lab`
> user surface — including branchable proposal bases, per-proposal
> criteria, the manual promotion tier, the typed `Run` / `Bundle` views
> of §1.1 (engine-room functions included), edit tombstones, and
> measurement reuse (§4.1) — with unit tests and a deterministic demo.
> Container injection of prompt/playbook/skills (§2.4), imp@k scheduling,
> and similarity-level novelty remain open.

## 0. Invariants (the rules that never break)

1. **Bundles are immutable and promoted whole.** A bundle is
   content-addressed; "current" is a pointer file. Context and weights
   co-adapt, so gating, decision logging, and rollback always operate on a bundle
   hash, never on a component.
2. **The substrate is not agentic and not evolvable.** `gate`, `decisions`,
   `archive`, pointer promotion, and the staging/promoted split are plain
   code that no candidate can modify (the Zombie-Agents containment line;
   the same line DGM-H draws around parent selection and evaluation).
3. **Nothing is deleted.** Rejected candidates and shadow lineages stay in
   the store (DGM-H ablation: stepping stones are load-bearing). Promotion
   and retention are separate concerns.
4. **A running agent never modifies itself.** Bundle loads happen at
   episode start; a promoted meta bundle takes effect at the *next*
   episode. Rollback = move the pointer back.
5. **Core API stays at two verbs.** `rollout` and `update` (plus the pure
   `compare`). Every concrete method — reflection, GEPA, skill induction,
   slime training — is a cookbook entry implementing the update protocol,
   never a framework module (the Tinker discipline: primitives, not
   pipelines).

The flip side keeps the framework honest as research instrumentation:
**everything outside this list is a policy point, never a framework
position.** Self-evolution is an open problem; the substrate fixes
guarantees, not search strategy. The policy points, all swappable plain
functions or parameters — **five questions**: what to run (`rollout`),
how to score (reward fn; multi-dimensional scores by criterion reference
— see §3.3), how to judge (criterion — per-Lab and per-proposal), what
to change (strategy / updater, which also decides the branch base, a
per-proposal criterion, and file retirement), and whether promotion is
automatic or human-confirmed (`auto_promote` + `Lab.promote`). Rarer
knobs live in the engine room, off the user surface: custom aggregation
(construct a `Measure`), forced re-measurement (`gate(reuse_runs=False)`),
and meta-episode scheduling (`meta_due`, ships with the imp@k increment). The defaults (single lineage, `improve("reward")`,
auto-promote) encode one search policy — greedy hill climbing — chosen as
the cheapest starting point, and a researcher replaces it without
touching the framework: a strategy that picks its `base` from the archive
(e.g. DGM's score × child-count-decay weighting, a cookbook recipe) *is*
tree search; namespaced pointers *are* populations. None of the defaults
is load-bearing for the guarantees.

## 1. On-disk layout

A workspace directory (default `.evolution/`, gitignored; location is a
parameter everywhere) holds all evolution state:

```text
.evolution/
  bundles/
    <hash>/                  # immutable bundle dirs (task and meta alike)
      manifest.json
      provider.json
      prompt.md
      playbook.md            # optional
      lessons.jsonl          # optional
      skills/                # optional; ordinary SKILL.md tree
  pointers/
    task.json                # {"hash": "<hash>", "updated": "..."}
    meta.json
    shadow/                  # imp@k namespaces, e.g. gate-000042/task.json
  runs/                      # rollout output, standard eval run trees:
    <run_id>/<instance_id>/{input/instance.json, out/result.json,
                            out/trajectory.jsonl}
  decisions.jsonl            # the decision log (append-only); see §4
  episodes/                  # one trace per evolution-agent episode
    <episode_id>.trajectory.json
```

Everything under `bundles/` and `decisions.jsonl` is append-only by
convention; `write_jsonl_atomic` (already in
`src/simple_agent_lab/trace/jsonl.py`) is reused for all writes.

### 1.1 Typed views over the layout

The layout above is the wire format and the source of truth; **no
function in this framework hands code a raw `Path`** — extension points
and engine-room functions alike (`resolve` and `stage_bundle` return
`Bundle`; `promote` and `gate` accept it). The spec's nouns are types:
`Run` (catalog.py — `instance_id` / `run_id` / `ref` / `ok` / `result` /
`reward` / `bundle` / lazy `events()`) and `Bundle` (bundle.py — `hash` /
`manifest` / `parent` / `read()` / `files()`) are read-only views whose
docstrings carry the directory contract, so "what is inside that path" is
answered by the type signature and by `ls` with the same words.

One convention rides on the views: **durable records carry
workspace-relative refs, never absolute paths.** Decision-log evidence
uses `Run.ref` (`<run_id>/<instance_id>`); the append-only log outlives
any machine, so an absolute path written into it today is a broken
reference after relocation or remote storage. Four rules keep
the views from growing into an ORM that hides files:

1. **Read-only** — no write methods; mutation stays with `stage_bundle` /
   `promote` (the safety boundary lives in functions, not objects).
2. **Lazy** — nothing parses until asked; no caching across processes,
   no hidden state. A view is a reader, never a store.
3. **`.dir` is always exposed** — anything not wrapped is one `open()`
   away; `ls`/`jq` and code see the same files.
4. **Growth discipline** — a property earns its place only when two call
   sites repeat the same path-poking; otherwise use `.dir`.

## 2. Bundle

### 2.1 Contents and manifest

A bundle directory contains `manifest.json` plus any subset of the
artifact files. `manifest.json`:

```json
{
  "schema": "simple-agent-lab.bundle.v1",
  "level": "task",                  // "task" | "meta"
  "parent": "<hash|null>",          // lineage
  "producer": "evolution-agent",    // who created it (provenance)
  "evidence": ["trace:..."],        // trace ids that motivated the change
  "note": "added lesson about pytest fixture paths",
  "created": "2026-06-10T12:00:00Z"
}
```

`provider.json` is the serialized `Provider` record
(`src/simple_agent_lab/llm/provider.py`) — pointing at a fine-tuned
checkpoint is editing this one file.

### 2.2 Hashing

`bundle_hash(dir)` = sha256 over the sorted relative-path walk of
`(path, file-sha256)` pairs, excluding `manifest.json` itself (the
manifest records lineage *about* the content; two bundles with identical
behavior content but different notes must collide). First 12 hex chars are
used in paths and the decision log.

### 2.3 Module `simple_agent_lab/evolution/bundle.py`

```python
class Bundle:                        # the typed view of §1.1
    dir: Path                        # escape hatch, always exposed
    hash: str; manifest: Manifest; parent: str | None
    def read(filename) -> str        # "" when absent
    def files() -> tuple[str, ...]

def bundle_hash(bundle_dir: Path) -> str      # low-level file primitive
def read_manifest(bundle_dir: Path) -> Manifest
def stage_bundle(workspace: Path, *, base: Bundle | None,
                 edits: Mapping[str, str | bytes | None],
                 manifest: Manifest) -> Bundle
    # copy base, apply edits, write manifest, store under bundles/<hash>/.
    # An edit value is full new content (str text / bytes binary) or None —
    # a tombstone removing the inherited file (how a skill is retired).
    # Never overwrites: re-staging identical content returns the existing
    # bundle with its ORIGINAL manifest (first provenance wins — rollback
    # walks manifest parents).
def resolve(workspace: Path, pointer: str, *, namespace: str = "") -> Bundle
    # pointer = "task" | "meta"; namespace selects pointers/shadow/<ns>/
def promote(workspace: Path, pointer: str, bundle: Bundle,
            *, namespace: str = "") -> None
    # atomic pointer-file rewrite; the ONLY mutation primitive
```

### 2.4 Loading a bundle into an agent

`load_bundle(bundle_dir)` returns a `LoadedBundle` consumed by `rollout`:

| Bundle file | Injection path (all existing machinery) |
|---|---|
| `provider.json` | `Provider` passed through `run_dataset(**run_kwargs)` → `run_suite_instance(provider=...)` |
| `prompt.md` | `AgentSpec.system_prompt` (suite `agent_spec()` override) |
| `playbook.md`, `lessons.jsonl` (top-k) | `user_message(..., kind="context")` messages prepended to the task — `kind="context"` is already in `DEFAULT_PRESERVE_KINDS`, so compression never drops them |
| `skills/` | skills root for `run_with_skills()` / flavor `"bash_skills"` |

No runtime changes are required; the bundle layer only *assembles*
existing injection channels.

## 3. The two verbs

### 3.1 `rollout` — `simple_agent_lab/evolution/rollout.py`

```python
# The injected callable the gate (and anything else) sees:
Rollout = Callable[[Bundle, EvalSlice, str], Sequence[Run]]
#          (bundle, slice, run_id) -> the per-instance runs it produced

# The containerized implementation is built by a factory; deployment
# concerns bind at construction, the slice arrives per call:
def dataset_rollout(*, suite: Suite, backend: ContainerBackend,
                    store: ArtifactStore, runs_root: Path,
                    concurrency: int = 1,
                    run_kwargs: Mapping[str, Any] | None = None) -> Rollout
```

A thin wrapper over the existing `run_dataset()`
(`src/simple_agent_lab/evals/dataset.py`). Two contract points:

- **The slice is a call argument, not a construction argument** — the
  same rollout serves the main gate, a guard slice (non-target probe),
  held-out rotation, and shadow evaluation, without being rebuilt.
- **One `Run` per instance, always.** A crashed instance has no
  `result.json` (`run.ok` is False) and the measures decide how to
  account for it — `REWARD` raises, so a gate never silently compares
  unequal sets.

Its added responsibilities over `run_dataset()`:

1. resolve the bundle, build provider/agent-spec/context injections;
2. stamp **run provenance** (`bundle.json`: bundle hash, level, slice;
   later `sampling` and `policy_logprobs` in trace meta — the
   pre-committed fields that cannot be backfilled);
3. require the suite's `result.json` to carry the standard reward key
   `{"reward": float, ...}` (verifier suites: 1.0/0.0 from the verdict).

The signature deliberately contains everything a future HTTP wrapper
(growth form B) needs; serving it later changes the executor, not the
contract.

### 3.2 `update` — the only extension point

```python
class UpdaterSpec(TypedDict):
    name: str
    reads: list[str]      # e.g. ["runs", "decisions", "lessons"]
    produces: str         # candidate kind: "lesson" | "playbook" | "skill"
                          # | "prompt" | "context_policy" | "provider" | "meta"
    cost: str             # "cheap" | "slow"  (slow = external training jobs)

UpdateFn = Callable[[UpdateInputs], Path]   # returns a staged bundle dir
```

`UpdateInputs` carries: workspace, base bundle dir, selected runs (paths),
decision-log view, and a scratch dir. Implementations live in `cookbook/`
(examples tree), e.g. `cookbook/reflect_lessons.py`,
`cookbook/induce_skill.py`, `cookbook/gepa_prompt.py`,
`cookbook/slime_sft.py`. The typed spec is what lets the evolution agent
reason about its toolbox ("which updater reads failures and produces
skills?") — the DSPy signature idea applied to updaters.

`cookbook/slime_sft.py` is the reference *slow* updater: export per-turn
pairs via `model_turns_from_events()`
(`src/simple_agent_lab/trace/training.py`) into the trainer's data
format, submit the external job, poll, then `stage_bundle()` with only
`provider.json` edited to the new checkpoint endpoint. The trainer's
inner loop (its own rollout/buffer/weight-sync cadence) is invisible
here — one expensive function call, gated like any other candidate.

### 3.3 The user surface: `Lab` — `simple_agent_lab/evolution/lab.py`

Design review found the six engine-room nouns too heavy an on-ramp for a
researcher. The benchmark is verl/slime integration: verl users plug in a
custom reward by writing one `compute_score()` function; slime users plug
in a custom rollout via `--rollout-function-path`. The public surface
mirrors that — **three plain functions, nothing else to learn**:

```python
lab = Lab(workspace, rollout=...)          # ① what to run (dataset_rollout
                                           #   for suites, or any callable)

def my_reward(run: Run) -> float: ...      # ② how to score one run —
                                           #   run.result / run.reward /
                                           #   run.events() / run.dir
                                           #   (optional: defaults to the
                                           #    result.json reward key)

def my_strategy(ctx: EpisodeContext):      # ③ how to change the agent —
    return ctx.propose(kind="lesson", edits={...}, note=..., evidence=[...])
                                           #   ctx.runs / ctx.failures /
                                           #   ctx.current / ctx.bundle(h) /
                                           #   ctx.decisions, all typed views
                                           #   (optional: omit and call
                                           #    lab.evolve() — agent-driven)

lab.step(my_strategy)   # observe -> propose -> compare -> promote/reject
lab.evolve(provider)    # the no-strategy, agent-driven mode
lab.history()           # the experiment log;  lab.rollback() moves back
```

Contracts the facade enforces (it is a facade — no new mechanism):

- A strategy **only proposes**; `step()` stages the proposal, runs the
  gate, logs the decision, and promotes host-side. Nothing a strategy
  returns can bypass the comparison (invariant 2 holds for human-written
  strategies exactly as for the evolution agent).
- A proposal may carry `base` (any archived bundle hash — rejected
  candidates included) to branch from a stepping stone instead of the
  current version; the gate baseline remains the current pointer either
  way. It may also carry its own `criterion` (e.g. an efficiency
  objective for one lateral step). `EpisodeContext` exposes `decisions`
  and `archived()` so tree-style selection policies are writable as plain
  strategies — no framework change between hill climbing and DGM-style
  branching search.
- `Lab(auto_promote=False)` inserts a confirmation tier: `step()` still
  gates and logs, promotion waits for an explicit `lab.promote(hash)`,
  and `promote` refuses a hash with no accepted decision in the log —
  evidence stays mandatory in both modes.
- **Referencing a dimension IS registering it.** A criterion's dimension
  names (`Criterion.requires`) resolve automatically: `"reward"` → this
  Lab's reward definition, built-ins (`cost_tokens`) by name, anything
  else → the same-named numeric `result.json` field. Multi-score benches
  need no measure plumbing — `guarded(improve("reward"),
  [not_worse("compile_ok")])` just works. A missing field fails the gate
  with a clear error. Custom extractors/aggregations construct a
  `Measure` and call `gate()` directly (engine room).
- `runs_root=` (one place to align where rollouts land; default
  `workspace/runs`) is deployment config, not a policy point. Edit values
  in a proposal may be `str` (text), `bytes` (binary), or `None` — the
  deletion tombstone that retires an inherited file.
- A custom `reward` replaces scoring everywhere at once: it becomes the
  gate's reward measure **and** re-scores the run index behind
  `ctx.failures`, so "what the strategy sees as failing" and "what the
  gate measures" are the same definition by construction.
- Relationship to §3.2: a `StrategyFn` is the user-facing projection of
  an updater — `Lab.step()` wraps it in staging + manifest provenance.
  The typed `UpdaterSpec` registration remains the engine-room form,
  needed when the evolution agent reasons about its toolbox; cookbook
  entries can expose either shape.

This also fixes one assembly point for deployment concerns: workspace and
(later) storage backends are configured on `Lab` construction only.

## 4. Substrate: gate, decision log, catalog

### 4.1 `simple_agent_lab/evolution/gate.py`

Evaluation is deliberately generic: a gate decision is **measures**
(what to quantify) combined with a **criterion** (how to judge two
measurement sets). Task quality is just one measure among several —
cost, latency, and robustness are first-class evolution objectives.

```python
@dataclass(frozen=True)
class MeasureFrame:                    # one measure over one run set
    name: str
    value: float                       # the aggregate (what the log records)
    per_run: Mapping[str, float]       # instance_id -> value: the frozen
                                       # slice pairs runs by instance, so
                                       # paired statistics stay possible

Measurement = Mapping[str, MeasureFrame]

@dataclass(frozen=True)
class Measure:                         # what to quantify
    name: str
    per_run: Callable[[Run], float]    # one scalar per typed Run
    aggregate: Callable[[Sequence[float]], float] = mean
                                       # aggregation is a policy point too:
                                       # mean for rates, sum for totals,
                                       # min / median / pass@k as needed

class Criterion(Protocol):             # how to judge
    requires: tuple[str, ...]          # dimension names it reads; the Lab
                                       # resolves them to measures (§3.3) —
                                       # built-in combinators fill this in
    def judge(self, baseline: Measurement, candidate: Measurement) -> Judgment

@dataclass(frozen=True)
class Judgment:
    accepted: bool
    deltas: Mapping[str, float]        # per-dimension candidate - baseline
    reason: str                        # human-readable; goes into the decision log

@dataclass(frozen=True)
class GateResult:
    judgment: Judgment
    baseline: Measurement
    candidate: Measurement
    runs: dict[str, str]               # {"baseline": run_id, "candidate": run_id}

def gate(
    workspace: Path, *,
    baseline: Bundle, candidate: Bundle,
    slice_: EvalSlice,           # frozen instance list + content hash + suite pin
    rollout: Rollout,            # injected; slice travels per call (§3.1)
    measures: Sequence[Measure] = (REWARD,),
    criterion: Criterion = improve("reward"),
    runs_root: Path | None = None,   # default workspace/runs
    reuse_runs: bool = True,         # policy point: measurement reuse
) -> GateResult
```

`gate` stays four steps: `rollout(baseline)`, `rollout(candidate)`, apply
measures + criterion, `decisions.append(...)`. (The anti-thrash budget is
enforced one layer up, where "per episode" is meaningful: the evolution
agent's `run_gate` tool counts calls; the gate itself stays a stateless
pure function.)

**Measurement reuse** (`reuse_runs`, default on): a side whose
`(bundle, slice)` was already measured — found via the `bundle.json`
provenance stamp under `runs_root` — reuses those run dirs instead of
re-rolling. The unchanged baseline is measured once per slice rather than
once per candidate, halving steady-state gate cost; the decision's `runs`
field references the run set actually used, so the evidence chain stays
honest. Unstamped run sets never match (stub rollouts simply re-roll), and
`reuse_runs=False` forces fresh measurements when provider-side drift is
the question.

**Built-in measures** — all computed from artifacts that already exist,
no new collection: `REWARD` (the standard `result.json` key, mean),
`COST_TOKENS` (trace usage events, sum); `latency_s`, `turns` (span
tree) and `tool_error_rate` (tool-execution events) follow the same
two-liner pattern. Custom measures are one function each.

**Built-in criteria** — declarative combinators, chosen over weighted
sums because constraint-style judgments produce auditable reasons (a
weighted score cannot explain *why* in the decision log):

```python
improve("reward", min_delta=0.0)            # MVP default: quality climb
paired_improve("reward", min_net_wins=1)    # sign-test style: wins minus
                                            #  losses over paired instances —
                                            #  noise-robust where two means
                                            #  a hair apart are not
minimize("cost_tokens", min_gain=0.10)      # efficiency objective
not_worse("reward", tol=0.01)               # a guard
guarded(objective=..., guards=[...])        # "optimize X subject to Y"
all_of(...) / lexicographic(...)            # composition
```

Examples this unlocks without new machinery: an *efficiency episode*
("same resolved rate, 10% fewer tokens" — `guarded(minimize(
"cost_tokens", 0.10), [not_worse("reward")])`), a robustness gate
(`not_worse("tool_error_rate")` as a guard on every promotion), and the
non-target regression probe (a guard measured on a second slice).

**Meta level:** `imp_at_k` is not a special code path — it is a Measure
whose unit of evaluation is evolution *episodes* instead of instance
runs: run k episodes under each meta bundle in an isolated pointer
**namespace** (`pointers/shadow/<gate-id>/`), starting from the same task
bundle and slice; the measurement is the best task-level improvement each
produces (and, being a Measurement, can also carry the episodes' cost).
Shadow promotions touch only the namespace; afterwards the namespace is
retired (retained, per invariant 3). Cost is ~2·k episodes — which is why
meta gating is scheduled, not routine.

Two pre-gate guards run before any rollout is spent:

- **Novelty rejection** (ShinkaEvolve), keyed on the whole comparison —
  the `(candidate, baseline, slice)` triple, not the candidate alone: the
  same content against a *moved* baseline or a different slice is a fresh
  question, so archived stepping stones stay re-testable. Exact triple
  match is auto-rejected with a decision-log entry and zero rollout cost;
  near-duplicate similarity (normalized diff within a kind) is the
  post-MVP refinement.
- **Budget cap**: a per-episode ceiling on gate calls (anti-thrash),
  enforced at the tool layer where "episode" exists
  (`max_gates_per_episode`); the gate function itself stays stateless.

### 4.2 `simple_agent_lab/evolution/decisions.py`

One JSONL record per gate decision (parent), referencing run dirs
(children) — the MLflow parent/child lineage pattern flattened into
references:

```json
{
  "schema": "simple-agent-lab.decision.v1",
  "id": "gate-000042",
  "ts": "2026-06-10T12:34:56Z",
  "level": "task",
  "episode": "ep-000017",
  "kind": "skill",
  "baseline": {"bundle": "ab12...",
                "measurements": {"reward": 0.34, "cost_tokens": 81000}},
  "candidate": {"bundle": "cd34...",
                 "parent": "ab12...",   // lineage in the log: tree analytics
                                        // (child counts) without manifest scans
                 "measurements": {"reward": 0.41, "cost_tokens": 78500},
                 "evidence": ["trace:..."], "note": "..."},
  "slice": {"suite": "swebench==0.3.1", "instances_sha": "ef56...", "n": 20},
  "criterion": "guarded(improve(reward), [not_worse(tool_error_rate)])",
  "deltas": {"reward": 0.07, "cost_tokens": -2500},
  "decision": "accepted",        // accepted | rejected | novelty_rejected
  "reason": "reward 0.34->0.41 (min_delta 0.0 met); guard tool_error_rate ok",
  "runs": {"baseline": "runs/r-101", "candidate": "runs/r-102"}
}
```

Query helpers (pure functions over the JSONL): `hit_rate(kind, window)`,
`recent(level)`, `lineage(bundle_hash)`. `hit_rate` doubles as the bandit
prior surface and the meta-episode trigger input.

### 4.3 `simple_agent_lab/evolution/catalog.py`

Read-side index over one or more run roots: one row per run —
`{run_id, instance_id, bundle, reward, turns, tokens, duration,
trace_path}` — built by scanning `out/result.json` + trajectory meta.
Pure, cache-friendly, no daemon. This is what `query_runs` serves.

## 5. The evolution agent

### 5.1 Construction

An ordinary `Agent` built with `make_llm_agent()` — its episodes are
themselves traces (saved under `episodes/`), so every diagnosis and
decision is auditable. It loads its own bundle (`pointers/meta.json`) at
episode start: prompt = `prompt.md`, playbook/lessons injected as
`kind="context"`, tools as below.

### 5.2 Tools (the complete set)

| Tool | Maps to | Notes |
|---|---|---|
| `query_runs(filter)` | catalog | "what keeps failing, at what cost" |
| `read_trace(run, tail)` | run result + trace tail | failure localization; takes a workspace-relative run ref |
| `read_decisions(query)` | decision log helpers | incl. hit rates (bandit prior) |
| `write_candidate(kind, edits, note, evidence, base?)` | `stage_bundle()` | **staging only**; target bundle implied by kind (`meta` kind targets the meta bundle); `base` branches from any archived bundle (stepping stones), default = current; a `null` edit value is a deletion tombstone |
| `run_gate(candidate, slice, objective?)` | `gate()` | measures/criterion resolved from candidate level and declared objective (quality / efficiency / robustness); enforces budget |

The tool layer is where authority is enforced: the agent cannot touch
pointers, the decision log, or promoted bundles except through these five
tools.

### 5.3 Episode loop (host-side driver, plain code)

```python
def run_episode(workspace, config) -> EpisodeReport:
    level = "meta" if meta_due(decisions, config) else "task"
    agent = build_evolution_agent(resolve(workspace, "meta"))
    state, events = agent.run(episode_task(level, workspace))  # agent loop
    # promotion is host-side, evidence-driven, not agent-side:
    for gate_id in accepted_gates(state):
        promote(workspace, level, candidate_of(gate_id))
    save_episode_trace(state)
```

`meta_due()` = hit-rate decay over the last W task episodes **or** every
N episodes (periodic floor), with "one meta promotion at a time" enforced
by requiring ≥M task episodes since the last meta promotion.

### 5.4 Task vs meta episodes

Identical machinery; three parameter differences (metric, effect time,
frequency):

| | task episode | meta episode |
|---|---|---|
| candidate target | task bundle | meta bundle (next incarnation) |
| gate criterion | over instance-run measures (reward/cost/...) on frozen slice | over the `imp_at_k` episode measure (shadow namespace) |
| cadence | steady state | triggered (`meta_due`) |
| takes effect | next rollout | next episode |

## 6. MVP scope and acceptance

**In:** `bundle.py`, `gate.py` (the `reward` and `cost_tokens` measures,
`improve` / `not_worse` / `guarded` criteria, novelty hash-check,
budget), `decisions.py`, `catalog.py`, `rollout.py`, the `Lab` user
surface (§3.3), the evolution agent with its five tools, episode driver,
and two cookbook updaters (`reflect_lessons`, `induce_skill`) — run
against a SWE-bench slice on `LocalDockerBackend` (unit tests on
`FakeBackend` + fake provider). Of these, everything except container
prompt/skills injection and the cookbook updaters is already merged (see
the status note at the top).

**Out (interfaces reserved):** `imp_at_k` and meta episodes (ship
right after MVP; the namespace mechanism is designed in from day one),
near-duplicate similarity (hash-equality only at first), bandit
scheduling (hit_rate query ships; the policy stays in the agent's
playbook), GEPA/slime cookbook entries, the HTTP rollout wrapper, and
population selection over the archive.

**Acceptance criteria (unchanged from the design memo):**
1. the decision log shows ≥2 intervention kinds attempted;
2. ≥1 promotion whose improvement reproduces on the frozen slice;
3. pointer rollback restores baseline behavior;
4. rejected candidates remain retrievable (archive = nothing deleted).

**Testing strategy:** every substrate function is pure or
filesystem-local → unit-testable with `tmp_path`; episode e2e runs on
`FakeBackend` with the deterministic fake adapter
(`src/simple_agent_lab/llm/adapters/fake.py`) scripting the evolution
agent's tool calls; one live smoke test mirrors
`runs/run_swebench_gold_smoke.sh` conventions.

## 7. Growth paths (what this spec already fixes)

- **Form B (services):** `rollout`'s signature is the future HTTP body;
  the decision log/bundle stores are already shared-filesystem-safe (atomic
  writes, immutable dirs). Serving = new executor, same contract.
- **Form C (population search):** replace the single `task` pointer with
  per-individual namespaces (mechanism already exists for shadow
  lineages) and add a selector over the archive; gate becomes a scorer.
  Stores, hashing, and decision-log schema are reused unchanged. The
  branch *mechanism* is already live (`base` on proposals), so Form C
  adds only concurrent lineages and a host-side selector; until then the
  selection policy lives in the strategy or playbook — matching
  HyperAgents, whose lineage-editable selector ships as uniform random
  over valid parents with host-side weighted fallbacks.
- **RL (online):** the slow-updater seam plus the pre-committed run
  provenance fields (`bundle`, `sampling`, `policy_logprobs`, reward key)
  are exactly what trajectory-consuming trainers need; nothing to
  retrofit.

### 7.1 Distributed deployment and the storage seam

Review raised the concern that the Path-based substrate limits a future
distributed deployment. The stance, made explicit here:

- **The data structures were chosen to distribute well.** Bundles are
  content-addressed and immutable — the storage model of git objects,
  OCI layers, and build CAS stores: object-store keys are the hashes,
  sync is conflict-free, caches never invalidate. The decision log is a
  single append-only stream (trivially a table or log service). Only
  **three points need coordination primitives**: pointer update (the one
  compare-and-swap), log append (the one multi-writer), and run-artifact
  locality.
- **The seam already exists: it is the module boundary.** All I/O goes
  through the functions of `bundle.py` / `decisions.py` / `catalog.py`;
  callers (gate, agent, Lab, cookbook) never touch files directly.
  Swapping those internals for an object-store or HTTP client changes no
  caller. Run-artifact distribution is already solved one layer down by
  the eval framework's `ArtifactStore` / `RemoteDockerBackend` seams,
  which `dataset_rollout` sits on.
- **Shared filesystems are a first-class first step**, not a stopgap —
  slime ships weight sync over disk/shared-FS transport in production.
- **No explicit store protocol yet, deliberately.** Adding a `Store`
  Protocol now would spend the concept budget §3.3 just reclaimed. The
  trigger to introduce it is concrete: the moment a second storage
  backend is actually needed, the refactor is mechanical because the
  call sites are already confined.

## 8. Open questions

1. Lesson retrieval at injection time: top-k by tag (MVP) — when does
   similarity search earn its dependency?
2. Default guards on every promotion: the criterion abstraction makes a
   non-target regression probe and a `tool_error_rate` guard cheap to
   attach — which guards are mandatory by default vs. opt-in per
   candidate kind?
3. Episode task templates (`episode_task()`): how much procedure lives in
   the meta bundle's playbook (evolvable) vs. the host driver (fixed)?
   Start maximally fixed, migrate into the playbook as evidence
   accumulates.
