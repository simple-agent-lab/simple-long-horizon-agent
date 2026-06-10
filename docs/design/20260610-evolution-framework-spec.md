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
both and comparing (gate); history is an append-only ledger. All
intelligence lives in the evolution agent and update functions; all
guarantees live in the substrate.*

## 0. Invariants (the rules that never break)

1. **Bundles are immutable and promoted whole.** A bundle is
   content-addressed; "current" is a pointer file. Context and weights
   co-adapt, so gating, ledgering, and rollback always operate on a bundle
   hash, never on a component.
2. **The substrate is not agentic and not evolvable.** `gate`, `ledger`,
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
  ledger.jsonl               # append-only; see §4
  episodes/                  # one trace per evolution-agent episode
    <episode_id>.trajectory.json
```

Everything under `bundles/` and `ledger.jsonl` is append-only by
convention; `write_jsonl_atomic` (already in
`src/simple_agent_lab/trajectory/jsonl.py`) is reused for all writes.

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
used in paths and the ledger.

### 2.3 Module `simple_agent_lab/evolution/bundle.py`

```python
def bundle_hash(bundle_dir: Path) -> str
def read_manifest(bundle_dir: Path) -> Manifest
def stage_bundle(workspace: Path, *, base: Path, edits: Mapping[str, str | bytes],
                 manifest: Manifest) -> Path
    # copy base, apply file edits, write manifest, store under
    # bundles/<hash>/, return the new immutable dir. Never overwrites.
def resolve(workspace: Path, pointer: str, *, namespace: str = "") -> Path
    # pointer = "task" | "meta"; namespace selects pointers/shadow/<ns>/
def promote(workspace: Path, pointer: str, bundle_dir: Path,
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
def rollout(
    bundle_dir: Path,
    instances: Sequence[Mapping[str, Any]],
    *,
    suite: Suite,
    backend: ContainerBackend,
    store: ArtifactStore,
    run_root: Path,
    run_id: str,
    concurrency: int = 1,
    sampling: Mapping[str, Any] | None = None,   # temperature etc.
) -> DatasetReport
```

A thin wrapper over the existing `run_dataset()`
(`src/simple_agent_lab/evals/dataset.py`). Its added responsibilities:

1. resolve the bundle, build provider/agent-spec/context injections;
2. stamp **run provenance** into each trace meta:
   `{"bundle": <hash>, "bundle_level": ..., "sampling": {...},
   "policy_logprobs": <when the endpoint provides them>}` — the
   pre-committed fields that cannot be backfilled later;
3. require the suite's `result.json` to carry the standard reward key
   `{"reward": float, ...}` (verifier suites: 1.0/0.0 from the verdict).

The signature deliberately contains everything a future HTTP wrapper
(growth form B) needs; serving it later changes the executor, not the
contract.

### 3.2 `update` — the only extension point

```python
class UpdaterSpec(TypedDict):
    name: str
    reads: list[str]      # e.g. ["runs", "ledger", "lessons"]
    produces: str         # candidate kind: "lesson" | "playbook" | "skill"
                          # | "prompt" | "context_policy" | "provider" | "meta"
    cost: str             # "cheap" | "slow"  (slow = external training jobs)

UpdateFn = Callable[[UpdateInputs], Path]   # returns a staged bundle dir
```

`UpdateInputs` carries: workspace, base bundle dir, selected runs (paths),
ledger view, and a scratch dir. Implementations live in `cookbook/`
(examples tree), e.g. `cookbook/reflect_lessons.py`,
`cookbook/induce_skill.py`, `cookbook/gepa_prompt.py`,
`cookbook/slime_sft.py`. The typed spec is what lets the evolution agent
reason about its toolbox ("which updater reads failures and produces
skills?") — the DSPy signature idea applied to updaters.

`cookbook/slime_sft.py` is the reference *slow* updater: export per-turn
pairs via `model_turns_from_events()`
(`src/simple_agent_lab/trajectory/training.py`) into the trainer's data
format, submit the external job, poll, then `stage_bundle()` with only
`provider.json` edited to the new checkpoint endpoint. The trainer's
inner loop (its own rollout/buffer/weight-sync cadence) is invisible
here — one expensive function call, gated like any other candidate.

## 4. Substrate: gate, ledger, catalog

### 4.1 `simple_agent_lab/evolution/gate.py`

Evaluation is deliberately generic: a gate decision is **measures**
(what to quantify) combined with a **criterion** (how to judge two
measurement sets). Task quality is just one measure among several —
cost, latency, and robustness are first-class evolution objectives.

```python
Measurement = Mapping[str, float]      # named scalars for one run set

class Measure(Protocol):               # what to quantify
    name: str
    def __call__(self, runs: Sequence[RunRef]) -> Measurement

class Criterion(Protocol):             # how to judge
    def judge(self, baseline: Measurement, candidate: Measurement) -> Judgment

@dataclass(frozen=True)
class Judgment:
    accepted: bool
    deltas: Mapping[str, float]        # per-dimension candidate - baseline
    reason: str                        # human-readable; goes into the ledger

@dataclass(frozen=True)
class GateResult:
    judgment: Judgment
    baseline: Measurement
    candidate: Measurement
    runs: dict[str, str]               # {"baseline": run_id, "candidate": run_id}

def gate(
    workspace: Path, *,
    baseline: Path, candidate: Path,
    slice_: EvalSlice,           # frozen instance list + content hash + suite pin
    measures: Sequence[Measure] = (reward,),
    criterion: Criterion = improve("reward"),
    budget: GateBudget,          # max runs / max gate calls per episode
) -> GateResult
```

`gate` stays four steps: `rollout(baseline)`, `rollout(candidate)`, apply
measures + criterion, `ledger.append(...)`.

**Built-in measures** — all computed from artifacts that already exist,
no new collection: `reward` (mean of the standard `result.json` key),
`cost_tokens` / `cost_usd` (trace `TokenUsage`), `latency_s` and `turns`
(span tree), `tool_error_rate` (tool-execution events). Custom measures
are one function each.

**Built-in criteria** — declarative combinators, chosen over weighted
sums because constraint-style judgments produce auditable reasons (a
weighted score cannot explain *why* in the ledger):

```python
improve("reward", min_delta=0.0)            # MVP default: quality climb
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

- **Novelty rejection** (ShinkaEvolve): exact-duplicate check by bundle
  hash, then near-duplicate check (normalized diff against archived
  candidates of the same kind); near-duplicates are auto-rejected with a
  ledger entry and zero rollout cost.
- **Budget cap**: a `GateBudget` ceiling on gate calls and total runs per
  episode (anti-thrash).

### 4.2 `simple_agent_lab/evolution/ledger.py`

One JSONL record per gate decision (parent), referencing run dirs
(children) — the MLflow parent/child lineage pattern flattened into
references:

```json
{
  "schema": "simple-agent-lab.ledger.v1",
  "id": "gate-000042",
  "ts": "2026-06-10T12:34:56Z",
  "level": "task",
  "episode": "ep-000017",
  "kind": "skill",
  "baseline": {"bundle": "ab12...",
                "measurements": {"reward": 0.34, "cost_tokens": 81000}},
  "candidate": {"bundle": "cd34...",
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
| `read_trace(trace_id, view)` | span tree / model turns | failure localization |
| `read_ledger(query)` | ledger helpers | incl. hit rates (bandit prior) |
| `write_candidate(kind, edits, note, evidence)` | `stage_bundle()` | **staging only**; target bundle implied by kind (`meta` kind targets the meta bundle) |
| `run_gate(candidate, slice, objective?)` | `gate()` | measures/criterion resolved from candidate level and declared objective (quality / efficiency / robustness); enforces budget |

The tool layer is where authority is enforced: the agent cannot touch
pointers, the ledger, or promoted bundles except through these five
tools.

### 5.3 Episode loop (host-side driver, plain code)

```python
def run_episode(workspace, config) -> EpisodeReport:
    level = "meta" if meta_due(ledger, config) else "task"
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
budget), `ledger.py`, `catalog.py`, `rollout.py`, the evolution agent with
its five tools, episode driver, and two cookbook updaters
(`reflect_lessons`, `induce_skill`) — run against a SWE-bench slice on
`LocalDockerBackend` (unit tests on `FakeBackend` + fake provider).

**Out (interfaces reserved):** `imp_at_k` and meta episodes (ship
right after MVP; the namespace mechanism is designed in from day one),
near-duplicate similarity (hash-equality only at first), bandit
scheduling (hit_rate query ships; the policy stays in the agent's
playbook), GEPA/slime cookbook entries, the HTTP rollout wrapper, and
population selection over the archive.

**Acceptance criteria (unchanged from the design memo):**
1. the ledger shows ≥2 intervention kinds attempted;
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
  the ledger/bundle stores are already shared-filesystem-safe (atomic
  writes, immutable dirs). Serving = new executor, same contract.
- **Form C (population search):** replace the single `task` pointer with
  per-individual namespaces (mechanism already exists for shadow
  lineages) and add a selector over the archive; gate becomes a scorer.
  Stores, hashing, and ledger schema are reused unchanged.
- **RL (online):** the slow-updater seam plus the pre-committed run
  provenance fields (`bundle`, `sampling`, `policy_logprobs`, reward key)
  are exactly what trajectory-consuming trainers need; nothing to
  retrofit.

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
