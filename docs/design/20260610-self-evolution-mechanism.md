---
title: "Self-Evolution Mechanism: Agent-Driven Evolution on a Deterministic Substrate"
status: Proposed
date: 2026-06-10
slug: self-evolution-mechanism
---

# Self-Evolution Mechanism: Agent-Driven Evolution on a Deterministic Substrate

Design memo for building a self-evolving agent loop on top of Simple Agent
Lab, consistent with the accepted ADR
[`docs/decisions/20260507-treat-self-evolution-as-harness-capability.md`](../decisions/20260507-treat-self-evolution-as-harness-capability.md).

The central design choice: **an evolution agent makes the decisions; a
deterministic substrate enforces the guarantees.** A fixed
collect → distill → inject pipeline is just trajectory post-processing —
the distillation prompt is frozen, so it can only ever produce one kind of
improvement. What makes the system *self*-evolving is that an agent
diagnoses failures, chooses the intervention (lesson, skill, prompt,
context policy), drafts the candidate, and interprets the comparison —
while promotion, comparison, and record-keeping stay as plain code it
cannot bypass.

```text
evolution episode (agent-driven):
  observe -> diagnose -> choose intervention -> draft candidate
          -> gate (A/B compare) -> interpret -> ledger entry -> next episode
```

## 1. Literature snapshot (late 2025 – mid 2026)

Grouped by the question each line of work answers.

### 1.1 Surveys / framing

- *A Survey of Self-Evolving Agents: What, When, How, and Where to Evolve*
  (arXiv 2507.21046, revised 2026-01) — organizes the field along **what**
  evolves (model / memory / prompt / tools / architecture), **when**
  (intra-run vs. cross-run), **how** (reward, demonstration, population
  search), and **where** it is evaluated. Already the framing of our ADR.
- *Adaptation of Agentic AI: A Survey of Post-Training, Memory, and Skills*
  (arXiv 2512.16301) — contrasts parameter adaptation (post-training) with
  non-parametric adaptation (memory, skill libraries, prompt refinement),
  and notes mature systems run both against the same experience stream.

### 1.2 Agent-driven evolution (the layer this memo centers on)

- *AgentEvolver: Towards Efficient Self-Evolving Agent System*
  (arXiv 2511.10395) — a master orchestrator drives task synthesis →
  rollout → experience summarization → sample construction, with
  self-questioning, self-navigating, and self-attribution. The orchestrator
  deciding *what to do next* — not the pipeline stages — is the part we
  adopt first.
- *Darwin Gödel Machine* (arXiv 2505.22954) and *Automated Design of
  Agentic Systems* (arXiv 2408.08435) — a meta-agent proposes candidate
  modifications, an empirical benchmark gate accepts or rejects them, and
  an archive of candidates supports branching search. Our gate + ledger is
  the conservative core of this pattern; the archive/branching search is a
  later extension.

### 1.3 Training-free intervention kinds (the evolution agent's toolbox)

- *Agentic Context Engineering (ACE)* (arXiv 2510.04618, ICLR'26) —
  context as an evolving playbook maintained via incremental, itemized
  delta updates; wholesale rewriting causes *context collapse* and
  *brevity bias*. +10.6% on agent benchmarks at 83.6% lower token cost.
- *EvolveR* (arXiv 2510.16079) — closed-loop lifecycle distilling the
  agent's own trajectories into a persistent, retrievable experience
  corpus.
- *Just-In-Time RL* (arXiv 2601.18510) — non-parametric memory of
  (state, action, outcome) with test-time advantage estimation; zero
  weight updates. Motivates helpful/harmful outcome counters on lessons.
- *Decocted Experience Improves Test-Time Inference* (arXiv 2604.04373) —
  distilled experience beats raw trajectory replay; store **lessons**, not
  logs.
- *SkillRL* (arXiv 2602.08234) and *SAGE: RL for Self-Improving Agent with
  Skill Library* (arXiv 2512.17102) — abstract trajectories into reusable
  hierarchical skills; execution outcomes alone decide library admission.
- *How Well Do Agentic Skills Work in the Wild* (arXiv 2604.04323) —
  skills regress performance when trigger descriptions over-fire; a gate
  must check non-target slices, not just the target slice.
- *MemoPilot* (arXiv 2606.08656) and *SuRe* (arXiv 2511.22367) — learned
  memory-update policies and surprise-prioritised replay; later-stage
  options for the memory layer.

### 1.4 Training-based evolution (deferred; informs the interface contract)

- *ASTRA* (arXiv 2601.21558) — automated trajectory synthesis + verifiable
  RL with trajectory-level rewards.
- *SWE-TRACE* (arXiv 2604.14820) — test-grounded SWE data curation, rubric
  process-reward models, memory-augmented GRPO; directly relevant to our
  SWE-bench suite. *SWE-Master* (arXiv 2602.03411) and *Klear-AgentForge*
  (arXiv 2511.05951) — SFT → RL post-training recipes for software agents.
- *SWE-AGILE* "Hindsight Backfill" (arXiv 2604.11716) and *Hindsight Hint
  Distillation* (arXiv 2605.11556) — cheap dataset-quality multipliers:
  backfill reasoning into sparse successes; scaffold failures with
  gold-derived hints, then strip the hints on export.
- Task supply at scale: *daVinci-Env* (arXiv 2603.13023), *Agent World
  Model* (arXiv 2602.10090), *SWE-rebench* (21k+ interactive SWE tasks,
  decontaminated evaluation), ICLR'25 *Learn-by-Interact*.

### 1.5 Safety of self-evolution

- *Zombie Agents* (arXiv 2602.15654) — adversarial content can enter an
  experience store and self-reinforce across runs. Every write path into
  the agent's context needs provenance, a review gate, and rollback. This
  makes the ADR's "every accepted evolution leaves a local record" rule a
  security control, not just hygiene.

## 2. What the repo already provides

Most of the loop's read side exists today:

| Loop stage | Existing support |
|---|---|
| run | `src/simple_agent_lab/core.py` loop; eval harness `src/simple_agent_lab/evals/runner.py`, `src/simple_agent_lab/evals/dataset.py` (concurrent rollout) |
| trace | three-layer trajectory (event → span → training pair): `src/simple_agent_lab/trajectory/spans.py`, `src/simple_agent_lab/trajectory/training.py`, `src/simple_agent_lab/trajectory/run_trace.py` (`simple-agent-lab.trajectory.v3`), live JSONL via `src/simple_agent_lab/trajectory/live.py` |
| evaluate | suite `evaluate()` hook writes verdicts into `out/result.json`; candidate+judge composition in `examples/bench_suite/` |
| distill (export) | `src/simple_agent_lab/trace.py` exports OpenAI fine-tuning records; `model_turns_from_events()` yields per-turn (visible context, response, tools) pairs |
| inject context | skills runtime `src/simple_agent_lab/skills/runtime.py` injects SKILL.md bodies; `src/simple_agent_lab/context_view.py` controls model-visible context; `kind="context"` messages are never compressed |
| swap model | `Provider` is a frozen data record (`src/simple_agent_lab/llm/provider.py`) — pointing at a new checkpoint or endpoint is a one-field change |
| build the evolution agent itself | the runtime: the evolution agent is an ordinary `Agent` with tools, so its own decision process is a trace |

Missing: the artifact store, catalog, gate, ledger, and the evolution agent
with its tools.

## 3. Architecture: two layers

```text
┌─ EVOLUTION AGENT (an ordinary Agent on this runtime; its run is a trace) ─┐
│                                                                           │
│  tools:  query_runs(filter)        - catalog query: what keeps failing?  │
│          read_trace(trace_id)      - span tree: localize failure modes   │
│          write_candidate(kind, ..) - staging area ONLY                   │
│          run_gate(candidate, slice)- trigger A/B, get comparison         │
│          read_ledger()             - what was tried before, what worked  │
│                                                                           │
│  per episode: observe -> diagnose -> CHOOSE INTERVENTION KIND             │
│               -> draft -> gate -> interpret -> ledger narrative           │
└──────────────────────────────┬────────────────────────────────────────────┘
                               │ may only write staging; may only promote via gate
┌─ DETERMINISTIC SUBSTRATE (plain code; the safety boundary) ───────────────┐
│  catalog   index over evals/out run roots: verdict, cost, trace paths     │
│  gate      frozen instance slice, baseline vs candidate via run_dataset() │
│  ledger    append-only JSONL: feedback signal, baseline, candidate hash,  │
│            comparison result, accept/reject reason (the ADR record)       │
│  artifacts staging/ and promoted/ stores for lessons, SKILL.md, prompts,  │
│            context policies — versioned, file-diffable, rollback-able     │
└───────────────────────────────────────────────────────────────────────────┘
```

Division of labor:

- **The substrate is deliberately not agentic.** Gate, ledger, and the
  staging/promoted split are the security boundary (the Zombie-Agents
  mitigation) and the evidence chain (the ADR record). The evolution agent
  cannot improvise around them: `write_candidate` only reaches staging,
  and the only path to promotion is `run_gate`.
- **The agent owns the open decisions.** Which failures matter, what the
  root cause is, which intervention kind fits, what the candidate says,
  whether a gate failure means "wrong content" or "wrong intervention
  kind". The same batch of failed traces might yield a lesson this episode
  and — after that lesson fails the gate — a SKILL.md next episode. A
  fixed pipeline is the degenerate case of this design (an agent that
  always picks the same intervention).
- **The loop feeds itself.** `read_ledger()` lets the agent learn which
  intervention kinds work on which failure classes — the evolution
  strategy itself accumulates experience. Because the agent runs on this
  harness, every episode leaves a full trace: why a hypothesis was
  proposed and how the gate result was read is itself auditable, which is
  exactly the ADR's inspectability requirement.

### Proposed layout

New code lives in one subpackage, `simple_agent_lab/evolution/` (package
paths; none exist yet):

```text
evolution/
  catalog.py     # substrate: scan run roots -> index rows (verdict, cost, paths)
  gate.py        # substrate: frozen-slice A/B via run_dataset(); returns comparison
  ledger.py      # substrate: append-only ledger JSONL
  artifacts.py   # substrate: staging/ and promoted/ stores, provenance fields
  agent.py       # evolution agent factory + the five tools above
  runtime.py     # run_with_artifacts(): inject promoted lessons/playbook as
                 #   kind="context" messages (mirrors run_with_skills())
```

Induced skills are ordinary SKILL.md files in a promoted learned-skills
root passed to `discover_skills()` — zero runtime changes.

## 4. MVP

> Three substrate modules (catalog, gate, ledger + a minimal artifact
> store) and one evolution agent with the five tools, run for N evolution
> episodes against a SWE-bench slice.

**Acceptance criteria:**

1. The ledger shows attempts of **at least two different intervention
   kinds** — this directly tests that an agent is deciding, not an ETL
   script running.
2. At least one candidate is promoted through the gate, and a re-run with
   the promoted artifact reproduces the improvement on the frozen slice.
3. Reverting the artifact file restores baseline behavior (rollback
   works).

**Deliberately out of MVP scope:** rubric LLM scoring and near-duplicate
dedup in the catalog (verifier verdicts suffice), similarity retrieval
(top-k by tag is enough), the ACE playbook curator as a dedicated
component (a "playbook delta" is just another candidate kind the agent may
draft), prompt A/B as a separate driver (a prompt variant is another
candidate kind), and all of Track B (section 6).

**Two cheap pre-commitments made now** (the only things that cannot be
backfilled later):

1. Trace meta records `policy_id` (model/version), sampling parameters,
   and logprobs when the endpoint provides them — required later for
   off-policy correction in RL; impossible to reconstruct after the fact.
2. `result.json` adopts a standard reward key
   (`{"reward": <float>, "breakdown": {...}}`) alongside suite-specific
   verdict fields.

## 5. Intervention kinds (the agent's toolbox, with literature cautions)

These are candidate *kinds*, not roadmap phases. The evolution agent picks
among them per episode; each lands as a reviewable file with provenance
(`evidence_trace_id`, producer) and goes through the same gate.

- **Lesson** — structured `{task_tags, situation, lesson, evidence_trace_id,
  helpful, harmful}` entries (EvolveR / Decocted-Experience style),
  injected as `kind="context"` messages. Post-run outcome accounting
  (JitRL-style) decays harmful lessons out of retrieval.
- **Playbook delta** — itemized add/amend/retire of id-tagged strategy
  bullets in a per-role playbook (ACE). Never wholesale rewrites: that is
  the documented context-collapse failure mode.
- **Skill** — a drafted SKILL.md abstracted from exemplar trajectories
  (SkillRL/SAGE: store the recipe, not the logs). Gate must include a
  non-target slice, because over-triggering descriptions regress
  performance (skills-in-the-wild finding).
- **Prompt variant** — a system-prompt edit; the gate comparison *is* the
  A/B test.
- **Context policy tweak** — visibility/compression adjustments
  (`ContextPolicy`), the most harness-native and least explored kind.
- **(later) Fine-tuned checkpoint** — a candidate `Provider` record; see
  section 6. The gate does not care whether a candidate is a prompt or a
  checkpoint — that symmetry is the point.

## 6. Deferred: training track and the unified RL contract

Per the ADR, weight training stays outside the runtime: this repo produces
datasets and consumes checkpoints. The design principle that keeps RL from
becoming a third system: **RL is not a new pipeline; it is the same
pipeline, sampled more densely and consumed online.** Five unified
interfaces:

1. **Environment = the eval harness.** `run_suite_instance()` is one
   episode: container = environment, agent loop = policy-environment
   interaction, `evaluate()` = verifiable reward. An external trainer
   (TRL / verl-style) integrates through a thin rollout-server wrapper
   around `run_dataset()`; gradients and weight sync never enter this
   repo.
2. **One sample schema** — `(context, response, signals)` with optional
   signal fields: SFT uses `verdict`; DPO uses `group_id` + paired
   verdicts; GRPO uses `group_id` + `reward` + `behavior_policy` +
   `logprobs`. All three are projections of the same export; catalog and
   filtering are shared. Per-turn export via `model_turns_from_events()`
   guarantees training inputs equal the post-compression context the model
   actually saw.
3. **Reward layered on the suite.** Outcome reward now (the standard
   `result.json` key); process rewards later as scores attached to span
   ids (SWE-TRACE-style rubric PRMs, AgentEvolver self-attribution) — the
   span tree already provides the attribution structure.
4. **Policy = a Provider record** behind an OpenAI-compatible endpoint.
   Rollout workers do not know whether they serve eval, SFT collection, or
   RL sampling.
5. **Same gate for promotion; extra monitoring for training.** Checkpoint
   promotion goes through the frozen-slice gate like every other
   candidate. RL additionally needs in-training monitors (reward curve, KL
   to reference, entropy, tool-call format error rate) — monitoring
   decides when to stop training; the gate decides what ships.

Sequencing note: DPO's group sampler ("same instance, N rollouts, pair by
verdict") is exactly GRPO's group-advantage collector — build it once.
Task synthesis (AgentEvolver self-questioning, daVinci-Env, SWE-rebench)
slots in as an upstream pipeline stage producing instance files
indistinguishable from benchmark instances. Under optimization pressure,
verdict weaknesses become reward-hacking targets (tests gamed, rubric
judges fooled), so label auditing upgrades from quality control to
adversarial review.

## 7. Validation: proving the data is fit for training the current model

Four falsifiable claims, cheap to expensive. MVP enables the first two
checks marked (*); the rest activate when Track B starts.

1. **The data is faithful** (training input = what the model experienced):
   replay check — re-render each exported turn's context via the context
   view and byte-diff against the export (unit-testable, permanent);
   chat-template/tool-format validation; length within
   `Provider.context_window`; (*) decontamination — zero instance- and
   content-level overlap between training traces and the gate's frozen
   slice.
2. **The data carries signal**: verdict-source labeling in the catalog
   (verifier vs. rubric; rubric-labeled data requires sampled human audit
   with reported agreement); current-model loss probe on the dataset
   (near-zero loss = nothing to learn; abnormal loss = distribution
   mismatch, usually a faithfulness bug); bucket by `policy_id` — own
   successful trajectories are rejection-sampling data, near-on-policy by
   construction; foreign-policy data earns its place separately.
3. **The improvement is causal** (the core evidence): ablation matrix
   through the same gate — filtered vs. equal-size unfiltered data
   (filtering earns its cost), 25/50/100% data-scaling curve (flat curve =
   data homogeneity, fix collection not volume), each augmentation
   (backfill, hint distillation) ablated separately, and failed-trajectory
   training as a negative control (must hurt).
4. **No side effects**: (*) no regression on non-target slices; no format
   collapse (tool-call error rate, trajectory-length drift vs. baseline);
   a held-out slice that distillation never reads, rotated periodically.

Each dataset release ships a dataset card (counts, length/tool
distributions, verdict-source mix, dedup rate, decontamination result,
loss stats, content hash); ledger entries reference the dataset hash so
"which data batch produced which improvement" stays traceable.

## 8. Risks and guardrails

- **Experience poisoning / self-reinforcing injection** (Zombie Agents):
  the staging/promoted split, provenance on every artifact, gate-only
  promotion, append-only ledger, one-command rollback. The evolution agent
  never writes directly to the promoted store.
- **Evolution-agent thrash** (novel to the agent-driven design): an agent
  that keeps proposing failing candidates burns eval budget. Cap gate runs
  per episode, and surface ledger-derived hit rates so the agent (and its
  operator) can see diminishing returns.
- **Context collapse / brevity bias** (ACE): playbook updates are itemized
  deltas; lesson volume is bounded by retrieval, not lossy rewriting.
- **Skill over-triggering** (skills-in-the-wild): gates include non-target
  slices.
- **Eval overfitting**: held-out slice never read by distillation, rotated
  periodically; decontamination for tasks drawn from public benchmarks.
- **Scope creep vs. the ADR**: weight training stays external; the runtime
  only ever consumes versioned artifacts (context files or a Provider
  record); the substrate stays non-agentic.
