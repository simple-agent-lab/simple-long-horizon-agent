---
title: "Self-Evolution Mechanism: Offline Data Pipeline + Online Adaptation"
status: Proposed
date: 2026-06-10
slug: self-evolution-mechanism
---

# Self-Evolution Mechanism: Offline Data Pipeline + Online Adaptation

Design memo for building a self-evolving agent loop on top of Simple Agent
Lab. It covers two complementary tracks:

- **Training-free evolution** (memory, playbook, skills, prompts) — the
  default path, consistent with the accepted ADR
  [`docs/decisions/20260507-treat-self-evolution-as-harness-capability.md`](../decisions/20260507-treat-self-evolution-as-harness-capability.md).
- **Training-based evolution** (offline trajectory dataset production →
  SFT / preference / RL fine-tuning) — an opt-in extension that reuses the
  same data pipeline but runs the weight updates outside this repo.

Both tracks share one closed loop:

```text
run -> trace -> evaluate -> distill -> propose candidate -> compare -> accept or reject -> (next run)
```

## 1. Literature snapshot (late 2025 – mid 2026)

The design choices below are grounded in the current research landscape.
Grouped by the question each line of work answers.

### 1.1 Surveys / framing

- *A Survey of Self-Evolving Agents: What, When, How, and Where to Evolve*
  (arXiv 2507.21046, revised 2026-01) — organizes the field along **what**
  evolves (model / memory / prompt / tools / architecture), **when**
  (intra-run vs. cross-run), **how** (reward, demonstration, population
  search), and **where** it is evaluated. This is already the framing of our
  self-evolution ADR.
- *Adaptation of Agentic AI: A Survey of Post-Training, Memory, and Skills*
  (arXiv 2512.16301) — explicitly contrasts the two tracks we adopt here:
  parameter adaptation (post-training) vs. non-parametric adaptation
  (memory databases, skill libraries, system-prompt refinement), and notes
  that mature systems run both against the same experience stream.

### 1.2 Training-free: evolving context, memory, experience

- *Agentic Context Engineering (ACE)* (arXiv 2510.04618, ICLR'26) — treats
  context as an **evolving playbook** maintained by a
  Generator → Reflector → Curator pipeline. Key findings we adopt: avoid
  *brevity bias* and *context collapse* by making **incremental, itemized
  delta updates** to the playbook instead of wholesale rewriting; reported
  +10.6% on agent benchmarks with 83.6% lower token cost than rewriting.
- *EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle*
  (arXiv 2510.16079) — closed-loop lifecycle where the agent distills its own
  trajectories into a persistent experience corpus and retrieves from it.
- *Just-In-Time RL: Continual Learning in LLM Agents Without Gradient
  Updates* (arXiv 2601.18510) — keeps a non-parametric memory of past
  (state, action, outcome) and estimates action advantages at test time;
  shows "RL-flavored" improvement is possible with zero weight updates.
- *Decocted Experience Improves Test-Time Inference in LLM Agents*
  (arXiv 2604.04373) — distilled ("decocted") experience beats raw
  trajectory replay; supports our choice to store **lessons**, not raw logs.
- *From Player to Master: Test-Time Learning via RL over Memory*
  (arXiv 2606.08656, "MemoPilot") — learns the **memory update policy**
  itself with downstream reward, a later-stage option for our memory layer.
- *SuRe: Surprise-Driven Prioritised Replay* (arXiv 2511.22367) — replay
  scheduling for continual learning; relevant to lesson retention/decay.

### 1.3 Skill-library evolution (bridge between the two tracks)

- *SkillRL: Evolving Agents via Recursive Skill-Augmented RL*
  (arXiv 2602.08234) — abstracts past trajectories into a hierarchical,
  reusable skill library rather than storing raw trajectories, then trains
  the policy with the library in context.
- *Reinforcement Learning for Self-Improving Agent with Skill Library*
  ("SAGE", arXiv 2512.17102) — execution outcomes alone decide which skills
  enter the library (no human annotation); sequential rollout lets skills
  accumulate across similar tasks. Reported +8.9% goal completion with 26%
  fewer steps.
- *SkillNet* (arXiv 2603.04448) and *How Well Do Agentic Skills Work in the
  Wild* (arXiv 2604.04323) — skill creation/evaluation and a caution: skills
  help only when their trigger descriptions are precise; bad skills regress
  performance. Hence our accept/reject gate before a skill lands.

### 1.4 Training-based: offline data pipelines and agentic post-training

- *AgentEvolver: Towards Efficient Self-Evolving Agent System*
  (arXiv 2511.10395) — the closest end-to-end reference: a master
  orchestrator drives **task synthesis → trajectory rollout → experience
  summarization → sample construction & model optimization**, with
  self-questioning (task generation), self-navigating (experience reuse
  during exploration), and self-attribution (fine-grained credit
  assignment).
- *ASTRA: Automated Synthesis of Agentic Trajectories and Reinforcement
  Arenas* (arXiv 2601.21558) — fully automated data synthesis + verifiable
  RL with trajectory-level rewards balancing completion and efficiency.
- *SWE-TRACE* (arXiv 2604.14820) — test-grounded SWE data curation,
  rubric process-reward models, memory-augmented GRPO for long-horizon SWE
  agents; directly relevant to our SWE-bench suite.
- *SWE-Master* (arXiv 2602.03411), *Klear-AgentForge* (arXiv 2511.05951) —
  post-training recipes (SFT → RL scaling) for software agents.
- *SWE-AGILE* (arXiv 2604.11716) — "Hindsight Backfill": augment successful
  trajectories with reconstructed reasoning before SFT. *Hindsight Hint
  Distillation* (arXiv 2605.11556) — scaffold failed tasks with hints
  derived from known answers, then distill. Both are cheap dataset-quality
  multipliers for our pipeline.
- Environment/task supply: *daVinci-Env* (arXiv 2603.13023, SWE environment
  synthesis at scale), *Agent World Model* (arXiv 2602.10090, synthetic
  environments for agentic RL), *SWE-rebench* (21k+ interactive Python SWE
  tasks with decontaminated evaluation), and ICLR'25 *Learn-by-Interact*
  (backward construction of instructions from interaction histories).

### 1.5 Safety of self-evolution

- *Zombie Agents: Persistent Control of Self-Evolving LLM Agents via
  Self-Reinforcing Injections* (arXiv 2602.15654) — adversarial content can
  enter the experience store and **self-reinforce** across runs. Any memory
  / skill / playbook write path therefore needs provenance, review gates,
  and easy rollback. This validates the ADR's "every accepted evolution
  leaves a local record" rule as a security control, not just hygiene.

## 2. What the repo already provides

The harness is unusually well positioned; most of the loop's "read side"
exists today:

| Loop stage | Existing support |
|---|---|
| run | `src/simple_agent_lab/core.py` loop; eval harness `src/simple_agent_lab/evals/runner.py`, `src/simple_agent_lab/evals/dataset.py` (concurrent rollout) |
| trace | three-layer trajectory (event → span → training pair): `src/simple_agent_lab/trajectory/spans.py`, `src/simple_agent_lab/trajectory/training.py`, `src/simple_agent_lab/trajectory/run_trace.py` (`simple-agent-lab.trajectory.v3`), live JSONL via `src/simple_agent_lab/trajectory/live.py` |
| evaluate | suite `evaluate()` hook writes verdicts into `out/result.json`; candidate+judge composition in `examples/bench_suite/` |
| distill (export) | `src/simple_agent_lab/trace.py` already exports OpenAI fine-tuning records; `model_turns_from_events()` yields SFT-ready (input messages, output message, tools) pairs |
| inject context | skills runtime `src/simple_agent_lab/skills/runtime.py` injects SKILL.md bodies; `src/simple_agent_lab/context_view.py` controls model-visible context |
| swap model | `Provider` is a frozen data record (`src/simple_agent_lab/llm/provider.py`) — pointing at a fine-tuned checkpoint is a one-field change |

Missing pieces (the "write side" of the loop): trajectory filtering/dataset
builder, persistent cross-run memory, skill induction, prompt comparison
driver, the evolution ledger, and any training orchestration.

## 3. Target architecture

```text
                        OFFLINE DATA PIPELINE (batch, scheduled)
  ┌──────────────────────────────────────────────────────────────────────┐
  │  evals/out/<suite>/<run>/<instance>/{out/trajectory.jsonl,           │
  │                                      out/result.json}                │
  │        │                                                             │
  │        ▼                                                             │
  │  [1] Ingest+Index ── [2] Filter/Score ── [3] Distill                 │
  │       (catalog of      (verdicts, dedup,    ├─ lessons (memory)      │
  │        RunTrace refs)   cost, rubric LLM)   ├─ skills (SKILL.md)     │
  │                                             ├─ playbook deltas       │
  │                                             ├─ SFT dataset (JSONL)   │
  │                                             └─ preference pairs      │
  └───────────────┬───────────────────────────────────────┬──────────────┘
                  │ training-free artifacts               │ training artifacts
                  ▼                                       ▼
        ONLINE ADAPTATION (per run)            EXTERNAL TRAINING (opt-in)
  ┌─────────────────────────────────┐   ┌──────────────────────────────────┐
  │ retrieve lessons/playbook/skills│   │ SFT (LoRA) / DPO / GRPO job      │
  │ inject as kind="context" msgs   │   │ outside the repo (TRL, OpenAI    │
  │ post-run: reflect → propose     │   │ FT API, ...) → new checkpoint    │
  │ candidate artifact              │   └───────────────┬──────────────────┘
  └───────────────┬─────────────────┘                   │
                  ▼                                     ▼
        GATE: eval-set A/B compare  ◄────── candidate Provider(model=ft-ckpt)
                  │ accept/reject + ledger entry
                  ▼
        promoted artifact store (versioned, rollback-able)
```

One principle ties it together: **artifacts, not behaviors, evolve.** A
candidate is always a reviewable file (a lesson JSONL entry, a SKILL.md, a
prompt text, a playbook delta, or a checkpoint id), it is always gated by an
eval comparison, and acceptance always appends to a ledger.

### 3.1 New package layout (proposed)

Proposed new code lives under one subpackage, `simple_agent_lab/evolution/`
(paths relative to the package root, none of these exist yet):

```text
evolution/
  catalog.py      # [1] ingest: scan run roots, index RunTrace + verdicts
  filtering.py    # [2] filter/score: verdict, dedup, cost, rubric judge
  distill.py      # [3] reflectors: trace -> lesson / skill draft / playbook delta
  datasets.py     # [3] exporters: SFT JSONL, preference pairs (reuses
                  #     model_turns_from_events + openai_training_record)
  memory.py       # LessonStore: JSONL-backed, retrieval by task tags
  playbook.py     # ACE-style itemized playbook with delta merge (no rewrite)
  ledger.py       # append-only evolution ledger (per ADR record fields)
  gate.py         # A/B compare driver on a frozen eval set; accept/reject
  runtime.py      # run_with_memory(): retrieve + inject kind="context" msgs
```

Skill induction extends the existing `src/simple_agent_lab/skills/`
package (new `induction` module) so induced skills are ordinary SKILL.md
files picked up by `discover_skills()` with zero runtime changes.

## 4. Track A — training-free evolution (build first)

### Phase A0: dataset plumbing (shared with Track B)

1. **Catalog** — walk one or more run roots, parse each
   `out/trajectory.jsonl` + `out/result.json`, emit an index row per run:
   `{trace_id, suite, instance_id, verdict, turns, tokens, duration,
   producer, paths}`. Pure read-side code; no runtime changes.
2. **Filtering** — composable predicates over index rows and `RunTrace`:
   - verdict == resolved (from suite `evaluate()`),
   - near-duplicate task/trajectory dedup,
   - cost ceilings (turns, tokens),
   - optional LLM rubric scoring (SWE-TRACE-style process rubrics) for
     suites without a verifier.

### Phase A1: lesson memory (EvolveR / Decocted-Experience style)

- **Distill**: a reflector agent reads a filtered `RunTrace` (span tree
  makes failure localization cheap: which tool call failed, what was
  retried) and emits structured lessons:
  `{id, task_tags, situation, lesson, evidence_trace_id, helpful: 0,
  harmful: 0}`.
- **Store**: JSONL file under a workspace dir (e.g. a gitignored
  `.evolution/lessons.jsonl`), human-diffable, append-only with tombstones.
- **Inject**: `run_with_memory()` wrapper (mirroring
  `run_with_skills()`) retrieves top-k lessons by tag/similarity and
  injects them as `kind="context"` messages — these are already in
  `DEFAULT_PRESERVE_KINDS`, so compression never eats them.
- **Feedback**: post-run, increment helpful/harmful counters on retrieved
  lessons based on the verdict (JitRL-style outcome accounting); decayed or
  harmful lessons fall out of retrieval.

### Phase A2: playbook evolution (ACE-style)

For each agent role, maintain one curated playbook document of itemized,
id-tagged strategy bullets. The curator merges **deltas** produced by the
reflector (add/amend/retire single bullets) — never a wholesale rewrite, to
avoid ACE's context-collapse failure mode. The playbook is injected after
the system prompt as a context message. The playbook file itself is the
candidate artifact for the gate.

### Phase A3: skill induction

- Mine the catalog for **recurring resolved task patterns** (≥N similar
  successes) and **recurring failure motifs**.
- A skill-writer agent drafts SKILL.md from exemplar trajectories
  (SkillRL/SAGE pattern: abstract trajectories into a reusable recipe, do
  not store raw logs).
- The draft lands in a staging root; the gate runs the relevant eval slice
  with and without the staged skill; only accepted skills move into a
  learned-skills root passed to `discover_skills()`. Per the
  skills-in-the-wild benchmark finding, the gate must also check the skill
  does not regress non-target tasks (description over-triggering).

### Phase A4: prompt evolution

A/B driver over `run_dataset()`: fan instances across system-prompt
variants (or playbook versions), aggregate `DatasetReport` verdicts,
promote the winner if the delta clears a threshold on a frozen eval set.
This reuses the harness as-is; only the driver and the ledger entry are new.

### The gate and the ledger (all phases)

`gate.py` runs baseline vs. candidate on a frozen instance slice and writes
a ledger entry with exactly the ADR-mandated fields: feedback signal
(eval set), baseline behavior, candidate change (artifact diff/hash),
comparison result, accept/reject reason. Rollback = revert the artifact
file; the ledger never rewrites history. Provenance fields on every lesson /
skill / bullet (`evidence_trace_id`, producer) are the Zombie-Agents
mitigation: nothing enters the agent's context whose origin cannot be
traced to a concrete run.

## 5. Track B — training-based evolution (opt-in, after A0)

Per the ADR, weight training stays **out of the runtime**: this repo
produces datasets and consumes checkpoints; the training job itself is
external (TRL/LoRA on open models, or a provider fine-tuning API).

1. **SFT dataset build** (`datasets.py`):
   - source = filtered successful runs from A0;
   - per-run export via the existing `openai_training_record()` (whole
     conversation) and per-turn export via `model_turns_from_events()`
     (each model call's exact visible context → response pair — this
     respects compression, so training data matches what the model actually
     saw);
   - optional **hindsight backfill** (SWE-AGILE): a rewriter adds reasoning
     digests to sparse successful trajectories before export;
   - optional **hint distillation** (arXiv 2605.11556): for failed
     instances with known gold patches (SWE-bench has them in
     `input/eval.json`), re-run with oracle-derived hints, keep newly
     successful trajectories, strip the hints from the exported context.
2. **Preference pairs**: same instance, multiple rollouts (the dataset
   driver already supports `max_attempts`/concurrent runs) → (resolved,
   unresolved) trajectory pairs, or per-turn pairs where two rollouts
   diverge; export DPO-format JSONL.
3. **Train + swap**: external job produces a checkpoint served behind an
   OpenAI-compatible endpoint; a candidate `Provider` (new `model` /
   `base_url`) goes through the **same gate** as a prompt variant. The gate
   does not care whether the candidate is a prompt or a checkpoint — that
   symmetry is the point.
4. **"Online" training** is a scheduled offline loop, not in-process:
   `collect (run_dataset) → build dataset → train → gate → promote`,
   cadenced (e.g. nightly) AgentEvolver-style. True on-policy RL (GRPO over
   the harness) is a research extension: the eval containers already
   provide verifiable rewards (`result.json`), and a rollout server could
   wrap `run_suite_instance()`, but that should live in a separate repo or
   a reference-architecture note.

## 6. Phasing and effort

| Phase | Deliverable | Depends on | Size |
|---|---|---|---|
| A0 | catalog + filtering + SFT/preference exporters | nothing (read-only) | S–M |
| A1 | lesson store + `run_with_memory()` + outcome feedback | A0 | M |
| GATE | A/B compare driver + evolution ledger | A0 | S |
| A2 | ACE-style playbook curator | A1, GATE | M |
| A3 | skill induction → staged SKILL.md | A0, GATE | M |
| A4 | prompt-variant driver | GATE | S |
| B1 | external SFT loop + candidate Provider through GATE | A0, GATE | M (mostly external) |
| B2 | preference/DPO loop; hindsight backfill; hint distillation | B1 | M |
| B3 | on-policy RL rollout server | B1 | L, separate repo / reference note |

Recommended order: **A0 → GATE → A1 → A4 → A2 → A3 → B1 → B2**. A0+GATE+A1
already closes a real self-evolution loop end to end and produces the
datasets Track B needs, so nothing is throwaway.

## 7. Risks and guardrails

- **Experience poisoning / self-reinforcing injection** (Zombie Agents):
  provenance on every artifact, gate before promotion, append-only ledger,
  one-command rollback. Never let an in-run agent write directly to the
  promoted store.
- **Context collapse / brevity bias** (ACE): playbook updates are itemized
  deltas; lessons are bounded in count via retrieval, not via lossy
  rewriting.
- **Skill over-triggering** (skills-in-the-wild): gate checks non-target
  slices, not just the target slice.
- **Eval overfitting**: keep a held-out instance slice that the gate uses
  but distillation never reads; rotate it periodically (SWE-rebench-style
  decontamination if tasks come from public benchmarks).
- **Train/serve context mismatch** (Track B): export per-turn pairs from
  `model_turns_from_events()` so training inputs equal the post-compression
  context the model actually received.
- **Scope creep vs. ADR**: weight training stays external; the runtime
  only ever consumes versioned artifacts (context files or a Provider
  record).
