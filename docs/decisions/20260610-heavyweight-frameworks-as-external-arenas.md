---
title: "Heavyweight Eval Frameworks Are External Arenas, Not the Backbone"
status: Accepted
date: 2026-06-10
slug: heavyweight-frameworks-as-external-arenas
---

# Heavyweight Eval Frameworks Are External Arenas, Not the Backbone

## Status

Accepted

## Context

Harbor (harborframework.com, the Terminal-Bench successor maintained by the
Laude Institute) prompted the question this ADR answers: it ships a benchmark
registry, cloud sandbox providers (Daytona, Modal, E2B, Runloop), and a dozen
pre-integrated CLI agents (Claude Code, Codex CLI, OpenHands, ...). Compared to
our in-house eval framework (ADR generic-containerized-eval-framework), it has
*more agents and more benchmarks*. Why not adopt it — or any similar
heavyweight harness — as our evaluation backbone? The same challenge will recur
with other frameworks, so we record the reasoning once.

What a Harbor-class framework offers and asks for:

- Tasks defined in its format (`task.toml` + `environment/` + `tests/`
  producing a reward file). The agent under test is a black box: it is given an
  environment, and what comes back out is an end-of-run reward.
- Custom agents integrate by implementing its agent interface
  (`BaseAgent` driving the environment externally, or `BaseInstalledAgent`
  installed into the container).

What this project is, and where it is going:

- The mission is explicit data flow over hidden framework behavior; the eval
  framework and the runtime are co-designed. The runner streams
  `out/trajectory.jsonl` live for the trace viewer, records the three-layer
  trace (ADR three-layer-trace-event-span-training), accounts tokens per
  request (ADR context-size-accounting), and supports offline scored runs via
  a wheelhouse with host-pull artifact movement.
- The roadmap is self-evolution of both the agent and the model
  (ADR treat-self-evolution-as-harness-capability). That loop is
  `run -> trace -> evaluate -> propose candidate -> compare -> accept or
  reject`, and every stage consumes white-box data: the full trajectory to
  diagnose *why* a run failed, the context view and token accounting to compare
  candidates fairly, and the training layer of the trace to produce examples
  for model evolution. A single reward number at the end of a black-box run
  cannot drive "propose" or "compare" — for self-evolution, **evaluation is the
  data-collection stage of the loop, not a leaderboard at the end of it**.

The category mismatch is the heart of it: Harbor's "many agents, many benches"
breadth serves benchmark *operators* who rank off-the-shelf agents. We are
agent *builders* — we run our own agent and need to see every step of it. The
two are complementary, not substitutes.

## Decision

1. **The in-house eval framework stays the backbone.** Anything the
   self-evolution loop consumes — trajectories, traces, token accounting,
   scoring, run comparison — runs on our own runner
   (ADR generic-containerized-eval-framework), where the runtime and the
   harness are co-designed and every intermediate artifact is ours.

2. **Heavyweight frameworks are integrated as external arenas, through thin
   adapters at existing boundaries.** The natural seam is the agent boundary:
   wrap our shipped wheel as the framework's installed-agent (for Harbor, a
   `BaseInstalledAgent` whose install step is `pip install simple-agent-lab` —
   the same bootstrap our own runner uses). The adapter lives in an optional
   directory under `evals/` (a harbor adapter next to the existing suites)
   behind an extra; the core gains no dependency on the framework.

3. **Our task and trace formats remain the source of truth.** We do not
   restructure suites into an external task format, and we do not adopt an
   external trace schema. If an arena run should feed the evolution loop, the
   adapter converts *outward* at the edge.

4. **Adoption test for any future heavyweight framework.** Integrate at the
   boundary when it offers breadth we lack — benchmark registries, cloud
   sandbox capacity, side-by-side baselines against pre-integrated agents.
   Refuse it as a backbone if any of these hold:

   - It returns only an end-of-run reward, not in-loop data (trajectory,
     context, per-request usage).
   - Integration cannot live in an optional adapter without touching core.
   - It would replace something co-designed with the runtime (live trace,
     three-layer records, accounting, offline mode).

   Shorthand: *breadth can be plugged in later through an adapter; depth,
   once outsourced, cannot be recovered.*

## Consequences

- The self-evolution direction keeps its substrate: every run on the backbone
  yields white-box feedback (full trajectory, span and training records, token
  accounting), so the evolve loop can be built without re-platforming.
- Running Terminal-Bench or any Harbor-registry benchmark, scaling onto cloud
  sandboxes, or benchmarking against Claude Code / OpenHands baselines becomes
  a bounded task: one adapter file, no core changes. We accept that this
  breadth is *deferred* until someone writes the adapter.
- We keep paying the maintenance cost of our own runner (backends, stores,
  in-container bootstrap). That cost already bought capabilities Harbor does
  not offer for our setup — offline scored runs, intranet workers with
  host-pull artifacts, unmodified upstream images — so it is not duplicated
  spend.
- Arena adapters track an external, fast-moving interface and may break on
  framework upgrades; they must stay thin enough to rewrite in an afternoon.
- This ADR doubles as the project's answer to "why not just use X?" for any
  evaluation framework with a bigger registry.

## Alternatives Considered

- **Adopt Harbor as the evaluation backbone.** Rejected. The agent under test
  becomes a black box that emits a reward; the live trace viewer, three-layer
  trace, and token accounting have no equivalent; offline wheelhouse runs and
  host-pull movement are bespoke to our infra; and it contradicts the
  explicit-data-flow mission. Decisive once self-evolution is on the roadmap:
  the loop needs in-loop data that a reward file cannot carry.
- **Ignore external frameworks entirely.** Rejected. Benchmark breadth,
  cloud-sandbox capacity, and baseline comparisons against widely used agents
  are genuinely valuable, and the agent-boundary adapter makes them cheap to
  reach. "Not the backbone" must not harden into "never interoperate".
- **Maintain suites in both task formats (dual-track).** Rejected. Two sources
  of truth for task definitions drift apart; conversion belongs in the adapter,
  at the edge, in one direction.
- **Wrap external frameworks as a `ContainerBackend`.** Rejected. The backend
  seam abstracts *where compute runs*, not *who owns the loop*; a Harbor-shaped
  backend would smuggle the framework's run loop, task format, and reward
  shape into the backbone through a seam designed for daemons.
