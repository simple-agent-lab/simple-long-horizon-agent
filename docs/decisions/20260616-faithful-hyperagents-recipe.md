---
title: "Faithful HyperAgents Recipe Evolves The Agent Version, Not The Kernel"
status: Proposed
date: 2026-06-16
slug: faithful-hyperagents-recipe
note: amended by `retarget-recipe-to-dgm` (reference method changed to DGM)
---

# Faithful HyperAgents Recipe Evolves The Agent Version, Not The Kernel

## Context

The evolution infra is the contribution we open-source ("VeRL for self-evolving
agents", see `docs/design/20260612-evolution-infra-redesign.md`). To prove the
substrate, we need a *faithful* demo recipe reproducing a real published method,
the way VeRL ships PPO/GRPO. We chose HyperAgents (DGM successor,
`.idea/hyperagents/`).

The recipe currently in the tree reduces HyperAgents to editing two pure prompt
functions (`scaffold.py`) under a `not_worse` hill-climb guard. That is not
HyperAgents: HyperAgents' defining mechanics are (1) the meta-agent may "modify
any part of the codebase" (whole-repo diff), (2) an **open-ended** archive that
retains any child which compiles and runs as a `valid_parent`, and (3)
self-reference (the meta-agent can rewrite itself).

This collides with `treat-self-evolution-as-harness-capability` (Accepted),
which keeps open-ended self-modification off the default path and rejects
"agents freely rewriting the repository." We need to decide how a faithful
HyperAgents recipe can exist without violating that ADR.

## Decision

Build the faithful HyperAgents recipe as an **advanced reference recipe** that
evolves the **agent `Version`** — never the SAL kernel — under the kernel's
existing guarantees.

Specifically:

- The mutation surface is the whole agent program carried in a `Version`
  (whole-file diffs via recipe-local `repo_edits` helpers), with the eval
  harness held host-side and protected (the `domains/` analog).
- Acceptance is **validity-admission**, not the `not_worse` guard: a child that
  builds/imports and produces a gradable run enters the archive as
  `valid_parent=True` even if its reward is worse; parent selection is
  open-ended via recipe-local archive selection.
- Untrusted model-written code (the evolved agent, and in the self-referential
  stage the meta-agent) executes **only** in a sandbox (`code_rollout`), never
  in the host process.
- The kernel (`store`, `log`, `loop`) stays non-agentic and non-evolvable. Every
  change remains a content-addressed `Version` plus an append-only `Decision`,
  so full lineage and traceability hold.
- Self-reference ships via a C→A build order: a host-driven whole-repo coding
  strategy first, then the meta-agent lifted into the `Version` and executed by a
  thin host shim.

This requires **no kernel change** — only the three component gaps the infra
redesign already named: a sandboxed `code_rollout`, a whole-repo coding-agent
`strategy`, and the `selector`/meta wiring (selection math belongs with the
advanced recipe, not the kernel).

## Consequences

- The infra's central claim becomes demonstrable: a published, non-trivial
  self-evolution method is reproduced as components on reserved seams, with the
  guarantees intact.
- "Open-ended self-modification" is admitted **only** as a sandboxed, fully
  traced, opt-in recipe — consistent with the prior ADR, which permits advanced
  reference examples while keeping the default path conservative. The boundary
  the prior ADR protects (no free repository rewrites, no kernel mutation,
  traceable feedback) is preserved.
- The current two-function scaffold recipe and its mis-titled plan
  (`docs/superpowers/plans/2026-06-16-hyperagents-faithful-reproduction.md`) are
  superseded by the new design spec; the scaffold approach may remain as a
  separate, simpler "prompt-only" example if still useful.
- New cost: executing arbitrary evolved agent programs in a sandbox, plus staged
  eval to bound cost and reward noise.

## Alternatives Considered

- **Keep the two-function scaffold and call it HyperAgents.** Rejected: it is a
  different, weaker algorithm; it would misrepresent the infra's validation
  target.
- **Reproduce HyperAgents by letting the agent rewrite the SAL repo directly.**
  Rejected: violates the kernel's non-evolvable safety boundary and the prior
  ADR's traceability requirement.
- **Host-side meta-agent only (no self-reference).** Rejected as the endpoint
  (drops HyperAgents' headline self-referential mechanic) but accepted as the
  first build milestone on the way to self-reference (Path B in C→A).
- **Defer indefinitely as a reference-architecture note.** Rejected: a runnable
  faithful recipe is the strongest proof of the substrate and is now in scope.
