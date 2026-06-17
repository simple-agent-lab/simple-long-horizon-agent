---
title: "Retarget The Self-Evolving Recipe To DGM, Not HyperAgents"
status: Proposed
date: 2026-06-17
slug: retarget-recipe-to-dgm
note: amends `faithful-hyperagents-recipe`
---

# Retarget The Self-Evolving Recipe To DGM, Not HyperAgents

## Context

`faithful-hyperagents-recipe` set HyperAgents as the reference method to
reproduce on the evolution substrate. After building the recipe (archive,
open-ended admission, whole-agent-program mutation, SWE-bench-in-sandbox eval)
and running it, two facts became clear:

1. **DGM and HyperAgents share one engine.** They are from the same group, and
   the reference repo (`.idea/hyperagents/`) ships DGM as a *baseline inside the
   HyperAgents framework* (`generate_loop.py` imports `baselines.dgm.utils`;
   `--run_baseline dgm` runs DGM on the same archive/selection/eval code). The
   parts we built are common to both.
2. **Our scope already matches DGM, not full HyperAgents.** HyperAgents'
   generalization is multi-domain (balrog, genesis, imo, paper_review, …) plus
   ensemble-of-archive evaluation. We deliberately run SWE-bench only, with no
   ensemble. That is DGM's scope, not HyperAgents'.

Measured against the three defining mechanics named in
`faithful-hyperagents-recipe`, the current recipe stands at:

| Mechanic | Status |
| --- | --- |
| 1. Meta-agent rewrites the agent's whole program (full-file edits under `agent/`, add/remove, imports the wheel; AST-validated, sandboxed) | Reproduced, scoped to the agent `Version` |
| 2. Open-ended archive (admit any child that builds + runs as `valid_parent`; `archive.select_parent` with `latest/best/score_prop/score_child_prop`) | Reproduced |
| 3. Self-reference (the agent improves its own improver) | Not yet (host-side meta-strategy; "Path B" of the prior ADR's C→A plan) |

Self-reference is required by DGM too (Darwin **Gödel** Machine), so switching
targets does not remove that gap. But DGM makes the gap *cheaper* to close,
because in DGM the meta-agent and the task-agent are the same coding agent:
self-improvement is the existing SWE-bench coding agent pointed at its own
`agent/` package with an "improve your resolve rate" task — reusing the sandbox
and agent we already have, instead of maintaining a distinct meta-agent.

## Decision

Retarget the reference recipe from HyperAgents to **DGM**.

- The recipe is named and documented as a faithful **DGM** reproduction. Its
  current state is a faithful DGM *substrate* demo (mechanics 1 and 2) with
  self-reference as the next milestone.
- All substrate choices from `faithful-hyperagents-recipe` stand unchanged
  (evolve the agent `Version` not the kernel; validity-admission; sandboxed
  `code_rollout`; content-addressed `Version` + append-only `Decision` lineage).
- The C→A self-reference plan stays, with one simplification: the self-reference
  milestone unifies meta-agent and task-agent (the evolved coding agent runs on
  its own package) rather than lifting a separate meta-agent into the `Version`.
- HyperAgents-only features (multi-domain, ensemble) are explicitly out of scope
  for this recipe.

## Consequences

- **Honesty of the claim improves.** The open-source claim becomes "we reproduce
  the DGM substrate (open-ended, archive-based, self-improving coding-agent loop)
  as composable components with full lineage," which is exactly what runs today —
  rather than implying HyperAgents' multi-domain/ensemble/self-reference scope we
  do not cover.
- **The remaining milestone shrinks.** Self-reference reuses the existing
  task-agent + sandbox instead of a separate meta-agent component.
- **Naming churn.** Files, run ids, and docs under the `hyperagents` name should
  migrate to `dgm` (or be clearly labeled as the DGM recipe) to avoid confusion;
  until migrated, treat "hyperagents recipe" and "DGM recipe" as the same path.
- `faithful-hyperagents-recipe` is amended (not withdrawn): its substrate
  decision is retained; only the reference-method label and the self-reference
  shape change.

## Alternatives Considered

- **Stay on HyperAgents.** Rejected for now: it implies multi-domain + ensemble
  scope we do not implement, overstating the claim, with no nearer-term payoff.
- **Drop the reference-method framing and ship a generic "self-evolving agent"
  demo.** Rejected: reproducing a named, published method (as VeRL ships
  PPO/GRPO) is the stronger proof of the substrate; DGM is the closest faithful
  match to current scope.
- **Reach full HyperAgents before publishing.** Rejected as the near-term target:
  it adds multi-domain and ensemble work that does not strengthen the substrate
  claim and delays a runnable faithful recipe.
