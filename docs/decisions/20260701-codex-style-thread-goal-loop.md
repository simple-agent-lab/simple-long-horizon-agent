---
title: "Add Codex-Style Thread Goal Loop Beside the Verifier-Driven Goal Loop"
status: Accepted
date: 2026-07-01
slug: codex-style-thread-goal-loop
---

# Add Codex-Style Thread Goal Loop Beside the Verifier-Driven Goal Loop

## Context

The existing `run_goal_loop` is intentionally verifier-driven: it keeps resuming
one conversation until a caller-supplied `CompletionCheck` says the objective is
complete. That is the right shape for benchmark repair tasks where completion
should be gated by a command, judge, or other independent verifier.

Codex `/goal` uses a different center of gravity. The goal is explicit
thread-level state owned by the host, the model can inspect that state through a
tool, and the model can only make narrow terminal updates such as `complete` or
`blocked`. The host continues by re-reading the authoritative goal state rather
than treating chat history or a verifier callback as the only source of truth.

We want that Codex-style control model available for teaching and experiments
without changing the public behavior of the verifier-driven loop.

## Decision

Add `workflow/thread_goal_loop.py` as a sibling of `workflow/goal_loop.py`, not a
replacement.

The new loop introduces:

- `ThreadGoal` and `ThreadGoalStore` as the authoritative in-memory goal state.
- `get_goal` and `update_goal` tools over that store.
- `run_thread_goal_loop`, which creates a goal, steers each continuation from
  the current goal state, and stops on terminal goal status or host-owned limits.
- The live `ThreadGoalStore` attached to `State.data["thread_goal_store"]`
  so callers inspecting the final state can find the current control object.
- `GoalStatusEvent` entries in `state.events`, so the explicit goal's lifecycle
  is visible in traces without making `state.data` the authority.
- A `goal` workflow flavor that runs the Codex-style loop, while the existing
  `loop` flavor continues to use verifier-gated `run_goal_loop`.

Model-facing lifecycle control stays narrow: `update_goal` accepts only
`complete` and `blocked`. Host-owned states such as `paused` and
`budget_limited` are set by host code.

## Consequences

- The repository now exposes two distinct goal-loop designs:
  - `loop`: verifier-driven, best when completion must be independently gated.
  - `goal`: Codex-style, best when the experiment is about explicit thread goal
    state and host-owned continuation.
- Existing `run_goal_loop` callers keep their behavior and public interface.
- `run_thread_goal_loop` derives a tool-bound copy of its input agent without
  mutating the caller's agent. LLM-backed agents expose a small
  `generate_for_tools` binding hook, so the derived copy sends the injected
  `get_goal` / `update_goal` schemas to the model and dispatches the same tools
  at runtime. Programmatic agents keep their original generate callable.
- A `ThreadGoalStore` still owns current goal state. `State.data` carries that
  live object for in-process consumers; `State.events` carries an append-only
  history for trace/replay.
- The first implementation uses an in-memory store. Persistence across process
  restarts remains a later extension.

## Alternatives Considered

- **Replace `run_goal_loop` with the Codex-style loop.** Rejected because it
  would remove the verifier-driven completion authority that is useful for
  SWE-bench and other testable tasks.
- **Add Codex goal state into `state.data`.** Rejected because the goal should
  have a clear ownership boundary and should not depend on mutable scratchpad
  conventions.
- **Only document the Codex pattern.** Rejected because the useful distinction
  is behavioral and needs tests: model-visible `get_goal`, narrow
  `update_goal`, host-owned status transitions, and a real workflow flavor.
