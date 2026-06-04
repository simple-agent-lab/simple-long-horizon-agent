# ADR 0022: Per-Turn Model Selection via a Provider Selector

## Status

Accepted

## Context

An LLM-backed agent was pinned to one model for its whole run:
`make_llm_agent(provider=...)` closed over a single `Provider`, and the
core loop calls `agent.generate(visible) -> Message` once per turn with no
turn argument. Common workflows want to vary the model per round — explore
with a cheap model and finish with a strong one, alternate models to
compare them, or escalate after a rough patch.

The constraint is the core contract. `Agent.generate(visible) -> Message`
is the runtime boundary fixed by ADR 0001 and ADR 0009 and is depended on
by the loop, the test fakes, and the demos. Threading a turn index through
`generate` would change that signature everywhere and erode the beginner
readability the project optimizes for (see `AGENTS.md`). `Provider` is also
deliberately pure data (no callables) per the `llm/` layer design, so the
per-turn decision cannot live on the provider itself.

## Decision

Resolve which provider to call **inside `make_llm_agent`**, before each
model step, leaving the core loop and the `generate` contract untouched.

`make_llm_agent(provider=...)` now accepts either:

- a single `Provider` — one model for the whole run (unchanged default), or
- a `ProviderSelector` — a `Callable[[int], Provider]` resolved each turn
  from the zero-based turn index.

The turn index is derived statelessly from the visible context: the count
of assistant messages this agent has already produced (`_turn_index`). The
loop calls `generate` once per turn and records the output before the next
turn, so the count is 0 on the first step, 1 on the second, and so on, and
it resets on its own for each fresh `agent.run(...)` — there is no mutable
counter to leak across runs.

A plain list of models is intentionally *not* a separate parameter: it is
just a one-line selector the caller writes —
`lambda turn: models[turn % len(models)]` (cycle) or
`lambda turn: models[min(turn, len(models) - 1)]` (escalate-then-hold) —
so the rotation policy stays visible in caller code rather than hiding
behind a list-exhaustion rule the framework would have to define.

The served model already rides on each response
(`AssistantMessage.model` / `ModelResponseEvent.model`), so a trace shows
which model answered each turn with no extra plumbing.

`ProviderLike = Provider | ProviderSelector` and `ProviderSelector` are
exported from the package root and accepted by the preset agents
(`make_bash_agent`, `make_bash_task_agent`).

## Consequences

- Per-turn model switching needs no change to `core.py`, the `GenerateFn`
  type, the test fakes, or the demos. The feature is additive and
  backward compatible — passing a `Provider` behaves exactly as before.
- The decision policy lives in caller code (any function of the turn), so
  cycling, escalation, and conditional routing are all expressible without
  new framework surface.
- The turn index is a model-choice policy, not a correctness invariant. If
  compression folds away some of the agent's own earlier messages the count
  can dip, which may repeat an earlier model choice after a compaction. That
  is acceptable for routing; nothing else depends on the count.
- Routing keyed only on the turn index cannot react to conversation content
  (e.g. "switch to a stronger model after an error"). A selector can close
  over external state for that; a richer signature is a future ADR if the
  need becomes common.

## Alternatives Considered

- **Thread a turn index through `generate(visible, turn)`.** Most explicit,
  but breaks the ADR 0001 / 0009 core contract and every fake/demo, and
  trades away beginner readability for a feature only LLM-backed agents use.
- **Accept a `Sequence[Provider]` and index by turn.** Declarative and
  JSON-friendly, but forces the framework to pick a list-exhaustion rule
  (cycle vs. clamp). A selector subsumes both and keeps the policy in caller
  code, so the list was dropped.
- **Put the model choice on `Provider` (callable field).** Violates the
  `llm/` rule that `Provider` is pure, JSON-serializable data with no
  callables.
- **Keep a mutable turn counter in the `generate` closure.** Simple, but
  leaks state across `agent.run(...)` calls on a reused agent. Deriving the
  index from the visible context avoids the hidden state entirely.
