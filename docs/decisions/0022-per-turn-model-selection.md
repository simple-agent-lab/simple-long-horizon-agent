# ADR 0022: Per-Round Model Selection via a Provider List

## Status

Accepted

## Context

An LLM-backed agent was pinned to one model for its whole run:
`make_llm_agent(provider=...)` closed over a single `Provider`, and the
core loop calls `agent.generate(visible) -> Message` once per round with no
round argument. Common workflows want to vary the model per round — explore
with a cheap model and finish with a strong one being the headline case.

The constraint is the core contract. `Agent.generate(visible) -> Message`
is the runtime boundary fixed by ADR 0001 and ADR 0009 and is depended on
by the loop, the test fakes, and the demos. Threading a round index through
`generate` would change that signature everywhere and erode the beginner
readability the project optimizes for (see `AGENTS.md`). `Provider` is also
deliberately pure data (no callables) per the `llm/` layer design, so the
per-round decision cannot live on the provider itself.

An earlier iteration of this ADR accepted a `ProviderSelector` callable
(`Callable[[int], Provider]`). It was flexible but the API felt heavy: it
added two exported type aliases and asked callers to hand-write a
`lambda round: ...`. The simpler shape below replaced it before the feature
shipped to `main`.

## Decision

Resolve which model to call **inside `make_llm_agent`**, before each model
step, leaving the core loop and the `generate` contract untouched.

`make_llm_agent(provider=...)` accepts either:

- a single `Provider` — one model for the whole run (unchanged default), or
- a **list of `Provider`s** — one model per round. Round N uses the Nth
  model, and the **last model sticks** once the list runs out.

So `[fast, strong]` runs round 0 on the fast model and every later round on
the strong one ("explore cheap, finish strong"); `[a, b, c]` gives a
different model to each of the first three rounds. There is exactly one new
idea to learn — "pass a list of models" — and no new vocabulary, callable,
or exported type. An empty list is rejected at construction.

The round index is derived statelessly from the visible context: the count
of assistant messages this agent has already produced (a bare `Provider`
skips that scan entirely). The loop calls `generate` once per round and
records the output before the next, so the count is 0 on the first step, 1
on the second, and so on, and it resets on its own for each fresh
`agent.run(...)` — there is no mutable counter to leak across runs.

The "last model sticks" (clamp) rule is chosen over cycling because it
matches the intuition of an escalation list and never surprises the caller
by quietly dropping back to a cheap model on a later round. A caller who
truly wants to alternate can repeat entries (`[a, b, a, b]`).

The served model already rides on each response
(`AssistantMessage.model` / `ModelResponseEvent.model`), so a trace shows
which model answered each round with no extra plumbing.

The single union type `ProviderLike = Provider | Sequence[Provider]` lives
in `llm_agent.py` for the signatures (including the preset agents) but is
**not** exported from the package root — it is an implementation detail of
the parameter, not a concept callers name.

## Consequences

- Per-round model switching needs no change to `core.py`, the `GenerateFn`
  type, the test fakes, or the demos. The feature is additive and
  backward compatible — passing a `Provider` behaves exactly as before.
- The public surface is minimal: one parameter that now also accepts a
  list. No new exported names, no callable contract to document.
- The round index is a model-choice policy, not a correctness invariant. If
  compression folds away some of the agent's own earlier messages the count
  can dip, which may repeat an earlier model choice after a compaction. That
  is acceptable for routing; nothing else depends on the count.
- Routing is keyed only on the round index, so it cannot react to
  conversation content (e.g. "switch to a stronger model after an error").
  That is intentionally out of scope; if the need becomes common, a richer
  mechanism is a future ADR rather than a heavier API today.

## Alternatives Considered

- **A `ProviderSelector` callable (`Callable[[int], Provider]`).** Strictly
  more flexible (cycle, clamp, or content-aware routing all expressible),
  but heavier: two exported type aliases and a hand-written lambda for the
  common case. The list covers the headline workflows with far less surface,
  so the callable was dropped before shipping.
- **Thread a round index through `generate(visible, round)`.** Most explicit,
  but breaks the ADR 0001 / 0009 core contract and every fake/demo, and
  trades away beginner readability for a feature only LLM-backed agents use.
- **Cycle instead of clamp at the end of the list.** Rejected as the default
  because silently returning to a cheaper model on a later round is
  surprising; clamp matches the escalation intuition, and cycling is still
  expressible by repeating list entries.
- **Put the model choice on `Provider` (callable field).** Violates the
  `llm/` rule that `Provider` is pure, JSON-serializable data with no
  callables.
- **Keep a mutable round counter in the `generate` closure.** Simple, but
  leaks state across `agent.run(...)` calls on a reused agent. Deriving the
  index from the visible context avoids the hidden state entirely.
