# ADR 0022: Per-Round Model Selection via a Model Map + Chooser

## Status

Accepted

## Context

An LLM-backed agent was pinned to one model for its whole run:
`make_llm_agent(provider=...)` closed over a single `Provider`, and the
core loop calls `agent.generate(visible) -> Message` once per round with no
round argument. Common workflows want to vary the model per round — explore
with a cheap model and finish with a strong one, or escalate to a stronger
model after a round fails.

The constraint is the core contract. `Agent.generate(visible) -> Message`
is the runtime boundary fixed by ADR 0001 and ADR 0009 and is depended on
by the loop, the test fakes, and the demos. Threading a round index through
`generate` would change that signature everywhere and erode the beginner
readability the project optimizes for (see `AGENTS.md`). `Provider` is also
deliberately pure data (no callables) per the `llm/` layer design, so the
per-round decision cannot live on the provider itself.

Two earlier iterations of this ADR were superseded before the feature
shipped to `main`: a `ProviderSelector` callable returning a `Provider`
(flexible but heavy — two exported type aliases and a hand-written lambda),
and a plain list of providers indexed by round (simple but it couldn't route
on conversation state, and "list exhaustion" needed an arbitrary rule). The
shape below separates the two concerns cleanly.

## Decision

Resolve which model to call **inside `make_llm_agent`**, before each model
step, leaving the core loop and the `generate` contract untouched.

Split "which models exist" from "which one this round":

- `provider` accepts a single `Provider` (one model for the whole run, the
  unchanged default) **or a map of named models** —
  `{"fast": ..., "strong": ...}`.
- `choose_model` is a `(RoundContext) -> name` function the factory calls
  before each round; it returns a key in the map and the factory calls
  `provider[key]`. It is required with a map and rejected with a single
  `Provider`.

```python
make_llm_agent(
    name="agent",
    provider={"fast": fast, "strong": strong},
    choose_model=lambda ctx: "fast" if ctx.round == 0 else "strong",
)
```

`RoundContext` is what the chooser sees: `round` (zero-based round index),
`messages` (the visible context for this round), and a `last_failed`
convenience (did the most recent tool result in view error out?). So a
chooser can route on the round number or on conversation state — e.g.
`"strong" if ctx.last_failed else "fast"` to escalate after a failed round.
Returning a *name* rather than a `Provider` keeps the map the single source
of available models and makes traces read in plain model names.

The round index is derived statelessly from the visible context: the count
of assistant messages this agent has already produced. The loop calls
`generate` once per round and records the output before the next, so the
count is 0 on the first step, 1 on the second, and so on, and it resets on
its own for each fresh `agent.run(...)` — there is no mutable counter to leak
across runs. A bare `Provider` skips that scan and the chooser entirely.

The served model already rides on each response
(`AssistantMessage.model` / `ModelResponseEvent.model`), so a trace shows
which model answered each round with no extra plumbing.

`RoundContext` is exported from the package root (callers annotate their
chooser with it). `ModelChooser` and `ProviderLike` stay internal to
`llm_agent.py` for signatures. The preset agents (`make_bash_agent`,
`make_bash_task_agent`) take and forward `choose_model`.

## Consequences

- Per-round model switching needs no change to `core.py`, the `GenerateFn`
  type, the test fakes, or the demos. The feature is additive and backward
  compatible — passing a single `Provider` behaves exactly as before.
- "Which models exist" is declarative (the map) and "which one this round"
  is a small function — the two can be written, read, and tested apart.
- The chooser can react to conversation state via `RoundContext`, so
  content-aware routing (escalate on error, switch after a tool runs) needs
  no further API.
- The round index is a model-choice policy, not a correctness invariant. If
  compression folds away some of the agent's own earlier messages the count
  can dip, which may repeat an earlier model choice after a compaction. That
  is acceptable for routing; nothing else depends on the count.
- A chooser returning a key not in the map raises a clear `KeyError` naming
  the bad key and the available keys.

## Alternatives Considered

- **A `ProviderSelector` callable returning a `Provider`
  (`Callable[[int], Provider]`).** Flexible, but heavier: two exported type
  aliases and a hand-written lambda for the common case, and it conflated
  "what models exist" with "which one now".
- **A plain list of providers indexed by round (`[fast, strong]`).** Simple
  and declarative, but it cannot route on conversation state, and it forces
  an arbitrary "list exhaustion" rule (cycle vs. clamp).
- **Thread a round index through `generate(visible, round)`.** Most explicit,
  but breaks the ADR 0001 / 0009 core contract and every fake/demo, and
  trades away beginner readability for a feature only LLM-backed agents use.
- **Have the chooser return a `Provider` instead of a key.** Removes the map
  as a source of truth, lets traces show ad-hoc providers, and drops the
  cheap "key must exist" check. Returning a name is the smaller, safer shape.
- **Put the model choice on `Provider` (callable field).** Violates the
  `llm/` rule that `Provider` is pure, JSON-serializable data with no
  callables.
- **Keep a mutable round counter in the `generate` closure.** Simple, but
  leaks state across `agent.run(...)` calls on a reused agent. Deriving the
  index from the visible context avoids the hidden state entirely.
