# ADR 0005: Make Balanced Runtime The Lead Core Candidate

## Status

Accepted

## Context

ADR 0004 defines self-evolution as a harness loop:

```text
run -> trace -> evaluate -> propose candidate -> compare -> accept or reject
```

That loop needs more than the minimal teaching runtime, but it should not force
the project into a heavy event-sourced or provider-specific framework.

The three current runtime candidates have different strengths:

- `01_functional_loop` is the clearest teaching baseline.
- `02_balanced_runtime` has multi-agent scheduling, context transforms, tool
  dispatch, and a small event stream.
- `03_event_runtime` has the richest request/response observability and
  provider boundary, but is single-agent and heavier for beginner recipes.

## Decision

Use `02_balanced_runtime` as the lead core candidate for self-evolution work.

Keep `01_functional_loop` as the minimal teaching reference. Keep
`03_event_runtime` as an observability and provider-boundary reference while
its useful ideas are folded into the balanced runtime.

The first concrete fold-in is request/response tracing:

- emit `model_request` before each `agent.act`
- emit `model_response` after each `agent.act`
- include the visible-message outline and LLM payload when available
- preserve an optional `state.data["candidate_id"]` on request and response
  events so eval code can compare baseline and candidate runs

This does not yet prune the other folders. Pruning should happen after the team
has run a self-evolution experiment and is confident the balanced runtime is the
right canonical implementation.

## Consequences

Implementation work should now prefer improving the promoted balanced runtime
in `src/simple_agent_lab/core.py` over adding parallel features to all three
versions. ADR 0009 records that promotion and removes the earlier steering /
follow-up queues from the canonical core.

Self-evolution examples should first target prompt text, context transforms,
tool descriptions, and small recipes on top of the balanced runtime.

The balanced runtime should stay readable. New observability should use plain
events and small helper functions instead of introducing a trace-store class,
eval framework, or hidden plugin system.

The tradeoff is that the repository temporarily keeps three runnable designs.
That is acceptable while the team uses `01` and `03` as references, but future
work should not let all three keep growing as independent frameworks.

## Alternatives Considered

- Select `01_functional_loop` as canonical. Rejected because it cannot express
  realistic self-evolution targets beyond prompt and context policy without
  editing the loop directly.
- Select `03_event_runtime` as canonical. Rejected for now because its
  observability is strong, but its single-agent shape makes workflow and recipe
  evolution less direct.
- Keep all three equally active. Rejected because it would split effort and
  make every architecture experiment three times larger.
