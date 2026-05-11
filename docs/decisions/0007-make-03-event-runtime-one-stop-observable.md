# ADR 0007: Make 03 Event Runtime the One-Stop Observable Baseline

## Status

Withdrawn (2026-05-11). Superseded by ADR 0009, which promoted the
balanced runtime to `src/simple_agent_lab/core.py` and retired the
`03_event_runtime` design version this ADR was scoped to. The
observability and replay direction this ADR proposed is no longer
attached to a live runtime; if a future "one-stop observable" surface
is wanted, capture it as a new ADR against the canonical `core.py`
instead of resuming this one.

## Context

`02_balanced_runtime` is the lead core for self-evolution (ADR 0005) because it supports multi-agent scheduling, agent-as-tool delegation, and generator-based event streams. ADR 0009 promotes the simplified version into `src/simple_agent_lab/core.py`.

`03_event_runtime` currently provides a clean event-sourced single-agent loop with `AgentLoop`, `RuntimeState.events`, and `LLMModelClient` backed by the shared `simple_agent_lab.llm` layer. However, users perceive it as "not very different" from 02: both have events, both support tools, both use the same `Message` / `Tool` shapes.

The goal for 03 is to become the **one-stop integrated reference** for:
- Single-agent scenarios with full observability.
- Easy provider swapping (fake → OpenAI → Anthropic) without touching core loop.
- Built-in replay / inspection / eval surface from the event log.
- Minimal wiring for beginners: one import, one `run()` call yields complete trace + result.

This keeps 01 as pure teaching baseline, 02 as multi-agent functional core, and gives 03 a clear, differentiated "batteries-included but still inspectable" role.

## Decision

Position `03_event_runtime` as the one-stop observable runtime by adding small, explicit integration helpers:

1. A convenience factory / facade (e.g., `make_loop` or `ObservableAgent`) that wires `AgentLoop` + `LLMModelClient` + default `RunConfig` + optional persistence hook.
2. Replay / fork capability: functions that can consume `state.events` to reconstruct state, re-run a prefix, or export for evals.
3. Optional streaming hook point in `ModelClient` (so 03 owns the streaming reference implementation that 02 deliberately left out).
4. Keep the core `AgentLoop.run()` and `print_trace()` unchanged so existing demos stay stable; new capabilities are additive and opt-in.

The feedback signal before any code change: the existing `demo.py` must continue to run and produce identical output. New capabilities will be proven by an additional small script (e.g., `replay_demo.py`) that shows "load events → replay decision" or "swap provider in one line".

## Consequences

- 03 becomes clearly the "I just want a traceable single agent with real LLM and zero boilerplate" choice.
- 02 stays lean for multi-agent control flow experiments.
- Smallest change principle: add helpers in a new `facade.py` or extend `models.py`, never bloat `core.py`.
- Readability preserved: every new helper has a one-line docstring and is exercised by a deterministic demo.
- Future self-evolution work can still start in 02; 03 serves as the eval harness target (full event log makes metric extraction trivial).

## Alternatives Considered

- Merge 02 and 03 into one runtime → rejected: would blur the teaching axis (simple / moderate / rich) and violate "small explicit modules".
- Keep 03 as-is and only update docs → insufficient, user feedback indicates the integration surface is not obvious enough.
- Add heavy framework features (registry, plugin system) → violates code-style: "avoid abstractions that only hide a few lines of code" and "no framework-style registries until plain lists become confusing".
