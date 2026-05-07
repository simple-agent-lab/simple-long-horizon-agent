# Design Versions

> **Status: focused candidates.**
> Three side-by-side runtime designs remain runnable. ADR 0005 makes
> `02_balanced_runtime` the lead core candidate for self-evolution work,
> while `01_functional_loop` stays the teaching baseline and
> `03_event_runtime` stays the graph orchestration / observability /
> provider-boundary reference.

The three folders move along a single axis - simple, moderate, complex - so
comparison stays about runtime architecture, not agent roles or workflow
recipes:

```text
01 simple:    messages -> context_view -> model -> append message
02 moderate:  next_agent(state) -> request event -> agent.step -> response event
03 complex:   graph nodes/edges -> event log -> observers/replay/reports -> provider boundary
```

Run any of the three with `uv`:

```bash
uv run python examples/design_versions/01_functional_loop/demo.py
uv run python examples/design_versions/02_balanced_runtime/demo.py
uv run python examples/design_versions/03_event_runtime/demo.py
```

Collect comparable raw trajectories for all three through temporary adapters:

```bash
PYTHONPATH=src python3 scripts/collect_design_version_trajectories.py
```

The project-level trajectory, evaluation, and training-data schemas are
runtime-neutral. These design-version adapters can be deleted once the three
runtime sketches collapse into one implementation.

## 01 Functional Loop (minimal)

One file, ~115 lines. No `State`, `Event`, or loop class. The core is just
`run_loop(messages, model)`. Best for teaching the loop in a few minutes. It
still uses the shared `Message` protocol, but keeps dispatch local.

## 02 Balanced Runtime

The lead design has been promoted into `src/simple_agent_lab/core.py`.
The local `core.py` and `agent.py` files are compatibility re-exports so
older demo imports still work. The canonical runtime has generator-based
`run()`, shared `Message`, `Event`, `State`, hooks, tool dispatch,
request/response trace events, and an `AgentRuntime` wrapper with
`subscribe / abort / prompt / resume`. Architecture borrowed from
[pi-mono](https://github.com/badlogic/pi-mono); see
[../../docs/reference-architectures/pi-mono-agent-runtime.md](../../docs/reference-architectures/pi-mono-agent-runtime.md).

Best when you want a small multi-agent core you would actually reuse in a
real project, with agent-as-tool delegation, tool dispatch, and comparable
request/response trace events for evals.

The first self-evolution probe lives here:

```bash
bash runs/run_self_evolution_probe.sh
```

## 03 Event Runtime (rich)

Graph/event runtime. `GraphRunner` runs named `GraphNode`s through ordered
`GraphEdge`s, records graph/node/edge/model/tool/message/stop events, and
supports live observers, transcript replay, and event-derived run reports.
Best when you want a tiny LangGraph/CrewAI-like shape for orchestration,
observability, replay, eval hooks, and provider-boundary experiments.

## Selection Criteria (draft)

The eventual decision should weigh, at minimum:

- Readability for new contributors (target: a student reads it in one sitting).
- Smallest core that supports the recipes the team actually uses.
- Cost of adding a real LLM provider and tool calling.
- Ability to capture a trace useful for debugging and evals.
- Friction when extending: hooks, custom message kinds, scheduling rules.

Runtime work should now start with `02_balanced_runtime` unless the goal is
specifically to teach the smallest possible loop (`01`) or study graph /
provider / event-boundary behavior (`03`).
