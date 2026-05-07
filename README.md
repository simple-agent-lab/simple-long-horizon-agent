# Simple Agent Lab

Simple Agent Lab is a docs-first project for learning, comparing, and building small agent systems.

The project is aimed at college students, small company teams, and agent learners who want something simple, hackable, and understandable before adopting heavier frameworks.

## Setup

This project supports Python 3.10 and newer.

It uses [uv](https://docs.astral.sh/uv/) to manage the Python
environment. The runtime itself is stdlib-only, but `uv run` makes the version
of Python explicit and lets future dependencies land without a global install.

```bash
# install uv (one of):
curl -LsSf https://astral.sh/uv/install.sh | sh
brew install uv

# from the repo root:
uv sync          # creates .venv and installs the package in editable mode
uv run python examples/design_versions/01_functional_loop/demo.py
uv run python examples/design_versions/02_balanced_runtime/demo.py
uv run python examples/design_versions/03_event_runtime/demo.py
```

Plain `python3` works too (no third-party deps yet); `uv run` is the
recommended path so the same command keeps working when dependencies are
added. For direct `python3` smoke runs from this source checkout, use the
repo scripts; they set `PYTHONPATH=src` for the current src-layout:

```bash
bash runs/run_design_versions.sh
bash runs/run_examples.sh
bash runs/run_self_evolution_probe.sh
bash runs/run_training_trace_eval.sh
```

## Current Status

The repository currently hosts **three focused runtime designs** living
side by side under
[`examples/design_versions/`](examples/design_versions/README.md):

| Version | Shape | Current size |
| --- | --- | --- |
| `01_functional_loop` (simple) | One file; direct message list loop with local tool dispatch | ~250 lines |
| `02_balanced_runtime` (moderate) | Promoted into `src`; generator-based runtime, request/response events, tools, agent-as-tool delegation | ~1050 lines in `src/simple_agent_lab/core.py` |
| `03_event_runtime` (complex) | Graph/event runtime: nodes, edges, handoffs, observers, replay, reports, provider boundary | ~650 lines across `core.py`, `models.py`, and `demo.py` |

ADR 0005 made `02_balanced_runtime` the lead core candidate for
self-evolution work. ADR 0009 promotes that design into
`src/simple_agent_lab/core.py`. `01_functional_loop` remains the smallest
teaching baseline, and `03_event_runtime` remains the richer graph
orchestration / observability / provider-boundary reference until the team is
ready to prune.

The shared message protocol across all three is deliberately small and
role-specific:

```text
Message =
  UserMessage
  | SystemMessage
  | AssistantMessage
  | ToolResultMessage

ModelMessage =
  ModelUserMessage
  | ModelSystemMessage
  | ModelAssistantMessage
  | ModelToolResultMessage
```

Runtime `Message` values keep `sender`, `target`, `kind`, `channel`, and rare
sidecar `data`. Provider-boundary `ModelMessage` values remove runtime routing
fields and keep structured content blocks for text, images, thinking, and tool
calls.

The committed architectural direction (a small, message-first runtime
rather than a heavy framework) is recorded in
[ADR 0001](docs/decisions/0001-use-tiny-message-runtime.md). The
self-evolution harness direction is recorded in
[ADR 0004](docs/decisions/0004-treat-self-evolution-as-harness-capability.md),
the lead runtime direction is recorded in
[ADR 0005](docs/decisions/0005-make-balanced-runtime-the-lead-core-candidate.md),
and the promotion into `src` is recorded in
[ADR 0009](docs/decisions/0009-promote-balanced-runtime-to-src-core.md).

## Project Goals

- Make agent architecture easy to read and modify.
- Keep the first implementation small enough to explain in one sitting.
- Prefer explicit data flow over hidden framework behavior.
- Support experiments, classroom use, and small internal team prototypes.
- Build shared context that both humans and coding agents can follow.

## Development Process

Development follows the
[harness engineering workflow](docs/context/harness-engineering.md): keep the
repo itself as the source of truth, make changes small and verifiable, and
improve docs, examples, scripts, or tests when an agent workflow is ambiguous.

The concrete test suite is intentionally deferred until the core architecture is
settled, but the repo now has a deterministic trajectory -> eval -> training
example pipeline for the three design-version demos. That training-data
direction is recorded in
[ADR 0008](docs/decisions/0008-collect-training-trajectories-across-design-versions.md):

```bash
bash runs/run_training_trace_eval.sh
```

## Non-Goals

- This is not intended to be a production agent platform at the start.
- This is not a wrapper around every available model or tool provider.
- This is not trying to hide the agent loop behind a large abstraction layer.
- This is not a benchmark project yet.

## Repository Map

- [AGENTS.md](AGENTS.md): collaboration rules for coding agents and contributors.
- [docs/context](docs/context/README.md): product intent, users, and design principles.
- [docs/architecture-options](docs/architecture-options/README.md): architecture documentation for the three focused runtime candidates (mirrors `examples/design_versions/`).
- [docs/reference-architectures](docs/reference-architectures/README.md): notes on agent architectures we want to learn from.
- [docs/decisions](docs/decisions/README.md): architecture decision records.
- [docs/tasks](docs/tasks/README.md): agent-friendly implementation task specs.
- [docs/glossary.md](docs/glossary.md): shared vocabulary.
- [examples/design_versions](examples/design_versions/README.md): the three runtime designs (simple / moderate / complex), with `02_balanced_runtime` as the lead candidate.
- [examples](examples/README.md): demo notes and earlier sketches.
- [src/simple_agent_lab](src/simple_agent_lab/core.py): the installable package - `core.py` (canonical balanced runtime), `messages.py` (shared message protocol), `tools.py` (shared tool values), `trajectory.py` / `evaluation.py` / `training_data.py` (runtime-neutral harness records), and `llm/` (shared LLM access layer and message bridge used by all design versions).
- [evals](evals/README.md): future behavior checks and comparisons.
- [tests](tests/README.md): future test strategy.
- [runs](runs/README.md): small reproducible commands for examples and future experiments.

## License

License is TBD.

No `LICENSE` file is included yet, so this repository should not be treated as legally open source until a license is chosen.
