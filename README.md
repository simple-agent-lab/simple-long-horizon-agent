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
uv sync --group dev   # creates .venv, installs the package + dev tools (ty)
uv run python -m unittest discover -s tests
uv run ty check src
```

Plain `python3` works too when it is Python 3.10 or newer (no third-party deps
yet). `uv run` is the recommended path so the same command keeps working when
dependencies are added. The repo smoke scripts prefer `uv run python` when `uv`
is installed and fall back to `python3`; they also set `PYTHONPATH=src` for the
current src-layout:

```bash
bash runs/run_examples.sh
bash runs/run_bash_agent_demo.sh
```

The same checks (`ty check src` + the unittest suite on Python 3.10 and 3.13)
run on every push and pull request via [GitHub Actions](.github/workflows/ci.yml).

## Current Status

The canonical runtime lives in `src/simple_agent_lab/core.py` — promoted from
the original balanced-runtime sketch per ADR 0009. The earlier
side-by-side architectural sketches under `examples/design_versions/` have been
folded into the package and removed; ADRs 0005 and 0009 record that decision.

The shared message protocol is deliberately small and role-specific:

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
[harness engineering workflow](docs/agent-native/harness-engineering.md): keep the
repo itself as the source of truth, make changes small and verifiable, and
improve docs, examples, scripts, or tests when an agent workflow is ambiguous.
Concrete day-to-day commands and the quality gate (ty + unittest, run by
`runs/run_ci.sh` locally and `.github/workflows/ci.yml` remotely) are spelled
out in [docs/agent-native/development.md](docs/agent-native/development.md).

The training-data direction (deterministic trajectory → eval → training
example pipeline) is recorded in
[ADR 0008](docs/decisions/0008-collect-training-trajectories-across-design-versions.md).
The original design-version pipeline that backed it has been retired alongside
`examples/design_versions/`; a replacement targeting the canonical runtime is
not yet wired up.

## Non-Goals

- This is not intended to be a production agent platform at the start.
- This is not a wrapper around every available model or tool provider.
- This is not trying to hide the agent loop behind a large abstraction layer.
- This is not a benchmark project yet.

## Repository Map

- [AGENTS.md](AGENTS.md): collaboration rules for coding agents and contributors.
- [docs/agent-native](docs/agent-native/README.md): the single future-agent loading map, plus project intent, code style, development workflow, doc inventory, source-of-truth routing, and unresolved owner questions.
- [docs/reference-architectures](docs/reference-architectures/README.md): notes on agent architectures we want to learn from.
- [docs/decisions](docs/decisions/README.md): architecture decision records.
- [docs/glossary.md](docs/glossary.md): shared vocabulary.
- [src/simple_agent_lab](src/simple_agent_lab/core.py): the installable package - `core.py` (canonical balanced runtime), `bash_tool.py` / `bash_agent.py` (minimal bash-use agent demo), `context_view.py` (model-visible context projection), `messages.py` (shared message protocol), `tools.py` (shared tool values), `trajectory.py` / `evaluation.py` / `training_data.py` (runtime-neutral harness records), and `llm/` (shared LLM access layer and message bridge).
- [evals](evals/README.md): future behavior checks and comparisons.
- [tests](tests/README.md): future test strategy.
- [runs](runs/README.md): small reproducible commands for examples and future experiments.

## Contributing

Contributions are welcome. Start with [AGENTS.md](AGENTS.md) for the
collaboration contract and [docs/agent-native/development.md](docs/agent-native/development.md)
for the local quality gate. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
short version.

## License

Licensed under the [Apache License, Version 2.0](LICENSE). By contributing
to this project you agree that your contributions will be licensed under the
same terms (see [CONTRIBUTING.md](CONTRIBUTING.md) for details).
