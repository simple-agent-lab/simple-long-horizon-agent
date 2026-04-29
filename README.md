# Simple Agent Lab

Simple Agent Lab is a docs-first project for learning, comparing, and building small agent systems.

The project is aimed at college students, small company teams, and agent learners who want something simple, hackable, and understandable before adopting heavier frameworks.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python
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
added.

## Current Status

This repository has a tiny runnable core demo.

The current direction is a message-first multi-agent runtime:

```text
Agent + Message + State + context_view() + run()
```

Reference architectures are still kept as research notes, but the implementation path is intentionally much smaller.
The committed architecture decision is [ADR 0001](docs/decisions/0001-use-tiny-message-runtime.md).

The `Message` shape is deliberately small:

```text
role + content
sender + target + kind + channel + data
```

`role` and `content` are the model-facing fields. The other fields are for
multi-agent routing, context views, and experiment analysis. Use
`model_messages(...)` to convert visible lab messages into a common chat-model
payload.

```python
Message("user", "hello")
```

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
settled, but each architecture proposal should describe its expected feedback
signals.

## Non-Goals

- This is not intended to be a production agent platform at the start.
- This is not a wrapper around every available model or tool provider.
- This is not trying to hide the agent loop behind a large abstraction layer.
- This is not a benchmark project yet.

## Repository Map

- [AGENTS.md](AGENTS.md): collaboration rules for coding agents and contributors.
- [docs/context](docs/context/README.md): product intent, users, and design principles.
- [docs/architecture-options](docs/architecture-options/README.md): earlier multi-agent architecture options and comparison notes.
- [docs/reference-architectures](docs/reference-architectures/README.md): notes on agent architectures we want to learn from.
- [docs/decisions](docs/decisions/README.md): architecture decision records.
- [docs/tasks](docs/tasks/README.md): agent-friendly implementation task specs.
- [docs/glossary.md](docs/glossary.md): shared vocabulary.
- [simple_agent_lab](simple_agent_lab/core.py): tiny message runtime core.
- [examples](examples/README.md): current demo notes, design versions, and future walkthroughs.
- [evals](evals/README.md): future behavior checks and comparisons.
- [src](src/README.md): future implementation landing zone.
- [tests](tests/README.md): future test strategy.
- [runs](runs/README.md): small reproducible commands for examples and future experiments.

## License

License is TBD.

No `LICENSE` file is included yet, so this repository should not be treated as legally open source until a license is chosen.
