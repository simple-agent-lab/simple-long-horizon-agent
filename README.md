# Simple Agent Lab

Simple Agent Lab is a docs-first project for learning, comparing, and building small agent systems.

The project is aimed at college students, small company teams, and agent learners who want something simple, hackable, and understandable before adopting heavier frameworks.

## Setup

This project supports Python 3.10 and newer.

It uses [uv](https://docs.astral.sh/uv/) to manage the Python
environment. The base package installs the supported model-provider SDKs; the
heavier benchmark tooling stays behind optional extras.

```bash
# install uv (one of):
curl -LsSf https://astral.sh/uv/install.sh | sh
brew install uv

# from the repo root:
uv sync --group dev   # creates .venv, installs package + dev tools (ruff, ty)
uv run python -m unittest discover -s tests/unit
uv run ruff format --check .
uv run python scripts/lint_docs.py
uv run ty check src
```

Plain `python3` works too when it is Python 3.10 or newer and dependencies are
installed. `uv run` is the recommended path so the same command keeps working
when optional extras are added. The repo smoke scripts prefer `uv run python`
when `uv` is installed and fall back to `python3`; they also set `PYTHONPATH=src`
for the current src-layout:

```bash
bash runs/run_bash_agent_demo.sh
```

The same checks (`ruff format --check .`, docs lint, `ty check src`, and the
unittest suite on Python 3.10 through 3.13) run on every push and pull request
via [GitHub Actions](.github/workflows/ci.yml).

Optional benchmark suites have their own setup notes under
[evals](evals/README.md) so the root setup stays focused on normal
development.

## Current Status

The canonical runtime lives in `src/simple_agent_lab/core.py`. Earlier
side-by-side architectural sketches under `examples/design_versions/` have
been folded into the package and removed.

The shared message protocol is deliberately small and role-specific:

```text
Message =
  UserMessage
  | RuntimeMessage
  | AssistantMessage

ContentBlock =
  TextBlock
  | ImageBlock
  | ThinkingBlock
  | ToolCallBlock
  | ToolResultBlock
```

Runtime `Message` values keep `sender`, `target`, `kind`, `channel`, and rare
sidecar `data`. The LLM bridge projects them into routing-free `LLMMessage`
values that preserve ordered content blocks for text, images, thinking, tool
calls, and tool results.

The architecture decision history lives in
[docs/decisions](docs/decisions/README.md); keep detailed rationale there
instead of expanding the public README.

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
Concrete day-to-day commands and the quality gate (ruff format, ty, and
unittest, run by `runs/run_ci.sh` locally and `.github/workflows/ci.yml`
remotely) are spelled out in
[docs/agent-native/development.md](docs/agent-native/development.md).

Training-data and eval architecture notes also live under
[docs/decisions](docs/decisions/README.md). The original design-version
pipeline has been retired alongside `examples/design_versions/`. The current
eval architecture is the generic containerized framework in
`src/simple_agent_lab/evals/` ([ADR 0017](docs/decisions/0017-generic-containerized-eval-framework.md)):
a suite is one host-side `Suite` plus a container half of two functions, run
through `run_suite_instance` over swappable `ContainerBackend` (in-process /
local Docker / remote) and `ArtifactStore` (local dir / host HTTP / S3) seams.

## Non-Goals

- This is not intended to be a production agent platform at the start.
- This is not a wrapper around every available model or tool provider.
- This is not trying to hide the agent loop behind a large abstraction layer.
- This is not a benchmark project yet.

## Repository Map

- [AGENTS.md](AGENTS.md): collaboration rules for coding agents and contributors.
- [docs/agent-native](docs/agent-native/README.md): the single future-agent loading map, plus project intent, code style, development workflow, source-of-truth routing, and unresolved owner questions.
- [docs/decisions](docs/decisions/README.md): architecture decision records.
- [docs/glossary.md](docs/glossary.md): shared vocabulary.
- [src/simple_agent_lab](src/simple_agent_lab/core.py): the installable package - `core.py` (canonical tiny run loop), `trace.py` (trace printing and OpenAI Chat JSONL export), `context_view.py` (model-visible context projection), `compression.py` (context-compression strategies run before each model request), `messages.py` (shared message protocol), `memory/` (small memory extension boundary plus notes-memory and filesystem-memory implementations, each kept in one implementation file), `tools/` (shared tool values plus concrete tools like bash and the sub-agent `task` tool), `trajectory/` (runtime-neutral trace records split into span/training transforms, record schema, JSONL IO, and the live-trace edge), `llm/` (shared LLM access layer and message bridge), `mcp/` (optional Model Context Protocol integration: wrap an MCP server's tools — including multimodal results — as `AgentTool`s, behind the `mcp` extra), `agents/` (preset agents built on the core layers: `bash/` and the multi-agent `bash_task/`), and `evals/` (the generic containerized eval framework - two seams `ContainerBackend` x `ArtifactStore`, the `run_suite_instance` entry point, and in-wheel suite container halves under `src/simple_agent_lab/evals/suites/`; see [ADR 0017](docs/decisions/0017-generic-containerized-eval-framework.md)).
- [evals](evals/README.md): suite adapters (host halves) and the "add a suite" guide; the framework itself ships in the package above.
- [examples/bench_suite](examples/bench_suite/README.md): agent-as-judge worked example - candidate + judge runs composed over the shared artifact store, runnable with no Docker.
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
