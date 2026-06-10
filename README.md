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
  | SystemMessage
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

## OpenClaw Benchmark Handoff

The `dev/claw` branch carries the current OpenClaw benchmark integration work.
It adds suite adapters for:

- `clawbench_tribe`
- `pinchbench`
- `clawbench_official`
- `skillsbench`
- `agentbench`
- `claweval`
- `zclawbench`

The main entry point is [run_benches.py](run_benches.py). The host-side suite
adapters live under [evals](evals/README.md); the in-run/container halves live
under `src/simple_agent_lab/evals/suites/`.

### Large Benchmark Data

This branch uses Git LFS for large benchmark fixtures. Run this before pulling
or checking out the branch:

```bash
git lfs install
git fetch origin
git switch dev/claw
git pull --ff-only
git lfs pull
```

The history rewrite that introduced LFS moved the large files from normal Git
blobs into LFS pointers. The rewritten OpenClaw adapter commit is `cf71adc`
(`Add OpenClaw benchmark adapters`), and current `origin/dev/claw` is
`657b64e`. Files above 5 MB are now LFS-managed; the branch still contains many
small benchmark files, so checkout can still take longer than a normal source
branch.

### ModelHub / GPT-5.5 High Runs

For ModelHub Responses API runs, create a repo-local `.env` file. It is
gitignored and should not be committed.

```plaintext
OPENAI_MODEL=gpt-5.5-2026-04-24
OPENAI_AUTH_TOKEN=<your modelhub AK>
OPENAI_BASE_URL=https://aidp.bytedance.net/api/modelhub/online/responses/openai/responses
OPENAI_SESSION_ID=sal-openclaw-gpt55high
OPENAI_LOG_ID=sal-openclaw-gpt55high
API_KIND=openai-responses
OPENAI_REASONING_EFFORT=high
```

`run_benches.py` also accepts these fields as CLI flags:

```bash
.venv/bin/python run_benches.py \
  --bench clawbench_official \
  --model gpt-5.5-2026-04-24 \
  --api-kind openai-responses \
  --reasoning-effort high \
  --session-id sal-openclaw-gpt55high \
  --log-id sal-openclaw-gpt55high-clawbench_official \
  --backend process \
  --sample 20 \
  --sample-strategy spread \
  --seed 0 \
  --max-turns 10 \
  --pass-threshold 60 \
  --run-root .tmp/openclaw_gpt55high_20_spread_m10/clawbench_official
```

To run all currently wired benches with the same 20-task spread subset:

```bash
for bench in clawbench_tribe pinchbench clawbench_official skillsbench agentbench claweval zclawbench; do
  .venv/bin/python run_benches.py \
    --bench "$bench" \
    --model gpt-5.5-2026-04-24 \
    --api-kind openai-responses \
    --reasoning-effort high \
    --session-id sal-openclaw-gpt55high \
    --log-id "sal-openclaw-gpt55high-$bench" \
    --backend process \
    --sample 20 \
    --sample-strategy spread \
    --seed 0 \
    --max-turns 10 \
    --pass-threshold 60 \
    --run-root ".tmp/openclaw_gpt55high_20_spread_m10/$bench"
done
```

Use `.venv/bin/python`, not macOS `/usr/bin/python3`; the latter is often
Python 3.9 and cannot import this codebase.

### Current GPT-5.5 High Snapshot

The latest local run used:

- model: `gpt-5.5-2026-04-24`
- served model observed in traces: `deployment-gpt-5.5-2026-04-24-platform-global`
- API kind: `openai-responses`
- reasoning effort: `high`
- backend: `process`
- sample strategy: `spread`
- max turns: `10`
- result root: `.tmp/openclaw_gpt55high_20_spread_m10/`

| Bench | Tasks | Passed | Mean | Zero | Nonzero | Nonzero Mean | Errors | Pending |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `clawbench_tribe` | 8 | 8 | 100.0 | 0 | 8 | 100.0 | 0 | 0 |
| `clawbench_official` | 20 | 9 | 54.845 | 1 | 19 | 57.73 | 1 | 0 |
| `agentbench` | 20 | 9 | 40.0 | 11 | 9 | 88.89 | 5 | 0 |
| `pinchbench` | 20 | 4 | 21.11 | 15 | 5 | 84.44 | 1 | 7 |
| `skillsbench` | 20 | 2 | 12.855 | 16 | 4 | 64.27 | 3 | 0 |
| `claweval` | 20 | 0 | 0.0 | 20 | 0 | - | 3 | 17 |
| `zclawbench` | 20 | 0 | 0.0 | 20 | 0 | - | 5 | 15 |

Interpretation:

- `clawbench_official` is the cleanest deterministic signal in this snapshot.
  It improved to `54.845` mean on the same 20-task spread subset.
- `pinchbench`, `claweval`, and `zclawbench` still need post-hoc judging for
  pending tasks before their final means are meaningful.
- `skillsbench` remains mostly blocked by tool/dependency/environment demands,
  even with GPT-5.5 high.
- Several errors are ModelHub API timeouts. Treat them separately from verifier
  failures when analyzing results.

Post-hoc judge entry point:

```bash
.venv/bin/python scripts/judge_openclaw_pending.py \
  --bench pinchbench \
  --run-root .tmp/openclaw_gpt55high_20_spread_m10/pinchbench \
  --model gpt-5.5-2026-04-24
```

The judge currently supports `pinchbench`, `claweval`, and `zclawbench`.
`zclawbench` still uses a generic completion judge because the current adapter
only has task ids/categories, not official prompts/rubrics.

Detailed Chinese reports are under [docs/agent-native](docs/agent-native/README.md):

- [OpenClaw benchmark experiment report](docs/agent-native/openclaw-benchmark-experiment-report.zh.md)
- [OpenClaw 20-task subset result notes](docs/agent-native/openclaw-20-subset-results.zh.md)
- [OpenClaw harness experiment plan](docs/agent-native/openclaw-harness-experiments.md)

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
- The base project is not a benchmark leaderboard; `dev/claw` uses benchmarks as
  an integration and harness-diagnosis workload.

## Repository Map

- [AGENTS.md](AGENTS.md): collaboration rules for coding agents and contributors.
- [docs/agent-native](docs/agent-native/README.md): the single future-agent loading map, plus project intent, code style, development workflow, source-of-truth routing, and unresolved owner questions.
- [docs/decisions](docs/decisions/README.md): architecture decision records.
- [docs/glossary.md](docs/glossary.md): shared vocabulary.
- [src/simple_agent_lab](src/simple_agent_lab/core.py): the installable package - `core.py` (canonical tiny run loop), `trace.py` (trace printing and OpenAI Chat JSONL export), `context_view.py` (model-visible context projection), `compression.py` (context-compression strategies run before each model request), `messages.py` (shared message protocol), `tools/` (shared tool values plus concrete tools like bash and the sub-agent `task` tool), `trajectory/` (runtime-neutral trace records split into span/training transforms, record schema, JSONL IO, and the live-trace edge), `llm/` (shared LLM access layer and message bridge), `mcp/` (optional Model Context Protocol integration: wrap an MCP server's tools — including multimodal results — as `AgentTool`s, behind the `mcp` extra), `agents/` (preset agents built on the core layers: `bash/` and the multi-agent `bash_task/`), and `evals/` (the generic containerized eval framework - two seams `ContainerBackend` x `ArtifactStore`, the `run_suite_instance` entry point, and in-wheel suite container halves under `src/simple_agent_lab/evals/suites/`; see [ADR 0017](docs/decisions/0017-generic-containerized-eval-framework.md)).
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
