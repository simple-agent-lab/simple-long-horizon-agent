# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the runtime is in `0.x`, public APIs may change between minor versions.

## [Unreleased]

## [0.1.0] - 2026-05-11

First public release. Promotes the canonical balanced runtime, ships a
minimal bash-use agent, adds a SWE-bench eval adapter, and wires up a
project-wide CI gate.

### Added
- `simple_agent_lab.core` — canonical, message-first agent runtime promoted
  from the prior balanced-runtime sketch (ADR 0009). Generator-driven event
  loop, request/response events, tools, and agent-as-tool delegation.
- `simple_agent_lab.bash_tool` and `simple_agent_lab.bash_agent` — minimal
  bash-use agent demo, including command interpretation, observation
  formatting, and a `until_final` driver.
- `simple_agent_lab.context_view` — explicit projection of the message
  history that an agent's model actually sees, with token estimation and a
  configurable clipping policy (ADR 0010).
- `simple_agent_lab.messages` — role-specific message protocol
  (`UserMessage` / `SystemMessage` / `AssistantMessage` / `ToolResultMessage`)
  and a parallel `ModelMessage` family for the provider boundary
  (ADR 0006). Includes `TokenUsage` records.
- `simple_agent_lab.llm` — shared LLM access layer and a `bridge` between
  runtime messages and provider-shaped messages, with a deterministic
  `FakeAdapter` for tests.
- `simple_agent_lab.trajectory`, `evaluation`, `training_data` —
  runtime-neutral records for trajectory capture, eval, and training-data
  export.
- `evals/swebench/` — SWE-bench eval adapter that drives the bash agent
  against benchmark instances (ADR 0011), with workspace prep, trajectory
  collection, and prediction evaluation entry points.
- `runs/run_ci.sh` — canonical local pre-push gate (ty + unittest).
- `runs/run_examples.sh`, `runs/run_bash_agent_demo.sh`,
  `runs/run_swebench_smoke.sh`, `runs/run_swebench_gold_smoke.sh` — small
  reproducible commands.
- `.github/workflows/ci.yml` — GitHub Actions workflow running ty + unittest
  on Python 3.10 and 3.13.
- Apache-2.0 `LICENSE`, `CONTRIBUTING.md`, this `CHANGELOG.md`, and GitHub
  issue / pull request templates under `.github/`.
- ADR 0010 (context view as explicit projection) and ADR 0011 (benchmark
  suites as eval adapters).
- Reference-architecture notes for `mini-swe-agent` and context-management
  pipelines that informed the design.
- Project metadata in `pyproject.toml`: version, license, authors,
  classifiers, keywords, and `[project.urls]` pointing at the GitHub repo.

### Removed
- `examples/design_versions/` (the side-by-side `01_functional_loop` /
  `02_balanced_runtime` / `03_event_runtime` sketches), now folded into the
  canonical runtime (ADRs 0005 and 0009). The historical architecture notes
  remain under `docs/architecture-options/` for reference.
- `evals/evaluate_design_version_traces.py`,
  `scripts/collect_design_version_trajectories.py`,
  `scripts/export_training_examples.py`, and the `runs/run_design_versions.sh`
  / `runs/run_self_evolution_probe.sh` / `runs/run_training_trace_eval.sh`
  scripts that drove the retired pipeline. A replacement targeting the
  canonical runtime is not yet wired up.

[Unreleased]: https://github.com/simple-agent-lab/simple-agent-lab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/simple-agent-lab/simple-agent-lab/releases/tag/v0.1.0
