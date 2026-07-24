<h1 align="center">Simple Agent Lab</h1>

<p align="center">
  <strong>Simple by design. Effective over long horizons.</strong>
</p>

<p align="center">
  A simple yet effective AI agent for learning, experimentation, and real work
  that takes more than one model turn.
</p>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://github.com/simple-agent-lab/simple-agent-lab/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/simple-agent-lab/simple-agent-lab/actions/workflows/ci.yml/badge.svg">
  </a>
  <a href="https://www.python.org/">
    <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&amp;logoColor=white">
  </a>
  <a href="LICENSE">
    <img alt="Apache-2.0 License" src="https://img.shields.io/badge/License-Apache--2.0-blue.svg">
  </a>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#results">Results</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#documentation">Documentation</a>
</p>

## Overview

Simple Agent Lab is small enough to understand and change, while capable
enough to take on real tasks with language models and tools. It gives you a
practical agent to learn from, experiment with, and adapt—without requiring a
large framework first.

Long-horizon work cannot be completed in a single response. It requires
sustained progress across many steps: planning, acting, checking results, and
iterating until the goal is complete. Simple Agent Lab is designed for that
shape of work while keeping the project clear and approachable.

## Highlights

- **Simple by design** — a compact project with a short setup path and concepts
  you can understand by inspection.
- **Effective in practice** — built to use tools, interact with real
  environments, and complete meaningful tasks.
- **Long-horizon ready** — designed for sustained progress, repeated tool use,
  verification, and iteration.
- **Easy to adapt** — a practical base for classes, research ideas, benchmarks,
  and focused team workflows.
- **Built to be evaluated** — runnable examples and benchmark integrations make
  behavior and results inspectable.

## Results

We focus on long-horizon evaluations across software engineering, terminal
work, and autonomous model post-training. Reproducible results will be
published here with the exact model, agent setup, and cost.

| Benchmark | Model | **Score ↑** | Baseline | **Δ vs. Baseline ↑** | Cost / Task |
| --- | --- | ---: | ---: | ---: | ---: |
| [SWE-bench Pro](evals/swebench/README.md)[^swe-chain]<br>Resolved (%) | GPT-5.4 (xHigh) | **63.20%** | [59.10%](https://labs.scale.com/leaderboard/swe_bench_pro_public) | **+6.94%** | $7.7823 |
| [Terminal-Bench](evals/harbor/README.md)<br>2.1 official score | GPT-5.3-Codex (xHigh) | **77.53%** | [64.70%](https://www.tbench.ai/leaderboard/terminal-bench/2.0) | **+19.83%** | $0.5667 |
| [PostTrainBench](https://posttrainbench.com/)[^posttrain]<br>Weighted average | GPT-5.5 (xHigh) | **45.88%** | [43.97%](https://posttrainbench.com/) | **+4.34%** | — |

[^swe-chain]: We use a chained workflow similar to [ChainSWE](https://arxiv.org/abs/2607.02606v1), augmented with the `task` tool.

[^posttrain]: We evaluate Qwen3-4B-Base on AIME 2025, BFCL, GSM8K, and HumanEval. Normalized rewards follow the [OpenAI PostTrainBench Lite method](https://deploymentsafety.openai.com/gpt-5-6-preview/performance-in-cases-flagged-by-users).

**Δ vs. Baseline** is the score improvement over a baseline using the same
model and task budget. It separates the value of the agent from the capability
of the underlying model.

An em dash means “not published yet,” not zero.

## Quick Start

Simple Agent Lab supports Python 3.10 and newer and uses
[uv](https://docs.astral.sh/uv/) for its environment.

```bash
git clone https://github.com/simple-agent-lab/simple-agent-lab.git
cd simple-agent-lab

uv sync
bash runs/demos/run_bash_agent_demo.sh
```

The default demo is deterministic and does not require an API key. It shows the
agent receiving a task, using a tool, and returning a result.

To try the same agent with a real model:

```bash
export OPENAI_MODEL="your-model"
export OPENAI_AUTH_TOKEN="your-token"
# Optional for an OpenAI-compatible endpoint:
export OPENAI_BASE_URL="https://your-provider.example/v1"

uv run python -m scripts.run_bash_agent_demo \
  --provider openai \
  --task "Inspect this repository and explain what it is for."
```

See [.env.example](.env.example) for the supported provider settings.

## Use Cases

- Learn how an agent behaves by running and modifying a complete example.
- Build focused agents for coding, research, teaching, or internal workflows.
- Explore tasks that require many steps, tool calls, checks, and revisions.
- Compare prompts, models, tools, and agent strategies under repeatable tasks.
- Evaluate agent behavior on small experiments or established benchmarks.

## Project Principles

- Keep the agent easy to understand.
- Prefer useful behavior over impressive abstraction.
- Make long-running work observable and verifiable.
- Make experiments reproducible.
- Add complexity only when it earns its place.

## Documentation

- [Runnable demos and experiments](runs/README.md)
- [Evaluation suites and benchmarks](evals/README.md)
- [Project documentation](docs/README.md)
- [Contributing guide](CONTRIBUTING.md)

## Project Status

Simple Agent Lab is early-stage and actively evolving. It is designed for
learning, research, and small-team experimentation rather than production-scale
infrastructure.

## Contributing

Contributions that make the agent simpler, more effective, or easier to learn
from are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) and run the local
quality gate before opening a pull request:

```bash
bash runs/dev/run_ci.sh
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
