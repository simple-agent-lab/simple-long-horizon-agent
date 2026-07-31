<h1 align="center">Simple Long Horizon Agent</h1>

<p align="center">
  <strong>设计上保持简单，在长程任务中依然有效。</strong>
</p>

<p align="center">
  一个简单但有效、紧凑且易改造的 AI Agent，面向学习、实验和需要多轮推进的真实任务。
</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

<p align="center">
  <a href="https://github.com/simple-agent-lab/simple-long-horizon-agent/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/simple-agent-lab/simple-long-horizon-agent/actions/workflows/ci.yml/badge.svg">
  </a>
  <a href="https://www.python.org/">
    <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&amp;logoColor=white">
  </a>
  <a href="LICENSE">
    <img alt="Apache-2.0 License" src="https://img.shields.io/badge/License-Apache--2.0-blue.svg">
  </a>
</p>

<p align="center">
  <a href="#项目介绍">项目介绍</a> ·
  <a href="#效果">效果</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#文档导航">文档导航</a>
</p>

## 项目介绍

Simple Long Horizon Agent 足够小，可以被完整理解和自由修改；同时也足够有效，能够结合语言模型与工具处理真实任务。你可以从一个实用的 Agent 出发，学习它、实验它，并将它改造成适合自己的样子，而不必先引入一个庞大的框架。

长程任务无法靠单轮回答完成，而是需要跨越多个步骤持续推进：规划、行动、检查结果，并不断迭代，直到真正完成目标。Simple Long Horizon Agent 为这类任务而设计，同时保持项目清晰、易懂。

## 特点

- **设计简单**：项目紧凑、启动路径短，核心概念可以通过阅读项目直接理解。
- **实际有效**：能够使用工具、与真实环境交互，并完成有意义的任务。
- **适配长程任务**：面向需要持续推进、反复调用工具、验证结果和迭代的工作。
- **易于改造**：适合课程教学、研究想法、Benchmark 和专注的团队工作流。
- **便于评测**：可运行示例和 Benchmark 集成让行为与结果都可以被检查。

## 效果

我们聚焦三类长程任务评测：软件工程、终端操作和自主模型后训练。可复现结果将在这里发布，并完整注明模型、Agent 配置和成本。

| Benchmark | 模型 | **分数 ↑** | 同模型基线 | **相对基线提升 ↑** | 单任务成本 |
| --- | --- | ---: | ---: | ---: | ---: |
| [SWE-bench Pro](evals/swebench/README.md)[^swe-chain]<br>Resolved (%) | GPT-5.4 (xHigh) | **63.20%** | [59.10%](https://labs.scale.com/leaderboard/swe_bench_pro_public) | **+6.94%** | $7.7823 |
| [Terminal-Bench](evals/harbor/README.md)<br>2.1 官方分数 | GPT-5.3-Codex (xHigh) | **77.53%** | [64.70%](https://www.tbench.ai/leaderboard/terminal-bench/2.0) | **+19.83%** | $0.5667 |
| [PostTrainBench](https://posttrainbench.com/)[^posttrain]<br>加权平均分 | GPT-5.5 (xHigh) | **45.88%** | [43.97%](https://posttrainbench.com/) | **+4.34%** | — |

[^swe-chain]: 我们采用与 [ChainSWE](https://arxiv.org/abs/2607.02606v1) 类似的链式运行方式，并加入 `task` 工具。

[^posttrain]: 我们在 Qwen3-4B-Base 模型上评估 AIME 2025、BFCL、GSM8K 和 HumanEval 四项任务；归一化奖励采用 [OpenAI PostTrainBench Lite 方法](https://deploymentsafety.openai.com/gpt-5-6-preview/performance-in-cases-flagged-by-users)。

**相对基线提升**表示：在使用相同模型和任务预算时，相比基础方案取得的分数增益。它用于区分 Agent 带来的价值与底层模型本身的能力。

破折号表示“尚未发布”，而不是零分。

## 快速开始

Simple Long Horizon Agent 支持 Python 3.10 及以上版本，并使用 [uv](https://docs.astral.sh/uv/) 管理环境。

```bash
git clone https://github.com/simple-agent-lab/simple-long-horizon-agent.git
cd simple-long-horizon-agent

uv sync
bash runs/demos/run_bash_agent_demo.sh
```

默认 Demo 是确定性的，不需要 API Key。它会展示 Agent 接收任务、使用工具并返回结果的完整过程。

如需使用真实模型运行同一个 Agent：

```bash
export OPENAI_MODEL="your-model"
export OPENAI_AUTH_TOKEN="your-token"
# 使用 OpenAI 兼容服务时可选：
export OPENAI_BASE_URL="https://your-provider.example/v1"

uv run python -m scripts.run_bash_agent_demo \
  --provider openai \
  --task "检查这个仓库并说明它的用途。"
```

更多模型配置请参考 [.env.example](.env.example)。

## 适用场景

- 通过运行和修改一个完整示例，理解 Agent 如何完成任务。
- 为编程、研究、教学或内部工作流构建专注的小型 Agent。
- 探索需要多个步骤、工具调用、结果检查和反复修正的任务。
- 在可重复的任务上比较不同提示词、模型、工具和 Agent 策略。
- 使用小型实验或成熟 Benchmark 评估 Agent 的实际效果。

## 项目原则

- 让 Agent 始终容易理解。
- 优先追求有用的行为，而不是华丽的抽象。
- 让长时间运行的任务可以被观察和验证。
- 让实验可以复现。
- 只在确有收益时增加复杂度。

## 文档导航

- [可运行的 Demo 与实验](runs/README.md)
- [评测套件与 Benchmark](evals/README.md)
- [项目文档](docs/README.md)
- [贡献指南](CONTRIBUTING.md)

## 项目状态

Simple Long Horizon Agent 仍处于早期阶段并在持续演进。目前主要面向学习、研究和小团队实验，而不是生产级基础设施。

## 参与贡献

欢迎任何能让 Agent 更简单、更有效或更容易学习的贡献。提交 Pull Request 前，请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 并运行本地质量检查：

```bash
bash runs/dev/run_ci.sh
```

## 许可证

本项目采用 [Apache License 2.0](LICENSE)。
