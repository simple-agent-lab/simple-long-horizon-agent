# <text color="blue">Simple Agent Lab × OpenClaw/ClawBench 集成实验复盘</text>

<callout emoji="🎯" background-color="light-blue" border-color="light-blue">
<text color="blue">**结论先行：老板给的“集成 ClawBench”目标已经完成。**</text>
<text color="gray">我在 simple-agent-lab 现有 Bash Agent 和 tiny runtime 基础上，新增了 evals/openclaw 适配层，打通了 PinchBench、AgentBench 初版和 ClawBench Official。当前已经能加载 ClawBench 303 个任务，用 SAL Bash Agent 在隔离 workspace 中执行，并调用 ClawBench pytest verifier 给出可复现分数。</text>
</callout>

<grid cols="3">
  <column width="33">
    <text color="blue">**目标完成度**</text>
    <text color="gray">ClawBench Official 已接入；303 个任务可加载；单任务 smoke 和样本评测均可跑通。</text>
  </column>
  <column width="33">
    <text color="blue">**初步结果**</text>
    <text color="gray">PinchBench 全量 23 题端到端无 crash，按日志汇总总分 60.0%；ClawBench 已跑 35/303，平均 78.9%。</text>
  </column>
  <column width="33">
    <text color="blue">**核心判断**</text>
    <text color="gray">SAL 的最小运行时适合做实验骨架，但要成为更强 benchmark harness，还缺环境隔离、artifact 管理、provider 标准化和失败诊断能力。</text>
  </column>
</grid>

---

# <text color="blue">一、我对 Simple Agent Lab 框架的理解</text>

## 1.1 这个项目的定位

Simple Agent Lab 不是一个“大而全”的生产级 Agent 平台，而是一个偏教学、偏实验、偏可解释的 agent runtime。它强调：

- **代码足够小**：核心 loop 在 `src/simple_agent_lab/core.py`，读完以后能理解 agent 是怎么跑起来的。
- **数据流显式**：Message、ContentBlock、State、Event、Tool 都是显式对象，不把行为藏在复杂框架里。
- **面向实验**：evals、trace、context_view、trajectory 这些模块说明项目更关心“如何比较 agent 行为”和“如何沉淀训练/评测轨迹”。
- **文档优先**：docs/decisions 和 docs/agent-native 记录了很多架构决策，说明这个项目希望人和 coding agent 都能理解上下文。

我的理解是：SAL 的核心价值不是“替代 LangChain/AutoGen”，而是提供一个足够干净的实验台，让我们可以控制 agent loop、工具调用、上下文投影、轨迹记录和 benchmark 适配。

<quote-container>
**SAL 的设计重心是：把 agent 行为拆成可观察、可替换、可评分的最小部件。**
</quote-container>

## 1.2 核心运行时怎么工作

SAL 的核心模型可以概括为：

```text
Agent + Message + State + ContextView + Tool + Event
```

运行过程是：

1. 用户任务被写入 `State`，成为一条 `task` message。
2. `run()` 每一轮构造 model-visible 的 context view。
3. 调用 `agent.generate(visible_messages)` 得到 assistant message。
4. 如果 assistant message 包含 tool call，就由 runtime dispatch 到绑定工具。
5. 工具结果再写回 state，进入下一轮。
6. 当 assistant 输出 `kind="final"`，或者达到 `max_turns`，run loop 结束。

这个 loop 的好处是非常清楚：agent 本身只负责“看上下文，生成下一条消息”；runtime 负责“记录事件、执行工具、维护状态、截断和结束”。这让 eval 适配层可以比较容易地挂在 runtime 外面。

## 1.3 Message / ContentBlock 是 SAL 的协议核心

SAL 的共享消息协议比较小：

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

这套协议有两个重要意义：

- 对内，它让 runtime 可以统一处理文本、图片、思考、工具调用和工具结果。
- 对外，它让不同 LLM provider 的返回可以被桥接成同一种 assistant message。

这次集成 ClawEvalkit/ClawBench，本质上就是把 SAL 的 `Event + Message + ContentBlock` 转成 ClawEvalkit 能评分的 transcript，把 SAL workspace 的产物交给 ClawBench verifier。

## 1.4 Bash Agent 的角色

这次实验主要复用的是 `src/simple_agent_lab/agents/bash/agent.py` 里的 Bash Agent。它是一个最小可用的 workspace agent：

- LLM 根据任务决定是否调用 bash。
- bash tool 在指定 `cwd/workspace` 下执行命令。
- 工具结果回到上下文。
- agent 最后总结或完成产物生成。

对于 ClawBench 这种“在 workspace 里读输入、生成输出、再用 verifier 检查”的任务，Bash Agent 是合适的第一版执行引擎，因为它天然具备文件操作、脚本执行和结果检查能力。

---

# <text color="blue">二、我的实现：把 SAL 接到 OpenClaw/ClawBench</text>

## 2.1 实现原则

我这次没有侵入 SAL 的核心 runtime，而是在 `evals/openclaw/` 下做 benchmark adapter。这样做的原因是：

- SAL 上游核心保持干净，不把某个 benchmark 的特殊逻辑塞进 runtime。
- benchmark 相关代码集中在 eval adapter，后续可以继续加 SkillsBench、ClawEval、ZClawBench。
- 适配层可以独立处理 ClawEvalkit 的目录结构、模型配置、评分函数和输出格式。

<callout emoji="💡" background-color="light-green" border-color="light-green">
<text color="green">**关键设计：换 agent 不换评测框架。**</text>
<text color="gray">ClawEvalkit 原本更依赖 OpenClawPro/NanoBotAgent。我这次让 SAL Bash Agent 也能进入同一套任务加载、执行和评分流程，从而验证 SAL agent 的通用执行能力。</text>
</callout>

## 2.2 新增模块结构

<lark-table>
  <lark-tr>
    <lark-td>**文件**</lark-td>
    <lark-td>**作用**</lark-td>
    <lark-td>**关键内容**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`evals/openclaw/config.py`</lark-td>
    <lark-td>模型配置适配</lark-td>
    <lark-td>把 ClawEvalkit provider 配置映射到 SAL Provider；支持 openrouter、glm、anthropic、azure_openai 等。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`evals/openclaw/adapter.py`</lark-td>
    <lark-td>执行和格式桥接</lark-td>
    <lark-td>负责 SAL event 到 ClawEvalkit transcript 的转换，以及 PinchBench/AgentBench/ClawBench 单任务执行。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`evals/openclaw/runner.py`</lark-td>
    <lark-td>benchmark runner</lark-td>
    <lark-td>负责加载任务、缓存结果、并行执行、汇总分数和 CLI 参数。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`evals/openclaw/test_integration.py`</lark-td>
    <lark-td>冒烟测试</lark-td>
    <lark-td>验证 transcript 转换、PinchBench pipeline、ClawEvalkit 任务加载、ClawBench verifier。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`docs/exp1_openclaw/`</lark-td>
    <lark-td>实验沉淀</lark-td>
    <lark-td>包含 README、dashboard、代码可视化页面、运行日志和结果分析。</lark-td>
  </lark-tr>
</lark-table>

## 2.3 SAL Event 到 ClawEvalkit Transcript

ClawEvalkit 的 PinchBench 评分函数期待 transcript 是一组 dict，例如：

```python
{"type": "message", "message": {"role": "assistant", "content": [...]}}
{"type": "tool_call", "name": "bash", "params": {...}}
{"type": "tool_result", "result": "..."}
```

但 SAL runtime 产生的是 dataclass event，例如 message event、tool execution start/end event。我的做法是在 `sal_events_to_claweval_transcript()` 中做显式转换：

- `message` event 转成 ClawEvalkit message。
- `tool_execution_start` 转成 tool_call。
- `tool_execution_end` 转成 tool_result。
- `TextBlock`、`ToolCallBlock`、`ToolResultBlock`、`ThinkingBlock`、`ImageBlock` 分别映射成 ClawEvalkit 可读的 content dict。

这一步是整个集成的协议桥。它跑通后，SAL agent 的轨迹才能被 ClawEvalkit 的评分代码消费。

## 2.4 AzureOpenAI / ModelHub 支持

真实评测使用的是 `gpt-4.1-2025-04-14`，通过字节内部 ModelHub GPT 代理访问。SAL 原生 provider 没有直接支持 AzureOpenAI，因此我在适配层里加了 `_build_azure_agent()`：

- 使用 `AzureOpenAI` 客户端调用 ModelHub endpoint。
- 复用 SAL 的 message bridge，把 SAL visible messages 转成 OpenAI Chat messages。
- 复用 SAL 的 tool schema bridge，把 bash tool 转成 OpenAI tool definition。
- 将 AzureOpenAI 返回的 `tool_calls` 重新封装为 SAL `ToolCallBlock`。
- 清空 `http_proxy/https_proxy`，避免内部 endpoint 403。

这里踩过几个关键点：

- SAL Provider 参数是 `id + api + model + base_url + api_key_env`，不是 ClawEvalkit 的 provider 字段。
- AzureOpenAI 返回的 `tc.function.arguments` 是 JSON 字符串，需要 `json.loads()`。
- `llm_response_to_assistant_message` 位于 `simple_agent_lab.llm.bridge`。
- tool tuple 需要在闭包定义前创建，否则 Python 作用域会出问题。

## 2.5 ClawBench Official 集成

ClawBench Official 和 PinchBench 不一样，它不是简单 markdown task + grade 函数，而是目录化任务：

```text
task_dir/
├── task.toml
├── instruction.md
├── environment/
│   ├── data/
│   └── setup.sh
└── verifier/
    └── test_output.py
```

我实现的 ClawBench pipeline 是：

1. `_load_clawbench_tasks()` 扫描 repo-local `assets/benchmarks/claw-bench/tasks/*/*/task.toml`，加载 303 个任务。
2. `_prepare_clawbench_workspace()` 创建临时 workspace，复制 `environment/data/`，执行 `environment/setup.sh`。
3. `_load_clawbench_instruction()` 读取 `instruction.md`，把 `workspace/` 替换成临时 workspace 的绝对路径，并额外强调只能写到该 workspace。
4. `run_task_with_sal_agent()` 用 SAL Bash Agent 执行任务。
5. `_verify_clawbench_task()` 调用 `claw_bench.core.verifier.verify_task(task_dir, workspace)`，用 pytest verifier 检查产物。
6. 保存 `result.json` 和 `transcript.json`，并在 summary 中汇总 score、passed、pending。

这意味着 SAL 已经能作为 native workspace agent 直接跑 ClawBench。这里的任务集在 `assets/benchmarks/claw-bench/` 下，执行路径是本地 workspace + pytest verifier，不需要 Docker/NanoBotAgent。

## 2.6 CLI 使用方式

```bash
cd /Users/bytedance/Documents/github/glm_dev/simple-agent-lab

# 冒烟测试：transcript、PinchBench、ClawBench verifier
uv run python evals/openclaw/test_integration.py

# PinchBench 全量
uv run python -m evals.openclaw \
  --bench pinchbench \
  --model gpt-4.1-2025-04-14 \
  --clawevalkit /path/to/ClawEvalkit \
  --api-url "https://search.bytedance.net/gpt/openapi/online/v2/crawl" \
  --api-key-env GPT_API_KEY \
  --provider azure_openai \
  --max-turns 10

# ClawBench 单任务
uv run python -m evals.openclaw \
  --bench clawbench \
  --model gpt-4.1-2025-04-14 \
  --clawevalkit /Users/bytedance/Documents/github/glm_dev/simple-agent-lab/assets/benchmarks/claw-bench \
  --task-ids cal-001 \
  --api-url "https://search.bytedance.net/gpt/openapi/online/v2/crawl" \
  --api-key-env GPT_API_KEY \
  --provider azure_openai

# ClawBench 样本评测
uv run python -m evals.openclaw \
  --bench clawbench \
  --model gpt-4.1-2025-04-14 \
  --clawevalkit /Users/bytedance/Documents/github/glm_dev/simple-agent-lab/assets/benchmarks/claw-bench \
  --sample 30 \
  --max-turns 15 \
  --api-url "https://search.bytedance.net/gpt/openapi/online/v2/crawl" \
  --api-key-env GPT_API_KEY \
  --provider azure_openai
```

---

# <text color="blue">三、当前实验结果</text>

## 3.1 冒烟测试结果

本地执行：

```bash
uv run python evals/openclaw/test_integration.py
```

结果：

```text
Results: 4 passed, 0 failed
```

覆盖项：

- SAL events 能转成 ClawEvalkit transcript。
- fake provider 下 PinchBench 最小 pipeline 可以跑。
- 能从外部 ClawEvalkit 加载 23 个 PinchBench task。
- 能从 repo-local `assets/benchmarks/claw-bench/` 加载 303 个 ClawBench task，并对 `cal-001` 跑通 SAL execution + pytest verifier。

## 3.2 PinchBench 全量结果

<callout emoji="gem" background-color="light-blue" border-color="light-blue">
<text color="blue">**PinchBench 23/23 全部执行成功，无 runtime crash。**</text>
<text color="gray">按运行日志汇总，gpt-4.1-2025-04-14 在 PinchBench 上总分 60.0%。这说明集成链路是稳定的，但 agent 能力在工具链/检索/结构化文件任务上仍有明显短板。</text>
</callout>

总体指标：

<lark-table>
  <lark-tr>
    <lark-td>**指标**</lark-td>
    <lark-td>**数值**</lark-td>
    <lark-td>**说明**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>任务数</lark-td>
    <lark-td>23/23</lark-td>
    <lark-td>全量 PinchBench 任务完成执行。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>总分</lark-td>
    <lark-td>60.0%</lark-td>
    <lark-td>来自运行日志 `run_pinchbench_gpt41.log` 的全量汇总。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>满分任务</lark-td>
    <lark-td>10 个</lark-td>
    <lark-td>文本生成、总结、简单文件/日程/邮件类任务表现较好。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>零分任务</lark-td>
    <lark-td>6 个</lark-td>
    <lark-td>股票、天气、邮件搜索、表格总结、OpenClaw 理解、second brain 失败明显。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>平均耗时</lark-td>
    <lark-td>约 7.9s/题</lark-td>
    <lark-td>按本地 per-task result 统计。</lark-td>
  </lark-tr>
</lark-table>

任务级结果：

<lark-table>
  <lark-tr>
    <lark-td>**任务**</lark-td>
    <lark-td>**分数**</lark-td>
    <lark-td>**观察**</lark-td>
  </lark-tr>
  <lark-tr><lark-td>`task_00_sanity`</lark-td><lark-td>1.0</lark-td><lark-td>基础响应能力正常。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_01_calendar`</lark-td><lark-td>1.0</lark-td><lark-td>日程类结构化输出可完成。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_02_stock`</lark-td><lark-td>0.0</lark-td><lark-td>缺少可靠实时数据/检索能力。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_03_blog`</lark-td><lark-td>1.0</lark-td><lark-td>文本生成能力强。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_04_weather`</lark-td><lark-td>0.0</lark-td><lark-td>同样暴露实时外部信息能力不足。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_05_summary`</lark-td><lark-td>1.0</lark-td><lark-td>摘要类表现稳定。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_06_events`</lark-td><lark-td>1.0</lark-td><lark-td>结构化提取/整理成功。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_07_email`</lark-td><lark-td>1.0</lark-td><lark-td>简单邮件任务可完成。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_08_memory`</lark-td><lark-td>0.8</lark-td><lark-td>简单记忆可做，但细节有遗漏。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_09_files`</lark-td><lark-td>0.857</lark-td><lark-td>文件操作基本可靠，但仍会漏边界条件。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_10_workflow`</lark-td><lark-td>0.167</lark-td><lark-td>多步流程计划和执行一致性较弱。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_11_clawdhub`</lark-td><lark-td>1.0</lark-td><lark-td>简单知识/文本任务完成好。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_12_skill_search`</lark-td><lark-td>0.833</lark-td><lark-td>搜索/选择类任务有一定能力，但不完全稳定。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_13_image_gen`</lark-td><lark-td>0.333</lark-td><lark-td>缺少真正图片生成/视觉工具支持。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_14_humanizer`</lark-td><lark-td>1.0</lark-td><lark-td>纯文本改写能力强。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_15_daily_summary`</lark-td><lark-td>1.0</lark-td><lark-td>聚合总结类表现好。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_16_email_triage`</lark-td><lark-td>日志为 0.0；本地 per-task 为 0.909</lark-td><lark-td>结果文件和日志存在一次重跑差异，需要后续统一缓存/summary 口径。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_16_market_research`</lark-td><lark-td>日志为 0.813；本地 per-task 为 0.875</lark-td><lark-td>检索式市场分析能做一部分，但可复现性和评分口径需整理。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_17_email_search`</lark-td><lark-td>0.0</lark-td><lark-td>检索/定位能力不足。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_18_spreadsheet_summary`</lark-td><lark-td>0.0</lark-td><lark-td>表格工具链能力不足。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_20_eli5_pdf_summary`</lark-td><lark-td>1.0</lark-td><lark-td>PDF/文本摘要类表现好。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_21_openclaw_comprehension`</lark-td><lark-td>0.0</lark-td><lark-td>对 benchmark/project 内部知识的定位能力不足。</lark-td></lark-tr>
  <lark-tr><lark-td>`task_22_second_brain`</lark-td><lark-td>0.0</lark-td><lark-td>长期记忆/知识库类能力缺失。</lark-td></lark-tr>
</lark-table>

## 3.3 ClawBench Official 样本结果

ClawBench Official 已经成功加载 303 个任务。目前跑出了 35 个任务样本，summary 为：

```json
{
  "model": "gpt-4.1-2025-04-14",
  "score": 78.9,
  "passed": 22,
  "scored": 35,
  "total": 303,
  "pending": 268
}
```

总体指标：

<lark-table>
  <lark-tr>
    <lark-td>**指标**</lark-td>
    <lark-td>**数值**</lark-td>
    <lark-td>**说明**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>已加载任务</lark-td>
    <lark-td>303</lark-td>
    <lark-td>ClawBench Official task.toml 扫描成功。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>已执行样本</lark-td>
    <lark-td>35</lark-td>
    <lark-td>覆盖 20 个左右 domain。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>平均分</lark-td>
    <lark-td>78.9%</lark-td>
    <lark-td>样本分，不代表全量最终分。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>完全通过</lark-td>
    <lark-td>22/35</lark-td>
    <lark-td>calendar、communication、data-analysis、document-editing 等表现较好。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>零分</lark-td>
    <lark-td>5/35</lark-td>
    <lark-td>bioinformatics、academic-research、time-series 等失败明显。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>平均耗时</lark-td>
    <lark-td>约 16.9s/题</lark-td>
    <lark-td>样本总耗时约 591.6s。</lark-td>
  </lark-tr>
</lark-table>

表现较好的任务类型：

- **结构化文件生成**：calendar、document-editing、file-operations 的多数任务得分高。
- **文本理解和改写**：communication、contract-review、regulatory-compliance、email 任务表现稳定。
- **数据分析基础题**：data-analysis 3 个样本均满分，说明 bash + Python/脚本执行路径是有效的。
- **跨域组合任务**：xdom-007、xdom-009 两个样本均满分，说明简单多步骤产物生成可以被 SAL 完成。

表现较差的任务类型：

- **Bioinformatics**：3 个样本全部 0 分，可能需要领域工具、格式约束和更强的数据处理模板。
- **Debugging/System Admin**：debug-004 仅 0.0625，sys-008 为 0.3333，说明复杂诊断流程和系统命令验证不足。
- **Time-series/Data Science 边界任务**：ds-004 为 0，可能因为 verifier 对文件格式/数值结果要求严格。
- **Academic Research**：plagiarism check 为 0，说明当前 agent 没有可靠的文献/相似度检测工具链。

## 3.4 和 PinchBench 结果的对照

PinchBench 和 ClawBench 给出的信号不完全一样：

- PinchBench 更像“小型综合能力检查”，失败集中在实时信息、搜索、外部工具和复杂 workflow。
- ClawBench 更像“workspace artifact benchmark”，只要输入文件和 verifier 明确，Bash Agent 往往可以通过脚本生成正确产物。
- 因此 ClawBench 样本分高于 PinchBench，不一定说明 agent 更强，而是任务形态更适配 bash workspace agent。

我的判断是：SAL Bash Agent 当前更擅长 **本地 workspace 内可验证的产物生成**，不擅长 **外部世界检索、长期记忆、多工具链协同和复杂自我校验**。

---

# <text color="blue">四、从结果看当前框架的不足</text>

## 4.1 Runtime 很清楚，但执行控制还比较薄

SAL 的 run loop 可读性很好，但作为 benchmark runner 还缺一些工程能力：

- 没有 per-tool timeout 的细粒度控制，主要依赖外层任务 timeout。
- 工具执行失败、模型失败、verifier 失败之间的错误边界还不够清楚。
- `max_turns` 是硬预算，但没有更智能的“任务未完成时继续规划/自检”机制。
- 对长任务没有内建 checkpoint/resume，只能靠结果缓存。

影响：简单任务很顺，但复杂任务容易“看起来执行成功，实际上产物不满足 verifier”。

## 4.2 工具生态过窄，Bash Agent 是好起点但不是完整 OpenClaw Agent

这次主要靠 bash tool 解决所有任务。它的优点是通用、简单、可控；缺点也明显：

- 实时信息类任务缺少浏览器/搜索/API 工具。
- 图片生成、多模态理解类任务缺少专门工具。
- 表格、PDF、文档、数据科学任务虽然可以用脚本做，但缺少高层工具封装和模板。
- 邮件、日历、知识库、second brain 等任务需要结构化应用状态或 mock service，而不是纯 bash。

影响：Bash Agent 在 ClawBench 的文件产物任务上表现不错，但在 PinchBench 的搜索/记忆/外部应用任务上掉分明显。

## 4.3 Provider 抽象还不够覆盖真实企业环境

SAL 的 provider 抽象本身比较干净，但真实接入 ModelHub 时暴露出：

- AzureOpenAI 不在原生 adapter 内，需要我在 eval adapter 里自定义 `_build_azure_agent()`。
- provider 配置与 ClawEvalkit YAML 之间需要手动映射。
- proxy、api_version、endpoint、tool call arguments 等企业代理细节没有统一入口。
- token usage 在自定义 Azure path 中没有完整沉淀到所有 trace 统计中。

影响：能跑通，但 provider 适配逻辑散在 eval adapter 里，不利于长期维护。后续应该把 Azure/OpenAI-compatible enterprise provider 正式纳入 SAL `llm/adapters`。

## 4.4 Benchmark harness 能跑，但结果管理还有口径问题

我在结果分析时发现一个重要工程问题：PinchBench summary 文件和 per-task result/log 之间出现过口径不一致。例如：

- `run_pinchbench_gpt41.log` 记录了 23/23 全量，总分 60.0%。
- 当前 summary JSON 一度显示 result_count 为 10，这是后续 sample/cache 重跑覆盖 summary 导致的。
- 个别任务如 `task_16_email_triage` 在日志和 per-task result 中分数不同，说明缓存、重跑、summary 保存策略需要更严格。

这不影响“集成已跑通”的结论，但会影响后续做正式 benchmark report。

建议：

- summary 文件名带上 run id、sample/full 标识和时间戳。
- 每次 run 生成独立 `run_manifest.json`。
- 区分 cached result、new result、overwritten summary。
- dashboard 读取 run manifest，而不是直接聚合当前目录下所有 result。

## 4.5 失败诊断还不够自动化

现在 result 里有 score、checks_passed、details、workspace_files、transcript，但还缺：

- 失败 verifier 的断言摘要自动抽取。
- agent 最后一轮想法/命令/产物和失败 check 的关联。
- workspace snapshot 保留策略；现在临时 workspace 会删除，只保存文件列表。
- 对 0 分任务的类别化诊断，例如“没写文件”“文件名错”“schema 错”“数值错”“依赖缺失”。

影响：跑分很快，但分析失败原因还要人工翻 transcript 和 verifier。

## 4.6 ClawBench 集成还没有全量稳定性结论

目前 ClawBench 已经证明：

- 303 任务可以加载。
- verifier 可以调用。
- SAL Bash Agent 可以跑样本。
- 35 个样本平均 78.9%。

但还不能说全量能力已经充分评估，因为还有 268 个 pending。全量评测可能暴露：

- 长耗时任务的 timeout 问题。
- 环境依赖差异。
- setup.sh 副作用。
- verifier 对路径、权限、依赖包的隐含要求。
- 并行运行时的资源竞争。

因此当前结论应表述为：**ClawBench integration complete，full-scale evaluation pending**。

---

# <text color="blue">五、我的阶段性结论</text>

## 5.1 已完成的事情

<grid cols="2">
  <column width="50">

**集成层面**

- 新增 `evals/openclaw/` 适配层。
- 支持 SAL Event 到 ClawEvalkit transcript。
- 支持 PinchBench 任务加载、执行和评分。
- 支持 AgentBench 初版 L0 文件存在性评分。
- 支持 ClawBench Official 303 任务加载和 pytest verifier。
- 支持 AzureOpenAI/ModelHub 真实模型调用。

  </column>
  <column width="50">

**实验层面**

- 冒烟测试 4/4 通过。
- PinchBench 全量 23 题跑通，总分 60.0%。
- ClawBench 样本 35/303 跑通，平均 78.9%。
- 保存了 result、transcript、dashboard 和运行日志。
- 初步定位了 SAL Bash Agent 的强项和短板。

  </column>
</grid>

## 5.2 我对 SAL 能力边界的判断

SAL 作为一个最小 agent lab，已经足够支撑 benchmark integration 和快速实验。它的优势是：

- 结构小，容易理解和改。
- event/message/trace 设计适合做可观测 eval。
- Bash Agent 很适合 workspace artifact 类型任务。
- 适配新 benchmark 的成本不高。

但如果要继续往“严肃 benchmark harness”发展，需要补齐：

- provider 标准化；
- 工具生态；
- run 级别结果管理；
- 失败诊断；
- workspace 保留和复现；
- 全量并行评测稳定性。

## 5.3 下一步建议

<lark-table>
  <lark-tr>
    <lark-td>**优先级**</lark-td>
    <lark-td>**事项**</lark-td>
    <lark-td>**原因**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>P0</lark-td>
    <lark-td>跑完 ClawBench 303 全量</lark-td>
    <lark-td>老板目标已经完成集成，但全量分数是下一步最硬的结果。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>P0</lark-td>
    <lark-td>修正 run summary/cache 口径</lark-td>
    <lark-td>避免 sample run 覆盖 full run，保证报告可信。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>P1</lark-td>
    <lark-td>把 AzureOpenAI adapter 下沉到 SAL llm/adapters</lark-td>
    <lark-td>减少 eval adapter 的 provider 特例，方便跑 GLM、GPT、Claude 对比。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>P1</lark-td>
    <lark-td>保留失败 workspace snapshot</lark-td>
    <lark-td>让 0 分任务可诊断，而不是只能看 transcript。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>P1</lark-td>
    <lark-td>为 spreadsheet、browser/search、image、document 增加专门工具</lark-td>
    <lark-td>PinchBench 掉分集中在这些能力缺口上。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>P2</lark-td>
    <lark-td>做失败类型自动聚类</lark-td>
    <lark-td>从“跑分”升级到“知道为什么掉分”。</lark-td>
  </lark-tr>
</lark-table>

---

# <text color="blue">六、附录：产物位置</text>

<lark-table>
  <lark-tr>
    <lark-td>**产物**</lark-td>
    <lark-td>**路径**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>OpenClaw 集成代码</lark-td>
    <lark-td>`evals/openclaw/`</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>实验 README</lark-td>
    <lark-td>`docs/exp1_openclaw/README.md`</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>PinchBench 结果</lark-td>
    <lark-td>`evals/openclaw/results/gpt-4.1-2025-04-14/`</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>ClawBench 结果</lark-td>
    <lark-td>`evals/openclaw/results/clawbench/gpt-4.1-2025-04-14/`</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>运行日志</lark-td>
    <lark-td>`docs/exp1_openclaw/assets/logs/`</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>Dashboard</lark-td>
    <lark-td>`docs/exp1_openclaw/dashboard.html` 和 `docs/exp1_openclaw/clawbench_dashboard.html`</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>SAL 源码解析页面</lark-td>
    <lark-td>`docs/simple_agent_lab_intro.html`</lark-td>
  </lark-tr>
</lark-table>

<callout emoji="sparkles" background-color="light-gray" border-color="light-gray">
<text color="gray">**一句话总结：这次工作已经把 simple-agent-lab 从一个小型 agent runtime 推进到了可以接入 OpenClaw/ClawBench 的实验框架；下一阶段重点不是“能不能跑”，而是“全量跑得稳、结果讲得清、失败能诊断”。**</text>
</callout>
