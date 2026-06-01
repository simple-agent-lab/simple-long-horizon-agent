# <text color="blue">Simple Agent Lab 代码剖析：从 Runtime 到 OpenClaw Adapter</text>

<callout emoji="book" background-color="light-blue" border-color="light-blue">
<text color="blue">**这篇文档从代码角度解释 simple-agent-lab 是怎么工作的，以及我如何把它接到 OpenClaw/ClawBench。**</text>
<text color="gray">重点不放在 API 罗列，而是解释核心对象之间的关系：Message 如何流动、State 如何记录、Tool 如何执行、Eval Adapter 如何把 SAL 变成 ClawBench runner。</text>
</callout>

---

# <text color="blue">一、源码地图</text>

<lark-table>
  <lark-tr>
    <lark-td>**路径**</lark-td>
    <lark-td>**模块角色**</lark-td>
    <lark-td>**我对它的理解**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`src/simple_agent_lab/core.py`</lark-td>
    <lark-td>核心 runtime</lark-td>
    <lark-td>定义 Agent 和 run loop，是整个框架最重要的文件。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`src/simple_agent_lab/messages.py`</lark-td>
    <lark-td>消息协议</lark-td>
    <lark-td>定义 Message 与各种 ContentBlock，是模型、工具和 trace 之间的公共语言。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`src/simple_agent_lab/state.py`</lark-td>
    <lark-td>状态容器</lark-td>
    <lark-td>保存 task、message history、event history 和实验 data。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`src/simple_agent_lab/protocols.py`</lark-td>
    <lark-td>事件协议</lark-td>
    <lark-td>定义 agent_start、turn_start、model_request、tool_execution 等事件形态。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`src/simple_agent_lab/context_view.py`</lark-td>
    <lark-td>上下文投影</lark-td>
    <lark-td>决定哪些 message 会被模型看到，是做压缩、裁剪和多 agent 路由的关键入口。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`src/simple_agent_lab/llm/bridge.py`</lark-td>
    <lark-td>LLM 桥接</lark-td>
    <lark-td>把 SAL Message 转成 LLMMessage，也把 LLMResponse 转回 AssistantMessage。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`src/simple_agent_lab/llm/adapters/`</lark-td>
    <lark-td>模型 provider adapter</lark-td>
    <lark-td>适配 OpenAI Chat、OpenAI Responses、Anthropic Messages、Fake provider。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`src/simple_agent_lab/tools/bash.py`</lark-td>
    <lark-td>Bash 工具</lark-td>
    <lark-td>让 agent 在 workspace 中执行命令，是这次 benchmark 执行的主要能力来源。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`src/simple_agent_lab/agents/bash/agent.py`</lark-td>
    <lark-td>预置 Bash Agent</lark-td>
    <lark-td>把 provider、system prompt、bash tool 包装成一个可运行 Agent。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`evals/openclaw/`</lark-td>
    <lark-td>OpenClaw 适配层</lark-td>
    <lark-td>我新增的 benchmark adapter，让 SAL 可以跑 PinchBench/AgentBench/ClawBench。</lark-td>
  </lark-tr>
</lark-table>

---

# <text color="blue">二、核心 Runtime：core.py</text>

## 2.1 Agent 是什么

`Agent` 是一个很薄的 dataclass：

```python
@dataclass
class Agent:
    name: str
    generate: GenerateFn
    role: str = ""
    tools: tuple[AgentTool, ...] = ()
    context_policy: ContextPolicy | None = None
```

我理解这里的设计重点是：Agent 本身不内置模型调用逻辑。它只持有一个 `generate` 函数：

```python
GenerateFn = Callable[[list[Message]], Message]
```

也就是说，只要能把可见消息列表变成下一条 assistant message，不管背后是真实 LLM、fake provider、rule-based function，还是 AzureOpenAI 闭包，都可以成为一个 SAL Agent。

这让 eval 场景特别好接：我可以在 OpenClaw adapter 里临时构造一个 AzureOpenAI-backed Agent，而不需要改 core runtime。

## 2.2 run loop 做了什么

`run(agent, state, max_turns)` 是 SAL 的中心循环。它每轮大致做这些事：

1. `state.turn_start(agent=name)`：记录轮次开始。
2. `maybe_compress_context()`：按 policy 做上下文压缩。
3. `build_context_view()`：生成当前 agent 可见消息。
4. `messages_to_llm_messages()`：生成 LLM payload，并写入 `state.data["last_llm_payload"]`。
5. `state.model_request(...)`：记录模型请求事件。
6. `agent.generate(visible)`：调用模型/生成函数。
7. `state.model_response(...)`：记录模型响应元信息。
8. `state.record(output)`：把 assistant message 写入状态。
9. 如果有 tool call，进入 `dispatch_tool_calls()`。
10. 如果输出是 final，结束；否则进入下一轮。

<quote-container>
**core.py 的好处是职责边界很清晰：Agent 负责生成，runtime 负责调度，State 负责留痕，Tool 负责执行。**
</quote-container>

## 2.3 dispatch_tool_calls 的意义

`dispatch_tool_calls()` 会从 assistant message 中取出 `ToolCallBlock`，找到同名工具，然后执行工具并把结果写回 state。

它还处理了一个重要细节：如果工具声明为 sequential，就按顺序执行；否则可以并发执行。这说明 SAL 的工具层不是简单字符串拼接，而是有基本的执行语义。

对 ClawBench 来说，bash tool 是唯一主要工具，因此绝大多数任务就是：

```text
model proposes bash command → bash executes in workspace → tool result returns → model writes final answer/artifacts
```

---

# <text color="blue">三、消息协议：messages.py</text>

SAL 的 Message 不是 OpenAI message 的简单复制，而是一个运行时中立协议。

核心字段包括：

- `role`：user / assistant / system。
- `sender`：消息发送方。
- `target`：消息目标。
- `kind`：task / thought / final / tool_result 等。
- `channel`：消息通道。
- `content`：一组 ContentBlock。

ContentBlock 包含：

- `TextBlock`
- `ImageBlock`
- `ThinkingBlock`
- `ToolCallBlock`
- `ToolResultBlock`

这个设计让 SAL 可以表达 OpenAI/Anthropic 等不同 provider 的 tool call，也可以在 trace 中保留 thinking、image、tool result 等结构化内容。

对我这次集成最重要的是：

- `ToolCallBlock` 可以转成 ClawEvalkit transcript 的 `toolCall`。
- `ToolResultBlock` 可以转成 ClawEvalkit transcript 的 `toolResult`。
- `TextBlock` 可以转成普通评分逻辑可读文本。

---

# <text color="blue">四、LLM 层：provider、bridge、adapter</text>

## 4.1 Provider 是配置，不是 runtime

SAL 的 `Provider` 描述模型调用配置，例如：

```python
Provider(
    id=config.name,
    api=config.api_kind,
    model=config.model,
    base_url=config.api_url or None,
    api_key_env=config.api_key_env,
)
```

这里的 `api` 是 SAL 内部的 ApiKind，例如：

- `fake`
- `openai-chat`
- `openai-responses`
- `anthropic-messages`

ClawEvalkit 的 provider 名称和 SAL 的 ApiKind 不完全一样，所以我在 `evals/openclaw/config.py` 做了映射，例如：

- `openrouter` → `openai-chat`
- `glm` → `openai-chat`
- `anthropic` → `anthropic-messages`
- `azure_openai` → `azure_openai`

## 4.2 bridge 是协议转换层

`simple_agent_lab.llm.bridge` 做两类转换：

- SAL Message → LLMMessage
- LLMResponse → SAL AssistantMessage

这层非常关键，因为它把 runtime 和 provider 解耦。runtime 永远处理 SAL Message；provider adapter 只需要处理自己的 API 格式。

## 4.3 为什么我额外写了 AzureOpenAI path

SAL 原生没有 AzureOpenAI adapter，而这次真实评测用的是字节内部 ModelHub GPT 代理。因此我在 `evals/openclaw/adapter.py` 写了 `_build_azure_agent()`。

这个函数做的事：

1. 构造 bash tool。
2. 定义 `azure_generate(visible)` 闭包。
3. 用 bridge 把 SAL visible messages 转成 OpenAI Chat messages。
4. 用 OpenAI Chat adapter 把 SAL tool schema 转成 OpenAI tools。
5. 调用 `AzureOpenAI.chat.completions.create()`。
6. 把返回的 text/tool_calls 重新封装为 SAL `TextBlock` 和 `ToolCallBlock`。
7. 用 `llm_response_to_assistant_message()` 转回 SAL assistant message。

这说明 SAL 的架构是可扩展的：即使 provider adapter 没有正式合入，也可以在 eval 层用闭包补上。

---

# <text color="blue">五、Bash Agent：为什么它适合跑 ClawBench</text>

`make_bash_agent()` 做了一层很薄的封装：

```python
return make_llm_agent(
    name=name,
    provider=provider,
    role=role,
    tools=[make_bash_tool(cwd=cwd)],
    system_prompt=system_prompt,
    target="user",
)
```

它把 LLM provider 和 bash tool 绑定成一个 agent。

ClawBench 的很多任务都是：

- 给一个输入文件目录。
- 要求生成某个 JSON/CSV/Markdown/配置/报告。
- verifier 检查输出文件是否符合预期。

这类任务天然适合 Bash Agent，因为 agent 可以：

- `ls/cat/find` 观察 workspace。
- 用 Python/awk/sed 等工具处理数据。
- 创建目标文件。
- 自己跑一些本地检查。

但它也解释了当前短板：如果任务需要浏览器、搜索、邮箱、日历、图像生成、长期记忆，纯 bash 就不够。

---

# <text color="blue">六、OpenClaw Adapter：新增 evals/openclaw</text>

## 6.1 config.py：模型配置桥接

`config.py` 的职责是把 ClawEvalkit 模型配置变成 SAL 可以理解的 `ModelConfig`。

关键对象：

```python
@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str
    api_url: str
    api_key_env: str
    api_kind: str = "openai-chat"
    timeout: int = 300
    max_turns: int = 20
```

它支持：

- 从 ClawEvalkit `configs/models/*.yaml` 读取配置。
- 从 CLI 参数覆盖 `api_url`、`api_key_env`、`provider`。
- provider → SAL ApiKind 的映射。
- provider → API key env 的映射。

这让 runner 可以用统一参数启动不同模型。

## 6.2 adapter.py：真正的协议桥

`adapter.py` 是最重的文件，包含四类逻辑：

1. `sal_events_to_claweval_transcript()`：SAL Event → ClawEvalkit transcript。
2. `run_task_with_sal_agent()`：构造 agent，运行任务，收集 content/transcript/usage/events。
3. `run_pinchbench_task()` 和 `run_agentbench_task()`：处理旧 benchmark 的任务执行与评分。
4. `run_clawbench_task()`：处理 ClawBench Official 的 workspace、instruction 和 verifier。

我认为这里最关键的是 `run_task_with_sal_agent()`，因为它把 SAL 和所有 benchmark 连接起来：

```text
prompt + ModelConfig + workspace
  → build SAL Bash Agent
  → agent.run(prompt)
  → collect events/state
  → transcript conversion
  → return normalized result
```

只要这个函数稳定，后面接不同 benchmark 就只是“任务加载”和“评分方式”的差异。

## 6.3 runner.py：benchmark orchestration

`runner.py` 负责更外层的编排：

- `_load_pinchbench_tasks()`
- `_load_agentbench_tasks()`
- `_load_clawbench_tasks()`
- `run_pinchbench()`
- `run_agentbench()`
- `run_clawbench()`
- `run_eval()`
- CLI `main()`

它还处理：

- sample。
- parallel。
- force rerun。
- cached result。
- summary JSON 保存。
- benchmark key 到 runner 的分发。

`BENCHMARK_RUNNERS` 当前支持：

```python
BENCHMARK_RUNNERS = {
    "pinchbench": run_pinchbench,
    "agentbench": run_agentbench,
    "clawbench": run_clawbench,
    "clawbench-official": run_clawbench,
}
```

这就是“换 benchmark 不换 agent”的入口。

---

# <text color="blue">七、ClawBench Pipeline 代码路径</text>

ClawBench 单任务执行的主函数是：

```python
run_clawbench_task(task, config, results_dir)
```

当前任务集来自 repo-local `assets/benchmarks/claw-bench/`，执行方式是 native workspace + pytest verifier。目录里虽然有 Docker 相关文件，但这条 SAL 集成路径不需要 Docker。

它的执行流程如下：

```text
task metadata
  → create temp workspace
  → copy environment/data
  → run environment/setup.sh
  → load instruction.md
  → rewrite workspace path to absolute path
  → run SAL Bash Agent
  → call claw_bench.core.verifier.verify_task
  → save result.json + transcript.json
  → remove temp workspace
```

这里最重要的两个实现细节：

1. **workspace 绝对路径注入**：我在 instruction 前面加了强约束，要求所有输出写到临时 workspace，避免 agent 写到 repo 或其他目录。
2. **pytest verifier 复用**：没有重写 ClawBench 的评分逻辑，而是直接调用 `verify_task(task_dir, workspace)`，保证评分方式与官方 benchmark 对齐。

---

# <text color="blue">八、测试与可验证性</text>

`evals/openclaw/test_integration.py` 当前覆盖 4 个 smoke test：

<lark-table>
  <lark-tr>
    <lark-td>**测试**</lark-td>
    <lark-td>**验证内容**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>Transcript conversion</lark-td>
    <lark-td>SAL events 可以转成 ClawEvalkit transcript。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>Full PinchBench task</lark-td>
    <lark-td>fake provider 下最小 PinchBench pipeline 可以执行并评分。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>Load from ClawEvalkit</lark-td>
    <lark-td>能从外部 ClawEvalkit 加载 23 个 PinchBench task。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>ClawBench task smoke</lark-td>
    <lark-td>能从 repo-local `assets/benchmarks/claw-bench/` 加载 303 个 ClawBench task，并跑通 cal-001 的 SAL execution + verifier；这一路径不需要 Docker。</lark-td>
  </lark-tr>
</lark-table>

本地结果：

```text
Results: 4 passed, 0 failed
```

---

# <text color="blue">九、代码层面的主要不足</text>

## 9.1 AzureOpenAI adapter 还在 eval 层

当前 `_build_azure_agent()` 写在 `evals/openclaw/adapter.py`。短期能跑，但长期更合理的位置是：

```text
src/simple_agent_lab/llm/adapters/azure_openai.py
```

这样 provider 配置、token usage、tool call 解析和错误处理都能统一。

## 9.2 result summary 口径需要重构

`runner.py` 当前会把 summary 写到固定文件：

```text
results_dir / f"{model_key}.json"
```

这导致 sample run、full run、force rerun 可能互相覆盖。后续建议改成：

```text
runs/
  2026-06-01T14-30-00_full/
    run_manifest.json
    summary.json
    tasks/*/result.json
```

这样 report 和 dashboard 会更可信。

## 9.3 失败任务缺少 workspace snapshot

`run_clawbench_task()` 结束时会删除临时 workspace，只保存 `workspace_files` 列表。这对节省空间有好处，但对失败分析不够友好。

建议：

- 满分任务删除 workspace。
- 失败任务保留 workspace snapshot。
- result 中记录 snapshot path。
- verifier details 抽取失败断言摘要。

## 9.4 工具能力还是单一

Bash tool 是一个好基座，但从 PinchBench 失败任务看，后续至少需要：

- browser/search tool；
- spreadsheet tool；
- document/PDF tool；
- image generation/vision tool；
- memory/knowledge base tool；
- application mock/service tool。

否则很多 OpenClaw 场景只能靠模型“猜”，不能稳定执行。

---

# <text color="blue">十、我的总体评价</text>

从代码角度看，Simple Agent Lab 的底层设计是健康的：小、清楚、容易扩展。它最值得保留的是：

- 显式 Message 协议。
- 生成函数与 runtime 解耦。
- State/Event 留痕。
- ContextView 投影。
- ToolCall/ToolResult 结构化。

这次 OpenClaw/ClawBench 集成能较快完成，正是因为这些边界足够清楚。

但要继续向正式 benchmark framework 演进，重点应放在工程化能力：

- provider adapter 下沉；
- benchmark run manifest；
- workspace snapshot；
- failure diagnosis；
- 多工具生态；
- 全量并行稳定性。

<callout emoji="sparkles" background-color="light-gray" border-color="light-gray">
<text color="gray">**一句话：SAL 的核心 runtime 已经足够干净，OpenClaw adapter 证明它能承载真实 benchmark；下一步要把临时适配代码沉淀成稳定、可复现、可诊断的评测基础设施。**</text>
</callout>
