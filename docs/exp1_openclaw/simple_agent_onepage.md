# <text color="blue">Simple Agent Lab OnePage</text>

<callout emoji="gem" background-color="light-blue" border-color="light-blue">
<text color="blue">**Simple Agent Lab 是一个“小而透明”的 Agent 实验框架。**</text>
<text color="gray">它用极少的核心抽象表达 agent loop、消息协议、工具调用、上下文投影和评测轨迹，适合快速理解、改造和接入 benchmark。</text>
</callout>

<grid cols="3">
  <column width="33">
    <text color="blue">**定位**</text>
    <text color="gray">不是生产平台，而是教学、研究、benchmark integration 和 agent runtime 设计实验台。</text>
  </column>
  <column width="33">
    <text color="blue">**核心抽象**</text>
    <text color="gray">Agent、Message、ContentBlock、State、Event、Tool、ContextView。</text>
  </column>
  <column width="33">
    <text color="blue">**当前进展**</text>
    <text color="gray">已完成 OpenClaw/ClawBench 集成，SAL Bash Agent 可以跑 PinchBench 和 ClawBench Official。</text>
  </column>
</grid>

---

# <text color="blue">一、框架一句话</text>

Simple Agent Lab 的核心不是“封装更多工具”，而是把 agent 运行过程拆成一组非常显式的对象：

```text
Agent + Message + State + ContextView + Tool + Event
```

它的设计哲学是：agent 运行时应该能被人读懂、被实验替换、被 benchmark 评分、被 trace 复盘。

<quote-container>
**Simple Agent Lab 更像一个 agent runtime 显微镜：它让我们看清楚每一轮模型输入、模型输出、工具调用和状态变化。**
</quote-container>

---

# <text color="blue">二、最小运行链路</text>

1. 用户任务进入 `State`，成为一条 task message。
2. runtime 根据 `ContextPolicy` 构造当前 agent 可见的上下文。
3. `Agent.generate(visible_messages)` 产出 assistant message。
4. 如果 message 包含 `ToolCallBlock`，runtime 调用对应 tool。
5. tool result 作为 message 写回 state。
6. agent 输出 final message 或达到 max turns 后结束。

这个链路的价值在于，每一步都有 event，可记录、可导出、可评分。

---

# <text color="blue">三、我这次完成的集成</text>

<lark-table>
  <lark-tr>
    <lark-td>**方向**</lark-td>
    <lark-td>**完成内容**</lark-td>
    <lark-td>**状态**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>OpenClaw/ClawEvalkit</lark-td>
    <lark-td>新增 `evals/openclaw/` 适配层，支持 SAL Event 到 ClawEvalkit transcript。</lark-td>
    <lark-td>已完成</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>PinchBench</lark-td>
    <lark-td>加载 23 个 task，执行 SAL Bash Agent，调用内嵌 grade 函数评分。</lark-td>
    <lark-td>已跑全量</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>ClawBench Official</lark-td>
    <lark-td>从 repo-local `assets/benchmarks/claw-bench/` 加载 303 个 task，准备 workspace，执行 agent，调用 pytest verifier；不需要 Docker。</lark-td>
    <lark-td>已接入并跑样本</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>ModelHub/AzureOpenAI</lark-td>
    <lark-td>在 adapter 中实现 AzureOpenAI path，支持 gpt-4.1-2025-04-14 真实评测。</lark-td>
    <lark-td>已跑通</lark-td>
  </lark-tr>
</lark-table>

---

# <text color="blue">四、实验结果快照</text>

<grid cols="2">
  <column width="50">

**PinchBench**

- 23/23 任务端到端执行成功。
- 全量日志汇总分数：60.0%。
- 满分任务 10 个。
- 失败集中在实时信息、搜索、表格、记忆和多工具链任务。

  </column>
  <column width="50">

**ClawBench Official**

- 303 个任务可加载。
- 已执行样本 35 个。
- 样本平均分：78.9%。
- 22/35 完全通过。
- 适合验证 workspace artifact 生成能力。

  </column>
</grid>

---

# <text color="blue">五、文档入口</text>

<callout emoji="💡" background-color="light-green" border-color="light-green">
<text color="green">**下面两篇是本 OnePage 的子文档。**</text>
<text color="gray">第一篇记录实验目标、实现和结果复盘；第二篇从源码角度剖析 Simple Agent Lab 和 OpenClaw 适配层。</text>
</callout>

- [实验复盘：Simple Agent Lab × OpenClaw/ClawBench 集成实验复盘](https://www.feishu.cn/wiki/BebXwX1NyirTkNkt9KmcKZMdnMe)
- [代码剖析：从 Runtime 到 OpenClaw Adapter](https://www.feishu.cn/wiki/Wk7QwPJXtiFRIDkNBWfcKRhhnSg)

---

# <text color="blue">六、下一步</text>

1. 跑完 ClawBench 303 全量，形成正式分数。
2. 修复 run summary/cache 口径，避免 sample 覆盖 full run。
3. 把 AzureOpenAI adapter 下沉到 SAL provider 层。
4. 增强 browser/search/spreadsheet/document/image 等工具能力。
5. 给失败任务做 workspace snapshot 和 verifier 失败归因。

<callout emoji="sparkles" background-color="light-gray" border-color="light-gray">
<text color="gray">**一句话：Simple Agent Lab 已经从“可解释的小 runtime”推进到“可接 benchmark 的实验框架”，下一步要从能跑升级为跑得稳、分数可信、失败可诊断。**</text>
</callout>
