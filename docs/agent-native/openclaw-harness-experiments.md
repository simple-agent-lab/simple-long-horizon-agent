# OpenClaw Harness 实验计划：把 simple-agent-lab 从跑分器变成诊断器

<callout emoji="🎯" background-color="light-blue" border-color="light-blue">
<text color="blue">**一句话目标**</text>
<text color="gray">这组实验不是再做一个“哪个模型分数更高”的 leaderboard，而是研究：在同一个 simple-agent-lab / OpenClaw benchmark 生态下，改动 harness 的不同维度，会分别修复哪些 agent 失败类型，又会破坏哪些原本能做对的任务。</text>
</callout>

本文档面向当前 exp2 分支：`dev/claw`。

当前代码已经把 OpenClaw 相关 benchmark 接入 simple-agent-lab，并能通过 `run_benches.py` 启动多个 suite。下一步应该把这套集成从“能跑 benchmark”升级为“能解释 harness 机制的实验平台”。

---

# <text color="blue">1. 为什么这里要研究 Harness</text>

`clawRecipe` 的核心不是“发明一堆 prompt trick”，而是把 agent 失败看成一个控制环里的断点，然后用 frozen recipe 去验证：某种结构性改动是不是更容易修复某类断点。

这和我们现在的 simple-agent-lab 非常契合。因为 simple-agent-lab 的设计本来就把 agent runtime 拆成了几个清晰边界：

- `Suite`：benchmark 的 host 半边，负责 load instance、launch spec、eval input。
- `Backend`：process / docker 执行环境。
- `Container module`：benchmark 的 container 半边，负责 build task、prepare workspace、extract result、evaluate。
- `ArtifactStore`：统一存放 `input/instance.json`、`input/eval.json`、`out/result.json`、`out/trajectory.jsonl`。
- `AgentSpec / build_agent`：定义 agent prompt、role、tool、flavor。

这意味着我们可以只改 harness 层，而不改 benchmark 题目本身。这样实验才干净。

<quote-container>
**实验对象应该是 `model + harness + benchmark + budget` 这个配置，而不是模型本身。**
</quote-container>

这个观点也和 Harness-Bench 的结论一致：agent workflow 的表现不只取决于 base model，还取决于 context、tools、state、constraints、permissions、tracing、recovery 等执行层配置。Harness-Bench 主张把能力报告在 model-harness configuration 层级，而不是归因给模型单体。

---

# <text color="blue">2. 从 clawRecipe 抽出的实验哲学</text>

## 2.1 不只看总分，要看“修复了哪类失败”

聚合分数回答的是：

```text
这个 agent 总体做对了多少？
```

但 harness 实验真正要回答的是：

```text
这个 harness 改动修好了什么？代价是什么？它有没有把原本做对的任务弄坏？
```

所以主指标不应该只有平均分，而应该至少包括：

- `CRR_k`：Category-conditional Repair Rate，某个 failure label `k` 下的 baseline 失败，有多少被该 harness variant 修复。
- `NIR`：Negative Interference Rate，baseline 原本通过的任务，有多少被该 variant 弄失败。
- `net_gain`：`CRR - NIR`。
- `cost`：token、tool call、wall time。
- `liveness`：是否完成、是否 timeout、是否因环境失败。
- `process_quality`：是否读了关键文件、是否验证输出、是否重复无效 action、是否正确响应 tool error。

## 2.2 failure label 是诊断假设，不是绝对因果

`clawRecipe` 的 taxonomy 不是声称 A/B/C/D/E 完全互斥。长程 agent 失败经常 cascade：一开始误解目标，后面可能表现成错误 tool 参数、错误文件路径、验证失败。

因此标注规则应该采用：

```text
earliest actionable breakpoint
```

也就是：标最早出现、并且如果修掉就可能防止后续错误的那个断点。

## 2.3 干预必须 frozen，比较必须 paired

要避免“看了结果再改 prompt”的实验污染：

- 每个 harness variant 的 prompt / tool wrapper / verifier / state format 必须先冻结。
- 所有 variant 跑同一批 task id。
- 同一个 benchmark、model、temperature、max turns、backend、budget 都要固定。
- 之后只比较同一 task 上 BL vs variant 的结果。

这也是为什么要用 paired CRR / NIR，而不是只看两个平均分。

## 2.4 需要 negative control

`clawRecipe` 里 T3b Commander-Executor 是一个非常重要的 negative control：它看起来更“结构化”，但在 ClawBench / ClawEval 上大幅掉分。这说明多 agent / 多角色不是天然更好，错误的分工会放大目标分解错误。

simple-agent-lab 里也应该保留一个类似 negative control：

```text
H7 Commander-Executor：预先分解 + 执行，但不做验证
```

如果 H7 也涨分，说明任务可能只是需要更多结构化 prompt；如果 H7 掉分而 verifier variant 涨分，说明关键不是“多角色”，而是“验证闭环”。

## 2.5 小差异不要过度解释

`clawRecipe` 的失败分析里有一个很现实的观察：同一 endpoint 的 PinchBench 分数跨度可以从 0.058 到 0.847，std=0.251。也就是说 agent benchmark 有很强运行噪声，特别是 API endpoint、模型服务、工具环境、初始化污染都会引入方差。

所以实验报告要遵守：

- 小于 5pp 的差异默认不解释，除非有重复运行和置信区间支持。
- 小样本 category，比如 C/D/E，要明确写“方向性证据”而不是“显著结论”。
- 任务数少的 bench，比如 PinchBench，最好重复运行或 bootstrap。

---

# <text color="blue">3. clawRecipe 的 Failure Taxonomy 如何映射到 Harness</text>

<lark-table>
  <lark-tr>
    <lark-td>**Label**</lark-td>
    <lark-td>**含义**</lark-td>
    <lark-td>**simple-agent-lab 里能观测到的迹象**</lark-td>
    <lark-td>**优先尝试的 harness 改动**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>**A Goal Grounding**</lark-td>
    <lark-td>目标定位失败：误解任务、忽略约束、执行中目标漂移。</lark-td>
    <lark-td>没有读题目要求的关键文件；创建无关文件；最终输出和任务 schema 不一致；中途改做别的事。</lark-td>
    <lark-td>Plan-first、success criteria、bounded replanning、任务前 contract extraction。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>**B Action Instantiation**</lark-td>
    <lark-td>动作实例化失败：知道要做什么，但 tool、参数、路径、格式映射错。</lark-td>
    <lark-td>路径错、参数错、没先读 config、写了错误文件名、格式差一层。</lark-td>
    <lark-td>Preflight check、tool schema grounding、path/input/output guard。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>**C State Representation**</lark-td>
    <lark-td>状态表示失败：中间事实、已完成步骤、待办、artifact 路径没有稳定维护。</lark-td>
    <lark-td>重复读同一信息；忘记刚生成的文件；多步任务后半段丢约束；前后结论矛盾。</lark-td>
    <lark-td>Structured state tracking、task-local memory、artifact registry。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>**D Outcome Verification**</lark-td>
    <lark-td>结果验证失败：不检查输出、不诊断 tool error、不恢复。</lark-td>
    <lark-td>pytest/verifier 明显失败但 agent 提前结束；tool 返回 error 后继续胡走；没有检查目标文件存在。</lark-td>
    <lark-td>Verifier pass、self-check hook、failure-triggered reflection、rollback/retry。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>**E Skill Activation / Composition**</lark-td>
    <lark-td>技能激活/组合失败：有工具或技能，但没选对、没按正确顺序组合。</lark-td>
    <lark-td>没有加载相关 SKILL.md；工具链顺序错；明明有 read/bash/skill 却绕路。</lark-td>
    <lark-td>Procedural support cards、skill retrieval、tool affordance hints。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>**F External / Infrastructure**</lark-td>
    <lark-td>外部/基础设施失败：依赖、权限、sandbox、 evaluator、网络、工具缺失。</lark-td>
    <lark-td>缺 pandas、缺 PDF/图像工具、Docker image 不存在、verifier 依赖失败。</lark-td>
    <lark-td>Docker base image、dependency lock、tool availability contract、infra label。</lark-td>
  </lark-tr>
</lark-table>

---

# <text color="blue">4. clawRecipe 的 Recipe 结果对我们的启发</text>

`clawRecipe` 里几个关键结果，应该直接进入我们这组实验的 prior。

<lark-table>
  <lark-tr>
    <lark-td>**Recipe**</lark-td>
    <lark-td>**机制**</lark-td>
    <lark-td>**主要目标 failure**</lark-td>
    <lark-td>**论文结果里的信号**</lark-td>
    <lark-td>**迁移到 simple-agent-lab 的做法**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>**T1 Control**</lark-td>
    <lark-td>Plan-first、preflight、bounded replanning、error reflection。</lark-td>
    <lark-td>A/B，兼顾 D。</lark-td>
    <lark-td>A 类 CRR 50.0%，B 类 CRR 62.5%；整体 single recipe 最强，CRR 48.7%。</lark-td>
    <lark-td>做成 H1/H2，先用 prompt 实现，再考虑工具前 hook。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>**T2 Memory**</lark-td>
    <lark-td>任务内 episodic memory，记录事实、路径、结论、未解决问题。</lark-td>
    <lark-td>C。</lark-td>
    <lark-td>C 类 CRR 71.4%，但 C 样本很小，需谨慎。</lark-td>
    <lark-td>做成 `state.md` 或 `memory.json`，每轮更新。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>**T2s Structured State**</lark-td>
    <lark-td>显式 slots：constraints、derived_facts、pending_subgoals、artifact_paths。</lark-td>
    <lark-td>C，也能帮助 B/D。</lark-td>
    <lark-td>C 类 CRR 71.4%，D 类也有 50.0%，说明结构化状态会跨类别迁移。</lark-td>
    <lark-td>优先做 H3，因为 simple-agent-lab 很适合把状态作为 artifact。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>**T3 Executor-Verifier**</lark-td>
    <lark-td>执行器做动作，验证器检查中间产物、目标、约束、格式。</lark-td>
    <lark-td>D。</lark-td>
    <lark-td>D 类 CRR 68.8%，是 D 类最强单 recipe。</lark-td>
    <lark-td>做成 H4 post-run verifier，后续再做 key-node verifier。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>**T3b Commander-Executor**</lark-td>
    <lark-td>先分解目标，再派执行器做子任务，不强调验证。</lark-td>
    <lark-td>negative control。</lark-td>
    <lark-td>在 ClawBench / ClawEval 明显掉分，说明“多角色”不等于“更好”。</lark-td>
    <lark-td>保留 H7，验证我们的 harness 实验是不是能识别坏结构。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>**T4 Procedural Cards**</lark-td>
    <lark-td>根据任务检索支持卡片，包含 goal、prerequisite、steps、pitfalls。</lark-td>
    <lark-td>E。</lark-td>
    <lark-td>E 类 CRR 52.2%，但 NIR 偏高，说明卡片会扰动原本正确路径。</lark-td>
    <lark-td>做成 H5，先 top-1/top-3 ablation，避免 context crowding。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>**T5 Combo**</lark-td>
    <lark-td>T2s + T1 + T3。</lark-td>
    <lark-td>A/B/C/D 结构性失败。</lark-td>
    <lark-td>整体 CRR 52.2%，最高；NIR 11.4%，没有因为组合而成倍干扰。</lark-td>
    <lark-td>做成 H6，但必须和 H1/H3/H4 分开跑，才能知道组合收益来自哪里。</lark-td>
  </lark-tr>
</lark-table>

---

# <text color="blue">5. 当前 exp2 OpenClaw 集成状态</text>

当前 `dev/claw` 已经接入这些 bench：

<lark-table>
  <lark-tr>
    <lark-td>**Bench**</lark-td>
    <lark-td>**任务规模**</lark-td>
    <lark-td>**当前 scoring 状态**</lark-td>
    <lark-td>**能否进入 CRR/NIR**</lark-td>
    <lark-td>**备注**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`clawbench_tribe`</lark-td>
    <lark-td>8 hardcoded tasks</lark-td>
    <lark-td>host-side trajectory check。</lark-td>
    <lark-td>可以。</lark-td>
    <lark-td>适合作 fast sanity，不代表复杂工具任务。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`pinchbench`</lark-td>
    <lark-td>23 markdown tasks</lark-td>
    <lark-td>in-run `grade()`，已修复 trajectory 输入。</lark-td>
    <lark-td>可以。</lark-td>
    <lark-td>任务少但非常适合看 trace-sensitive 机制。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`clawbench_official`</lark-td>
    <lark-td>303 task.toml</lark-td>
    <lark-td>pytest/verifier score。</lark-td>
    <lark-td>可以。</lark-td>
    <lark-td>适合 artifact-based scoring。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`skillsbench`</lark-td>
    <lark-td>88 task.toml</lark-td>
    <lark-td>pytest/verifier score。</lark-td>
    <lark-td>可以。</lark-td>
    <lark-td>适合研究 skill activation / procedural support。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`agentbench`</lark-td>
    <lark-td>39 task.yaml</lark-td>
    <lark-td>只有 Layer-0 structural score。</lark-td>
    <lark-td>可以，但标 partial。</lark-td>
    <lark-td>不要把 Layer-0 当完整 agentbench 成绩。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`zclawbench`</lark-td>
    <lark-td>116 tasks</lark-td>
    <lark-td>缺 host-side judge，目前 `scoring_pending`。</lark-td>
    <lark-td>暂不进入 CRR。</lark-td>
    <lark-td>先用于 trace collection。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`claweval`</lark-td>
    <lark-td>300 task.yaml</lark-td>
    <lark-td>缺 user-agent / LLM judge，目前 `scoring_pending`。</lark-td>
    <lark-td>暂不进入 CRR。</lark-td>
    <lark-td>必须补 judge 后再做正式结论。</lark-td>
  </lark-tr>
</lark-table>

---

# <text color="blue">6. Harness Variant 矩阵</text>

下面是建议第一轮实现的 harness profiles。它们应该放在 runner 侧统一注册，而不是写进每个 benchmark suite。

```text
run_benches.py --harness baseline
run_benches.py --harness control
run_benches.py --harness state
run_benches.py --harness verifier
...
```

<lark-table>
  <lark-tr>
    <lark-td>**ID**</lark-td>
    <lark-td>**名称**</lark-td>
    <lark-td>**改动维度**</lark-td>
    <lark-td>**目标 label**</lark-td>
    <lark-td>**最小实现**</lark-td>
    <lark-td>**预期风险**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>H0</lark-td>
    <lark-td>Baseline</lark-td>
    <lark-td>当前 agent spec。</lark-td>
    <lark-td>对照组。</lark-td>
    <lark-td>不改 prompt/tool/backend。</lark-td>
    <lark-td>无。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>H1</lark-td>
    <lark-td>Plan + Success Criteria</lark-td>
    <lark-td>控制环前置。</lark-td>
    <lark-td>A。</lark-td>
    <lark-td>system prompt 要求先写 plan、constraints、success criteria，再 action。</lark-td>
    <lark-td>增加 token / 可能过度规划。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>H2</lark-td>
    <lark-td>Preflight Check</lark-td>
    <lark-td>tool call 前检查。</lark-td>
    <lark-td>B。</lark-td>
    <lark-td>prompt 版先做：每次 bash 前确认路径、输入、输出、格式。</lark-td>
    <lark-td>慢；可能反复确认。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>H3</lark-td>
    <lark-td>Structured State</lark-td>
    <lark-td>状态表示。</lark-td>
    <lark-td>C。</lark-td>
    <lark-td>workspace 下维护 `state.md` 或 `state.json`：constraints、facts、artifacts、pending。</lark-td>
    <lark-td>状态写入错误会污染后续步骤。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>H4</lark-td>
    <lark-td>Verifier Pass</lark-td>
    <lark-td>结果验证。</lark-td>
    <lark-td>D。</lark-td>
    <lark-td>final 前或 final 后跑一个 verifier agent/checklist，要求修正缺失文件/格式。</lark-td>
    <lark-td>成本高；verifier 自己可能误判。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>H5</lark-td>
    <lark-td>Procedural Cards</lark-td>
    <lark-td>技能激活。</lark-td>
    <lark-td>E。</lark-td>
    <lark-td>根据 task text 检索 top-k cards，注入 steps/pitfalls。</lark-td>
    <lark-td>context crowding；NIR 可能升高。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>H6</lark-td>
    <lark-td>Combo</lark-td>
    <lark-td>H1 + H3 + H4。</lark-td>
    <lark-td>A/B/C/D。</lark-td>
    <lark-td>组合结构性机制，不含 H5。</lark-td>
    <lark-td>上下文膨胀、角色冲突。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>H7</lark-td>
    <lark-td>Commander-Executor</lark-td>
    <lark-td>负对照。</lark-td>
    <lark-td>验证结构性干预是否真有效。</lark-td>
    <lark-td>只做 pre-hoc decomposition，不做 verifier。</lark-td>
    <lark-td>可能像 clawRecipe T3b 一样大幅掉分。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>H8</lark-td>
    <lark-td>Strict Sandbox</lark-td>
    <lark-td>执行环境。</lark-td>
    <lark-td>F / reproducibility。</lark-td>
    <lark-td>Docker backend、clean workspace、no network、dependency lock。</lark-td>
    <lark-td>会暴露更多依赖缺失，短期分数可能下降。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>H9</lark-td>
    <lark-td>Budget / Step Ablation</lark-td>
    <lark-td>预算。</lark-td>
    <lark-td>A/D/F。</lark-td>
    <lark-td>max-turns = 3/10/20/40；记录 pass 和 cost curve。</lark-td>
    <lark-td>不是单一机制，主要用于成本边界。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>H10</lark-td>
    <lark-td>Trace Visibility</lark-td>
    <lark-td>可观测性。</lark-td>
    <lark-td>D/F。</lark-td>
    <lark-td>把 tool feedback、artifact list、recent errors 用结构化摘要注入上下文。</lark-td>
    <lark-td>摘要错误会误导 agent。</lark-td>
  </lark-tr>
</lark-table>

---

# <text color="blue">7. 实验阶段设计</text>

## 7.1 Phase 0：只验证 runner 和 scoring

目标：确认所有 bench 都能跑完，并且 summary 可信。

当前已经完成一次 1-task smoke：

<lark-table>
  <lark-tr>
    <lark-td>**Bench**</lark-td>
    <lark-td>**Smoke 结果**</lark-td>
    <lark-td>**解释**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`clawbench_tribe`</lark-td>
    <lark-td>1/1，score 100。</lark-td>
    <lark-td>host trajectory check 正常。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`pinchbench`</lark-td>
    <lark-td>1/1，score 100。</lark-td>
    <lark-td>trajectory context 修复有效。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`clawbench_official`</lark-td>
    <lark-td>跑通，score 0。</lark-td>
    <lark-td>分数来自 verifier，任务没做对但 scoring 成功。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`skillsbench`</lark-td>
    <lark-td>跑通，score 0。</lark-td>
    <lark-td>分数来自 verifier。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`agentbench`</lark-td>
    <lark-td>跑通，Layer-0 score 0。</lark-td>
    <lark-td>只代表 structural validators。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`zclawbench`</lark-td>
    <lark-td>跑通，`scoring_pending`。</lark-td>
    <lark-td>缺 host judge。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`claweval`</lark-td>
    <lark-td>跑通，`scoring_pending`。</lark-td>
    <lark-td>缺 user-agent / LLM judge。</lark-td>
  </lark-tr>
</lark-table>

## 7.2 Phase 1：小样本诊断集

目标：便宜地找到 harness variant 是否有明显方向。

建议：

- `clawbench_tribe`：全部 8 题。
- `pinchbench`：全部 23 题。
- `clawbench_official`：抽 30 题，覆盖不同 domain。
- `skillsbench`：抽 20 题，优先覆盖工具/skill-heavy 任务。
- `agentbench`：抽 20 题，记录 Layer-0 partial score。
- `zclawbench / claweval`：只收 trace，不进 CRR。

每个 task 至少跑：

```text
H0 baseline
H1 control
H3 structured state
H4 verifier
H6 combo
H7 negative control
```

## 7.3 Phase 2：failure-labeled CRR 实验

目标：把 baseline 失败先标注，再看 variant 是否修复。

流程：

1. 跑 H0 baseline。
2. 找出 baseline failed tasks。
3. 读取 `trajectory.jsonl` + `result.json` + verifier output。
4. 标 A-F failure label。
5. 跑 H1/H2/H3/H4/H5/H6/H7。
6. 计算 per-label CRR。
7. 在 baseline passed tasks 上计算 NIR。

## 7.4 Phase 3：重复运行与置信区间

目标：处理 API 方差和随机性。

对 Phase 2 中效果明显的 variant：

- 每个 task 重复 3 次。
- 或者在模型服务支持时固定 seed / temperature 0。
- 报告 bootstrap 95% CI。
- 对 pass/fail 用 McNemar test。
- 对连续 score 用 paired bootstrap / Wilcoxon signed-rank。

---

# <text color="blue">8. 指标定义</text>

## 8.1 基础结果字段

每个 run 的 summary 至少应该包含：

```json
{
  "bench": "pinchbench",
  "task_id": "task_00_sanity",
  "model": "gpt-4o-2024-11-20",
  "harness": "control",
  "backend": "process",
  "max_turns": 10,
  "status": "completed",
  "scoring_status": "scored",
  "score_source": "score",
  "score": 100.0,
  "passed": true,
  "token_input": 1234,
  "token_output": 456,
  "tool_calls": 7,
  "wall_time_s": 31.4,
  "failure_label": null
}
```

## 8.2 CRR

对 label `k`：

```text
CRR_k(harness) =
  count(task baseline failed with label k, harness passes)
  /
  count(task baseline failed with label k)
```

注意：

- 只在有正式 score 的 bench 上算。
- `agentbench` 如果只有 Layer-0，则必须标 `partial_score`。
- `zclawbench / claweval` 在补 judge 前不能进入正式 CRR。

## 8.3 NIR

```text
NIR(harness) =
  count(task baseline passed, harness fails)
  /
  count(task baseline passed)
```

NIR 很重要，因为很多 harness 改动会提高失败任务表现，但扰动原本正确任务。`clawRecipe` 里 T4 Procedural Cards 就有这种倾向。

## 8.4 成本与效率

除了分数，还要报告：

- 平均 tool calls。
- 平均 wall time。
- 平均 input/output tokens。
- 每提升 1pp CRR 的成本。
- timeout rate。
- verifier 触发修复次数。

这能防止 H6 combo 虽然涨分，但成本爆炸。

---

# <text color="blue">9. Failure Label 标注协议</text>

标注输入：

```text
task instruction
instance.json
result.json
trajectory.jsonl
workspace file list
verifier / pytest output
```

标注输出：

```json
{
  "bench": "clawbench_official",
  "task_id": "acad-001-citation-network",
  "baseline_score": 0,
  "failure_label": "A",
  "confidence": "medium",
  "earliest_breakpoint": "Agent never converted the required output schema into success criteria.",
  "evidence": [
    "Created temp_parsed_bibtex.csv but not required analysis.json.",
    "Verifier failed all 5 tests.",
    "Trajectory shows no final schema check."
  ],
  "possible_secondary_labels": ["D"]
}
```

## 9.1 决策树

1. 如果是 dependency、Docker、权限、工具缺失、verifier 异常：标 F。
2. 如果 agent 从一开始就误解目标、忽略关键约束、做了无关任务：标 A。
3. 如果目标理解正确，但 tool/path/参数/输出格式映射错：标 B。
4. 如果多步过程中忘记之前事实、文件、约束、进度：标 C。
5. 如果已经有输出或 tool feedback，但没有检查、没有修复、没有 recovery：标 D。
6. 如果工具/技能存在，但没激活、没按正确顺序组合：标 E。

## 9.2 边界例子

- 错路径是因为误解 workspace 结构：A。
- 错路径但目标和文件名都理解了，只是参数写错：B。
- 忘记前面已经计算出的值：C。
- 生成了文件但没跑检查，格式错就结束：D。
- 有 PDF/read skill 但没加载，或者 tool chain 顺序错：E。
- 没有 PDF 工具或 pandas 缺失：F。

---

# <text color="blue">10. 代码实现建议</text>

## 10.1 加 `--harness`

在 `run_benches.py` 增加：

```bash
uv run python run_benches.py \
  --bench pinchbench \
  --model gpt-4o-2024-11-20 \
  --sample 3 \
  --max-turns 10 \
  --backend process \
  --harness control \
  --run-root runs_harness
```

## 10.2 建 harness registry

建议新增：

```text
src/simple_agent_lab/evals/harness_profiles.py
```

提供：

```python
@dataclass(frozen=True)
class HarnessProfile:
    name: str
    system_prompt_suffix: str = ""
    state_tracking: bool = False
    verifier: bool = False
    procedural_cards: bool = False
    negative_control: bool = False
```

runner 根据 profile patch agent spec，而不是改 benchmark suite。

## 10.3 不要先大改 agent core

第一版先做 prompt / context 注入：

- H1/H2 可以只改 system prompt。
- H3 可以要求 agent 维护 `state.md`。
- H4 可以先做 final verifier，不做每步 verifier。
- H5 可以用手写 card registry，不急着上 embedding retrieval。

等 small sweep 有信号，再做 tool wrapper / post-turn hook。

## 10.4 summary schema

`summary.json` 应该扩展：

```json
{
  "bench": "pinchbench",
  "model": "gpt-4o-2024-11-20",
  "harness": "control",
  "backend": "process",
  "sample": 3,
  "passed": 2,
  "total": 3,
  "average_score": 66.7,
  "results": []
}
```

---

# <text color="blue">11. 当前最重要的两个 Judge 缺口</text>

## 11.1 zclawbench

当前 `zclawbench` 只生成 output 文件并标 completed，但没有 judge。它可以用于收 trace，但不能进入正式 CRR。

优先级：

1. 找到原 benchmark 的 prompt / rubric / expected outcome。
2. 如果没有 deterministic verifier，用 host-side LLM judge。
3. Judge 输出必须结构化：score、passed、rationale、rubric components。
4. Judge prompt 也要 frozen。

## 11.2 claweval

`claweval` 更复杂，因为它包含 user-agent simulation / tool-use scenario / 多轮任务。不能只靠最终文本 judge。

优先级：

1. 补 user simulator 或 mock service 启动。
2. 把 scoring_components / judge_rubric 接入 host-side judge。
3. 对 multi-turn 任务记录 conversation state。
4. 把 final answer、tool trace、workspace artifacts 一起给 judge。

---

# <text color="blue">12. 推荐第一轮执行表</text>

<lark-table>
  <lark-tr>
    <lark-td>**阶段**</lark-td>
    <lark-td>**Bench**</lark-td>
    <lark-td>**任务数**</lark-td>
    <lark-td>**Harness**</lark-td>
    <lark-td>**目的**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>Smoke</lark-td>
    <lark-td>所有 bench</lark-td>
    <lark-td>1 each</lark-td>
    <lark-td>H0</lark-td>
    <lark-td>确认 runner/scoring。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>Mini CRR</lark-td>
    <lark-td>tribe + pinch + official + skills + agentbench</lark-td>
    <lark-td>约 60-80</lark-td>
    <lark-td>H0/H1/H3/H4/H6/H7</lark-td>
    <lark-td>看主要方向和负干扰。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>Focused</lark-td>
    <lark-td>失败高发 domain</lark-td>
    <lark-td>按 label 采样</lark-td>
    <lark-td>H1-H5</lark-td>
    <lark-td>验证 category-conditioned repair。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>Full scored</lark-td>
    <lark-td>已评分 bench</lark-td>
    <lark-td>全量或预算内最大</lark-td>
    <lark-td>最优 2-3 个 harness</lark-td>
    <lark-td>正式报告。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>Judge-complete</lark-td>
    <lark-td>zclawbench + claweval</lark-td>
    <lark-td>补 judge 后</lark-td>
    <lark-td>H0/H1/H4/H6</lark-td>
    <lark-td>扩展到更复杂 user/tool scenarios。</lark-td>
  </lark-tr>
</lark-table>

---

# <text color="blue">13. 和最新 benchmark 论文的结合点</text>

## Harness-Bench

Harness-Bench 明确把 harness 定义为管理 context、tools、state、constraints、permissions、tracing、recovery 的系统层。它强调 model-harness pairing，并记录 artifacts、execution traces、usage statistics、validator outputs。

我们应该直接借这个 reporting style：

```text
不要写 “gpt-4o score = X”
要写 “gpt-4o + simple-agent-lab/H4-verifier + process backend + max_turns=10 score = X”
```

## ToolSandbox

ToolSandbox 强调 stateful tool execution、implicit state dependency、user simulator、intermediate/final milestone evaluation。它提醒我们：只看 final output 不够，很多 tool-use 任务要看中间状态是否正确推进。

对 simple-agent-lab 的启发：

- `trajectory.jsonl` 要成为一等评估对象。
- H3 structured state 和 H4 verifier 不应该只看最终文本。
- claweval 的 user-agent simulation 需要认真补，不然会低估多轮状态问题。

## tau-bench

tau-bench 强调 tool-agent-user interaction 和 end-state evaluation，并提出多次运行的 `pass^k` 用来衡量一致性。

对我们启发：

- 对同一 task 多次运行，不能只报 best/mean。
- 应该报 pass^1 / pass^3 或者 consistency。
- user / policy / tool state 相关任务要按最终状态评分。

## OSWorld / OSWorld-Human

OSWorld 强调真实交互环境和 execution-based evaluation；OSWorld-Human 进一步提醒 agent 的 latency 和 step cost 是核心问题，越到后面步骤可能越慢。

对我们启发：

- H6 combo 如果提高分数但 wall time 翻倍，不一定值得。
- 需要报告 wall-time/token/tool-call frontier。
- Docker backend 的环境可复现性要成为正式变量。

---

# <text color="blue">14. 风险与解释纪律</text>

1. `agentbench` 当前只有 Layer-0，不能宣称完整 AgentBench 成绩。
2. `zclawbench` / `claweval` 当前没有正式 judge，不能进入 CRR。
3. H5 procedural cards 容易把 benchmark-specific 知识泄漏进上下文，必须只用 task text 检索，不看 gold。
4. H4 verifier 如果用同一个强模型，可能把任务变成“第二个 agent 补做”，需要区分 verifier 是检查还是执行。
5. H8 strict sandbox 可能短期降低分数，但这是基础设施问题，不一定是 agent 能力下降。
6. 小样本 category 不要写成确定性机制结论。
7. 所有 prompt / profile 必须在 evaluation 前冻结。

---

# <text color="blue">15. 最小下一步</text>

建议下一步做 4 个小 PR / commit：

1. `--harness` 参数 + `HarnessProfile` registry。
2. H1/H2/H3 prompt-level profiles。
3. failure label 文件格式 + summary aggregation。
4. Mini CRR runner：给定 label file，自动输出 CRR/NIR table。

第一轮不要急着做复杂 verifier 和 embedding cards。先让实验闭环跑起来：

```text
baseline -> label failures -> run variants -> compute CRR/NIR -> inspect traces
```

<callout emoji="gem" background-color="light-green" border-color="light-green">
<text color="green">**最终判断标准**</text>
<text color="gray">如果一个 harness 改动只能提高平均分，但不能解释它修复了哪类失败、代价是什么、是否破坏 baseline pass，那么它还不是一个合格的实验结论。</text>
</callout>

---

# <text color="blue">16. 参考资料</text>

- 本地参考：`BrainHao/🤠 Self/paper/agentic/clawRecipe`
- Harness-Bench: https://arxiv.org/abs/2605.27922
- ToolSandbox: https://machinelearning.apple.com/research/toolsandbox-stateful-conversational-llm-benchmark
- tau-bench: https://arxiv.org/abs/2406.12045
- OSWorld: https://arxiv.org/abs/2404.07972
- OSWorld-Human: https://arxiv.org/abs/2506.16042
