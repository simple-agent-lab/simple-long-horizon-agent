# OpenClaw Benchmark 集成与 20 题子集实验报告

<callout emoji="🎯" background-color="light-blue" border-color="light-blue">
<text color="blue">**一句话结论**</text>
<text color="gray">当前 exp2 / dev/claw 已把 7 组 OpenClaw 相关 benchmark 接入 simple-agent-lab，并跑出了 20 题 spread 子集结果。整体上，runner / artifact / verifier 链路已经打通；但分数仍明显受限于工具栈、judge 完整度、max-turns 预算和部分 benchmark 的原始数据可用性。</text>
</callout>

本文档记录 2026-06-08 当前分支 `dev/claw` 的状态：我怎么把 bench 集成进 simple-agent-lab、怎么跑出结果、结果怎么看、哪些数字可信、哪些数字只能当诊断信号。

<quote-container>
**这次实验不应该被理解成“最终 leaderboard 分数”，更准确地说，它是 simple-agent-lab 接入 OpenClaw benchmark 生态后的第一轮可复盘诊断报告。**
</quote-container>

---

# <text color="blue">1. 当前仓库与分支状态</text>

<lark-table>
  <lark-tr>
    <lark-td>**项目**</lark-td>
    <lark-td>**当前值**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>仓库路径</lark-td>
    <lark-td>`/Users/bytedance/Documents/github/work/exp2_simple_agent_merge_claw_by_glm/simple-agent-lab`</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>分支</lark-td>
    <lark-td>`dev/claw`</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>最近关键提交</lark-td>
    <lark-td>`678b93e Add OpenClaw benchmark adapters`；`1a70810 Improve OpenClaw bench scoring summaries`；`21e3936 Add OpenClaw 20-task subset reporting`；`83664b2 Add post-hoc judges for OpenClaw pending scores`；`85e114f Fix ZClawBench category prompts for subset scoring`</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>主要运行入口</lark-td>
    <lark-td>`run_benches.py`</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>主要结果目录</lark-td>
    <lark-td>`.tmp/openclaw_20_spread_m10/`；ZClaw promptfix 结果在 `.tmp/openclaw_20_spread_m10_zclaw_promptfix/`</lark-td>
  </lark-tr>
</lark-table>

工作区还有一个未跟踪的 `0001_utils` symlink，它用于读取本地 API 配置，不属于这次 benchmark 代码改动。

---

# <text color="blue">2. 集成设计：把 benchmark 当作 Suite，而不是写死在 runner 里</text>

simple-agent-lab 的核心哲学是“小而透明”：agent loop、上下文、工具调用、状态、trace、评测都尽量拆成清楚的边界。OpenClaw benchmark 的集成也沿用了这个哲学。

当前没有把某个 benchmark 的逻辑硬塞进 agent runtime，而是把每个 benchmark 拆成两半：

- `evals/<bench>/suite.py`：host 半边，负责加载 task、整理 agent-visible input、整理 private eval input、声明 launch spec。
- `src/simple_agent_lab/evals/suites/<bench>/container.py`：run 半边，负责构造任务、准备 workspace、运行 agent 后抽取结果、必要时执行 verifier。

中间通过 generic eval runner 串起来：

- `src/simple_agent_lab/evals/runner.py`：统一创建 run directory、写入 `input/instance.json`、写入 `input/eval.json`、启动 backend、收集输出。
- `src/simple_agent_lab/evals/in_container.py`：统一读取 instance、构造 agent、执行 agent loop、写 `trajectory.jsonl` 和 `result.json`，并在 suite 提供 `evaluate()` 时合并评分结果。
- backend 可换：`LocalProcessBackend` 用于本地快速实验，`LocalDockerBackend` 用于更接近 benchmark 环境的隔离运行。

这个结构的好处是：新增 benchmark 时，只需要补一个 suite 和一个 container module；runner 不需要知道具体 benchmark 的领域细节。

---

# <text color="blue">3. 本次接入了哪些 benchmark</text>

<lark-table>
  <lark-tr>
    <lark-td>**Bench**</lark-td>
    <lark-td>**可加载任务数**</lark-td>
    <lark-td>**当前接入方式**</lark-td>
    <lark-td>**评分方式**</lark-td>
    <lark-td>**可信度备注**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`clawbench_tribe`</lark-td>
    <lark-td>8</lark-td>
    <lark-td>轻量 sanity suite，直接从本地 data 读取任务。</lark-td>
    <lark-td>host 侧 trajectory 文本检查。</lark-td>
    <lark-td>适合验证 runner/API/trace 链路，不代表复杂 OpenClaw 任务能力。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`pinchbench`</lark-td>
    <lark-td>23</lark-td>
    <lark-td>读取 PinchBench skill/task markdown，构造任务。</lark-td>
    <lark-td>一部分任务有 deterministic grade；一部分通过任务自带 rubric 做 post-hoc LLM judge。</lark-td>
    <lark-td>比 smoke 更可信，但 rubric judge 不是 leaderboard 官方执行环境。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`clawbench_official`</lark-td>
    <lark-td>303</lark-td>
    <lark-td>加载 `task.toml`、`instruction.md`、environment setup、pytest verifier。</lark-td>
    <lark-td>container 侧运行 pytest verifier，按 passed/total 得分。</lark-td>
    <lark-td>当前最接近正式 deterministic score 的一组。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`skillsbench`</lark-td>
    <lark-td>88</lark-td>
    <lark-td>加载 task.toml、instruction、environment files、pytest tests。</lark-td>
    <lark-td>container 侧运行 pytest。</lark-td>
    <lark-td>分数偏低主要反映工具/依赖/skill 激活不足，不能简单归因给 runner。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`agentbench`</lark-td>
    <lark-td>39</lark-td>
    <lark-td>接入 OpenClaw AgentBench task 数据。</lark-td>
    <lark-td>当前实现 Layer-0 structural validators / partial score。</lark-td>
    <lark-td>不能宣称完整 AgentBench 成绩，只能当结构产物检查。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`claweval`</lark-td>
    <lark-td>300</lark-td>
    <lark-td>加载 task.yaml 中的人设、prompt、reference、judge rubric。</lark-td>
    <lark-td>post-hoc LLM judge，使用 `judge_rubric`、`reference_solution`、`scoring_components`。</lark-td>
    <lark-td>当前还缺 user simulator/mock services，因此比正式 ClawEval 简化很多。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`zclawbench`</lark-td>
    <lark-td>116</lark-td>
    <lark-td>当前数据只有 task_id/category，adapter 用 category-derived prompt 构造任务。</lark-td>
    <lark-td>generic host LLM judge。</lark-td>
    <lark-td>不是官方 zclawbench 分数；只能说明 promptfix 后 agent 能做出具体 deliverable。</lark-td>
  </lark-tr>
</lark-table>

---

# <text color="blue">4. 关键脚本和文件</text>

## 4.1 `run_benches.py`

这是这轮实验的总入口。它做了几件事：

1. 从 `0001_utils/api/.env` 加载模型调用配置。
2. 设置 `OPENAI_MODEL`、`OPENAI_AUTH_TOKEN`、`OPENAI_BASE_URL`，供 simple-agent-lab 的 OpenAI-compatible provider 使用。
3. 根据 `--bench` 选择 suite。
4. 根据 `--sample-strategy` 选择实例。
5. 通过 `LocalProcessBackend` 或 `LocalDockerBackend` 运行每个任务。
6. 汇总每个任务的 `score`、`passed`、`scoring_status`、`score_source`。
7. 写出 `summary.json`。

这次我补了几个对 benchmark 很关键的参数：

```bash
--sample-strategy head|spread|random
--seed 0
--pass-threshold 60
--max-turns 10
```

这里最重要的是 `spread`。很多 benchmark 的 task id 是按 domain 或 slug 排序的，直接取前 N 题会严重偏某几个 domain。`spread` 会在加载后的任务序列上均匀抽样，虽然不是最终严谨分层采样，但已经比 first-N 更适合诊断。

## 4.2 `scripts/judge_openclaw_pending.py`

这个脚本负责给 “runner 已经跑完，但没有 deterministic verifier” 的任务补 post-hoc judge。它有意和 `run_benches.py` 分开，避免把正式 benchmark score 和补救性 judge 混在一起。

当前支持：

- `pinchbench`：使用 task 自带 `llm_judge_rubric` / grading criteria。
- `claweval`：使用 `judge_rubric`、`reference_solution`、`scoring_components`。
- `zclawbench`：因为当前 adapter 没有原始 prompt/rubric，只能使用 generic completion judge。

这个分离很重要：报告里必须区分 `score_source`，否则会把 deterministic verifier 和 generic LLM judge 混为一谈。

## 4.3 Docker 与 executor

老板之前提到的 docker / executor，在 simple-agent-lab 里对应的是：

- `LaunchSpec`：suite 声明要在哪个 image、哪个 workdir 里运行。
- `LocalProcessBackend`：直接在本地 process 跑，速度快，适合开发。
- `LocalDockerBackend`：通过 Docker 隔离运行，适合更接近正式评测。
- `docker/Dockerfile.clawbase`：为 OpenClaw benchmark 准备基础镜像，并显式去掉 OpenClawPro/NanoBotAgent 依赖，避免测试对象被污染。

这符合 simple-agent-lab 的哲学：benchmark 的运行环境由 `Suite + Backend` 数据化表达，agent runtime 本身不写死 Docker 细节。

---

# <text color="blue">5. 实验运行方式</text>

## 5.1 统一配置

本轮主要结果使用同一组配置：

<lark-table>
  <lark-tr>
    <lark-td>**配置项**</lark-td>
    <lark-td>**值**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>模型</lark-td>
    <lark-td>`gpt-4o-2024-11-20`</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>backend</lark-td>
    <lark-td>`process`</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>sample</lark-td>
    <lark-td>每个 bench 20 题；`clawbench_tribe` 只有 8 题，因此跑全量 8 题。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>sample strategy</lark-td>
    <lark-td>`spread`</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>seed</lark-td>
    <lark-td>`0`</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>max turns</lark-td>
    <lark-td>`10`</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>pass threshold</lark-td>
    <lark-td>`60`，仅用于没有 explicit `passed` 字段时判定 pass/fail；平均分不受影响。</lark-td>
  </lark-tr>
</lark-table>

## 5.2 主运行命令

主结果目录是：

```bash
.tmp/openclaw_20_spread_m10/
```

典型命令如下：

```bash
python3 run_benches.py \
  --bench clawbench_official \
  --model gpt-4o-2024-11-20 \
  --backend process \
  --sample 20 \
  --sample-strategy spread \
  --seed 0 \
  --max-turns 10 \
  --pass-threshold 60 \
  --run-root .tmp/openclaw_20_spread_m10/clawbench_official
```

其他 bench 只替换 `--bench` 和 `--run-root`。

## 5.3 post-hoc judge 命令

对需要补 judge 的任务，使用：

```bash
python3 scripts/judge_openclaw_pending.py \
  --bench pinchbench \
  --run-root .tmp/openclaw_20_spread_m10/pinchbench \
  --model gpt-4o-2024-11-20
```

`claweval` 和 `zclawbench` 同理。ZClaw 在第一轮 generic judge 全 0 后，我修了 adapter prompt，让它从“只有 category 的抽象任务”变成“根据 category 直接做一个具体 deliverable，并写入 `output.md`”。修复后的 ZClaw 结果单独放在：

```bash
.tmp/openclaw_20_spread_m10_zclaw_promptfix/zclawbench/
```

---

# <text color="blue">6. 结果总览</text>

<callout emoji="💡" background-color="light-orange" border-color="light-orange">
<text color="red">**重要口径说明**</text>
<text color="gray">下面表格里的 `zclawbench_promptfix` 不是官方 zclawbench 成绩。它使用 category-derived prompt 和 generic judge，只能作为当前 adapter 修复后的诊断信号。更保守的整体分析会单独去掉 ZClaw generic judge。</text>
</callout>

<lark-table>
  <lark-tr>
    <lark-td>**Bench**</lark-td>
    <lark-td>**可用任务**</lark-td>
    <lark-td>**本轮题数**</lark-td>
    <lark-td>**整体均分**</lark-td>
    <lark-td>**0 分数**</lark-td>
    <lark-td>**0 分率**</lark-td>
    <lark-td>**非零题数**</lark-td>
    <lark-td>**去零均分**</lark-td>
    <lark-td>**score >= 60**</lark-td>
    <lark-td>**满分数**</lark-td>
    <lark-td>**评分来源**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`clawbench_tribe`</lark-td>
    <lark-td>8</lark-td>
    <lark-td>8</lark-td>
    <lark-td>100.0000</lark-td>
    <lark-td>0</lark-td>
    <lark-td>0.0%</lark-td>
    <lark-td>8</lark-td>
    <lark-td>100.0000</lark-td>
    <lark-td>8</lark-td>
    <lark-td>8</lark-td>
    <lark-td>host trajectory check</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`pinchbench`</lark-td>
    <lark-td>23</lark-td>
    <lark-td>20</lark-td>
    <lark-td>23.5000</lark-td>
    <lark-td>15</lark-td>
    <lark-td>75.0%</lark-td>
    <lark-td>5</lark-td>
    <lark-td>94.0000</lark-td>
    <lark-td>5</lark-td>
    <lark-td>3</lark-td>
    <lark-td>deterministic grade + rubric LLM judge</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`clawbench_official`</lark-td>
    <lark-td>303</lark-td>
    <lark-td>20</lark-td>
    <lark-td>35.5150</lark-td>
    <lark-td>3</lark-td>
    <lark-td>15.0%</lark-td>
    <lark-td>17</lark-td>
    <lark-td>41.7824</lark-td>
    <lark-td>4</lark-td>
    <lark-td>2</lark-td>
    <lark-td>pytest verifier</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`skillsbench`</lark-td>
    <lark-td>88</lark-td>
    <lark-td>20</lark-td>
    <lark-td>6.9050</lark-td>
    <lark-td>18</lark-td>
    <lark-td>90.0%</lark-td>
    <lark-td>2</lark-td>
    <lark-td>69.0500</lark-td>
    <lark-td>2</lark-td>
    <lark-td>0</lark-td>
    <lark-td>pytest verifier</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`agentbench`</lark-td>
    <lark-td>39</lark-td>
    <lark-td>20</lark-td>
    <lark-td>29.1650</lark-td>
    <lark-td>14</lark-td>
    <lark-td>70.0%</lark-td>
    <lark-td>6</lark-td>
    <lark-td>97.2167</lark-td>
    <lark-td>6</lark-td>
    <lark-td>5</lark-td>
    <lark-td>Layer-0 partial validators</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`claweval`</lark-td>
    <lark-td>300</lark-td>
    <lark-td>20</lark-td>
    <lark-td>7.5000</lark-td>
    <lark-td>16</lark-td>
    <lark-td>80.0%</lark-td>
    <lark-td>4</lark-td>
    <lark-td>37.5000</lark-td>
    <lark-td>1</lark-td>
    <lark-td>0</lark-td>
    <lark-td>rubric LLM judge</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`zclawbench_promptfix`</lark-td>
    <lark-td>116</lark-td>
    <lark-td>20</lark-td>
    <lark-td>95.0000</lark-td>
    <lark-td>1</lark-td>
    <lark-td>5.0%</lark-td>
    <lark-td>19</lark-td>
    <lark-td>100.0000</lark-td>
    <lark-td>19</lark-td>
    <lark-td>19</lark-td>
    <lark-td>generic LLM judge，非官方</lark-td>
  </lark-tr>
</lark-table>

## 6.1 整体汇总口径

<grid cols="2">
  <column width="50">
    <text color="blue">**包含 ZClaw generic judge**</text>
    <text color="gray">共 128 个 scoreable task。整体均分 37.12；0 分 67 个，0 分率 52.34%；非零题 61 个，去零均分 77.90；score >= 60 的任务 49 个。</text>
  </column>
  <column width="50">
    <text color="blue">**保守口径：去掉 ZClaw generic judge**</text>
    <text color="gray">共 108 个 scoreable task。整体均分 26.40；0 分 66 个，0 分率 61.11%；非零题 42 个，去零均分 67.90；score >= 60 的任务 30 个。</text>
  </column>
</grid>

我更建议后续对外汇报时采用保守口径，把 ZClaw promptfix 单独作为 adapter 修复案例，而不是并入正式整体分数。

---

# <text color="blue">7. 各 bench 结果解读</text>

## 7.1 `clawbench_tribe`

这组 8/8 全部 100 分，说明基础运行链路是通的：

- API 配置可用。
- agent loop 可以跑完。
- trajectory 能写出。
- host 侧能从 trajectory 提取最后 assistant 文本并打分。

但它是 sanity suite，任务比较轻，不能说明复杂文件、工具、长链路任务已经做得好。

## 7.2 `pinchbench`

20 题均分 23.5，15 个 0 分，5 个非零；非零题去零均分 94.0。

这个结果的含义是：当前 agent 不是完全不能跑 PinchBench，而是表现很两极。一旦任务落在 simple bash agent 能处理的范围内，分数很高；但遇到日历、邮件、实时信息、复杂工具链、记忆/多步检索时，就容易直接 0 分。

这和 simple-agent-lab 当前工具栈有关：现在主要是 bash/read 类能力，不具备完整 OpenClaw app/tool ecosystem。PinchBench 原本很多任务是在 OpenClaw agent 生态里测真实工具使用，简单本地 workspace agent 会天然吃亏。

## 7.3 `clawbench_official`

20 题均分 35.515，只有 3 个 0 分，17 个非零；非零题均分 41.7824。

这组是当前最值得认真看的 deterministic benchmark。它的信号是：

- adapter 能加载 303 题。
- instruction/environment/verifier 链路基本打通。
- 大多数任务至少能拿到部分 pytest 分。
- 但拿到高分的任务不多，score >= 60 的只有 4 个。

这说明当前 agent 具备一定 workspace artifact 生成能力，但缺乏稳定的 preflight、验证、错误恢复和多步状态管理。很多任务不是“完全没做”，而是只做对了一部分输出或格式。

## 7.4 `skillsbench`

20 题均分 6.905，18 个 0 分，2 个非零；非零题均分 69.05。

这是本轮最低的一组。原因很可能不是单一的：

- 任务常需要特定依赖、数据处理库、音视频/PDF/Office/可视化等工具链。
- 当前 process backend 不一定提供任务原始 Docker environment 的完整依赖。
- agent 没有系统化的 skill activation / procedural cards。
- max-turns=10 对重工程任务仍然偏小。

所以 skillsbench 的低分更像 “当前 harness/tooling 不够” 的信号。后续如果要提升它，应优先做 Docker image、依赖锁定、skill retrieval、procedural cards，而不是只调 prompt。

## 7.5 `agentbench`

20 题均分 29.165，14 个 0 分，6 个非零；非零题均分 97.2167。

它和 PinchBench 一样呈现两极化：能过的题基本很高，不能过的题直接 0。当前实现是 Layer-0 partial validators，因此更像检查：

- 文件有没有创建。
- 结构是否符合预期。
- 某些简单字段是否存在。

它不能代表完整 AgentBench 成绩，尤其对 memory、research、多步工作流这种题，当前 simple-agent-lab 还缺稳定的 state/memory 机制。

## 7.6 `claweval`

20 题均分 7.5，16 个 0 分，4 个非零；非零题均分 37.5。

ClawEval 本身更接近真实用户交互任务，很多题需要：

- user simulator
- mock services
- 多轮状态
- 情绪/意图理解
- reference/rubric judge

当前只是把 task.yaml 的 prompt/rubric/reference 接进来，用 post-hoc LLM judge 看最终 trace。这个版本能提供诊断信号，但远不是完整 ClawEval harness。

因此 Claweval 低分不能简单理解成模型不会，而应理解成当前 harness 还没有复原正式交互环境。

## 7.7 `zclawbench_promptfix`

promptfix 后 20 题均分 95.0，只有 1 个 0 分，19 个满分。

这个数字表面很好，但必须非常谨慎。当前 zclawbench adapter 只能读到 task_id/category，没有原始具体 prompt、rubric、expected outcome。第一轮任务太抽象时，agent 经常反问用户要 topic，generic judge 给 0。之后我修了 container prompt，让 agent 根据 category 直接完成一个具体 deliverable 并保存 `output.md`，所以分数大幅上升。

因此这 95.0 的真实含义是：

```text
当前 ZClaw adapter prompt 已经不会再让 agent 卡在“请提供具体题目”。
```

它不意味着：

```text
simple-agent-lab 在 zclawbench 官方任务上达到了 95 分。
```

后续必须找回 zclawbench 原始 task prompt/rubric，才能进入正式比较。

---

# <text color="blue">8. 为什么之前会有很多 0 分</text>

这轮结果解释了之前 smoke run 里“很多 0 分”的来源。主要有 6 类原因。

## 8.1 1 题 smoke 样本太小

每个 bench 只跑 1 题时，抽到的题如果刚好是难题或依赖缺失题，就会把整个 bench 看成 0。20 题 spread 后可以看到，official/agentbench/pinchbench 都有非零任务。

## 8.2 first-N 采样有 domain 偏置

很多 task 是按目录或 id 排序的。前 20 题不一定代表整个 bench，而可能集中在某个 domain。`spread` 至少让样本覆盖更广。

## 8.3 max-turns 太低

之前 3 turns 经常只够读题、做一点准备、写半个文件。当前 max-turns=10 后，official 和 agentbench 的非零结果明显更丰富。但 clawRecipe 风格的长程 agent 实验通常需要更大 step budget，10 turns 仍偏诊断性质。

## 8.4 pending score 被看成 0

`zclawbench` 和 `claweval` 初始只能写出 result/trajectory，没有 host-side judge。如果 summary 不区分 `scoring_pending`，就很容易误解成 agent 真正得 0。现在 `summarize_result()` 已经把 `scoring_status` 和 `score_source` 写清楚。

## 8.5 部分 benchmark 需要完整工具生态

PinchBench、SkillsBench、ClawEval 很多任务不是“只靠 bash 生成文本”就能完成，而是需要 browser/search/email/calendar/spreadsheet/document/image/PDF/audio/video/mock service 等工具。当前 simple-agent-lab 的 baseline agent 还很轻。

## 8.6 ZClaw 当前数据缺原始任务细节

ZClaw 的 `task_id/category` 太粗，如果不构造具体 deliverable，agent 很自然会反问用户要更多信息。promptfix 后能跑出非零，但这也说明它当前测的是 adapter 构造任务能力，不是官方 ZClaw 任务能力。

---

# <text color="blue">9. 和 clawRecipe 哲学的关系</text>

这次接入总体上是沿着 clawRecipe 的哲学走的，但还只是第一阶段。

## 9.1 已经符合的部分

- 不只看总分：报告拆出了 0 分数、0 分率、去零均分、score>=60、满分数。
- 区分评分来源：deterministic verifier、rubric judge、generic judge 分开标。
- 保留 trace：每个任务都有 `trajectory.jsonl`，后续能做 failure attribution。
- 使用 paired-friendly sample：同一批 selected instance id 可以用于后续 harness variant 对比。
- 不把 Docker/executor 写死进 runner：backend 是可替换实验变量。

## 9.2 还没有完全做到的部分

clawRecipe 真正强调的是 “harness 改动修复了哪类 failure”。当前这轮还没有做到完整 CRR/NIR，因为缺：

- baseline failure label：A Goal Grounding、B Action Instantiation、C State Representation、D Outcome Verification、E Skill Activation、F Infrastructure。
- paired harness variants：例如 plan-first、structured state、executor-verifier、procedural cards。
- negative interference：baseline 原本做对的题，variant 是否弄坏。
- 重复运行或 bootstrap CI：小样本差异不能过度解释。

所以这篇报告是下一步 harness 实验的 baseline data，不是 clawRecipe 风格实验的终点。

---

# <text color="blue">10. 当前结论</text>

<grid cols="3">
  <column width="33">
    <text color="blue">**能跑**</text>
    <text color="gray">7 组 OpenClaw 相关 bench 都已经有 suite/container 接入，主 runner 能统一运行、写 summary、保留 trace。</text>
  </column>
  <column width="33">
    <text color="blue">**能评分**</text>
    <text color="gray">official/skills/agentbench 有 deterministic 或 partial verifier；pinch/claweval 有 rubric judge；zclaw 有临时 generic judge。</text>
  </column>
  <column width="33">
    <text color="blue">**还不够正式**</text>
    <text color="gray">工具生态、Docker 环境、ClawEval simulator、ZClaw 原始 rubric、failure labels 仍需补齐。</text>
  </column>
</grid>

如果只看保守口径，当前 108 个 scoreable task 的整体均分是 26.40，0 分率 61.11%，去零均分 67.90。这说明当前 baseline agent 的问题不是“所有任务都做不了”，而是“成功覆盖面窄，但做对的任务质量并不差”。

换句话说，后续最有价值的方向不是盲目调模型，而是做 harness 实验：

- 加 strong preflight 和 success criteria，解决 A/B 类失败。
- 加 structured state / artifact registry，解决 C 类失败。
- 加 verifier loop 和 error recovery，解决 D 类失败。
- 加 procedural cards / skill retrieval，解决 E 类失败。
- 加 Docker base image 和依赖契约，隔离 F 类 infrastructure 失败。

---

# <text color="blue">11. 下一步建议</text>

1. 固定这批 `selected_instances`，作为 harness 实验 baseline。
2. 从 `clawbench_official`、`agentbench`、`pinchbench` 中抽取失败样本，先做人工 failure label。
3. 实现 H1 Plan-first / Preflight、H3 Structured State、H4 Executor-Verifier 三个最优先 variant。
4. 每个 variant 跑同一批 task，计算 CRR、NIR、net_gain、cost。
5. 对 SkillsBench 单独补 Docker/依赖/skill activation，否则它会把 harness 问题和 infra 问题混在一起。
6. 对 Claweval 补 user simulator/mock services；对 ZClaw 找回原始 prompt/rubric。
7. 报告时同时展示 macro bench score、micro task score、0 分率、去零均分，避免只用一个平均分掩盖失败结构。

<callout emoji="sparkles" background-color="light-gray" border-color="gray">
<text color="gray">**最终判断：当前 exp2 的 OpenClaw bench 集成已经从“能不能跑”进入“能不能诊断”的阶段。下一步要把分数表变成 failure taxonomy + harness variant 的 paired 实验表，这才真正贴近 clawRecipe 的实验哲学。**</text>
</callout>
