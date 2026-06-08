# OpenClaw 20 题子集实验说明：为什么之前很多 0 分，以及现在怎么看结果

<callout emoji="🎯" background-color="light-blue" border-color="light-blue">
<text color="blue">**结论先行**</text>
<text color="gray">之前 1 题 smoke 出现很多 0，不应该被当成 benchmark 结论。主要原因是样本太小、first-N 采样偏置、max-turns 太低，以及部分 bench 根本还没有正式 judge。这次已经按 spread strategy 跑了每个 bench 的 20 题子集（tribe 只有 8 题），结果显示：official / agentbench / pinchbench 都有非零分，skillsbench 仍偏低但不是全 0，zclawbench 和 claweval 是 pending judge，不是模型真实 0 分。</text>
</callout>

本文档对应 repo：

```text
/Users/bytedance/Documents/github/work/exp2_simple_agent_merge_claw_by_glm/simple-agent-lab
branch: dev/claw
```

本轮运行目录：

```text
.tmp/openclaw_20_spread_m10/
```

---

# <text color="blue">1. 这次实验和之前 1 题 smoke 有什么不同</text>

之前我跑的 1 题 smoke 只是在验证：

- runner 能否启动；
- API key / provider env 是否可用；
- `result.json` / `trajectory.jsonl` 是否写出；
- summary 是否能聚合。

它不是一个可以解释 benchmark 表现的实验。

这次改成：

```bash
uv run python run_benches.py \
  --bench <bench> \
  --model gpt-4o-2024-11-20 \
  --sample 20 \
  --sample-strategy spread \
  --seed 0 \
  --max-turns 10 \
  --backend process \
  --run-root .tmp/openclaw_20_spread_m10/<bench>
```

关键变化：

- `sample=20`：每个 bench 尽量跑 20 题。
- `sample-strategy=spread`：从全量任务序列均匀抽样，避免只取前 20 个同一 domain 的任务。
- `max-turns=10`：比之前 smoke 的 3 turns 更接近真实任务预算，但仍低于 clawRecipe 里的 max 40 steps。
- `process backend`：先用本地 process backend，避免 Docker base image 缺失影响本轮结论。

<quote-container>
**1 题 smoke 只能证明“能不能跑”；20 题 spread 子集才开始能讨论“分数为什么这样”。**
</quote-container>

---

# <text color="blue">2. 为什么之前很多 bench 是 0 分</text>

## 2.1 样本太小：1 题不能代表一个 bench

之前每个 bench 只跑 1 题，刚好：

- `clawbench_official` 的第一题是 `acad-001-citation-network`，本轮 20 题里它仍然是 0，但 official 其他 17/20 都有非零分。
- `skillsbench` 的第一题是 `3d-scan-calc`，本轮 20 题里也仍然是 0，但第 12/16 题有非零分。
- `agentbench` 的第一题是 `data-analysis-cross-reference`，本轮仍然是 0，但 agentbench 20 题里有 6 个非零分。

所以之前的 0 更像是“抽到了失败题”，不是 bench 整体 0。

## 2.2 first-N 采样偏置：排序顺序会按 domain 聚集

原 runner 默认：

```python
instances = suite.load_instances()[:args.sample]
```

这会导致：

- `clawbench_official` 前 20 题集中在 academic / accounting / bio / calendar 等开头 domain。
- `skillsbench` 前 20 题集中在按 slug 字母序排列的重任务。
- `agentbench` 前 20 题集中在 data-analysis / error-handling / file-creation。

这不符合 clawRecipe 跨 benchmark / 跨 category 的实验精神。

因此我补了：

```text
--sample-strategy head | spread | random
```

本轮使用 `spread`，均匀覆盖全量任务列表。

## 2.3 max-turns=3 太低

很多 OpenClaw 类任务需要：

```text
读题 -> 查文件 -> 写产物 -> 自检 -> 修正
```

3 turns 经常只能完成前两步，尤其 official / skillsbench / agentbench 这种需要产出文件并跑 verifier 的任务。

本轮改成 `max-turns=10`，已经明显改善 official 和 agentbench 的非零分数。但这仍低于 clawRecipe 使用的 max 40 steps，所以它更像 mini diagnostic subset，不是最终 paper-grade 实验。

## 2.4 scoring_pending 不是 0 分

`zclawbench` 和 `claweval` 当前适配能跑出 `result.json` 和 `trajectory.jsonl`，但缺 host-side judge：

- `zclawbench` 缺 LLM judge / rubric scoring。
- `claweval` 缺 user-agent simulator / mock services / judge_rubric 接入。

所以它们现在显示 average score 0，是因为 runner 没有 judge 分数，不代表模型实际 0。

本轮文档里把它们单独标为：

```text
scoring_pending
```

## 2.5 之前 pass/fail 口径也需要修正

runner 原先逻辑是：

```python
passed = score > 0
```

这会把 official 里 10 分、22 分、28 分的任务都显示成 PASSED。这个和 clawRecipe 的 thresholded pass/fail 不一致。

我已经把 runner 改为支持：

```text
--pass-threshold
```

默认 `100.0`。也就是说：

- `average_score` 仍然保留连续分数。
- `nonzero score` 表示 verifier 有部分通过。
- `strict pass` 表示 score 达到 threshold。

这更接近 clawRecipe 里 CRR/NIR 的 pass/fail 二值化要求。

---

# <text color="blue">3. 本轮 20 题子集总览</text>

<lark-table>
  <lark-tr>
    <lark-td>**Bench**</lark-td>
    <lark-td>**可用任务**</lark-td>
    <lark-td>**本轮任务**</lark-td>
    <lark-td>**Scored**</lark-td>
    <lark-td>**Pending judge**</lark-td>
    <lark-td>**平均分**</lark-td>
    <lark-td>**非零分题数**</lark-td>
    <lark-td>**严格通过(=100)**</lark-td>
    <lark-td>**备注**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`clawbench_tribe`</lark-td>
    <lark-td>8</lark-td>
    <lark-td>8</lark-td>
    <lark-td>8</lark-td>
    <lark-td>0</lark-td>
    <lark-td>100.0</lark-td>
    <lark-td>8</lark-td>
    <lark-td>8</lark-td>
    <lark-td>简单 sanity bench，全部通过。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`pinchbench`</lark-td>
    <lark-td>23</lark-td>
    <lark-td>20</lark-td>
    <lark-td>13</lark-td>
    <lark-td>7</lark-td>
    <lark-td>15.0</lark-td>
    <lark-td>3</lark-td>
    <lark-td>3</lark-td>
    <lark-td>部分任务没有 automated grade code。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`clawbench_official`</lark-td>
    <lark-td>303</lark-td>
    <lark-td>20</lark-td>
    <lark-td>20</lark-td>
    <lark-td>0</lark-td>
    <lark-td>35.5</lark-td>
    <lark-td>17</lark-td>
    <lark-td>2</lark-td>
    <lark-td>最能说明不是全 0，很多题有部分 verifier 通过。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`skillsbench`</lark-td>
    <lark-td>88</lark-td>
    <lark-td>20</lark-td>
    <lark-td>20</lark-td>
    <lark-td>0</lark-td>
    <lark-td>6.9</lark-td>
    <lark-td>2</lark-td>
    <lark-td>0</lark-td>
    <lark-td>仍偏低，任务重且需要 skill/tool/procedure 支持。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`agentbench`</lark-td>
    <lark-td>39</lark-td>
    <lark-td>20</lark-td>
    <lark-td>20</lark-td>
    <lark-td>0</lark-td>
    <lark-td>29.2</lark-td>
    <lark-td>6</lark-td>
    <lark-td>5</lark-td>
    <lark-td>只有 Layer-0 structural score，不是完整 AgentBench。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`zclawbench`</lark-td>
    <lark-td>116</lark-td>
    <lark-td>20</lark-td>
    <lark-td>0</lark-td>
    <lark-td>20</lark-td>
    <lark-td>0.0</lark-td>
    <lark-td>0</lark-td>
    <lark-td>0</lark-td>
    <lark-td>20 题跑通，但缺 judge。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>`claweval`</lark-td>
    <lark-td>300</lark-td>
    <lark-td>20</lark-td>
    <lark-td>0</lark-td>
    <lark-td>20</lark-td>
    <lark-td>0.0</lark-td>
    <lark-td>0</lark-td>
    <lark-td>0</lark-td>
    <lark-td>20 题跑通，但缺 user-agent / LLM judge。</lark-td>
  </lark-tr>
</lark-table>

---

# <text color="blue">4. 每个 bench 的细节解释</text>

## 4.1 clawbench_tribe

结果：

```text
8/8 scored
average_score = 100.0
nonzero = 8
strict pass = 8
```

解释：

- 这是最轻量的 sanity bench。
- 主要用于验证 agent 基础对话、简单推理、JSON 输出、指令遵循。
- 它的高分不能说明复杂 OpenClaw 工具任务也没问题，但说明 API / runner / summary pipeline 正常。

## 4.2 pinchbench

结果：

```text
20 selected
13 scored
7 scoring_pending
average_score = 15.0
nonzero = 3
strict pass = 3
```

3 个 100 分任务：

```text
task_00_sanity
task_01_calendar
task_09_files
```

典型 0 分/低分原因：

- `task_13_image_gen`：需要图像生成能力，当前 runtime 没有生成图像工具，属于 F/E 型混合，prompt 很难修。
- `task_20_eli5_pdf_summary`：PDF 阅读能力不足，并且本轮是 pending judge。
- `task_10_workflow`：典型 B 型，需要先读配置/credential，再生成脚本，适合 H2 preflight。
- `task_08_memory`：C 型，当前 baseline 没有 structured state。

注意：

- 有 7 个任务没有 automated grade code，所以本轮显示 pending。
- 要做正式 CRR，需要给这些任务补 LLM judge 或只在 automated subset 上算。

## 4.3 clawbench_official

结果：

```text
20 selected
20 scored
average_score = 35.5
nonzero = 17
strict pass = 2
```

严格 100 分：

```text
cal-002
mkt-005-market-sizing
```

高/中等非零例子：

```text
file-011: 76.9
med-003-drug-interaction: 66.7
wfl-001: 57.1
cs-004-ci-pipeline: 50.0
data-015: 35.7
```

解释：

- 这组结果最能说明“不是 bench 全 0”。
- 大量任务拿到部分 pytest 通过，说明 agent 产物和目标有一定重叠，但没有完全满足 verifier。
- 这类任务非常适合 clawRecipe 式诊断：A/B/D 都会出现。

下一步最值得做：

- 对 0 分的 `acad-001-citation-network`、`edu-002-item-analysis`、`sci-003-matrix-decomposition` 做 failure label。
- 对 20-50 分的任务看是否是 D 型：已有部分正确产物，但没有 self-check 修完。
- 跑 H1 control / H4 verifier，看 CRR 是否提高。

## 4.4 skillsbench

结果：

```text
20 selected
20 scored
average_score = 6.9
nonzero = 2
strict pass = 0
```

非零任务：

```text
paper-anonymizer: 66.7
shock-analysis-demand: 71.4
```

解释：

- skillsbench 本轮仍偏低，但不是 runner 完全坏。
- 很多任务需要专门工具链、文件格式处理、依赖、甚至多媒体处理。
- 当前 baseline harness 没有 procedural cards，也没有强制 skill activation。
- 这和 clawRecipe 里的 E 型非常吻合：不是“模型不知道英语题目”，而是没有正确激活/组合技能。

典型风险：

- PDF、PPT、XLSX、video/audio、多媒体任务会触发工具缺失或依赖缺失。
- process backend 不是每个 task Dockerfile 的真实环境。
- 如果要对齐 SkillsBench 原始设定，后续应优先用 Docker backend + 对应 image/dependencies。

## 4.5 agentbench

结果：

```text
20 selected
20 scored
average_score = 29.2
nonzero = 6
strict pass = 5
```

严格 100 分：

```text
data-analysis-summary-statistics
error-handling-corrupted-input
file-creation-linked-project-scaffold
file-creation-pitch-deck-outline
research-compare-technologies
```

解释：

- 当前 agentbench 只实现 Layer-0 structural validators。
- 这意味着它能判断“有没有生成目标文件、关键词/字数是否满足”，但不是完整 semantic judge。
- memory 类任务全部 0，这和 clawRecipe 失败分析里提到的 C 型结构性问题一致：baseline 没有稳定 state/memory。

下一步：

- H3 structured state 应优先在 agentbench memory domain 上验证。
- 不要把当前结果写成完整 AgentBench 分数，只能写 partial Layer-0 score。

## 4.6 zclawbench

结果：

```text
20 selected
20 completed
20 scoring_pending
```

解释：

- 当前 container half 只负责 build task / extract result。
- 没有 host-side LLM judge。
- 所以它不是 0 分，而是“待评分”。

下一步：

- 接入 rubric / expected outcome。
- judge 输出结构化：`score`, `passed`, `rationale`, `components`。
- judge prompt 必须 frozen，避免实验后调 prompt。

## 4.7 claweval

结果：

```text
20 selected
20 completed
20 scoring_pending
```

解释：

- ClawEval 包含多轮 user-agent simulation、tool scenario 和 mock services。
- 当前适配没有真正启动 user simulator/mock services，也没有接 judge rubric。
- 所以它现在只能作为 trace collection，不应该进入 CRR/NIR。

下一步：

- 补 user-agent simulator。
- 补 mock service/tool state。
- 接入 scoring_components / judge_rubric。
- final answer + trajectory + artifacts 一起给 judge。

---

# <text color="blue">5. 这次结果如何更接近 clawRecipe</text>

已经更接近的地方：

- 从 1 题 smoke 改成 20 题 spread subset。
- 同一模型、同一 backend、同一 max-turns。
- 把 scored / pending judge 分开。
- 同时报告 average score、nonzero score、strict pass。
- 修正 runner 的 pass threshold，避免 `score > 0` 被误叫 pass。

还没有达到 clawRecipe 的地方：

- 没有 baseline failure labels。
- 没有 H1/H2/H3/H4/H5/H6/H7 harness variants。
- 没有 CRR/NIR。
- 没有重复运行和置信区间。
- zclawbench / claweval 缺正式 judge。
- max-turns=10 低于 clawRecipe 的 max 40 steps。

所以这轮应定位为：

```text
20-task scored baseline diagnostic subset
```

而不是完整 clawRecipe replication。

---

# <text color="blue">6. 下一步怎样让指标更“正常”</text>

## 6.1 不要用 first-N，固定 spread 子集

已经完成 runner 改动：

```text
--sample-strategy spread
--seed 0
```

建议后续报告固定使用这批 selected task ids，直到 full run。

## 6.2 把 max-turns 提到 20 或 40

本轮 max-turns=10，official 已经有 17 个非零分。要更接近 clawRecipe：

```text
max-turns=20: 预算折中
max-turns=40: 更接近 clawRecipe
```

预计：

- official 会更高；
- agentbench 多步任务会更高；
- skillsbench 仍需要工具/skill 支持，否则不会只靠 turns 修好。

## 6.3 先做 H1/H3/H4，而不是立刻全量

最小有效 harness variants：

```text
H1 control: plan + success criteria + preflight
H3 structured state: state.md / state.json
H4 verifier: final self-check / artifact checklist
```

对应 clawRecipe：

- H1 对 A/B；
- H3 对 C；
- H4 对 D；
- H6 = H1 + H3 + H4。

## 6.4 skillsbench 要引入 H5 procedural cards / skill activation

skillsbench 的低分很可能不是单纯 turns 不够，而是：

- 工具链顺序不清楚；
- task-specific procedure 缺失；
- 多媒体/文档/表格工具不可用；
- process backend 和原 Docker env 不一致。

所以对 skillsbench：

- 优先 H5 procedural cards；
- 再尝试 Docker backend；
- 标 F 类 infra failure，避免混入 agent failure。

## 6.5 补 zclawbench / claweval judge

否则这两个 bench 永远只能写：

```text
20 completed, 20 scoring_pending
```

这对“每个 bench 跑出分数”不够。

---

# <text color="blue">7. 建议的下一轮实验</text>

<lark-table>
  <lark-tr>
    <lark-td>**阶段**</lark-td>
    <lark-td>**Bench**</lark-td>
    <lark-td>**任务数**</lark-td>
    <lark-td>**Harness**</lark-td>
    <lark-td>**预算**</lark-td>
    <lark-td>**目标**</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>Baseline-20-v2</lark-td>
    <lark-td>tribe/pinch/official/skills/agentbench</lark-td>
    <lark-td>20 spread</lark-td>
    <lark-td>H0</lark-td>
    <lark-td>max-turns=20</lark-td>
    <lark-td>看 turns 是否能修掉 A/B/D 浅层失败。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>Control</lark-td>
    <lark-td>official + agentbench</lark-td>
    <lark-td>20 spread</lark-td>
    <lark-td>H1</lark-td>
    <lark-td>max-turns=20</lark-td>
    <lark-td>测试 A/B 修复。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>State</lark-td>
    <lark-td>agentbench memory + pinch memory</lark-td>
    <lark-td>定向 subset</lark-td>
    <lark-td>H3</lark-td>
    <lark-td>max-turns=20</lark-td>
    <lark-td>测试 C 型修复。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>Verifier</lark-td>
    <lark-td>official + pinch</lark-td>
    <lark-td>20 spread</lark-td>
    <lark-td>H4</lark-td>
    <lark-td>max-turns=20</lark-td>
    <lark-td>测试 D 型修复。</lark-td>
  </lark-tr>
  <lark-tr>
    <lark-td>Skills</lark-td>
    <lark-td>skillsbench</lark-td>
    <lark-td>20 spread</lark-td>
    <lark-td>H5</lark-td>
    <lark-td>max-turns=20</lark-td>
    <lark-td>测试 E 型 skill activation。</lark-td>
  </lark-tr>
</lark-table>

---

# <text color="blue">8. 当前代码改动</text>

为了支持这次实验，我改了 `run_benches.py`：

```text
新增 --sample-strategy head|spread|random
新增 --seed
新增 --pass-threshold
summary.json 记录 available_instances / selected_instances / sample_strategy / pass_threshold
```

这几个改动的意义：

- `spread` 解决 first-N domain bias。
- `seed` 保证 random subset 可复现。
- `pass-threshold` 让 CRR/NIR 口径能对齐 clawRecipe。
- `selected_instances` 保证以后同一 subset 可以复跑。

---

# <text color="blue">9. 本轮 selected task ids</text>

为了可复现，本轮 selected ids 已写入各 bench 的 `summary.json`。

示例：

```text
.tmp/openclaw_20_spread_m10/clawbench_official/summary.json
.tmp/openclaw_20_spread_m10/skillsbench/summary.json
.tmp/openclaw_20_spread_m10/agentbench/summary.json
```

后续如果要做 H1/H3/H4，必须复用同一批 ids，才可以 paired comparison。

---

# <text color="blue">10. 最后判断</text>

这次 20 题子集已经说明：

- “很多 bench 是 0 分”不是准确说法。
- 更准确说法是：

```text
1. 有些 bench 有正式分数，但 baseline harness 分数偏低；
2. 有些 bench 是 pending judge，不能算 0；
3. skillsbench 的低分主要是 skill/tool/procedure/infra 问题；
4. official/agentbench 已经有大量非零分，说明 runner/scoring pipeline 不是坏的；
5. 要对齐 clawRecipe，下一步必须做 failure label + harness variants + CRR/NIR。
```

<callout emoji="gem" background-color="light-green" border-color="light-green">
<text color="green">**下一步最值得做的不是继续盲目跑更大样本，而是先把 H1/H3/H4 跑在同一 20-task subset 上，计算每类 failure 的 repair rate。这样才是真正靠近 clawRecipe。**</text>
</callout>
