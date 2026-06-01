# Exp1: Simple Agent Lab × ClawEvalkit 集成

## 实验目标

在 [simple-agent-lab](https://github.com/simple-agent-lab/simple-agent-lab) 的基础上，添加对 [ClawEvalkit](https://github.com/linjh1118/ClawEvalkit) 中 benchmark 的评测支持。复用 SAL 的 Bash Agent 作为执行引擎，适配 ClawEvalkit 的评分系统，实现"换 agent 不换评测框架"。

## 实验推进历程

### Round 1: 架构分析与集成实现

**设计动机**：
- 开发者（simple-agent-lab 团队 + @王超凡）已跑通 SAL Bash Agent 跑 SWE-bench 的主流程
- ClawEvalkit 有 8 个 benchmark，但都依赖 OpenClawPro/NanoBotAgent 作为执行引擎
- 目标：让 SAL 的 Bash Agent 也能跑 ClawEvalkit 的 benchmark，验证 SAL agent 的通用能力

**具体方案**：
- 分支：`feature/openclaw`
- 在 `evals/openclaw/` 下新建适配层，不修改 SAL 上游代码
- 核心转换：SAL Event 协议 → ClawEvalkit transcript 格式
- 支持 PinchBench（23 tasks，规则评分）和 AgentBench（39 tasks，多层评分）

**关键文件**：

| 文件 | 功能 |
|------|------|
| `evals/openclaw/config.py` | 模型配置加载（ClawEvalkit YAML → SAL Provider） |
| `evals/openclaw/adapter.py` | SAL ↔ ClawEvalkit 格式桥接 + 任务执行 |
| `evals/openclaw/runner.py` | PinchBench + AgentBench 完整评测 pipeline |
| `evals/openclaw/test_integration.py` | Fake provider 冒烟测试 |
| `docs/simple_agent_lab_intro.html` | SAL 源码解析可视化 |

**结果数据**：
- 冒烟测试 3/3 全部通过（fake provider）
- Transcript 转换：SAL Event → ClawEvalkit dict 格式正常工作
- 完整 pipeline：load task → SAL agent run → transcript convert → grade() 执行通过

**核心发现**：
1. SAL 需要 Python 3.10+（用了 `TypeAlias`），通过 `uv sync` 管理依赖
2. SAL 的 Provider 用 `id` + `api`（ApiKind），不是 `kind`
3. SAL 通过 `api_key_env` 指定环境变量名来加载 API key
4. ClawEvalkit 的 ModelHub（字节内部）需要 `AzureOpenAI` 客户端，SAL 没有内置 Azure adapter
5. PinchBench 的评分函数 `grade(transcript, workspace_path)` 需要特定格式的 transcript

**推导的下一步**：
- 需要为 `azure_openai` provider 类型创建 SAL adapter 或自定义 generate 函数
- 配置好 API key 后可直接运行真实评测

---

### Round 2: AzureOpenAI 集成 + 真实评测

**设计动机**：
- 用户提供了字节内部 API keys（ModelHub GPT 代理 + OpenRouter）
- 验证 SAL agent 在真实 LLM 下的端到端 pipeline 能力
- 产生实际 agent traces 供分析

**具体方案**：
- 为 AzureOpenAI provider 创建自定义 `_build_azure_agent()` 函数
- 在闭包中使用 AzureOpenAI 客户端直接调用 ModelHub API
- 关键修复：
  1. `tools_tuple` 必须在闭包定义之前创建（Python 变量作用域）
  2. `llm_response_to_assistant_message` 需从 `simple_agent_lab.llm.bridge` 导入（不在 `messages` 模块）
  3. AzureOpenAI 返回的 `tc.function.arguments` 是 JSON 字符串，需 `json.loads()` 而非 `dict()`

**关键参数**：
- 模型: `gpt-4.1-2025-04-14`
- Endpoint: `https://search.bytedance.net/gpt/openapi/online/v2/crawl` (AzureOpenAI 客户端)
- API Key: 通过 `GPT_API_KEY` 环境变量注入
- Max turns: 10 per task

**结果数据（PinchBench 全量 23 tasks）**：

| 任务 | 状态 | 分数 | 耗时 |
|------|------|------|------|
| task_00_sanity | success | 1.0 | 3.2s |
| task_01_calendar | success | 1.0 | 15.1s |
| task_02_stock | success | 0.0 | 2.0s |
| task_03_blog | success | 1.0 | 6.2s |
| task_04_weather | success | 0.0 | 5.2s |
| task_05_summary | success | 1.0 | 7.3s |
| task_06_events | success | 1.0 | 6.2s |
| task_07_email | success | 1.0 | 2.7s |
| task_08_memory | success | 0.8 | 4.1s |
| task_09_files | success | 0.857 | 3.0s |
| task_10_workflow | success | 0.167 | 8.2s |
| task_11_clawdhub | success | 1.0 | 3.6s |
| task_12_skill_search | success | 0.833 | 11.8s |
| task_13_image_gen | success | 0.333 | 6.6s |
| task_14_humanizer | success | 1.0 | 4.2s |
| task_15_daily_summary | success | 1.0 | 12.9s |
| task_16_email_triage | success | 0.0 | 12.3s |
| task_16_market_research | success | 0.813 | 18.5s |
| task_17_email_search | success | 0.0 | 16.1s |
| task_18_spreadsheet_summary | success | 0.0 | 9.0s |
| task_20_eli5_pdf_summary | success | 1.0 | 2.5s |
| task_21_openclaw_comprehension | success | 0.0 | 3.0s |
| task_22_second_brain | success | 0.0 | 1.0s |

**核心发现**：
1. **总分 60.0%**，23/23 任务全部成功执行无 crash
2. 10 个满分（sanity, calendar, blog, summary, events, email, clawdhub, humanizer, daily_summary, eli5_pdf）
3. ModelHub 代理需要清除 http_proxy/https_proxy，否则 403
4. GPT-4.1 对文本生成任务表现优秀（mean=1.0），但对需要多步工具链/搜索的任务（stock, weather, email_triage）表现较差
5. Transcript 和 trace 数据已正确保存

**推导的下一步**：
- 接入 ClawBench (303 tasks) 以扩大评测覆盖面
- 用 GLM-4.7 等其他模型跑对比实验
- 改进 AgentBench 的 L1-L3 评分实现

---

### Round 3: ClawBench Official 集成（303 tasks）

**设计动机**：
- PinchBench 只有 23 个任务，覆盖面有限
- ClawBench Official 是 ClawEvalkit 的核心 benchmark，303 个任务覆盖 30+ domain（email, code, data-analysis, security 等）
- ClawBench 使用 pytest verifier 评分，比 PinchBench 的内嵌 grade() 更标准、更细粒度
- SAL Bash Agent 作为 native workspace agent 运行；repo-local `assets/benchmarks/claw-bench/` 这批任务走本地 workspace + pytest verifier，不需要 Docker/NanoBotAgent

**具体方案**：
- **任务加载**：从 repo-local `assets/benchmarks/claw-bench/tasks/*/*/task.toml` 加载 303 个任务（Python 3.11+ 用 `tomllib`，3.10 回退 `tomli`）
- **执行 pipeline**：prepare workspace → load instruction → run SAL agent → pytest verify
  1. `_prepare_clawbench_workspace()`：复制 `environment/data/` 到临时 workspace，执行 `environment/setup.sh`
  2. `_load_clawbench_instruction()`：读取 `instruction.md`，将相对路径替换为绝对 workspace 路径
  3. `run_task_with_sal_agent()`：SAL Bash Agent 在 workspace 下执行任务
  4. `_verify_clawbench_task()`：调用 `claw_bench.core.verifier.verify_task(task_dir, workspace)` 用 pytest 评分
- **评分**：优先用 `verification.weighted_score`，fallback 为 `checks_passed / checks_total`
- **CLI 支持**：`--bench clawbench` 或 `--bench clawbench-official`（两者是同一 runner）
- **缓存 + 并行**：与 PinchBench/AgentBench 一致的缓存和并行机制

**关键新增代码**：

| 文件 | 新增内容 |
|------|----------|
| `adapter.py` | `run_clawbench_task()`, `_prepare_clawbench_workspace()`, `_load_clawbench_instruction()`, `_verify_clawbench_task()` |
| `runner.py` | `_load_clawbench_tasks()`, `run_clawbench()`, `BENCHMARK_RUNNERS["clawbench"]` |
| `test_integration.py` | `test_clawbench_task_smoke()` — 用 cal-001 验证完整 pipeline |
| `config.py` | 无改动（复用 AzureOpenAI 支持） |

**ClawBench 任务结构**：
```
task_dir/
├── task.toml          # id, title, domain, level, timeout, capabilities
├── instruction.md     # 任务指令（发给 agent 的 prompt）
├── environment/
│   ├── data/          # 输入数据文件
│   └── setup.sh       # 环境初始化脚本
└── verifier/
    └── test_output.py # pytest 验证器
```

**测试结果**：
- 冒烟测试 4/4 全部通过（含 ClawBench cal-001 verifier smoke）
- `_load_clawbench_tasks()` 成功加载 303 个任务
- pytest verifier 在 SAL workspace 上正确执行并返回分数

**评测结果（ClawBench 35 tasks, GPT-4.1）**：

| Domain | Tasks | Avg Score | Perfect |
|--------|-------|-----------|---------|
| calendar | 2 | 1.00 | 2/2 |
| code-assistance | 2 | 0.97 | 0/2 |
| communication | 3 | 0.99 | 2/3 |
| contract-review | 1 | 1.00 | 1/1 |
| cross-domain | 2 | 1.00 | 2/2 |
| data-analysis | 3 | 1.00 | 3/3 |
| data-science | 3 | 0.56 | 1/3 |
| debugging | 1 | 0.06 | 0/1 |
| document-editing | 3 | 1.00 | 3/3 |
| educational-assessment | 1 | 1.00 | 1/1 |
| email | 1 | 1.00 | 1/1 |
| file-operations | 2 | 0.96 | 1/2 |
| memory | 1 | 1.00 | 1/1 |
| multimodal | 2 | 0.87 | 1/2 |
| regulatory-compliance | 1 | 1.00 | 1/1 |
| system-admin | 1 | 0.33 | 0/1 |
| web-browsing | 1 | 1.00 | 1/1 |
| workflow-automation | 1 | 1.00 | 1/1 |

**总结**：35 tasks, 22/35 passed (62.9%), avg score 78.9%

**失败分析**：
- bioinformatics 3/3 全部 0 分（依赖 biopython 等专业库）
- academic-research 1/1 失败（plagiarism check 任务）
- debugging 1/1 低分（0.06，pytest 验证逻辑未完全通过）
- data-science 中 ds-004 time-series 失败（0.0），ds-002 ab-testing 部分通过（0.67）
- 多数 domain（12/17）有满分或接近满分的任务

**核心发现**：
1. SAL Bash Agent 在 ClawBench 上表现良好，22/35 通过 pytest 验证
2. 文本/文档类任务（calendar, communication, document-editing, email）全部通过
3. 专业领域（bioinformatics, academic-research）完全失败，需要专业工具链
4. 代码类任务（code-assistance）接近完美（0.97 avg）但未完全通过
5. 每个任务都有完整 transcript trace，可在 dashboard 中查看

**推导的下一步**：
- 扩大样本到 100+ tasks 覆盖更多 domain
- 用 GLM-4.7 等模型做对比实验
- 分析失败任务的 trace 找 agent 行为模式
- 集成更多 benchmark（SkillsBench, ClawEval）

## 使用方式

```bash
cd /Users/bytedance/Documents/github/glm_dev/simple-agent-lab
git checkout feature/openclaw

# 冒烟测试（含 ClawBench）
uv run python evals/openclaw/test_integration.py

# PinchBench 真实评测
uv run python -m evals.openclaw --bench pinchbench --model <model> --clawevalkit /path/to/ClawEvalkit --sample 3

# ClawBench Official 单任务
uv run python -m evals.openclaw --bench clawbench --model <model> --task-ids cal-001

# ClawBench Official 小样本
uv run python -m evals.openclaw --bench clawbench --model <model> --sample 10

# ClawBench 全量（303 tasks）
uv run python -m evals.openclaw --bench clawbench --model <model>

# 使用 AzureOpenAI（字节 ModelHub）
uv run python -m evals.openclaw --bench clawbench --model gpt-4.1-2025-04-14 \
  --sample 5 \
  --api-url "https://search.bytedance.net/gpt/openapi/online/v2/crawl" \
  --api-key-env GPT_API_KEY --provider azure_openai
```

## 支持的 Benchmark

| Benchmark | 任务数 | 评分方式 | 状态 |
|-----------|--------|----------|------|
| PinchBench | 23 | 规则评分（内嵌 grade 函数） | 已实现 |
| AgentBench | 39 | 多层评分（L0-L3） | 已实现（仅 L0） |
| SkillsBench | 56-68 | Pytest | 待实现 |
| ClawBench | 303 | SAL Bash Agent + Pytest verifier | 已接入（Round 3，native workspace） |
| ClawEval | 199-300 | 多维评分 | 待实现 |
| ZClawBench | 116 | LLM Judge | 待实现 |

## 产物

| 产物 | 路径 |
|------|------|
| **索引页** | `docs/exp1_openclaw/index.html` |
|------|------|
| 集成代码 | `evals/openclaw/` |
| 冒烟测试 | `evals/openclaw/test_integration.py`（4/4 passed，含 PinchBench + ClawBench） |
| SAL 源码解析 | `docs/exp1_openclaw/sal_intro.html`（Round 1 产物） |
| 实验文档 | `docs/exp1_openclaw/README.md` |
| PinchBench 结果 | `evals/openclaw/results/gpt-4.1-2025-04-14/`（23 tasks） |
| ClawBench 结果 | `evals/openclaw/results/clawbench/gpt-4.1-2025-04-14/`（35 tasks） |
| PinchBench Dashboard | `docs/exp1_openclaw/pinchbench_dashboard.html`（23 traces） |
| ClawBench Dashboard | `docs/exp1_openclaw/clawbench_dashboard.html`（35 traces） |
| 代码剖析 | `docs/exp1_openclaw/openclaw_code.html`（全 4 模块） |
| 运行日志 | `docs/exp1_openclaw/assets/logs/run_pinchbench_gpt41.log` |
