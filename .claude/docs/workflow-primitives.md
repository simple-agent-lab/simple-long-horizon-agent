# Workflow 原语与 Schema 参考

本文档汇总了 `Workflow` 工具脚本中可用的**全部原语、全局对象、配置 schema 与编排模式**。
脚本是普通 **JavaScript**(不是 TypeScript),运行在 async 上下文中,可直接使用 `await`。

> 适用范围:本仓库 `simple-agent-lab` 下通过 `Workflow({...})` 启动的所有后台编排脚本
> (例如 `/simplicity-review`、`/code-review ultra` 等)。

---

## 1. 脚本结构

每个脚本必须以一个 **纯字面量** 的 `meta` 导出开头,随后是脚本主体:

```js
export const meta = {
  name: 'find-flaky-tests',                              // 必填,kebab-case 标识
  description: 'Find flaky tests and propose fixes',     // 必填,一行说明(权限弹窗里展示)
  whenToUse: 'When CI is intermittently red',            // 可选,workflow 列表里展示
  phases: [                                              // 可选,每个 phase() 调用对应一条
    { title: 'Scan',  detail: 'grep test logs for retries' },
    { title: 'Fix',   detail: 'one agent per flaky test' },
  ],
  model: 'sonnet',                                       // 可选,标注某 phase 的模型覆盖
}

// 主体从这里开始 —— 使用 agent()/parallel()/pipeline()/phase()/log()
phase('Scan')
const flaky = await agent('grep CI logs for retry markers', { schema: FLAKY_SCHEMA })
```

### `meta` 约束
- **必须是纯字面量**:不能有变量、函数调用、展开运算符 `...` 或模板插值。
- **必填字段**:`name`、`description`。
- **可选字段**:`whenToUse`、`phases`、`model`。
- `meta.phases` 里的 `title` 与脚本中 `phase('...')` 的字符串 **完全匹配**;没匹配上的 `phase()` 调用会自成一组进度框。

---

## 2. 核心原语

### `agent(prompt, opts?) → Promise<any>`
启动一个子 agent。

- **不带 schema**:返回该 agent 的最终文本(string)。
- **带 schema**:强制子 agent 调用 `StructuredOutput` 工具,返回**已校验的对象**(无需自己解析;不匹配会自动重试)。
- 用户中途跳过该 agent 时返回 `null` —— 用 `.filter(Boolean)` 过滤。

`opts` 字段:

| 字段 | 类型 | 说明 |
|---|---|---|
| `label` | string | 覆盖进度显示的标签 |
| `phase` | string | 显式归入某进度组(在 `pipeline()`/`parallel()` 内**务必显式指定**,避免和全局 `phase()` 抢状态) |
| `schema` | JSON Schema | 启用结构化输出,返回校验后的对象 |
| `model` | `'sonnet'`/`'opus'`/`'haiku'` | 覆盖该次调用的模型。**默认应省略**,继承主循环模型;仅在非常确定某档位更合适时才设 |
| `isolation` | `'worktree'` | 给 agent 独立 git worktree。**昂贵**(每个约 200–500ms + 磁盘),仅在多个 agent 并行改文件会冲突时用;未改动会自动清理 |
| `agentType` | string | 用自定义 agent 类型(如 `'Explore'`、`'code-reviewer'`)替代默认 workflow 子 agent。与 `agentType` 自带的 `model` frontmatter 有优先级关系(见下) |

> **模型解析优先级**:`opts.model` > `agentType` 定义的 `model` frontmatter > 继承父级(主循环模型)。
> 例如内置 `Explore` agent 默认是 haiku;不传 `opts.model` 就会落到 haiku。

### `parallel(thunks) → Promise<any[]>`
并发运行一批任务,**是一个 barrier**:等所有 thunk 完成才返回。

- 入参是 **函数数组**:`Array<() => Promise<any>>`(注意是 thunk,不是已启动的 Promise)。
- 某个 thunk 抛错 → 对应结果为 `null`,调用本身**不会 reject**。使用前 `.filter(Boolean)`。
- 仅在**确实需要所有结果一起拿到**时才用(见第 5 节"何时用 barrier")。

```js
const all = await parallel(DIMENSIONS.map(d => () => agent(d.prompt, { schema: S })))
```

### `pipeline(items, stage1, stage2, ...) → Promise<any[]>`
让每个 item **独立**流过所有 stage,**stage 之间没有 barrier**。item A 可以在 stage 3,而 item B 还在 stage 1。

- 墙钟时间 = 最慢单条链路,而非"每阶段最慢之和"。
- 每个 stage 回调签名:`(prevResult, originalItem, index)` —— 后续 stage 可用 `originalItem`/`index` 打标签,无需把上下文串进 stage1 的返回值。
- 某 stage 抛错 → 该 item 降为 `null`,跳过其剩余 stage。
- **这是多阶段工作的默认选择。**

```js
const results = await pipeline(
  DIMENSIONS,
  d      => agent(d.prompt, { label: `review:${d.key}`, phase: 'Review', schema: FINDINGS_SCHEMA }),
  review => parallel(review.findings.map(f => () =>
    agent(`Adversarially verify: ${f.title}`, { phase: 'Verify', schema: VERDICT_SCHEMA })
      .then(v => ({ ...f, verdict: v })))),
)
```

### `log(message) → void`
向用户发一条进度消息(显示在进度树上方的叙述行)。

### `phase(title) → void`
开启一个新 phase;之后的 `agent()` 调用在进度显示里归入该标题。
(在 `pipeline`/`parallel` 内部,优先用 `opts.phase` 显式归组而非依赖全局 `phase()`。)

### `workflow(nameOrRef, args?) → Promise<any>`
把另一个 workflow 当作子步骤内联运行,返回其返回值。

- 传 **名字** 调用已保存的 workflow;或传 `{ scriptPath }` 运行已写到磁盘的脚本文件。
- 子 workflow 共享本次运行的并发上限、agent 计数、abort 信号、token 预算;其 agent 在 `/workflows` 里显示为 `▸ name` 组,token 计入 `budget.spent()`。
- **只能嵌套一层**:子 workflow 里再调 `workflow()` 会抛错。
- 未知名字 / 不可读 scriptPath / 子脚本语法错误都会抛错,需 `catch` 处理。

---

## 3. 全局对象

### `args`
传给 `Workflow` 的 `args` 输入,**原样**透传(未提供则为 `undefined`)。

- 传数组/对象要传**真正的 JSON 值**,不要传 JSON 字符串:
  `args: ["a.ts", "b.ts"]` ✅ ,而非 `args: "[\"a.ts\"]"` ❌(后者到脚本里是单个字符串,`.filter`/`.map` 会抛错)。
- 用于参数化已命名 workflow(研究问题、目标路径、配置对象等)。

### `budget`
本轮的 token 目标(来自用户类似 "+500k" 的指令)。

| 成员 | 说明 |
|---|---|
| `budget.total` | 目标值;未设置时为 `null` |
| `budget.spent()` | 本轮已花的 output token(主循环 + 所有 workflow 共享同一池,非每 workflow 独立) |
| `budget.remaining()` | `max(0, total - spent())`;未设目标时为 `Infinity` |

- 目标是**硬上限**:`spent()` 达到 `total` 后,后续 `agent()` 调用会抛错。
- 动态循环要 **守卫 `budget.total`**,否则未设目标时 `remaining()` 是 `Infinity`,会一直跑到 1000-agent 上限:

```js
const FLEET = budget.total ? Math.floor(budget.total / 100_000) : 5
while (budget.total && budget.remaining() > 50_000) { /* ... */ }
```

---

## 4. Schema 写法(结构化输出)

`schema` 是标准 **JSON Schema**,校验发生在工具调用层 —— 不匹配模型会自动重试,所以脚本里拿到的对象一定合法。

惯用写法:`type: 'object'` + `additionalProperties: false` + `required` 列全字段。

本仓库 `simplicity-review` 用到的两个 schema 作为范例:

### 发现项 schema(reviewer 输出)
```js
const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['findings'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'file', 'line', 'severity', 'problem', 'suggestion'],
        properties: {
          title:      { type: 'string', description: 'short imperative title' },
          file:       { type: 'string' },
          line:       { type: 'string', description: 'line number or range' },
          severity:   { type: 'string', enum: ['minor', 'moderate', 'significant'] },
          problem:    { type: 'string' },
          suggestion: { type: 'string', description: 'concrete simpler alternative' },
        },
      },
    },
  },
}
```

### 裁决 schema(verify 输出)
```js
const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['verdicts'],
  properties: {
    verdicts: {
      type: 'array',
      description: 'one entry per finding, in the same order',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['index', 'keep', 'reason'],
        properties: {
          index:  { type: 'integer', description: '0-based index of the finding being judged' },
          keep:   { type: 'boolean', description: 'true only if real, behavior-preserving, and genuinely simpler' },
          reason: { type: 'string' },
        },
      },
    },
  },
}
```

**Schema 提示**
- 每个字段都写 `description`,等于给模型的 inline 指令,显著提升输出质量。
- 用 `enum` 约束分类字段(如 `severity`)。
- 让数组项顺序对应输入顺序时,在数组的 `description` 里讲清楚("one entry per finding, in the same order")。

---

## 5. 编排选择:pipeline vs parallel(barrier)

**默认用 `pipeline()`。** 只有当 stage N 需要 stage N-1 的**全部跨 item 结果**时,barrier 才正确:

barrier **正确**的场景:
- 在昂贵的下游工作前,对完整结果集做去重/合并;
- 总数为零时早退("0 个 bug → 整段验证直接跳过");
- stage N 的 prompt 要引用"其它所有发现"做对比。

barrier **不成立**的理由:
- "我得先 flatten/map/filter" → 在 pipeline 的 stage 里做:`pipeline(items, A, r => transform([r]).flat(), B)`;
- "这些阶段概念上是分开的" → 那正是 pipeline 建模的东西,分开 ≠ 同步;
- "代码更干净" → barrier 的延迟是真实代价。

**嗅探判据**:如果你写了
```js
const a = await parallel(...)
const b = transform(a)          // flatten/map/filter,无跨 item 依赖
const c = await parallel(b.map(...))
```
那个中间 transform 不需要 barrier —— 改写成把 transform 放进 stage 的 pipeline。拿不准时:用 pipeline。

---

## 6. 并发与规模限制

- 单 workflow 内并发 `agent()` 上限 = `min(16, CPU 核数 - 2)`,超出的排队,有空位再跑。
  → 可以放心给 `parallel()`/`pipeline()` 传 100 个 item,它们都会完成,只是同一时刻约 10 个在跑。
- 单 workflow 生命周期内 agent 总数上限 **1000**(防失控循环的兜底,远高于任何真实 workflow)。

---

## 7. 运行环境限制

- 脚本是 **纯 JavaScript**:类型注解(`: string[]`)、interface、泛型都会解析失败。
- 标准 JS 内建可用(`JSON`、`Math`、`Array`…),**但以下会抛错**(它们会破坏 resume 的确定性):
  - `Date.now()`
  - `Math.random()`
  - 无参 `new Date()`
  - → 时间戳通过 `args` 传入,或在 workflow 返回后再打戳;需要随机性就用 index 改变 agent 的 prompt/label。
- 无文件系统、无 Node.js API。
- workflow agent 可通过 `ToolSearch` 访问所有会话连接的 MCP 工具(schema 按需逐 agent 加载)。
  注意:交互式认证的 MCP server(如 claude.ai)在 headless/cron 运行里可能不可用。

---

## 8. Resume(断点续跑)

工具返回里含 `runId`。暂停/kill/改脚本后,用
`Workflow({ scriptPath, resumeFromRunId })` 重新拉起:

- **最长未改动前缀**的 `agent()` 调用瞬间返回缓存结果;第一个被改/新增的调用及其之后全部 live 重跑。
- 同脚本 + 同 args → 100% 命中缓存。
- 改了某次 `agent()` 的 `(prompt, opts)`(**包括加 `model`**)→ 该调用缓存失效,live 重跑。
- **续跑前先 `TaskStop` 停掉旧 run。**
- 兜底(无 journal 时):读 transcript 目录下的 `agent-<id>.jsonl`,手写续跑脚本。

> 实例:本会话把 `simplicity-review` 两处 `agent()` 从 haiku 改成 `model: 'sonnet'` 后,
> `TaskStop` 旧 run → `Workflow({ scriptPath, resumeFromRunId })` 续跑,因 `opts` 变了所以两处均 live 重跑。

---

## 9. 质量编排模式(按需组合)

- **对抗式验证(Adversarial verify)**:每个发现派 N 个独立"反驳者",被要求去**推翻**它;多数反驳则丢弃。
- **视角多样验证**:一个发现可能以多种方式出错时,给每个验证者不同视角(correctness / security / perf / 能否复现),比 N 个相同反驳者更能覆盖失败模式。
- **评审团(Judge panel)**:从不同角度生成 N 个独立方案 → 并行打分 → 从胜者综合并嫁接亚军的好点子。解空间大时优于"单方案迭代"。
- **跑到枯竭(Loop-until-dry)**:未知规模的发现(bug、edge case),持续派 finder 直到连续 K 轮无新增。简单计数器 `while (count < N)` 会漏尾巴。
- **多模态扫荡(Multi-modal sweep)**:多个 agent 各用不同方式搜索(按容器 / 按内容 / 按实体 / 按时间),彼此盲视;单一搜索角度找不全时用。
- **完整性批判者(Completeness critic)**:最后一个 agent 专问"还缺什么 —— 哪种模态没跑、哪个论断没验、哪个来源没读",其产出成为下一轮工作。
- **无声截断要 `log`**:若 workflow 限了覆盖(top-N、不重试、抽样),用 `log()` 说明丢了什么,否则读起来像"全覆盖"实则不然。

> 规模随需求伸缩:"找找有没有 bug" → 少量 finder + 单票验证;"彻底审计/要全面" → 更大 finder 池 + 3–5 票对抗验证 + 综合阶段。
