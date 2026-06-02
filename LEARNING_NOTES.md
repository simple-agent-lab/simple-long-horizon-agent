# 学习清单：Agent Trace Replay（从轨迹任意位置重跑）

> 维护中的理解检查清单。✅=已确认掌握　🔄=正在学　⬜=未开始
> 每一项都要同时理解 **是什么 / 怎么做 / 为什么（多层）**。

---

## 阶段 1：问题本身（最关键）✅
- ✅ 1.1 调试 agent 的痛点：要验证「第 N 步换个做法」，传统做法是从第 0 步整跑
- ✅ 1.2 为什么这很糟（三种代价叠加）：**时间**（每步一次 LLM 往返）、**金钱**（重跑前 N 步白烧 token）、**可复现**（采样→重跑可能走另一条路，bug 不复现）
  - 关键反转：replay「冻结已记录前缀」→ 不可复现从 bug 变成 feature
- ✅ 1.3 分水岭 = **工具有无「消息日志之外」的副作用**：
  - (a) 纯函数工具 → 容易分支，**重放消息就够**（消息日志记全了对话）
  - (b) 改文件/库/网络的工具 → 困难分支，**世界不会自己倒回**
  - 困难分支还叠第二层：重新执行 ≠ 当时结果（$RANDOM/时钟/网络 = 确定性问题）
- ✅ 1.4 「位置」= 消息序列里的一个点（细节留到阶段 3：event vs message）

## 阶段 2：为什么 simple-agent-lab 天然适合做这件事（架构洞察）✅
- ✅ 2.1 State 是 append-only 的事件日志
- ✅ 2.2 run() 在「轮与轮之间」是无状态的：每轮从 messages 重建上下文 → run() 分不清「自然历史」和「切出来的前缀」，所以一行不用改
- ✅ 2.3 State.__post_init__ 能从 events 重建快照（messages/active context）
- ✅ 2.4 => fork = 切一段 events + 新建 State，几乎免费
- ✅ 关键坑（=1.4/3.1）：events≠messages。事件流里混着 turn/model_request/tool_execution 等记账事件；24 事件可能只有 6 条消息。**消息号必须经 message_event_indices 翻译成事件下标**，不能 events[:N]。fork(state,k)=events[:translate(k)+1]

## 阶段 3：内存版 replay 方案 ✅
- ✅ 3.1 fork_at_message / resume / message_event_indices（消息号→事件下标翻译，见阶段2）
- ✅ 3.2 edit-and-continue（replace_tail）：切到第 i 条并**替换**它再续跑 → 改一处看后果（改 tool_result / 改模型上轮输出）
- ✅ 3.3 设计决策：
  - **fork 选在 task/tool_result 边界**：因为 run() 下一步是 generate()（轮到模型），无缝接上；run() 没脑子不回头检查（软约定，责任在调用者）。选错点：fake provider 静默跳语义；真实 API 会因「tool_use 没跟 tool_result」400 拒绝
  - **原 State 永不改动**（新建+dataclasses.replace 拷贝事件）：好处=能从同一点反复 fork 出独立分支，A/B 对比互不污染

## 阶段 4：容器副作用问题 ✅
- ✅ 4.1 内存 fork 倒回了「对话」，但**文件系统没倒回**（fork 回第4条，/testbed 里的文件还停在旧状态）→ 困难分支的根
- ✅ 4.2 恢复外部状态两类思路：**A 事先存快照**（docker commit/git commit/btrfs/tar 每步存）vs **B 事后重建**（啥都不存，从 baseline 重放记录的工具调用）
- ✅ 4.3 「每步 git commit」属 A 的一种，硬伤不是「难看」而是：SWE-bench 用 `git diff baseline` 抽 model_patch，狂打 commit **污染/搅乱这个抽取**，把「调试机制」和「被测产物」混在一起

## 阶段 5：replay-to-rebuild（B 方案）
- ⬜ 5.1 recorded_tool_calls / replay_side_effects
- ⬜ 5.2 为什么串行重放、为什么忽略 execution_mode
- ⬜ 5.3 on_fork 钩子：为什么让 resume 保持「中立」
- ⬜ 5.4 确定性的坑（时钟/随机/网络）

## 阶段 6：eval 侧胶水 + 反序列化基石
- ⬜ 6.1 trace_record 原来是单向的，需要 state_from_trace_record
- ⬜ 6.2 为什么只重建 events 就够（messages 由 __post_init__ 重算）
- ⬜ 6.3 resume_in_container 的流程
- ⬜ 6.4 往返幂等测试为什么是强校验
- ⬜ 6.5 ty 类型修复（visible blocks、sidecar cast）

## 阶段 7：真实 Docker 验证
- ⬜ 7.1 跨容器证明（#2 全新 /testbed 被重建）
- ⬜ 7.2 对照组（关掉 rebuild 只剩 gamma）为什么是关键证据

## 阶段 8：更广背景 + DinD 的坑
- ⬜ 8.1 这件事为什么重要、影响什么
- ⬜ 8.2 Docker-in-Docker 的主要陷阱分类
