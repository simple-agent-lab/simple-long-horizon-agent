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

## 阶段 2：为什么 simple-agent-lab 天然适合做这件事（架构洞察）
- ⬜ 2.1 State 是 append-only 的事件日志
- ⬜ 2.2 run() 在「轮与轮之间」是无状态的：每轮从 messages 重建上下文
- ⬜ 2.3 State.__post_init__ 能从 events 重建快照
- ⬜ 2.4 => fork = 切一段 events + 新建 State，几乎免费

## 阶段 3：内存版 replay 方案
- ⬜ 3.1 fork_at_message / resume / message_event_indices
- ⬜ 3.2 edit-and-continue（replace_tail）
- ⬜ 3.3 设计决策：以 message 为「位置」、在 user/tool-result 边界 fork、原 State 不可变、resume 开头有新的 AgentStartEvent

## 阶段 4：容器副作用问题
- ⬜ 4.1 为什么内存 fork 不够：文件系统不会被倒回
- ⬜ 4.2 恢复外部状态的两类思路：快照 vs 重建
- ⬜ 4.3 为什么「每步 git commit」不行（污染 testbed 历史 → 影响 SWE-bench 的 git diff 抽 patch）

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
