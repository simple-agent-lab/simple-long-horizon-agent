# Nano Simple Agent Lab

这个目录是对 `exp3_simple_agent_merge_claw_by_codex/simple-agent-lab` 的
nano 级解释版。它不追求完整可运行，而是追求把老板说的几个关键词讲明白：

- `docker` 不是核心框架本身，只是“在哪里跑”的一个后端。
- `executor` 有两层含义：agent loop 里并发执行工具；eval 框架里把一次任务交给某个运行环境。
- 核心设计不是“大而全 agent 框架”，而是“清楚的边界 + 可替换的小协议”。

## 我理解的设计理念

### 1. 核心 loop 要小

真正的 agent runtime 可以被压成一句话：

> 维护一段消息历史，投影出模型能看到的上下文，让 agent 生成下一条消息，如果里面有工具调用就执行工具，把工具结果写回历史，直到 final。

对应 nano 文件是 [`nano_agent.py`](nano_agent.py)。

原项目里的 `core.py` 做的也是这个：`Agent + State + Message + run()`。它没有把 Docker、SWE-bench、provider 细节塞进 loop。

### 2. Message 是项目自己的协议，不等于模型 API payload

老板这个框架非常在意“边界语言”：

- runtime 里保存 `Message`
- 调模型前投影成 provider-neutral 的 `LLMMessage`
- 具体 OpenAI / Anthropic 怎么发包，是 adapter 的事

这样做的好处是：agent loop 不被某个模型厂商绑死，轨迹、回放、训练数据也有稳定形状。

### 3. Tool call 是普通消息流的一部分

工具调用不是隐藏魔法，而是 assistant 消息里的结构化块：

1. assistant 说“我要调用 bash”
2. runtime 找到这个 tool
3. executor 执行它
4. 结果变成一条 `tool_result` message
5. 下一轮模型把这个结果当上下文继续看

原项目里用 `ThreadPoolExecutor` 支持并发工具调用。nano 版也保留了这个点。

### 4. Docker 是 eval backend，不是 agent runtime

老板说的 Docker 主要来自评测体系。它的抽象是：

- `Suite`：一个 benchmark 只描述自己特殊的部分，例如镜像、workdir、任务怎么构造、结果怎么提取。
- `ContainerBackend`：任务在哪里跑。可以是本进程、local Docker、remote Docker、以后也可以是 k8s。
- `ArtifactStore`：输入、输出、轨迹这些 bytes 放哪里。可以是本地目录、HTTP、对象存储。
- `run_suite_instance(...)`：把 suite/backend/store 接起来，跑一个 instance。

对应 nano 文件是 [`nano_eval.py`](nano_eval.py)。

关键点：`run_suite_instance` 自己不应该写 Docker 分支。它只生产 `RunSpec`，然后交给 backend。

### 5. Host half / container half

一个 Docker eval 天然是两个程序：

- host half：在调度机上决定用什么镜像、隐藏 gold/private 字段、准备输入。
- container half：在容器里构造 agent 任务、跑 agent、提取结果、可选评分。

这可以避免每个 benchmark 都复制一份大 launcher。新 benchmark 只补自己的薄薄一层。

### 6. ArtifactStore 是总线

这个设计里没有单独的 trace sink、input transport、output collector。

全部都是 store 的 `put/get`：

- `input/instance.json`
- `input/eval.json`
- `out/result.json`
- `out/trajectory.jsonl`

这就是为什么它能从本地 bind mount 演进到 remote Docker 或对象存储。

## 文件说明

- [`index.html`](index.html)：中文图解版思想说明，可以直接浏览器打开。
- [`nano_agent.py`](nano_agent.py)：最小 agent loop + tool executor。
- [`nano_eval.py`](nano_eval.py)：最小 eval runner + backend/store 抽象。
- [`mini_suite_example.py`](mini_suite_example.py)：展示一个 benchmark suite 应该长什么样。

## 一句话版本

这个框架的核心不是“我有 Docker”，而是：

> agent loop 保持小而透明；所有容易变化的东西都放在边界上，用数据和协议接起来。Docker 只是 `ContainerBackend`，工具只是 `AgentTool`，模型只是 `Provider`，评测只是 `Suite + Backend + Store` 的一次组合。

