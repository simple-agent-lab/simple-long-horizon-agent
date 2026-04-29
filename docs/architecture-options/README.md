# Architecture Options

This directory contains three alternative architecture options for Simple Agent Lab.

They are not implementation phases. They are comparable designs with similar scope and different organizing ideas.

Current implementation direction is smaller than all three options:

```text
Agent + Message + State + context_view() + run()
```

The options below are kept as comparison notes. In code, debate, pipeline, and workspace should be recipes over the tiny message runtime rather than separate runtimes.

Shared constraints:

- Keep the system simple enough for students to read and modify.
- Optimize for multi-agent research rather than production automation.
- Do not include permission, security, approval, or sandboxing in the core architecture yet.
- Keep traces and evaluations readable through `state.events`.
- Keep tools lightweight and secondary to agent communication, organization, and context.

## Options

| Option | Core abstraction | Best for |
| --- | --- | --- |
| [Message-Centric Agents](message-centric/README.md) | Message bus and communication protocols | Communication, debate, voting, information flow |
| [Pipeline-Centric Agents](pipeline-centric/README.md) | Task pipeline and artifact handoff | Reproducible workflows, staged research, map-reduce |
| [Workspace-Centric Agents](workspace-centric/README.md) | Shared workspace and task board | Organization structure, role coordination, shared state |

## Current Runtime Mapping

The current implementation does not create separate classes for artifacts,
traces, runtime stores, or evaluators. Those ideas are folded into the tiny
runtime:

```text
Agent
  A named role with one act function.

Message
  The communication and model unit: role, content, sender, target, kind,
  channel, and data.

State
  The shared world: task, events, and small experiment data.

context_view()
  The context management boundary. It decides which messages an agent sees.

run()
  The schedule loop. It calls agents in order.

Event
  A timestamped message in `State.events`, used for trace and replay.
```

Earlier component names such as `ArtifactStore`, `RunTraceStore`, and
`Evaluator` are useful research vocabulary, but they are not current core
implementation targets. Artifacts can be messages or entries in `State.data`.
The run trace is `State.events`. Evaluation code can read the same state later.

The options differ in how they coordinate agents:

```text
Message-Centric  -> coordination through messages and channels
Pipeline-Centric -> coordination through nodes, edges, and artifacts
Workspace-Centric -> coordination through shared boards and blackboards
```

## Recommended Reading Order

1. Start with [Message-Centric Agents](message-centric/README.md) for the most general multi-agent model.
2. Read [Pipeline-Centric Agents](pipeline-centric/README.md) for the most reproducible experiment model.
3. Read [Workspace-Centric Agents](workspace-centric/README.md) for organization and shared-state research.
