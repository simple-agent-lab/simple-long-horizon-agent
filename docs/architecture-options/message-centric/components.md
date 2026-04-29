# Components

## AgentStep

Runs one step for one agent.

Input:

```text
agent spec
visible messages
task objective
artifacts
round metadata
```

Output:

```text
assistant message
optional artifact writes
optional final signal
```

## MessageBus

Responsible for message persistence and delivery.

It should not decide what agents think. It only records messages and applies routing rules.

## Router

Controls communication topology.

Examples:

- Direct messages only.
- Broadcast all messages.
- Channel-based delivery.
- Judge-only final delivery.
- Limited message visibility.

## Scheduler

Controls execution order.

Examples:

- Round-robin.
- All agents once per round.
- Only agents with inbox messages.
- Fixed script for an experiment.

## ContextBuilder

Builds the model input for one agent step.

It is the boundary between communication structure and model behavior.

## RunTraceStore and Evaluator

The trace store records facts. The evaluator reads facts and computes reports.

Keep them close in code organization, but keep their responsibilities separate.

