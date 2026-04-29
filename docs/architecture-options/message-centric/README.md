# Message-Centric Agents

Message-Centric Agents treats a multi-agent system as a communication system.

The central question is:

```text
Who can send what information to whom, when, and through which channel?
```

This option is the best fit when the project wants to study communication protocols, information flow, debate, voting, and coordination patterns.

## Runtime Shape

```text
User task
  -> seed message
  -> MessageBus routes messages to agent inboxes
  -> Scheduler selects which agents step
  -> AgentStep reads context and emits messages
  -> MessageBus routes new messages
  -> stop condition
  -> RunTraceStore + Evaluator
```

## Core Components

- `AgentSpec`: role, instruction, model, and lightweight capabilities.
- `Message`: the unit of communication between agents.
- `Channel`: a named communication space such as `direct`, `broadcast`, `debate`, or `review`.
- `Mailbox`: per-agent inbox and outbox.
- `MessageBus`: stores and routes messages.
- `Router`: decides message delivery rules.
- `Scheduler`: decides which agents step in each round.
- `ContextBuilder`: turns inbox, memory, artifacts, and shared rules into model context.
- `RunTraceStore`: records every message, step, state transition, and final artifact.
- `Evaluator`: computes communication and task metrics from the trace.

## Minimal Data Model

```text
AgentSpec
  - id
  - role
  - instruction
  - model
  - capabilities

Message
  - role
  - content
  - sender
  - target
  - kind
  - channel
  - data

Delivery
  - message_id
  - recipient
  - delivered_at_round

RunConfig
  - agents
  - router
  - scheduler
  - stop_condition
  - max_rounds
```

## Built-In Coordination Patterns

The first implementation of this option should support a few small presets:

- `direct`: one agent sends to another agent.
- `broadcast`: every message is visible to all agents.
- `round_robin`: agents speak in a fixed order.
- `debate`: proposer, critic, defender, judge.
- `vote`: agents produce private or public votes, then an aggregator decides.
- `limited_context`: agents only see the latest N messages or selected channels.

## Context Strategy

Each agent context should be built from:

```text
agent instruction
visible messages
selected artifacts
optional private notes
task objective
current round metadata
```

Context visibility is the main research control surface.

Examples:

- Agent sees all messages.
- Agent sees only direct messages.
- Agent sees broadcast plus private inbox.
- Agent sees only summaries from other agents.
- Judge sees final claims but not hidden deliberation.

## Trace and Evaluation

Message-Centric traces should record:

- All sent messages.
- All deliveries.
- Agent step order.
- Context visibility for each step.
- Final answer or artifact.
- Token usage, latency, and model calls.

Useful metrics:

- Number of rounds.
- Number of messages.
- Message fan-out.
- Agent participation balance.
- Repeated message ratio.
- Agreement or disagreement rate.
- Final answer score.
- Communication cost per score point.

## First Experiments

- Debate vs single agent on research summarization.
- Broadcast vs direct message for collaborative Q&A.
- Round-robin vs free routing for synthesis quality.
- Hidden critic vs public critic in review tasks.
- Limited context vs full transcript for answer quality.

## What This Option Avoids

- No complex tool registry.
- No permission or approval layer.
- No heavy organization model at the start.
- No long-term memory system in the core.
- No recursive subagent tree as the primary abstraction.
