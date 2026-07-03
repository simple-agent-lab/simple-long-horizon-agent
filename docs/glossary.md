# Glossary

## Agent

A system that uses a model to decide what to do next, often across multiple steps.

## Agent Loop

The repeated control flow where the system observes state, asks a model for the next action, executes that action, and records the result.

## Message

The runtime transcript unit exchanged between agents. In the current core it
has model-adjacent `role` and `content`, plus lab-facing `sender`, `target`,
`kind`, `channel`, and structured `data`. It is projected into an `LLMMessage`
before a provider call.

## RuntimeMessage

A message variant for system, instruction, summary, or runtime guidance that should stay visible in transcript state.

## LLMMessage

A provider-agnostic model-call payload derived from a runtime message. It
keeps role and ordered content blocks, but drops runtime routing fields.

## Content Block

A typed unit of model-visible content such as text, image, thinking, tool call,
or tool result.

## Message Sidecar

Rare escape-hatch metadata attached to a message before a stable field or type exists.

## Event

A typed record in the run trace. In the current core, message events wrap
transcript messages, while lifecycle, model request/response, and tool
execution events record the rest of the run.

## Tool

A callable capability exposed to the agent, such as reading a file, searching
data, calling an API, or running a calculation. In this repo, shared tool
values live in `simple_agent_lab.tools`; each runtime owns its own dispatch
semantics.

## Model Adapter

The boundary between project code and a model provider or model API.

## State

The data the system carries between steps in an agent loop.

## Memory

Information preserved beyond a single immediate step or turn.

## Evaluation

A repeatable check that helps compare agent behavior, correctness, reliability, or usability.

## Reference Architecture

An external or internal architecture studied before deciding what Simple Agent Lab should borrow or avoid.

## Candidate

One point in an evolution run's search space: a JSON-able payload (prompt
text, program source, agent config) plus its lineage (parents, generation,
the operator that produced it).

## Evolution Archive

The append-only record of an evolution run: every candidate with its
evaluation and the accept/reject decision and reason, one JSON line each.
The archive is the audit trail, the resume point, and the dataset a run
leaves behind.

## Fitness

The scalar an evolution run maximizes, produced by the run's evaluator.
Richer signals ride alongside it as metrics (structured) and feedback
(free text shown to the proposer).

## Genome Component

One typed, evolvable slot in a candidate's payload (text, code, or JSON),
declared with proposer-facing docs, a mutability flag, and validation. The
component schema (`GenomeSpec`) is what lets the harness prompt for, check,
and attribute mutations per component.
