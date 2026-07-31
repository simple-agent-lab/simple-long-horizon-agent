# Glossary

General vocabulary used across the repo. The message-protocol terms —
`Message`, `LLMMessage`, `RuntimeMessage`, content blocks, provider adapters,
and message sidecars — are defined once in [`CONTEXT.md`](../CONTEXT.md); this
file does not repeat them.

## Agent

A system that uses a model to decide what to do next, often across multiple steps.

## Agent Loop

The repeated control flow where the system observes state, asks a model for the next action, executes that action, and records the result.

## Event

A typed record in the run trace. In the current core, message events wrap
transcript messages, while lifecycle, model request/response, and tool
execution events record the rest of the run.

## Tool

A callable capability exposed to the agent, such as reading a file, searching
data, calling an API, or running a calculation. In this repo, shared tool
values live in `simple_long_horizon_agent.tools`; each runtime owns its own dispatch
semantics.

## State

The data the system carries between steps in an agent loop.

## Memory

Information preserved beyond a single immediate step or turn.

## Evaluation

A repeatable check that helps compare agent behavior, correctness, reliability, or usability.

## Reference Architecture

An external or internal architecture studied before deciding what Simple Long Horizon Agent should borrow or avoid.
