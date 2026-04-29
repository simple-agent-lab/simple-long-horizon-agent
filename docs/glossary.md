# Glossary

## Agent

A system that uses a model to decide what to do next, often across multiple steps.

## Agent Loop

The repeated control flow where the system observes state, asks a model for the next action, executes that action, and records the result.

## Message

The unit of communication between agents and the provider-neutral model message.
In the current core it has model-facing `role` and `content`, plus lab-facing
`sender`, `target`, `kind`, `channel`, and structured `data`.

## Event

A timestamped record in the run trace. In the current core, each event wraps one message and gives it a step number.

## Tool

A callable capability exposed to the agent, such as reading a file, searching data, calling an API, or running a calculation.

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
