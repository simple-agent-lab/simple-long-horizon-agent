# Components

## Workspace

The shared state for a run.

It should be plain and inspectable. The first version only needs a task board, blackboard, artifact list, and decision list.

## TaskBoard

Represents work items.

It supports research into assignment strategies, dependency handling, task claiming, and review loops.

## Blackboard

Stores shared notes.

Notes should be typed lightly:

```text
claim
question
evidence
concern
decision
```

## WorkspacePolicy

Controls what each role can observe and where it normally writes.

This is not a security policy. It is an experimental visibility and coordination policy.

## Scheduler

Chooses the next agent or set of agents based on workspace state.

Examples:

- Manager first, then workers.
- Any agent with assigned tasks.
- Reviewer after artifacts appear.
- Coordinator after all tasks are done.

## RunTraceStore and Evaluator

Trace every workspace update.

Evaluation should focus on organization behavior, not only final answer quality.

