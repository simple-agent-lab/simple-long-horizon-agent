# Harness Engineering Workflow

Simple Agent Lab should be developed as an agent-first project. Humans define
intent, constraints, taste, and acceptance criteria. Agents do the mechanical
work: reading context, editing files, running checks, recording decisions, and
improving the repository so later agents can work with less ambiguity.

Reference:
[Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/)

## Core Idea

The repository is the harness.

For this project, harness engineering means making the codebase easy for future
agents to inspect, modify, verify, and clean up. A good change does not only
change code. It also improves the local feedback loop that lets the next agent
understand what happened and what to do next.

## Development Loop

Use this loop for non-trivial work:

```text
read map -> define feedback signal -> inspect source of truth
  -> make the smallest useful change -> run the local check
  -> update docs or decisions -> leave a clear trace
```

In practice:

1. Read `AGENTS.md`, `README.md`, and the most relevant file under
   `docs/context/`.
2. Define the feedback signal before editing: test, smoke script, eval,
   screenshot, trace, or explicit review checklist.
3. If the task depends on external architecture ideas, check or update
   `docs/reference-architectures/`.
4. If the task changes architecture, add or update an ADR in `docs/decisions/`.
5. If the task is large enough to hand off, write or update a task spec in
   `docs/tasks/`.
6. Make the smallest implementation or documentation change that satisfies the
   task.
7. Run the narrowest useful command from `runs/`, `scripts/`, or `tests/`.
8. If the check is missing or hard to run, improve the harness instead of only
   explaining the gap.

## Testing And Feedback First

Testing and feedback are the first priority for implementation work.

Before changing code, identify what will prove the change works. If the feedback
signal does not exist, add the smallest useful one before or alongside the
implementation.

Use this order:

```text
feedback signal -> red or baseline check -> implementation -> green check
```

There is no default test suite yet because the core architecture is still being
settled. Until then, document the intended feedback signal in the task, ADR, or
architecture note.

For Simple Agent Lab, useful feedback signals include:

- Unit tests for message shape, context visibility, scheduling, and model
  payload conversion.
- Smoke runs for public examples and design versions.
- Tiny evals for prompt, recipe, or context-view behavior.
- Trace output that shows what an agent saw and emitted.
- A short review checklist when the question is architectural rather than
  executable.

Once implementation starts, treat a missing test, confusing failure, or
hard-to-run demo as a harness bug. Fixing the feedback loop is part of the task,
not cleanup after the task.

## Agent Legibility

Optimize for future agents reading the repository cold.

- Keep `AGENTS.md` short. It is a table of contents, not the full manual.
- Put durable knowledge in `docs/`, not in chat history.
- Keep runnable commands in `runs/` when they become reference workflows.
- Keep examples deterministic unless the example is about live model behavior.
- Make important state printable through `Message`, `State`, and trace output.
- Prefer explicit data conversion functions over hidden adapters.
- Write errors and test failures so another agent can act on them.

Anything important that only exists in a conversation, a screenshot, or a
person's memory is not part of the harness yet.

## Progressive Context

Context is scarce. Do not solve that by making one giant instruction file.

Use progressive disclosure:

- `AGENTS.md`: entry point and working contract.
- `README.md`: public project map and current status.
- `docs/context/`: product intent, code style, and development process.
- `docs/reference-architectures/`: external systems we learn from.
- `docs/decisions/`: accepted architecture choices.
- `docs/tasks/`: executable task specs and acceptance criteria.
- `runs/`: reproducible commands for demos, checks, and experiments.

When adding documentation, prefer a small page with a clear owner and purpose
over a broad page that mixes unrelated rules.

## Invariants Over Instructions

Do not rely on prose when a small check would be clearer.

Good harness improvements include:

- A deterministic demo command.
- A focused test for message visibility, scheduling, or model payload shape.
- A script that reproduces an experiment.
- A validation rule for docs links or example commands.
- A task template that forces acceptance criteria and verification.

This project is still intentionally small, so mechanical enforcement should stay
lightweight. The principle is not "add CI for everything." The principle is:
when agents keep making the same mistake, encode the correction into docs,
examples, tests, scripts, or local checks.

## Taste And Cleanup

Human taste should become repository-local guidance.

If a review discovers a better convention, do one of these:

- Update `docs/context/code-style.md` for style or readability preferences.
- Update an ADR when the choice affects architecture.
- Update a task template when future work should include a missing step.
- Add a small test or run script when the rule can be checked mechanically.

Do cleanup continuously in small patches. Avoid saving all simplification work
for a large rewrite.

## Acceptance Criteria For Changes

Every meaningful change should answer:

- What became easier for a student or future agent to understand?
- What source of truth changed?
- What command proves the change still works?
- Did this add a new abstraction, and if so, what confusion did it remove?
- Is any important context still only present in chat?

If the answer to the last question is yes, update the repository before ending
the task.
