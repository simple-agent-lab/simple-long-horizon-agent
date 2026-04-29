# Code Style

Simple Agent Lab should follow a nanochat-inspired style: minimal, direct, and easy to modify.

Reference: [karpathy/nanochat](https://github.com/karpathy/nanochat)

## Principles

- Prefer readable scripts over framework scaffolding.
- Keep one clear executable path for each experiment.
- Put shared utilities in small modules with plain names.
- Use explicit data structures and direct control flow.
- Keep configuration close to the script that uses it.
- Make the default run work without extra services.
- Favor comments that explain intent, tradeoffs, or a non-obvious trick.
- Avoid abstractions that only hide a few lines of code.

## Python Practice

The Python code should feel strict, testable, small, readable, and natural for
both humans and coding agents.

### Strong Types, Strong Validation

- Type public functions, dataclasses, and protocol boundaries.
- Keep internal data shapes explicit. Prefer small dataclasses over loose dicts
  when a value crosses module boundaries.
- Validate inputs at the boundary where they enter the system, not deep inside
  unrelated logic.
- Use narrow literals or enums only when they prevent real mistakes. Do not add
  an enum for a value that is still experimental.
- Keep provider-specific response objects out of core state. Convert them into
  project-owned types such as `Message`.
- Raise clear errors with the invalid value and the expected shape.

### Testability

- Keep pure logic separate from I/O. A function that routes messages should not
  also call a model API.
- Prefer functions that accept data and return data. This makes recipes easy to
  unit test.
- Make side effects explicit through names such as `send`, `run`, `save`, or
  `load`.
- Add small tests around invariants: message visibility, schedule order, model
  payload conversion, and trace recording.
- Keep demo agents deterministic unless the example is specifically about live
  model behavior.
- Design model clients so they can be replaced by a fake client in tests.

### Simplicity

- Start with the smallest type that explains the idea.
- Add a new module only when a file is doing two clearly different jobs.
- Add a new class only when it owns state or gives a repeated concept a clear
  name.
- Prefer one obvious code path over many extension points.
- Keep defaults runnable without external services.
- Do not introduce framework-style registries until plain lists or functions
  become genuinely confusing.

### Readability

- Use names that match the project vocabulary: `Message`, `State`,
  `context_view`, `run`, `model_messages`.
- Put the important fields first. For example, `Message(role, content, ...)`
  should feel close to the model API shape.
- Keep control flow direct. Avoid decorators, metaclasses, dynamic imports, or
  implicit plugin loading in the core.
- Prefer short files with a clear top-to-bottom reading order.
- Use comments to explain why a boundary exists, not what each line does.
- Keep examples copy-pasteable and beginner-friendly.

### Natural for Agents

- Make the state inspectable by printing or reading one object, not by chasing
  hidden callbacks.
- Keep important transitions visible in `state.events`.
- Prefer explicit conversion functions such as `model_messages(...)` over magic
  adapter behavior.
- Keep errors actionable for another agent: include the field, value, and
  expected contract.
- Avoid clever compression of concepts. If two ideas are different in the
  research question, give them different names.
- Preserve local reasoning: an agent should understand a recipe by reading one
  script plus `simple_agent_lab/core.py`.

## Repository Shape

Use this shape as the project grows:

```text
simple_agent_lab/   # small reusable runtime pieces
scripts/            # command-line entrypoints
runs/               # reproducible experiment shell scripts
examples/           # readable standalone examples
tests/              # focused behavioral tests
docs/               # design notes and decisions
```

For now, `simple_agent_lab/core.py` is the small reusable runtime, and
`scripts/run_tiny_demo.py` is the reference demo.

## Script Style

Each runnable script should:

- Start with a module docstring that says what it demonstrates and how to run it.
- Use `argparse` for the small number of knobs that matter.
- Keep the default path simple.
- Print a useful final result and a compact trace or report.
- Make it obvious where to replace toy logic with a real model call.

## Experiment Style

Experiments should expose one clear research variable at a time:

- Message topology.
- Pipeline handoff.
- Workspace visibility.
- Agent role assignment.
- Context size.
- Evaluation metric.

If a script needs many knobs, add a `runs/*.sh` file with the recommended command.
