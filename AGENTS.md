# Agent Collaboration Guide

The working contract for AI agents and human contributors. This is the only
meta-document in the repo: everything else under `docs/` is a guide to one
subsystem, and code plus tests are the source of truth for behavior.

## Mission

Simple Agent Lab should make agent systems easier to understand, modify, and
teach. Every implementation choice should preserve that.

- A minimal agent loop understandable by inspection.
- Clear separation between model calls, tools, memory/state, and orchestration.
- Easy to customize for students, workshops, and small team experiments.
- Documentation that explains *why* the system works, not only how to run it.

Non-goals:

- No heavy framework before the simple implementation exists.
- No provider-specific complexity unless it teaches a concrete idea.
- No optimizing for production scale before the educational path is clear.
- No hiding core behavior behind magic configuration.

## Working Loop

1. Read `README.md` and this file. Inspect the current source of truth before
   editing — do not rely on chat memory.
2. Define the feedback signal *first*: unit test, smoke run, eval, trace, or an
   explicit review checklist. If it does not exist, add the smallest useful one
   alongside the change.
3. Make the smallest useful change.
4. Run the narrowest useful check and report the command. The full gate is
   `bash runs/dev/run_ci.sh`.

If the task depends on an external architecture idea, capture a local note under
`docs/reference-architectures/` first (gitignored; only the convention is
shared). If it changes an architectural boundary, update that subsystem's doc
and its validation in the same change.

## Environment and Commands

`uv`-managed project (`[tool.uv] managed = true`). Run **every** command through
`uv` so it uses the pinned environment.

- `uv run python -m ...`, never a bare `python` / `python3` / `pip`.
- `uv add` / `uv remove` / `uv sync` for dependencies; edit `pyproject.toml` for
  dependency-group or optional-dependency changes, then `uv sync`.
- Tests use the stdlib `unittest` runner:
  `uv run python -m unittest tests.unit.test_programbench_suite -v`.

See [`docs/development.md`](docs/development.md) for the full quality gate and
the known type-checker false positives.

## Editing Expectations

- Prefer small, explicit modules over clever abstractions. Add a new module only
  when a file is doing two clearly different jobs; a new class only when it owns
  state or names a repeated concept.
- Use plain names, direct control flow, and short files with a top-to-bottom
  reading order. Avoid decorators, metaclasses, dynamic imports, and implicit
  plugin loading in the core.
- Type public functions, dataclasses, and protocol boundaries. Validate inputs
  where they enter the system, not deep inside unrelated logic. Raise errors
  carrying the invalid value and the expected shape.
- Keep provider-specific response objects out of core state — convert them to
  project-owned types such as `Message` at the boundary.
- Keep pure logic separate from I/O, so recipes are unit-testable and model
  clients can be replaced by a fake in tests.
- Comments explain intent, tradeoffs, or a non-obvious trick — not what a line
  does. If a comment restates the code, delete it.
- Preserve beginner readability; keep examples small, runnable, and free of
  external services by default.

## Documentation

Keep the set small. A doc under `docs/` earns its place only by holding
decisions, constraints, and contracts that the code does not contain — owner
guidance, stop conditions, architectural boundaries, operational runbooks.

- Do not rewrite visible implementation details into prose. Code wins.
- Do not document an API that does not exist yet. Add the doc with the code.
- Do not create a one-off doc for a single bug, feature, or test.
- Do not add "updated on" / "as of <date>" stamps; express freshness through
  paths, commands, and exports that can be checked against the repo.
- When two docs would state the same fact, one of them links instead.
