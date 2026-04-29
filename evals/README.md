# Evals

This directory will hold higher-level behavior checks and comparison cases.

Unit tests will live in `tests/`. Evals are for feedback that compares agent
behavior across prompts, recipes, context views, or model adapters.

Future evals should help answer:

- Does the agent follow the expected loop?
- Does tool use behave predictably?
- Are changes making the system easier or harder to understand?
- Do reference architecture ideas improve the project in practice?

No eval harness is implemented yet. Architecture notes should describe what
would become an eval once the core is selected.
