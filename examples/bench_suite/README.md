# Example bench suite

A worked, end-to-end example of integrating a benchmark on the eval framework
(ADR generic-containerized-eval-framework): a small but **real** "fix the bug" suite — the candidate agent edits
`app.py` so it prints `42` — scored two ways: **in the run environment** (the
suite's `evaluate` hook *runs* the fixed program and writes the verdict into
`result.json`) and via **agent-as-judge** (a second agent re-runs the program and
judges). The "pipeline" between runs is plain Python over the shared
`ArtifactStore` — no pipeline engine, no separate scoring driver (ADR collapse-scorer-seam-into-run-primitive).

```bash
export OPENAI_MODEL=gpt-4o-mini
export OPENAI_AUTH_TOKEN=sk-...
# export OPENAI_BASE_URL=https://...   # optional, for a compatible gateway
# export API_KIND=openai-chat          # or openai-responses
uv run python -m examples.bench_suite.demo
```

It runs in-process (`LocalProcessBackend`, no Docker) against a real
OpenAI-compatible model — the task needs a model to locate and fix the bug.

## Files

- `candidate.py` — the bench suite. The host half `ExampleBenchSuite` is the
  whole surface: `launch_spec`, `task_input` (hides the gold `expected` output),
  `eval_inputs` (stages that gold under EVAL_KEY, which also turns on the hook).
  The container half is `build_task` + `extract_result` (which *runs* the edited
  `app.py` and captures its stdout as the product) plus `evaluate`, which runs the
  program, compares the output to the staged gold, and merges the verdict into
  `result.json`.
- `judge.py` — judge suite. The judge is *another agent* with a bash tool, so it
  can investigate the workspace to check the candidate's claim is grounded — the
  thing that distinguishes agent-as-judge from a one-shot score. Its verdict is
  written to the workspace and read back by `extract_result` as a `Judgment`.
- `demo.py` — the host glue: run the candidate with `run_dataset` (its verdict is
  already in `result.json`) → feed its `result.json` / `trajectory.jsonl`
  (product **and** process) as the judge's instance → run judge (agent-as-judge).

## What it demonstrates

- **Scoring in the run environment (ADR collapse-scorer-seam-into-run-primitive)** — the gold output rides on the
  instance as `expected`; `task_input` hides it from the agent; `eval_inputs`
  stages it (gold the agent never sees); the container-half `evaluate` hook *runs
  the fixed program* and writes the verdict beside the product in `result.json`.
  No separate scoring phase, and it reuses the run's fault tolerance.
- **A judge is just another agent run** — same `Suite` / container-half shape as
  any benchmark; nothing judge-specific in the framework. Scoring elsewhere is
  simply a *second run* (the judge) wired up through the store — the same
  primitive, not a bespoke scoring API.
- **The store is the bus between runs** — candidate artifacts become judge input
  by reading keys, so composing runs needs no new abstraction.
- **Same code, swappable backend** — the demo uses `LocalProcessBackend`; the
  identical suites run containerized by switching to `LocalDockerBackend()` (and
  a remote daemon by also switching the store). The suite code does not change.

## Going further

- **Different models for candidate vs judge**: pass distinct `provider` /
  `provider_env` to the candidate run (`run_dataset`) and the judge run.
- **Judge a panel**: run several judge suites (or the same one N times) over the
  same candidate artifacts and aggregate the verdicts in plain Python.
- **Judge the process, not just the result**: the judge already receives
  `candidate_steps` from the candidate's trajectory; give it the full
  `trajectory.jsonl` from the store to reason over how the answer was reached.
