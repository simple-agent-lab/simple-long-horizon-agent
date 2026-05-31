# Agent-as-Judge demo

A worked example of the eval framework (ADR 0017) used for **agent-as-judge**:
one agent produces a candidate answer, a second agent *judges* it. Both are
ordinary runs through `run_suite_instance`; the "pipeline" between them is plain
Python over the shared `ArtifactStore` — no pipeline engine.

```bash
uv run python -m examples.agent_judge.demo
```

It runs entirely in-process with the deterministic fake provider
(`LocalProcessBackend`, no Docker, no network), so it works anywhere.

## Files

- `candidate.py` — candidate suite: `build_task` + `extract_result` (host half
  is a trivial `Suite`). The agent inspects a workspace and answers.
- `judge.py` — judge suite. The judge is *another agent* with a bash tool, so it
  can investigate the workspace to check the candidate's claim is grounded — the
  thing that distinguishes agent-as-judge from a one-shot score. Its verdict is
  written to the workspace and read back by `extract_result` as a `Judgment`.
- `demo.py` — the host glue: run candidate → read its `result.json` /
  `trajectory.jsonl` from the store → feed both (product **and** process) as the
  judge's instance → run judge.

## What it demonstrates

- **A judge is just another agent run** — same `Suite` / container-half shape as
  any benchmark; nothing judge-specific in the framework.
- **The store is the bus between runs** — candidate artifacts become judge input
  by reading keys, so composing runs needs no new abstraction.
- **Same code, swappable backend** — the demo uses `LocalProcessBackend`; the
  identical suites run containerized by switching to `LocalDockerBackend()` (and
  a remote daemon by also switching the store). The suite code does not change.

## Going further

- **Different models for candidate vs judge**: pass distinct `provider` /
  `provider_env` to each `run_suite_instance` call.
- **Judge a panel**: run several judge suites (or the same one N times) over the
  same candidate artifacts and aggregate the verdicts in plain Python.
- **Judge the process, not just the result**: the judge already receives
  `candidate_steps` from the candidate's trajectory; give it the full
  `trajectory.jsonl` from the store to reason over how the answer was reached.
