# Evals

This directory holds higher-level behavior checks and comparison cases.

Unit tests live in `tests/`. Evals are for feedback that compares agent
behavior across prompts, recipes, context views, runtime designs, or model
adapters.

Trajectory collection lives outside this directory because trajectories are
fact records, not scores. The temporary design-version collector is:

```bash
PYTHONPATH=src python3 scripts/collect_design_version_trajectories.py
```

The first runnable eval reads those trajectory records and writes separate
evaluation results:

```bash
PYTHONPATH=src python3 evals/evaluate_design_version_traces.py
```

Generated files under `evals/out/` are local artifacts and are ignored by git.

Record types:

- `trajectory`: raw messages/events/model turns for one run, written by scripts
- `eval_result`: pass/fail score and compact metrics, written by evals
- `training_example`: model-visible input plus assistant output, written by export scripts

Evals should help answer:

- Does the agent follow the expected loop?
- Does tool use behave predictably?
- Can we score model-call trajectories without scraping terminal output?
- Are changes making the system easier or harder to understand?
- Do reference architecture ideas improve the project in practice?
