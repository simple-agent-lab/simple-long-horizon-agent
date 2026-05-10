# ADR 0011: Keep Benchmark Suites As Eval Adapters

## Status

Accepted

## Context

The project is starting to add built-in evaluation suites. SWE-bench is the first
reference suite because it exercises a realistic coding-agent loop: read an
issue, edit a repository, produce a patch, and let an external harness decide
whether the issue is resolved.

ADR 0008 already separates raw trajectory records, evaluation results, and
training examples. That split should continue to hold for large external
benchmarks.

SWE-bench also brings heavy requirements: a separate package, dataset access,
Docker images, and repository-specific test execution. Those requirements do
not belong in the minimal runtime core.

## Decision

Represent benchmark integrations as suite-specific eval adapters under `evals/`.

For SWE-bench, the first adapter lives under `evals/swebench/`:

- `prepare_workspace.py` creates a per-instance agent workspace from the
  SWE-bench repo and base commit.
- `collect_trajectories.py` writes Simple Agent Lab trajectory records plus the
  official SWE-bench prediction JSONL shape.
- `evaluate_predictions.py` invokes or normalizes the official SWE-bench harness
  output into `EvalResult` records.
- `runs/run_swebench_smoke.sh` verifies local adapter plumbing without Docker.
- `runs/run_swebench_gold_smoke.sh` verifies the external SWE-bench harness when
  the optional dependency and Docker are available.

The core runtime should not know about SWE-bench datasets, Docker, gold patches,
or official report directories.

## Consequences

New scene-level suites can follow the same shape without creating a benchmark
framework up front:

```text
evals/<suite>/collect_trajectories.py
evals/<suite>/evaluate_predictions.py
runs/run_<suite>_smoke.sh
```

If several suites begin duplicating the same small data conversion code, promote
only that repeated shape into `src/simple_agent_lab/`. Do not add a registry,
plugin loader, or database before the second or third suite proves the need.

For training, suite-specific scores remain labels attached after collection.
Raw trajectories must stay reusable when scoring rules change.
