# ADR 0011: Keep Benchmark Suites As Eval Adapters

## Status

Accepted. Superseded for the containerized case by ADR 0017 (generic
containerized eval framework): the host launcher described below
(`containerized_agent.py` / `in_container_runner.py`) has since been replaced by
`SwebenchSuite` + `run_suite_instance`, with shared host helpers in
`evals/swebench/harness.py`. The file names in this record are kept as the
historical "before" state. Its core principles still hold (raw trajectories are
fact records, the runtime core stays Docker/dataset-free, scoring is a separate
path).

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

- `containerized_agent.py` starts the SWE-bench instance container and runs the
  Simple Agent Lab runner inside it.
- `in_container_runner.py` writes Simple Agent Lab trajectory records plus the
  official SWE-bench prediction JSONL shape from inside the container.
- `patch_extract.py` collects the final `model_patch` while filtering generated
  files.
- `evaluate_predictions.py` invokes or normalizes the official SWE-bench harness
  output into `EvalResult` records.
- The SWE-bench adapter unit tests (run via `run_ci.sh`) verify the adapter's
  unit-smoke path without Docker.
- `runs/run_swebench_gold_smoke.sh` verifies the external SWE-bench harness when
  the optional dependency and Docker are available.

The core runtime should not know about SWE-bench datasets, Docker, gold patches,
or official report directories.

## Consequences

New scene-level suites can follow the same shape without creating a benchmark
framework up front:

```text
evals/<suite>/containerized_agent.py
evals/<suite>/evaluate_predictions.py
runs/run_<suite>_smoke.sh
```

If several suites begin duplicating the same small data conversion code, promote
only that repeated shape into `src/simple_agent_lab/`. Do not add a registry,
plugin loader, or database before the second or third suite proves the need.

For training, suite-specific scores remain labels attached after collection.
Raw trajectories must stay reusable when scoring rules change.
