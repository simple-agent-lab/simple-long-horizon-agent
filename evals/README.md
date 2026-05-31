# Evals

This directory holds higher-level behavior checks and comparison cases.

Unit tests live in `tests/`. Evals are for feedback that compares agent
behavior across prompts, recipes, context views, runtime designs, or model
adapters.

Trajectory collection for shared demos can live outside this directory because
trajectories are fact records, not scores. Scene-level suite adapters can keep
their collector beside their scorer when that makes the suite easier to read.
The first suite example is the local SWE-bench adapter smoke check:

```bash
uv run python -m unittest \
  tests.unit.test_swebench_patch_extract \
  tests.unit.test_swebench_containerized_agent \
  tests.unit.test_swebench_evaluate_predictions
```

That script runs the focused patch extraction, containerized-agent, and
prediction-evaluation unit tests without installing SWE-bench or running
Docker.

Generated files under `evals/out/` are local artifacts and are ignored by git.
Each subdirectory keeps a committed README describing the expected output layout
so that every user sees the same directory skeleton after cloning. See
[`evals/out/README.md`](out/README.md) for the full structure.

Record types:

- `trajectory`: raw messages/events/model turns for one run, written by scripts
- `eval_result`: pass/fail score and compact metrics, written by evals
- `training_example`: model-visible input plus assistant output, written by export scripts

## Adding a Containerized Suite

The generic framework lives in `simple_agent_lab.evals` (ADR 0017). It supplies
the container lifecycle, the Python/uv bootstrap, the run-directory convention,
and artifact transport, so a new Docker benchmark only implements what is
genuinely suite-specific:

1. A **host half** — a `Suite` (3 methods + 2 attributes):

   - `container_plan(instance)` -> `ContainerPlan` (image, workdir, shell,
     entrypoint as *data* — no `if`-branching in the runner).
   - `sanitize_instance(instance)` — drop gold/private fields before the agent
     sees the record.
   - `prediction_record(instance, *, model_name, result)` — shape the
     scorer-facing row.
   - `name` and `container_module` (dotted path to the container half).

2. A **container half** — a module exposing `build_task(instance, *, workdir)`
   and `extract_result(workspace, instance)` (the "product", e.g. a `git diff`).
   Optional: `prepare(workspace, instance)` for pre-run setup (its return value
   is threaded back into `extract_result` as `context`) and `agent_spec()` for
   the prompt/role and bash-vs-bash_task flavor. The generic in-container runner
   (`simple_agent_lab.evals.in_container`) imports this module and owns the
   agent loop, retry, and trace push — so a suite never re-implements them.

Then drive one instance:

```python
from simple_agent_lab.evals import (
    run_suite_instance, LocalDockerBackend, BindMountTransport, bootstrap_script,
)

run_suite_instance(
    suite=MySuite(),
    instance=instance,
    backend=LocalDockerBackend(),        # cloud later: a remote/k8s backend
    transport=BindMountTransport(),      # cloud later: CopyOutTransport
    command=("bash", "-lc", bootstrap_script(runner_argv=(...,))),
    run_root=Path("evals/out/mysuite"),
    run_id="run-1",
)
```

The container writes its raw product to ``out/result.json``; the host shapes
``out/prediction.jsonl`` from it via `prediction_record`, so prediction
formatting stays on the host with the rest of the suite config.

The same suite runs against a cloud daemon by swapping the `backend` /
`transport` / trace `TraceSink` — the seams are `Protocol`s, so the suite and
the runner do not change. `evals/swebench/{suite.py,container.py}` is the
reference: `suite.py` is the host half, `container.py` is the two functions.
`FakeBackend` runs the whole flow without Docker for tests. Still follow the
output-directory checklist in [`out/README.md`](out/README.md).

Evals should help answer:

- Does the agent follow the expected loop?
- Does tool use behave predictably?
- Can we score model-call trajectories without scraping terminal output?
- Are changes making the system easier or harder to understand?
- Do reference architecture ideas improve the project in practice?
