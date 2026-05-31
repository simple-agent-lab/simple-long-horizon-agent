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
and the one artifact seam, so a new Docker benchmark only implements what is
genuinely suite-specific:

1. A **host half** — a `Suite` (3 methods + 2 attributes), under
   `evals/<suite>/suite.py`:

   - `container_plan(instance)` -> `ContainerPlan` (image, workdir, shell,
     entrypoint as *data* — no `if`-branching in the runner).
   - `sanitize_instance(instance)` — drop gold/private fields before the agent
     sees the record.
   - `prediction_record(instance, *, model_name, result)` — shape the
     scorer-facing row from the container's result.
   - `name` and `container_module` (dotted path to the container half).

2. A **container half** — a module exposing `build_task(instance, *, workdir)`
   and `extract_result(workspace, instance)` (the "product", e.g. a `git diff`).
   Optional: `prepare(workspace, instance)` for pre-run setup (its return value
   is threaded back into `extract_result` as `context`), `agent_spec()` for the
   prompt/role and bash-vs-bash_task flavor, or `build_agent(...)` for full
   control. It **ships in the wheel** under
   `src/simple_agent_lab/evals/suites/<suite>/` (importing only stdlib + the
   installed runtime), so the in-container runner imports it with zero copying.

Then drive one instance. The same call runs locally or across machines — swap
the backend, not the code:

```python
from simple_agent_lab.evals import (
    run_suite_instance, LocalProcessBackend, LocalDockerBackend, LocalDirStore,
)

backend = LocalProcessBackend(workspace=ws)  # local dev: in-process, no Docker
# backend = LocalDockerBackend()             # one machine, containerized
# backend = RemoteDockerBackend(host="...")  # multi-machine (future)

run_suite_instance(
    suite=MySuite(),
    instance=instance,
    backend=backend,
    store=LocalDirStore(run_root),       # remote daemon: HostHttpStore (no S3) / S3Store
    run_root=Path("evals/out/mysuite"),
    run_id="run-1",
    provider_env={"OPENAI_MODEL": "...", "OPENAI_AUTH_TOKEN": "..."},
)
```

`LocalProcessBackend` runs the *exact* suite container half a container would
run, just in this process — fast iteration and a debugger during development —
then deploy containerized by swapping the backend. The execution environment is
the only real difference: a container's workspace + toolchain come from the
image; in-process you pass a local `workspace`. See
`examples/agent_judge/demo.py` for a candidate → agent-judge pipeline built this
way (plain Python over the shared store, no pipeline abstraction).

Inputs, the result, and the live trajectory all flow through the one `store`:
the host writes `input/instance.json`, the container writes `out/result.json`
and re-`put`s `out/trajectory.jsonl` on a cadence (the live trace), and the host
shapes `out/prediction.jsonl` from the result via `prediction_record`.

The same suite runs against a remote daemon by swapping `backend` / `store` —
both are `Protocol`s, so the suite and runner do not change. `HostHttpStore`
needs **no third-party middleware** (the host runs a stdlib HTTP server).
Reference: `evals/swebench/suite.py` (host half) +
`simple_agent_lab.evals.suites.swebench.container` (the two functions).
`FakeBackend` runs the whole flow without Docker for tests. Still follow the
output-directory checklist in [`out/README.md`](out/README.md).

Evals should help answer:

- Does the agent follow the expected loop?
- Does tool use behave predictably?
- Can we score model-call trajectories without scraping terminal output?
- Are changes making the system easier or harder to understand?
- Do reference architecture ideas improve the project in practice?
