# Adding a Docker-based Eval Suite

The concrete "how" for the framework in [`evals/README.md`](../evals/README.md).
Read when adding a benchmark that runs an agent inside a Docker image.

## The contract

A suite is **two halves plus a registration**:

```text
evals/<name>/suite.py                      ← host half (a Suite): which image, hide gold
src/simple_agent_lab/evals/suites/<name>/  ← container half (ships in the wheel)
    container.py                             build_task / extract_result
```

The framework supplies everything else: container lifecycle, Python/uv
bootstrap, run-directory layout, artifact movement, concurrency, and
host-reentrant batches. You do **not** modify the eval image or copy the agent
into it — it is `pip install`ed at container start (see
[multi-machine-eval.md](multi-machine-eval.md)).

## Container half

Imports **only the standard library and the installed `simple_agent_lab`
wheel** — it runs inside the image, where nothing else is guaranteed.

Required:

- `build_task(instance, *, workdir) -> ContentInput` — the model-visible task.
- `extract_result(workspace, instance, *, context=None) -> Mapping[str, Any]` —
  the run's product, read from the *workspace* (the agent's effect on the
  world), not from chat. **Declare `context` even if you ignore it**: the runner
  only passes `prepare()`'s output when the signature has it.

Optional: `prepare` (pre-run setup; its return is threaded in as `context`),
`agent_spec` / `build_agent`, `evaluate` (in-environment scoring), and
`memory_artifacts` (durable products for persistent memory).

Reference: `src/simple_agent_lab/evals/suites/swebench/container.py`. For a
product that is the *whole workspace* rather than a diff, and for isolating each
agent command's network, see
[`evals/programbench/README.md`](../evals/programbench/README.md).

## Host half

A class satisfying the `Suite` protocol, with `name`, `container_module`, and:

- `launch_spec(instance) -> LaunchSpec` — **every per-instance launch difference
  as data**: image, workdir, shell, entrypoint, platform, network_mode, and
  `cap_add`. No `if`-branching belongs in the runner.
- `task_input(instance) -> dict` — drop gold/private fields before the agent
  sees the record.
- `eval_inputs(instance) -> Mapping | None` — gold scoring inputs staged under
  `input/eval.json`. **Staging gold is the toggle** that turns the container
  half's `evaluate` hook on; return `None` to score elsewhere.

The host half may use heavy deps (the official harness, docker-py) because it
lives under `evals/`, outside the core package. Reference:
`evals/swebench/suite.py`.

## Register and test

- Add `runs/_benches/<name>.py` and register it in `runs/run_bench.py`.
- Document setup in `evals/<name>/README.md`; note the output root in
  [`evals/out/README.md`](../evals/out/README.md) if its layout differs from the
  shape documented there.
- Cover it in `tests/unit/` with **no Docker and no network**: `FakeBackend` for
  orchestration, and `LocalProcessBackend` + `provider="fake"` to run the real
  container half in-process against a temp workspace (see
  `tests/unit/test_evals_framework.py`).

## Common pitfalls

- **Forgetting `cap_add`** → capability-dependent tests fail with confusing
  errors. Pull the set from the image's test spec.
- **Omitting `*, context` on `extract_result`** → `prepare()`'s baseline never
  reaches extraction, so the product is computed against the wrong state.
- **Heavy imports in the container half** → it fails to import inside the image.
- **`if dataset == ...` in the runner** → express it as `LaunchSpec` data.
- **Expecting the trace viewer to tail a remote run** → it tails local files.
