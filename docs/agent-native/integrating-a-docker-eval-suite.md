# Integrating a Docker-based Eval Suite

A step-by-step guide for a developer or coding agent adding a new benchmark that
runs an agent **inside a Docker image** (like SWE-bench). It is the concrete
"how", on top of the architecture in
[ADR generic-containerized-eval-framework](../decisions/20260531-generic-containerized-eval-framework.md); for
running across machines see
[multi-machine-deployment.md](multi-machine-deployment.md).

Read when: adding a containerized benchmark. Skip for non-containerized checks
(unit tests, in-process demos).

## The contract in one paragraph

A suite is **two halves plus a registration**. The *host half* — a `Suite` under
`evals/<name>/suite.py` — says which image to run, hides gold from the agent, and
(optionally) stages gold for in-environment scoring. The *container half* — a module shipped in the wheel at
`src/simple_agent_lab/evals/suites/<name>/` — says how to turn one instance into
an agent task and how to read the run's product back out. The framework supplies
everything else (container lifecycle, Python/uv bootstrap, the run-directory
layout, artifact movement, concurrency, host-reentrant batches). You write ~2
small files; you do **not** modify the eval image and you do **not** copy the
agent into it (it is `pip install`ed at container start — see "Runtime
injection" in ADR generic-containerized-eval-framework).

```
your suite =
  evals/<name>/suite.py                              ← host half (a Suite)
  src/simple_agent_lab/evals/suites/<name>/          ← container half (ships in wheel)
      __init__.py
      container.py   build_task / extract_result (+ optional prepare / agent_spec)
```

## Step 1 — the container half (runs inside the image)

Create `src/simple_agent_lab/evals/suites/<name>/container.py`. It must import
**only the standard library and the installed `simple_agent_lab` runtime** — it
runs inside the image, where nothing else is guaranteed.

Required:

- `build_task(instance, *, workdir) -> str` — the model-visible task. `workdir`
  is where the agent's bash tool runs (the repo/work area in the image).
- `extract_result(workspace, instance, *, context=None) -> Mapping[str, Any]` —
  the run's **product**, read from the *workspace* (the agent's effect on the
  world), not from chat. For SWE-bench this is `{"model_patch": <git diff>}`.
  **Declare the `context` keyword** even if you ignore it — the runner only
  passes `prepare()`'s output when the signature has it.

Optional:

- `prepare(workspace, instance) -> Mapping[str, Any]` — pre-run setup that must
  happen *before* the agent edits (snapshot a baseline commit, install
  ignore-rules, checkout). Its return value is threaded into `extract_result` as
  `context`.
- `agent_spec() -> AgentSpec` — the agent's name/role/system-prompt and flavor
  (`"bash"` or `"bash_task"`). Or, for full control, define
  `build_agent(*, provider, cwd, request_extra) -> Agent` and the runner uses it
  instead.
- `memory_artifacts(workspace, instance, *, context=None)` — durable products to
  save with persistent memory, returned as `FilesystemArtifact` values. Use this
  for products like final patches or report files. The generic runner injects it
  into memory's artifact builder when `SAL_MEMORY_HOME` is set; no suite-specific
  memory branch belongs in the runner.

Reference: `src/simple_agent_lab/evals/suites/swebench/container.py`. For a
product that is the *whole workspace* rather than a diff (base64-encoded tar in
`result.json`) and for isolating *each agent command's* network (`unshare --net`
via the bash tool's `exec_prefix`), see the ProgramBench adapter and ADR
`programbench-reverse-engineering-adapter`.

## Step 2 — the host half (a `Suite`)

Create `evals/<name>/suite.py` with a class satisfying the `Suite` protocol:

- `name: str` and `container_module: str` (the dotted path from Step 1, e.g.
  `"simple_agent_lab.evals.suites.<name>.container"`).
- `launch_spec(instance) -> LaunchSpec` — **all per-instance launch
  differences as data**: `image`, `workdir`, `shell`, `entrypoint`, `platform`,
  `network_mode`, and **`cap_add`** (capabilities the image's tests need, e.g.
  `("SYS_PTRACE",)` — easy to forget; an under-privileged container fails
  capability-dependent steps). No `if`-branching belongs in the runner; express
  variants here.
- `task_input(instance) -> dict` — drop gold/private fields before the
  agent sees the record (it is what gets written to `input/instance.json`).
- `eval_inputs(instance) -> Mapping | None` — gold/private scoring inputs staged
  under `input/eval.json` (EVAL_KEY, kept out of the agent-visible instance) for
  the container half's `evaluate` hook. **Staging gold is the toggle** that turns
  in-environment scoring on; **return `None`** to score elsewhere (a follow-up
  run or the official harness). There is no `scorer()` method and no separate
  score driver (ADR collapse-scorer-seam-into-run-primitive): in-environment scoring is the `evaluate` hook, whose
  verdict is merged into `out/result.json`. See
  [`evals/README.md`](../../evals/README.md#scoring).

The host half may use heavy deps (the official harness, docker-py) — they stay
out of the core package because this file lives under `evals/`. Reference:
`evals/swebench/suite.py` (+ `evals/swebench/evaluate_predictions.py` for the
official harness and the `reuse_eval_row` host-grading helper).

## Step 3 — run it

Local development (no Docker, fast, debuggable) uses the in-process backend; the
*same suite* runs containerized by swapping the backend:

```python
from simple_agent_lab.evals import run_suite_instance, LocalDockerBackend, LocalDirStore

run_suite_instance(
    suite=MySuite(),
    instance=one_instance,
    backend=LocalDockerBackend(),       # or LocalProcessBackend(workspace=...) for local dev
    store=LocalDirStore("evals/out/<name>"),
    run_root="evals/out/<name>",
    run_id="run-1",
    provider="openai", provider_env={"OPENAI_MODEL": "...", "OPENAI_AUTH_TOKEN": "..."},
)
```

Persistent memory is optional. For local Docker runs, pass
`LocalDockerBackend(memory_home="evals/out/<name>/memory")`; the backend
bind-mounts that directory and sets `SAL_MEMORY_HOME` for the in-container
runner. The container half's optional `memory_artifacts(...)` hook then lets
memory capture run products before `extract_result` reads the final result.

A whole dataset: `run_dataset(..., concurrency=N)` (blocking, parallel) or, for
long runs the host should not babysit, `submit_dataset(...)` + `reconcile_dataset(...)`
(detach, leave, re-attach). See [`evals/README.md`](../../evals/README.md) for
both and the backend×store selection table.

## Step 4 — wire up outputs and a runner (ADR eval-output-directory-convention)

- `evals/out/<name>/README.md` documenting the output layout, plus a
  `.gitignore` negation so that README survives.
- `runs/run_<name>.sh` convenience runner; `runs/setup_<name>.sh` if external
  resources (images, datasets) are needed.

## Step 5 — test without Docker

Cover the suite in `tests/unit/` with **no Docker and no network**:

- `FakeBackend` — drive `run_suite_instance` / `run_dataset` to check
  orchestration (sanitization, store wiring).
- `LocalProcessBackend` + `provider="fake"` — run the *real* container half
  (`build_task` → loop → `extract_result`, plus the `evaluate` hook when gold is
  staged) in-process against a temp workspace, exactly as
  `tests/unit/test_evals_framework.py` does for SWE-bench and the in-env scoring
  test.

## Checklist

- [ ] `container.py`: `build_task`, `extract_result(..., *, context=None)`;
      optional `prepare` / `agent_spec` / `evaluate` (in-env scoring) /
      `memory_artifacts` (persistent-memory products);
      **stdlib + wheel imports only**.
- [ ] `suite.py`: `name`, `container_module`, `launch_spec` (incl. `cap_add`),
      `task_input`, `eval_inputs()` (return `None` to score elsewhere; non-`None`
      stages gold and turns the `evaluate` hook on).
- [ ] Image is reachable by the daemon (built locally / pulled from a registry —
      `LocalDockerBackend(pull=...)` controls the policy).
- [ ] Container can reach the model API; if running offline, a wheelhouse is
      mounted (see deployment doc).
- [ ] `evals/out/<name>/README.md` + `.gitignore` negation + `runs/run_<name>.sh`.
- [ ] Docker-free unit tests via `FakeBackend` and `LocalProcessBackend`.

## Common pitfalls

- **Forgetting `cap_add`** in `launch_spec` → capability-dependent tests fail
  with confusing errors. Pull it from the image's test spec.
- **Omitting `*, context` on `extract_result`** → `prepare()`'s baseline/setup
  never reaches extraction, so the product is computed against the wrong state
  (e.g. a patch diffed against the wrong base).
- **Heavy imports in the container half** → it fails to import inside the image.
  Keep it stdlib + the installed wheel; put harness/docker deps in the host half.
- **Putting `if dataset == ...` branching in the runner** → instead express the
  difference as `LaunchSpec` data in `launch_spec`.
- **Expecting the trajectory viewer to tail a remote run** → it tails local
  files; with a remote daemon use host-pull or point it where artifacts land
  (see deployment doc).
