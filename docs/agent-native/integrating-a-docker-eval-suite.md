# Integrating a Docker-based Eval Suite

A step-by-step guide for a developer or coding agent adding a new benchmark that
runs an agent **inside a Docker image** (like SWE-bench). It is the concrete
"how", on top of the architecture in
[ADR 0017](../decisions/0017-generic-containerized-eval-framework.md); for
running across machines see
[multi-machine-deployment.md](multi-machine-deployment.md).

Read when: adding a containerized benchmark. Skip for non-containerized checks
(unit tests, in-process demos).

## The contract in one paragraph

A suite is **two halves plus a registration**. The *host half* — a `Suite` under
`evals/<name>/suite.py` — says which image to run, sanitizes the instance, and
shapes the scorer row. The *container half* — a module shipped in the wheel at
`src/simple_agent_lab/evals/suites/<name>/` — says how to turn one instance into
an agent task and how to read the run's product back out. The framework supplies
everything else (container lifecycle, Python/uv bootstrap, the run-directory
layout, artifact movement, concurrency, host-reentrant batches). You write ~2
small files; you do **not** modify the eval image and you do **not** copy the
agent into it (it is `pip install`ed at container start — see "Runtime
injection" in ADR 0017).

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

Reference: `src/simple_agent_lab/evals/suites/swebench/container.py`.

## Step 2 — the host half (a `Suite`)

Create `evals/<name>/suite.py` with a class satisfying the `Suite` protocol:

- `name: str` and `container_module: str` (the dotted path from Step 1, e.g.
  `"simple_agent_lab.evals.suites.<name>.container"`).
- `container_plan(instance) -> ContainerPlan` — **all per-instance launch
  differences as data**: `image`, `workdir`, `shell`, `entrypoint`, `platform`,
  `network_mode`, and **`cap_add`** (capabilities the image's tests need, e.g.
  `("SYS_PTRACE",)` — easy to forget; an under-privileged container fails
  capability-dependent steps). No `if`-branching belongs in the runner; express
  variants here.
- `sanitize_instance(instance) -> dict` — drop gold/private fields before the
  agent sees the record (it is what gets written to `input/instance.json`).
- `prediction_record(instance, *, model_name, result) -> dict` — shape the
  scorer-facing row from `extract_result`'s product.

The host half may use heavy deps (the official harness, docker-py) — they stay
out of the core package because this file lives under `evals/`. Reference:
`evals/swebench/suite.py`.

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

A whole dataset: `run_dataset(..., concurrency=N)` (blocking, parallel) or, for
long runs the host should not babysit, `submit_dataset(...)` + `reconcile_dataset(...)`
(detach, leave, re-attach). See [`evals/README.md`](../../evals/README.md) for
both and the backend×store selection table.

## Step 4 — wire up outputs and a runner (ADR 0016)

- `evals/out/<name>/README.md` documenting the output layout, plus a
  `.gitignore` negation so that README survives.
- `runs/run_<name>.sh` convenience runner; `runs/setup_<name>.sh` if external
  resources (images, datasets) are needed.

## Step 5 — test without Docker

Cover the suite in `tests/unit/` with **no Docker and no network**:

- `FakeBackend` — drive `run_suite_instance` / `run_dataset` to check
  orchestration (sanitization, prediction shaping, store wiring).
- `LocalProcessBackend` + `provider="fake"` — run the *real* container half
  (`build_task` → loop → `extract_result`) in-process against a temp workspace,
  exactly as `tests/unit/test_evals_framework.py` does for SWE-bench.

## Checklist

- [ ] `container.py`: `build_task`, `extract_result(..., *, context=None)`;
      optional `prepare` / `agent_spec`; **stdlib + wheel imports only**.
- [ ] `suite.py`: `name`, `container_module`, `container_plan` (incl. `cap_add`),
      `sanitize_instance`, `prediction_record`.
- [ ] Image is reachable by the daemon (built locally / pulled from a registry —
      `LocalDockerBackend(pull=...)` controls the policy).
- [ ] Container can reach the model API; if running offline, a wheelhouse is
      mounted (see deployment doc).
- [ ] `evals/out/<name>/README.md` + `.gitignore` negation + `runs/run_<name>.sh`.
- [ ] Docker-free unit tests via `FakeBackend` and `LocalProcessBackend`.

## Common pitfalls

- **Forgetting `cap_add`** in `container_plan` → capability-dependent tests fail
  with confusing errors. Pull it from the image's test spec.
- **Omitting `*, context` on `extract_result`** → `prepare()`'s baseline/setup
  never reaches extraction, so the product is computed against the wrong state
  (e.g. a patch diffed against the wrong base).
- **Heavy imports in the container half** → it fails to import inside the image.
  Keep it stdlib + the installed wheel; put harness/docker deps in the host half.
- **Putting `if dataset == ...` branching in the runner** → instead express the
  difference as `ContainerPlan` data in `container_plan`.
- **Expecting the trajectory viewer to tail a remote run** → it tails local
  files; with a remote daemon use host-pull or point it where artifacts land
  (see deployment doc).
