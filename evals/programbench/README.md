# ProgramBench Eval Adapter

[ProgramBench](https://github.com/facebookresearch/programbench) is a
*reverse-engineering* benchmark: each instance's workspace holds a compiled
`./executable` plus its bundled docs, and the agent must write a brand-new
codebase from scratch whose `./compile.sh` rebuilds an executable with identical
behavior — inferring behavior **only** by running `./executable` and reading the
docs (no original source, no wrapping the binary, no decompiling).

This adapter maps ProgramBench onto the generic `Suite` protocol
(`generic-containerized-eval-framework`) as a peer of the SWE-bench adapter:

- `suite.py` — `ProgrambenchSuite`: `launch_spec` (image `:task_cleanroom`,
  workdir `/workspace`, `network_mode=host`, `cap_add=("SYS_ADMIN",)`),
  `task_input` (drops repo/commit/gold), `eval_inputs` (returns `None` — scoring
  is a follow-up CLI, not an in-environment hook).
- `harness.py` — host helpers: load instances from the installed `programbench`
  package (the task set ships *in the wheel*, not via HuggingFace), resolve the
  Docker image name, provider env / dotenv, and the offline wheelhouse (reused
  from the SWE-bench host helpers).
- `evaluate_submissions.py` — the scorer driver: collect each run's
  `result.json`, decode the submission, and run the official ProgramBench
  evaluator.
- The container half ships in the wheel at
  `simple_agent_lab.evals.suites.programbench` (`build_task`, `build_agent`,
  `prepare`, `extract_result`).
- `ProgrambenchDynamicWorkflowSuite` swaps in the sibling
  `dynamic_workflow_container` facade: an agent-written JavaScript workflow
  orchestrates normal ProgramBench workers while preserving the same isolated
  bash tool, workspace product, and official scorer.

## What differs from SWE-bench

ProgramBench forces two adaptations versus the SWE-bench reference:

1. **The product is the whole workspace, not a `git diff`.** The agent writes a
   new codebase; the submission is `tar -czf` of `/workspace`. A container half
   can only return bytes through `out/result.json`, so `extract_result` returns
   the gzipped workspace base64-encoded as `submission_tar_b64`.
   `evaluate_submissions.py` decodes it into the `<id>/submission.tar.gz` layout
   the official scorer expects.

2. **Network isolation is per-command, not per-container.** ProgramBench's
   anti-cheat relies on the agent having no network while it works
   (`--network none`). Our agent runs *inside* the container and must reach the
   model API, so we keep the container online but run **every agent bash command
   in sealed user + network namespaces**. The prefix drops to outer uid/gid
   65534 before `unshare --user --map-root-user --net` (which needs
   `CAP_SYS_ADMIN` to bootstrap), and preparation hands the scored workspace to
   that identity. The untrusted process therefore cannot use the container's
   capability to rejoin PID 1's online namespace or overwrite the root-owned
   isolation helpers used by a later command. Model calls keep the container
   network; agent commands do not, so the agent cannot `git clone` / `cargo
   install` / `curl` source code. If namespace creation is unavailable, the run
   **fails closed** rather than silently dropping the anti-cheat; pass
   `--no-network-isolation` to deliberately run un-isolated, which records
   `network_isolated: false` in `result.json`.

Scoring stays the **official** ProgramBench evaluator (compile → restore
`./executable` with a sha256 check → per-branch pytest → JUnit), so scores match
the official tool regardless of the inference-time isolation difference.

## Quick Start

```bash
# 1. Install deps (docker-py + the programbench scorer/task set)
uv sync --extra programbench

# 2. Pull the instance image
bash runs/programbench/setup_programbench.sh abishekvashok__cmatrix.5c082c6

# 3. Configure .env with your model provider
cat > .env <<'EOF'
OPENAI_MODEL=your-model-name
OPENAI_AUTH_TOKEN=your-api-key
OPENAI_BASE_URL=https://your-provider/v1
API_KIND=openai-chat
EOF

# 4. Run the agent, then score it
bash runs/programbench/run_programbench.sh abishekvashok__cmatrix.5c082c6
uv run python evals/programbench/evaluate_submissions.py --run-id <run-id>
```

Run the same instance through an agent-written JavaScript workflow:

```bash
uv run --extra programbench python runs/run_bench.py programbench \
  abishekvashok__cmatrix.5c082c6 --agent-flavor dynamic
```

ProgramBench images do not include Node. On the first dynamic run, the runner
downloads the pinned official Linux x64 Node archive, verifies both the archive
and extracted binary SHA-256, and extracts only `bin/node` inside the ignored
ProgramBench wheelhouse. The verified archive remains cached there too. That
wheelhouse is already mounted read-only in the container, so no image rebuild
or extra Docker mount is required. Pass `--workflow-node-binary <container-path>`
to use a pre-provisioned binary instead.

For the whole task set:

```bash
bash runs/programbench/run_programbench.sh --all --parallel 4
# equivalent Python entry used by that thin wrapper:
uv run --extra programbench python runs/run_bench.py batch programbench \
  --all --parallel 4
# dynamic workflow batch:
uv run --extra programbench python runs/run_bench.py batch programbench \
  --all --parallel 4 --agent-flavor dynamic
```

## Running the Agent

The agent runs inside the ProgramBench `:task_cleanroom` image, driven through
`run_suite_instance(ProgrambenchSuite, LocalDockerBackend, LocalDirStore)`. The
image is a language-toolchain image (c/rust/go/...) that need not ship Python
3.11, so the Python harness fetches and caches a static Linux `uv` under
`evals/out/uv-linux/` and mounts it with an offline wheelhouse, exactly like
SWE-bench.

From the agent's point of view, `/workspace` holds `./executable` + docs and the
bash tool is the normal local bash tool — except each command runs network-less
(see above). The runtime modules come from the installed wheel, not from `src/`.

### Dynamic workflow mode

`--agent-flavor dynamic` selects `ProgrambenchDynamicWorkflowSuite`. Its facade
generates a task-specific `workflow.js` (or accepts `--workflow-script`), then
executes JavaScript phases that call ordinary ProgramBench subagents. Subagents
share `/workspace`, are serialized by default, and both the Node orchestration
process and every worker bash call run in sealed user + network namespaces.
Generated scripts cannot request git worktrees because the scored product is
the original shared workspace.

Workflow artifacts are attached to `result.json` under `dynamic_workflow`:

- `workflow_js`
- `result` and `journal`
- `agent_calls`
- `subagent_traces`

Raw workflow files live under the run's `out/dynamic_workflow/` artifact
directory, outside `/workspace`; they therefore cannot enter the submission,
including indirectly through `.git/objects`. Useful controls are
`--workflow-max-concurrency`, `--workflow-max-agents`, and
`--workflow-timeout`. In dynamic mode, `--max-turns` is a per-worker ceiling;
the facade itself runs one outer turn. The workflow timeout is also capped by
`--wall-time-seconds`. Keep concurrency at `1` when workers may edit files.

## Scoring (official harness)

The authoritative scorer is the official ProgramBench evaluator, run on the host
(it needs Docker + the `programbench` package + access to the HF test blobs
`programbench/ProgramBench-Tests`). If that dataset needs auth, set `HF_TOKEN`
in `.env` or the environment — `evaluate_submissions.py` loads `.env` without
overriding the environment, so the evaluator inherits it and huggingface-hub uses
it:

```bash
uv run python evals/programbench/evaluate_submissions.py \
  --run-root evals/out/programbench --run-id <run-id> --workers 4
```

It rebuilds `submission.tar.gz` under `<run-id>_eval/`, runs the official
evaluator with `--image-tag task`, writes a `scores.json` manifest, and prints
the authoritative per-instance scores via `programbench info`.

## Local Adapter Smoke

This does not install `programbench` and does not run Docker — it exercises the
suite + container half in-process with a fake provider (also in `runs/dev/run_ci.sh`):
The dynamic test requires a Node version with the permission model on `PATH`:

```bash
uv run python -m unittest tests.unit.test_programbench_suite
uv run python -m unittest tests.unit.test_programbench_dynamic_workflow
```

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Run aborts: "requires per-command network isolation" | user/network namespace creation unavailable (`CAP_SYS_ADMIN` missing or restrictive kernel) | Use a kernel/daemon that permits the sealed namespaces (`CAP_SYS_ADMIN` is added by default); pass `--no-network-isolation` only to deliberately run un-isolated |
| `ModuleNotFoundError: programbench` during scoring | scorer package not installed | `uv sync --extra programbench` |
| `programbench: command not found` during final score summary | optional `programbench info` binary not on PATH | `uv sync --extra programbench`, pass `--programbench-info-bin`, or use `--no-info` |
| `programbench eval` can't download the HF test blobs | gated/private dataset or anonymous rate limit | Set `HF_TOKEN` in `.env` or the environment (the scorer loads `.env`) |
| Image pull fails | image not on the daemon | `bash runs/programbench/setup_programbench.sh <id> --scoring` |
| `OPENAI_AUTH_TOKEN` / `OPENAI_MODEL` missing | `.env` not configured | Create `.env` (see Quick Start) |
