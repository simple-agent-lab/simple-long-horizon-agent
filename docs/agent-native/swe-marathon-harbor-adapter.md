# SWE-Marathon Harbor Adapter

Read when:

- You are adapting Simple Agent Lab to run SWE-Marathon tasks.
- You are changing `adapters/harbor_agent.py` or adding run scripts around it.
- You need a clean-machine setup path for Harbor plus SWE-Marathon.

Do not read when:

- You are adding a normal Simple Agent Lab containerized eval suite. Use
  `docs/agent-native/integrating-a-docker-eval-suite.md` for that path.
- You are touching SWE-bench or ProgramBench without Harbor.

## Recommendation

Use Harbor as the primary integration path. SWE-Marathon tasks are already
Harbor tasks, so the shortest useful adaptation is to harden the existing
Harbor installed-agent adapter and let Harbor continue to own task loading,
Docker startup, network policy, verifier execution, restore paths, resources,
and result collection.

Do not start by writing a native Simple Agent Lab `Suite` for SWE-Marathon.
That would duplicate Harbor behavior for task config parsing, verifier modes,
network allowlists, GPU placement, task artifacts, and log collection before it
improves agent behavior. A native suite can be reconsidered only after the
Harbor adapter is reliable and there is a concrete reason to bypass Harbor.

## Clean-Machine Contract

Future code and docs must not assume the developer already has local sibling
checkouts of SWE-Marathon or the Harbor fork. The research that led to this
page used local clones, but implementation must work from a clean machine with
only this repository, Docker, Python or uv, and model credentials.

Support these inputs explicitly:

- `SWE_MARATHON_TASKS_DIR`: optional path to an existing task checkout.
- `SWE_MARATHON_REPO_URL`: optional repo URL, defaulting to
  `https://github.com/abundant-ai/swe-marathon`.
- `SWE_MARATHON_REF`: optional branch, tag, or commit to checkout.
- `HARBOR_SOURCE`: optional Harbor install source, defaulting to
  `git+https://github.com/RishiDesai/harbor.git`.
- `SIMPLE_AGENT_LAB_SOURCE`: optional source used inside task containers;
  closed-internet tasks should use a local wheel or source tree mounted into the
  container instead of a network install.

Any helper script should either use `SWE_MARATHON_TASKS_DIR` or clone into an
ignored local artifact directory such as `evals/out/swe-marathon/deps/`. It
should verify Docker, Harbor, the task directory, and model env vars up front,
then print actionable errors. It should never mutate a sibling checkout outside
the path it created or the path the user explicitly supplied.

## Current Adapter State

The branch has `adapters/harbor_agent.py`, a Harbor `BaseInstalledAgent` that:

- installs Simple Agent Lab in the task container;
- forwards a Harbor model like `openai/<model_id>` as `OPENAI_MODEL`;
- maps `OPENAI_AUTH_TOKEN` or `OPENAI_API_KEY` to Simple Agent Lab's
  `OPENAI_AUTH_TOKEN`;
- runs `make_bash_agent(...)` in the container workspace;
- logs stdout and stderr to `/logs/agent/sal.log`.

That is enough for an early smoke path, but not enough for reliable
SWE-Marathon runs.

Known gaps:

- The adapter assumes `/app`, while inspected SWE-Marathon tasks also use
  `/workspace`, `/workspace/rust-java-lsp`, and `/app/rj-rust`.
- Container install currently depends on network access, which fails for
  closed-internet tasks unless a local source or wheel is mounted.
- The adapter forwards only a small env subset and should use Harbor
  `extra_env` accessors rather than direct `os.environ` reads.
- It does not write Harbor ATIF `agent/trajectory.json`.
- It leaves `AgentContext` token and cost fields empty.
- It only exposes the plain bash agent, while long-horizon tasks may need
  `bash_task` or `bash_skills`.

## SWE-Marathon Shape

The inspected SWE-Marathon checkout has 20 task directories under `tasks/`.
Tasks are long-horizon Harbor tasks with agent timeouts from roughly 2 to 10
hours. Several tasks require GPU workers:

| Task | GPU |
| --- | --- |
| `embedding-eval` | T4 |
| `jax-pytorch-rewrite` | A100 |
| `parameter-golf` | H100 |
| `trimul-cuda` | H100 |

The inspected closed-internet tasks are:

- `embedding-eval`
- `kubernetes-rust-rewrite`
- `nextjs-vite-rewrite`
- `parameter-golf`
- `trimul-cuda`
- `zstd-decoder`

Verifier output follows Harbor/SWE-Marathon conventions: `reward.txt` lives
under `/logs/verifier/`, and tasks with shaped or diagnostic scoring also write
`metrics.json` with a normalized `partial_score`.

## Adaptation Plan

| Step | Change | Expected files | Effort | Estimated LOC | Time |
| --- | --- | --- | --- | ---: | ---: |
| 1 | Make workdir configurable and detectable. Prefer `SAL_WORKDIR`; otherwise use the image workdir or a small known-path probe. | `adapters/harbor_agent.py` | M | 40-80 | 0.5 day |
| 2 | Add clean-machine setup. Provide a run helper that can install Harbor, clone SWE-Marathon when missing, or use `SWE_MARATHON_TASKS_DIR`. | `runs/`, docs | M | 100-180 | 1 day |
| 3 | Support closed-internet Simple Agent Lab install through local wheel/source mounts and clear install-mode errors. | `adapters/harbor_agent.py`, `runs/` | M | 80-150 | 1 day |
| 4 | Forward provider and gateway env consistently: `OPENAI_BASE_URL`, `API_KIND`, `REASONING_EFFORT`, `OPENAI_SESSION_ID`, `OPENAI_LOG_ID`, and max-turn/flavor knobs. | `adapters/harbor_agent.py` | S-M | 60-110 | 0.5 day |
| 5 | Add agent flavor selection for `bash`, `bash_task`, and `bash_skills`; default marathon runs should prefer `bash_task` unless a smoke test shows it regresses. | `adapters/harbor_agent.py` | S | 40-80 | 0.5 day |
| 6 | Export Simple Agent Lab trace artifacts and convert the main run to Harbor ATIF `trajectory.json`; backfill `AgentContext` token totals from SAL usage events. | `adapters/harbor_agent.py`, maybe a small helper module | L | 200-350 | 1.5-2 days |
| 7 | Add focused tests for env construction, workdir selection, script generation, token aggregation, and ATIF conversion. | `tests/unit/` | M | 150-260 | 1 day |
| 8 | Document smoke commands and resource tiers. Keep full SWE-Marathon runs opt-in and outside required CI. | `runs/`, docs | S | 80-160 | 0.5 day |

Minimal "can run" scope is steps 1-5 plus a smoke doc. Expected size:
roughly 320-600 LOC over 2-3 days.

Reliable "can run and analyze" scope is steps 1-8. Expected size:
roughly 750-1,300 LOC over 4-6 days.

## Validation Ladder

Use the narrowest check that proves the layer being changed.

1. Unit tests with no Docker and no network:
   - env precedence and token mapping;
   - workdir resolution;
   - generated in-container Python script;
   - SAL usage aggregation;
   - ATIF JSON shape.
2. Harbor install-only or a tiny task smoke, if available.
3. CPU SWE-Marathon smoke:
   - `find-network-alignments` for ordinary internet-enabled behavior;
   - `zstd-decoder` for closed-internet behavior with local SAL source.
4. One longer CPU task after the smoke path is stable.
5. GPU tasks only on machines with the requested accelerator type.

Do not make live SWE-Marathon runs part of the required local CI gate. They are
external, expensive, and resource-dependent.

## Run-Time Notes

For model access in restricted tasks, use Harbor's agent-phase allowlist for the
model gateway host. For setup-time network access, use environment allowlists
only when the task policy permits it; closed-internet tasks should avoid setup
network by using mounted local artifacts.

The first useful user-facing command should make these assumptions explicit:

```bash
SWE_MARATHON_TASKS_DIR=... \
SIMPLE_AGENT_LAB_SOURCE=... \
harbor run \
  -p "$SWE_MARATHON_TASKS_DIR/find-network-alignments" \
  --agent adapters.harbor_agent:SimpleAgentLab \
  --model openai/<model-id> \
  --allow-agent-host <model-gateway-host>
```

The actual helper script may wrap this, but it should preserve the same
configuration surface instead of baking in local absolute paths.
