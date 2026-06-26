# PostTrainBench Integration Design

Date: 2026-06-26

## Purpose

Add PostTrainBench to Simple Agent Lab as a benchmark suite while keeping the
real execution path simple enough to debug under time and GPU cost pressure.

The immediate goal is a private-first workflow that works for the owner:
development happens mostly on the MacBook, DevBoxS provides company-network
access to the private OpenAI-compatible endpoint, and Runpod H100 runs the
actual PostTrainBench job. Open-source cleanup remains a constraint, but not the
primary objective for this first branch.

## Sources And Observations

- The repository already has a generic eval-suite contract: a host half under
  `evals/<suite>/` plus a lightweight container/runtime half under
  `src/simple_agent_lab/evals/suites/<suite>/`.
- PostTrainBench is an external benchmark focused on post-training agents. Its
  public site describes H100-class execution, multiple base models/tasks, and
  long per-run time, so full runs must remain explicit and cost-aware.
- The upstream PostTrainBench repository appears mature enough that Simple Agent
  Lab should wrap it, not rebuild its internals.
- Runpod is the intended H100 provider for final/probe runs.
- DevBoxS inspection on 2026-06-26:
  - 8 vCPUs, 15 GiB RAM.
  - `/` has about 950 GiB free; `/data00` has about 4.1 TiB free.
  - login workspace is `/data00/home/zimuwang`.
  - Docker and Git are available; `uv` is not installed; system Python is 3.7.
  - no local GPU.
  - GitHub, PyPI, PostTrainBench, Runpod, and Runpod console were reachable.
  - Hugging Face and public `api.openai.com` timed out from the shell.
- The owner clarified that the private OpenAI-compatible endpoint is reachable
  from the company environment, and Hugging Face can be handled by proxy/cache.

## Approved Direction

Use a private-first three-machine workflow:

```text
MacBook       -> development, tests, branch work
DevBoxS       -> company-network bridge, private config, Runpod coordination
Runpod H100   -> actual PostTrainBench execution
```

The H100 machine should run PostTrainBench and the CLI/agent process locally for
benchmark fidelity. DevBoxS should bridge LLM traffic to the private
OpenAI-compatible endpoint, and coordinate logs/artifacts as needed.

The execution on the benchmark machine should be as obvious as possible:
prepare environment, set env vars, run PostTrainBench, collect artifacts. Avoid
building a bespoke platform around a benchmark that is already meant to be used
directly.

## Architecture

Add PostTrainBench using the existing eval-suite shape:

- host-side evals/posttrainbench package: adapter, config loading, command helpers,
  dataset/task metadata helpers, and README.
- package runtime half under `src/simple_agent_lab/evals/suites`: lightweight runtime half
  shipped with the package. Keep imports to stdlib plus `simple_agent_lab`.
- `runs/run_posttrainbench_*.py` or `.sh`: small reproducible entrypoints for
  local smoke, DevBoxS probe, Runpod probe, and later full runs.
- `evals/out/posttrainbench/README.md`: committed output layout contract.

Keep the boundary clear:

- Suite code describes what benchmark task/run should happen.
- DevBoxS/Runpod scripts describe where and how to run it.

## Remote Execution Mechanics

DevBoxS owns private configuration outside the repository:

- private `OPENAI_BASE_URL`
- `OPENAI_AUTH_TOKEN`
- model name
- Hugging Face proxy/cache settings
- Runpod API token or SSH target, if automation is added

Preferred LLM bridge:

1. DevBoxS can reach the private OpenAI-compatible endpoint.
2. Runpod H100 gets an OpenAI-compatible local endpoint through SSH
   forwarding or a simple HTTP proxy through DevBoxS.
3. On Runpod, PostTrainBench sees ordinary env:
   `OPENAI_BASE_URL=http://127.0.0.1:<port>/v1`,
   `OPENAI_AUTH_TOKEN=...`, and model env.

Artifacts:

- Runpod writes logs/results locally during the job.
- DevBoxS pulls artifacts back with `scp`/`rsync` or a similarly simple command.
- MacBook can pull summaries from DevBoxS when needed.

Failure checks should be early and plain:

- no H100 visible,
- private endpoint route unavailable,
- Hugging Face proxy/cache unavailable,
- PostTrainBench repo/image missing,
- output directory not writable.

## Validation Ladder

1. **MacBook local smoke**
   Validate suite code, config parsing, task generation, command construction,
   artifact layout, and dry-run behavior with no Docker, GPU, Runpod, Hugging
   Face, or live LLM calls.

2. **DevBoxS control-plane smoke**
   Validate private env loading, Docker availability, private endpoint
   reachability, Runpod reachability, and proxy/HF assumptions.

3. **Runpod boot probe**
   Start a short H100 session. Check `nvidia-smi`, disk, container/runtime,
   PostTrainBench availability, HF proxy/cache, and the LLM route through
   DevBoxS. Terminate promptly.

4. **Tiny benchmark probe**
   Run the smallest practical PostTrainBench path that proves the agent can call
   the model, launch training/eval code, and write artifacts.

5. **Full benchmark run**
   Run the real 10-hour-class task only after the first four stages are boring.

## First Branch Scope

Include:

- PostTrainBench suite skeleton following existing eval patterns.
- Local smoke/dry-run path that works on the MacBook.
- Config format for model/task/run selection.
- DevBoxS probe script or runbook.
- Runpod probe script or runbook.
- Output directory contract.
- Unit tests that avoid Docker, GPU, Runpod, Hugging Face, and live LLM calls.
- A concise design/ADR only if the implementation creates a durable
  architectural commitment.

Do not include yet:

- automated Runpod provisioning unless it becomes trivially necessary,
- full benchmark dataset execution,
- self-evolving integration,
- public-polished generalized cloud backend,
- required CI that touches GPU or network.

## Branch Strategy

Create a fresh branch from latest `origin/main`, independent of current
self-evolving work, such as `codex/posttrainbench-suite`.

The current working tree contains unrelated self-evolving changes. Do not base
the PostTrainBench implementation on that branch unless the owner explicitly
changes this decision.

## Open Questions For Implementation Planning

- What is the exact upstream PostTrainBench command sequence for the smallest
  useful probe?
- Should the first DevBoxS setup install a project-local `uv`/Python toolchain
  under `/data00/home/zimuwang`, or should DevBoxS stay only a shell/SSH
  control host?
- Is Runpod provisioning manual for the first phase, or should the repo include
  a thin helper once the manual path is stable?
- Which Hugging Face proxy/cache mechanism will be used for Runpod?
- What artifact subset is enough for the MacBook to inspect a failed run without
  copying large model outputs?
