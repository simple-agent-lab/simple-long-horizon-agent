# ADR 0022: ProgramBench Adapter — Workspace-as-Product and Per-Command Network Isolation

## Status

Accepted

## Context

[ProgramBench](https://github.com/facebookresearch/programbench) is a
*reverse-engineering* benchmark: each instance's container holds a compiled
`./executable` plus its bundled docs, and the agent must write a brand-new
codebase whose `./compile.sh` rebuilds an executable with identical behavior,
inferring that behavior **only** by running `./executable` and reading the docs.
We want it as a peer of the SWE-bench adapter (ADR 0011, ADR 0017), reusing the
generic containerized framework rather than growing a second harness.

Two facts about ProgramBench do not fit the SWE-bench reference shape, and both
are hard to reverse once runs and scores exist, so they need a recorded
decision:

1. **The product is the whole workspace, not a `git diff`.** The agent authors a
   new codebase; the submission ProgramBench scores is `tar -czf` of the
   workspace. The framework's container half can only return *bytes through
   `out/result.json`* (ADR 0017) — there is no "return a file/dir" channel.

2. **The anti-cheat assumes the agent has no network.** ProgramBench's official
   runner uses `--network none` so the agent cannot fetch the original source
   (clone the repo, `cargo install` the tool, `pip install` the project, web
   search). But our agent runs *inside* the container and must reach the model
   API; bootstrap also `pip install`s the runtime wheel. A fully offline
   container would cut off the agent's own reasoning.

Scoring, as with SWE-bench, should stay the **official** tool (`programbench
eval`) so our numbers match published results.

## Decision

Integrate ProgramBench as a peer `Suite` (`evals/programbench/suite.py` +
container half `simple_agent_lab.evals.suites.programbench.container`) with four
specific choices:

1. **Workspace-as-product via base64 in `result.json`.** `extract_result` tars +
   gzips the workspace and returns it base64-encoded as `submission_tar_b64`
   (with `submission_tar_bytes` and `network_isolated`). The host driver
   `evals/programbench/evaluate_submissions.py` decodes it back into the
   `<id>/submission.tar.gz` layout the official scorer expects. We do **not** add
   a new "file/directory product" type to the framework protocols.

2. **Per-command network isolation, not per-container.** The container runs
   online (`launch_spec.network_mode="host"`) so the model API and the wheel
   bootstrap work, but **every agent bash command runs in a no-network namespace**
   via `unshare --net`. Model calls keep the network; agent commands do not, so
   the agent cannot fetch source. This needs `CAP_SYS_ADMIN`, supplied as
   `launch_spec.cap_add=("SYS_ADMIN",)`. A fresh net namespace ships only a *down*
   loopback (unlike `docker run --network none`, which auto-ups `lo`), so the
   command is wrapped in `sh -c 'ip link set lo up; exec "$@"'` to raise loopback
   first — keeping `127.0.0.1` usable for local self-tests inside the sandbox.

3. **A generic `exec_prefix` on the bash tool is the mechanism.** `make_bash_tool`
   / `run_bash` gained an optional `exec_prefix` argv prefix so the launched
   process is `[*exec_prefix, "bash", "-lc", command]`. The ProgramBench container
   half passes an `unshare --net --` prefix (followed by the loopback-raising `sh`
   shim above). This is a *general* seam (any suite needing a sandbox/cgroup/
   firejail wrapper can reuse it), not ProgramBench plumbing, and the
   model-visible command string is unchanged.

4. **Graceful fallback, recorded.** The container half probes `unshare --net`
   once; if it is unavailable (no `CAP_SYS_ADMIN`, restrictive kernel/seccomp) it
   falls back to un-isolated commands and records `network_isolated: false` in
   `result.json` instead of failing the run.

Scoring is the official `programbench eval` CLI run on the host — the
"official-harness as a standalone CLI" shape of ADR 0020, not a framework seam —
so `eval_inputs` returns `None` and the container half exposes no `evaluate`
hook.

## Consequences

- **Easier.** ProgramBench reuses the whole framework unchanged: run primitive,
  backends, store, batch submit/reconcile, live trace, wheelhouse/uv offline
  path. Scores stay official. The `exec_prefix` seam is now available for any
  future suite that must wrap agent commands.
- **The anti-cheat is restored without blinding the agent.** The agent reasons
  with a live model but executes blind to the network — closer in spirit to
  ProgramBench's offline rule than a single global `--network none` we could not
  use.
- **The prompt self-inspects in-container.** Because `build_task` runs inside the
  container, it states the real OS/kernel/arch via `os.uname()` and adds a `tmux`
  TUI hint only when `shutil.which("tmux")` finds it — more accurate than
  mini-swe-agent rendering `{{system}}` or asserting tool availability on the host.
- **Costs / harder.** The container needs `CAP_SYS_ADMIN`, which widens its
  privilege (acceptable for a throwaway eval container, and dropped by passing
  `--no-network-isolation`, which then weakens the anti-cheat and is recorded).
  `result.json` now carries the gzipped workspace (a few KB to a few MB), so it
  is larger than a diff. Isolation depends on the kernel/daemon permitting new
  network namespaces; where it does not, runs silently weaken to online commands
  (surfaced only via `network_isolated: false`).
- **Out of scope.** No `cap_drop` / non-root `user` field on `LaunchSpec` (we
  keep root + `cap_add`); no in-environment scoring; no new product protocol.

## Alternatives Considered

- **`--network none` for the whole container (official default).** Cuts off the
  agent's own model API calls and the bootstrap `pip install`. Unusable for an
  in-container agent.
- **A new "product is a file/directory" channel in the framework.** Larger blast
  radius than one suite warrants and against "smallest useful change to the
  core" (AGENTS.md). Base64-in-`result.json` keeps the single decoupling artifact
  intact.
- **An egress allowlist proxy (only the model API reachable).** More moving parts
  than `unshare --net`, easy to misconfigure, and still lets commands open
  connections to allowlisted hosts. Per-command `unshare --net` is both simpler
  and stricter (commands have *no* network), at per-command granularity.
- **Host firewall / global seccomp.** Heavier than a per-command namespace and
  would also have to carve out the model API, reintroducing the same leak risk.
