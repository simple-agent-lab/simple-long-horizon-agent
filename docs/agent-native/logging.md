# Logging And Progress Design

This is the shared design surface for logging and progress reporting work in
Simple Agent Lab. Keep it current when changing terminal output, experiment
status lines, trace links, or logging-related contracts. It is intentionally a
living topic doc, not an ADR: future agents may update it as requirements become
clearer.

## Current Goal

Long-running `simple`, `ahe`, and `dgm` self-evolving experiments should make
their progress and performance condition visible in ordinary terminal output.
The first implementation should be lightweight, always on, and useful in plain
log files captured by `tee`, `screen`, launchd, SSH, or shell redirection.

The near-term design is terminal progress lines, not a dashboard and not a new
persistent event-log system.

## Reader Routing

Load this doc when work touches:

- terminal output for `recipes/simple/`, `recipes/ahe/`, or `recipes/dgm/`;
- `src/simple_agent_lab/evolution/run.py`;
- `src/simple_agent_lab/evolution/components/rollout.py`;
- `src/simple_agent_lab/evals/dataset.py` progress callbacks;
- experiment status, per-instance completion lines, heldout reporting, or
  crash summaries;
- meta-agent-readable self-evolution context or evidence artifacts;
- a future viewer that consumes experiment-level status rather than
  per-agent `trajectory.jsonl` records.

Also load:

- `docs/agent-native/self-evolving.md` for the evolution loop and recipe
  boundaries.
- `docs/agent-native/docker-live-trace.md` when linking terminal output to
  per-instance trajectories.
- `src/simple_agent_lab/evolution/README.md` for output artifact layout and
  framework entry points.

## Design Principles

- **Terminal first.** A user watching a long run should know what phase it is
  in, whether instances are completing, and whether scores are improving.
- **Always on for experiment commands.** Sparse logs are the current failure
  mode, so the default should be useful progress output without requiring a
  flag.
- **Grep-friendly, stable shape.** Use compact one-line records with a stable
  prefix and key-value fields. Avoid progress bars and dynamic terminal control
  because they age poorly in captured logs.
- **Shared formatting, local ownership.** Prefer one tiny helper for formatting
  common progress lines, while keeping recipe-specific meaning in the recipe.
- **No outcome changes.** Logging must not change promotion, scoring, retry, or
  failure behavior. If progress formatting fails to inspect an artifact, print
  the best available line and continue.
- **Trace links, not transcript dumps.** Terminal output should point to
  `result.json`, `failure.json`, and `trajectory.jsonl` artifacts instead of
  printing model/tool transcripts inline.

## Proposed Output Style

Use a stable prefix:

```text
[progress] ...
```

Use short event names and key-value fields:

```text
[progress] run start id=dgm-real root=evals/out/dgm_swebench/dgm-real rounds=4 branches=3 parallel=3 train=60 heldout=60
[progress] rollout start label=candidate version=abc123 slice=train instances=60 run=abc123-deadbeef
[progress] rollout instance 07/60 ok id=django__django-11400 attempt=1 status=0 result=.../out/result.json trace=.../out/trajectory.jsonl
[progress] rollout instance 08/60 error id=pytest-dev__pytest-7521 attempt=1 error="DockerException: daemon unavailable"
[progress] decision accepted candidate=def456 baseline_reward=0.467 candidate_reward=0.533 delta=+0.067 reason="not worse"
[progress] heldout skipped label=baseline reason=skip_baseline_heldout
[progress] run complete id=dgm-real decisions=12 accepted=7 best=def456
```

Exact fields can vary by event, but keep these properties:

- one event per line;
- no terminal control sequences;
- no multi-line JSON blobs;
- IDs and paths are explicit enough for a follow-up command or viewer;
- long errors are clipped to a readable single-line summary.

## Event Coverage

The first implementation should cover these events.

### Run Setup

Print once before real execution begins:

- run id and output root;
- config path when available;
- train and heldout instance counts;
- rounds, branches, parent-selection mode, and parallelism where applicable;
- model/api kind when already shown by the command.

Dry-run plan output can keep its current human-readable shape, but real
execution should include `[progress] run start ...`.

### Rollout Start

Print whenever a version is about to be rolled out over a slice:

- label: `baseline`, `candidate`, `heldout`, `round-N`, or recipe-specific
  equivalent;
- version hash;
- slice id and instance count;
- deterministic run id when available.

### Per-Instance Completion

Use the existing `run_dataset(on_result=...)` hook for immediate completion
lines from the calling thread. Each line should include:

- finished index and total for that rollout;
- instance id;
- `ok` or `error`;
- attempt count;
- backend status code when available;
- paths to `out/result.json` and `out/trajectory.jsonl` when the run directory
  is known;
- a short error reason for raised infra/transient failures.

Distinguish these cases:

- completed run with `result.json`;
- completed container with nonzero status code;
- raised infra/transient error after retry attempts;
- missing `result.json` when existing rollout logic observes it.

### Decisions And Scores

After a candidate is scored and judged, print:

- accepted/rejected outcome;
- candidate hash and parent/baseline hash when useful;
- baseline and candidate mean reward;
- signed delta;
- reason from the criterion;
- decision id if already available.

DGM's open-ended archive can add archive-specific fields such as valid parent
status, archive size, best train reward, and selected parent. Keep those fields
on the same line instead of inventing a separate output dialect.

### Heldout And Final Summary

Heldout lines should show:

- label (`baseline`, `round-N`, `final`);
- version hash;
- reward mean;
- resolved count/rate when available;
- summary artifact path.

When baseline heldout is skipped, print an explicit skip line so the absence of
baseline scoring does not look like missing progress.

Final lines should show:

- run id;
- completed rounds/candidates;
- decisions and accepted count;
- current version or best archive version;
- heldout delta when available;
- summary artifact path when written.

### Crash Summaries

Top-level commands should print a short `[progress] run error ...` line before
propagating the existing exception. Include:

- phase when known;
- exception class and clipped message;
- run root or relevant artifact path;
- hint to inspect `failure.json`, `result.json`, or `trajectory.jsonl` when the
  path is known.

Do not swallow exceptions or convert failures into successful exits.

## Self-Evolution Internal Evidence

This is separate from monitor logging. Terminal progress lines are for humans
watching a long run. Internal evidence files are for the meta-agent inside a
proposal workspace, so it can choose when to inspect details instead of having
everything stuffed into a single prompt.

The source-tree meta-strategy writes these files into the temporary repository
copy before the meta-agent runs:

```text
SELF_EVOLUTION_CONTEXT.md
self_evolution/
  README.md
  current_manifest.json
  baseline_runs.json
  prior_decisions.jsonl
  failures/<instance-id>/result.json
  failures/<instance-id>/trace_excerpt.md
```

These artifacts should follow the same rules:

- write them into both the base and candidate temporary trees before the
  meta-agent runs, so simply reading them does not create a proposal diff;
- keep them local and inspectable with ordinary read/bash tools;
- keep summaries compact, with paths to deeper artifacts when available;
- never let these files change scoring, promotion, or rollout behavior;
- validate proposals against the selected `AgentSurface`, so edits to
  `SELF_EVOLUTION_CONTEXT.md` or `self_evolution/**` are ignored or rejected
  unless a future surface explicitly makes them editable.

Use `SELF_EVOLUTION_CONTEXT.md` as the first-read briefing. Use
`self_evolution/` for optional deeper evidence: parent/current version metadata,
prior decisions, baseline run summaries, and failure artifacts. This supports
the "meta-agent as an agent" direction: the meta-agent can read local evidence
with tools when it needs more context.

Do not confuse these artifacts with monitor logs. A monitor or future progress
viewer should consume run artifacts and progress lines outside the candidate
workspace; it should not depend on files that exist only inside a temporary
meta-agent workspace.

## Proposed Code Shape

Use a tiny stdout helper, for example:

```text
src/simple_agent_lab/evolution/progress.py
```

The helper may expose a small `ProgressReporter` or plain functions that wrap
`print(..., flush=True)`. It should format common events and normalize values,
but it should not own experiment state.

Primary call sites:

- `src/simple_agent_lab/evolution/run.py`: generic simple/AHE setup, heldout,
  round, decision, skip, final, and crash lines.
- `src/simple_agent_lab/evolution/components/rollout.py`: rollout start and
  per-instance completion, passing a callback into `run_dataset(...)`.
- `recipes/ahe/evolve.py`: AHE-specific ledger/round status using the shared
  format.
- `recipes/dgm/evolve.py` and `recipes/dgm/algorithm/open_ended.py`: DGM round,
  branch, archive, candidate, and heldout status.

Prefer passing a reporter or callback through existing function boundaries over
introducing global state. Avoid configuring Python's `logging` module in the
first implementation; plain stdout is easier to teach, capture, and test.

## Relationship To Traces And Viewers

The trace system already owns per-agent trajectories:

- `src/simple_agent_lab/trace/` records Event -> Span -> Training layers.
- `docs/agent-native/docker-live-trace.md` defines live `trajectory.jsonl`
  export for Docker/container runs.
- `studio/trace-viewer/` renders trajectory records and can poll output dirs.

Progress lines should complement that system by saying what the experiment is
doing and where to inspect details. They should not duplicate trace rendering
or become a second trajectory schema.

A future experiment-level viewer may consume structured progress artifacts, but
that is deliberately out of scope for the first terminal-line implementation.
If that need becomes concrete, update this doc before adding a persistent
progress event contract.

## Validation Expectations

Use deterministic local checks first:

- unit tests for reporter formatting and clipping;
- fake-suite tests proving `dataset_rollout` passes an `on_result` callback and
  prints per-instance completion lines;
- CLI tests proving generic evolution execution prints run, decision, skip, and
  final lines;
- DGM/AHE unit tests for representative progress lines without live providers
  or Docker;
- source-tree strategy tests proving meta-agent evidence files are present but
  excluded from candidate proposals.

Avoid validation that requires live model providers, Docker, or SWE-bench unless
the task explicitly targets those integrations.

## Open Design Questions

Keep these visible until implementation settles them:

- Should the shared helper live under `simple_agent_lab.evolution` only, or
  under `simple_agent_lab.evals` so non-evolution dataset runs can reuse it?
- Should there be a later `--quiet` option, or should experiment commands stay
  unconditionally verbose?
- How should status-code and artifact-path fields be populated when a backend
  raises before `RunArtifacts` exists?
- Should DGM archive progress print best-valid train score every candidate, or
  only once per round?
