---
title: "Scorer Seam and Per-Suite Scoring Topology"
status: Proposed
date: 2026-06-01
slug: scorer-seam-and-scoring-topology
note: "amends `generic-containerized-eval-framework`, builds on `oracle-run-mode-for-suite-self-check`, amended by `collapse-scorer-seam-into-run-primitive`"
---

# Scorer Seam and Per-Suite Scoring Topology

## Status

Amended by [ADR collapse-scorer-seam-into-run-primitive](20260601-collapse-scorer-seam-into-run-primitive.md). The
durable parts here still hold — `result.json` as the decoupling artifact, the
in-environment `evaluate` hook, private gold staging via `eval_inputs`/EVAL_KEY,
and the official-harness parity requirement. ADR collapse-scorer-seam-into-run-primitive **removes** the `Scorer`
protocol, `ScoreRequest`, `Suite.scorer()`, and the `score_dataset` driver: in
the run environment scoring is the `evaluate` hook, and scoring elsewhere is a
follow-up run or the official harness CLI — not a separate framework seam.

## Context

ADR generic-containerized-eval-framework made a containerized eval "implement a `Suite` + a container module,"
and kept scoring a separate, official step: the run phase produced
`prediction.jsonl` via the `Suite.prediction_record` method, and an external
tool (`evals/swebench/evaluate_predictions.py`) ran the official harness over it.
ADR oracle-run-mode-for-suite-self-check added an oracle run mode for wiring self-checks but preserved that
run/score split.

Two pressures surfaced:

- **The `Suite` protocol carried scoring-shaped concerns it should not.**
  `prediction_record` is "prepare the scorer's input," yet it lived on the
  run-side suite surface and ran in the run phase, even though it computes
  nothing about a run — it only reshapes `extract_result`'s output for a
  downstream scorer.
- **There is no single seam for "how does this suite score?"** Today SWE-bench
  scoring is a bespoke CLI; an agent-judge-with-rubric scorer (future) would be
  another bespoke path. We want one optional seam a `Suite` can implement, the
  way slime exposes a unified verifier abstraction — but as an *eval* seam, not
  coupled to rollout/reward as in an RL trainer.
- **Where scoring runs differs by benchmark.** SWE-bench's authoritative scorer
  is the official harness in fresh containers. Other benches are cheapest to
  score in the same environment the run used (in-container for Docker,
  in-process for `LocalProcessBackend`). Both must be supported.

## Decision

Add an optional **`Scorer` seam** and make "where scoring runs" a per-suite
choice behind it.

- **`Scorer` protocol** (`protocols.py`): `score(requests) -> list[dict]`,
  batch-oriented. A `Suite` exposes one via `Suite.scorer() -> Scorer | None`
  (a required method that returns `None` to opt out of in-framework scoring).
  Input is a `ScoreRequest` (`instance_id`, `instance`, raw `result`,
  `run_dir`). There is **no new reward type**: the scorer returns the
  eval-result dict rows the project already emits (`evaluate_predictions.py`
  `eval_result_record` / `EvalResult`), keyed by `instance_id`.
- **`Suite` drops `prediction_record`.** The host surface is now
  `name`, `container_module`, `launch_spec`, `task_input`, `scorer()`, and
  `eval_inputs()` — the last two are explicit methods that return `None` to opt
  out (rather than getattr-probed hooks), so the surface stays small but the
  capability is typed and discoverable. Prediction shaping moves into the scorer
  (`Scorer.prediction_rows`), which `score_dataset` uses to emit the official
  `prediction.jsonl` as a *scorer output*.
- **`result.json` is the single decoupling artifact.** The run phase produces
  only `out/result.json` (raw `extract_result`) and `out/trajectory.jsonl`; it
  shapes no prediction and computes no score. `model_name` leaves the run/dataset
  entry points and re-enters at the score phase (it labels a submission, not a
  run). Scoring is a separate, re-runnable phase: `score_dataset` reads each
  run's `result.json` back through the store, builds `ScoreRequest`s, and calls
  the suite's scorer once.
- **Two topologies behind the one seam**, chosen per suite:
  - `separate` — score in a fresh environment via the official harness
    (`SeparateScorer`). The default, re-scorable path; parity by definition.
  - `reuse` — reuse the run environment. The container half exposes an optional
    `evaluate(workspace, instance, *, context)` hook; the generic runner calls it
    after `extract_result` and merges its verdict into `result.json`, so the host
    scorer (`ReuseScorer`) is a passthrough. The hook is **environment-neutral**:
    the same generic runner drives it in-process (`LocalProcessBackend`) and
    in-container (Docker), like the ADR oracle-run-mode-for-suite-self-check in-process oracle.
- **Official-harness parity is a hard requirement for `reuse`.** A `reuse`
  verdict must agree with the `separate` official harness. We enforce this by
  (a) driving the *official* eval script — the host generates it from
  `make_test_spec` and stages it for the run environment via the optional
  `Suite.eval_inputs()` under a private `input/eval.json` (EVAL_KEY), separate
  from the agent-visible `input/instance.json`; and (b) a parity gate
  (`scorer.verify_parity` / `evaluate_predictions.parity_mismatches` +
  `--verify-parity`) that cross-checks `reuse` against `separate` on a sample and
  fails on any disagreement. Both topologies route through the *same*
  `eval_result_from_official` mapping, so rows are interchangeable.

### Backend × score_mode validity (SWE-bench)

- `LocalDockerBackend` + `separate`: official harness in fresh containers — parity by definition (default).
- `LocalDockerBackend` + `reuse`: official eval script in the same instance image (predict image == eval image) — parity-grade, no second image build.
- `LocalProcessBackend` + `reuse`: in-process eval against the workspace — ideal for no-Docker dev and self-contained benches; SWE-bench parity needs the image's deps, so treat in-process reuse as wiring/dev unless the host has them.
- `LocalProcessBackend` + `separate`: not supported for SWE-bench (the official harness needs Docker); the only fully no-Docker SWE-bench scoring path is `reuse`.

## Consequences

- The `Suite` host surface shrinks (`prediction_record` gone) while gaining one
  well-named seam (`scorer()`, returns `None` to opt out); scoring concerns
  leave the run side.
- Runs and scores are cleanly decoupled by `result.json`: a batch can be
  re-scored without re-running, and scoring is testable without Docker (the
  whole scorer path runs over `LocalProcessBackend`/`FakeBackend`).
- A future agent-judge-with-rubric scorer is just another `Scorer`; no new
  framework machinery is needed.
- `reuse` introduces a parity obligation; the gate makes that obligation
  explicit and checkable rather than assumed.
- One private gold-staging path now exists (`eval_inputs` → EVAL_KEY) for the
  `reuse` topology. It is narrow, opt-in, and separate from the agent-visible
  instance; `task_input` still governs what the agent sees.

## Alternatives Considered

- **Keep `prediction_record` on `Suite`.** Rejected — it is scorer input
  shaping, not a run-side concern; folding it into the scorer shrinks the
  required surface and keeps the run phase to a single artifact.
- **A `Reward` value type (slime-style).** Rejected — this is an eval seam, not
  an RL trainer; the existing eval-result dict already carries pass/fail, score,
  and metrics, so a parallel type would duplicate it.
- **One global scoring topology.** Rejected — SWE-bench's authoritative scorer is
  the official harness, while other benches are cheapest to score in place;
  making topology a per-suite choice serves both without a flag in the runner.
- **Score inside the run phase (no separate score step).** Rejected — it would
  recouple run and score, prevent cheap re-scoring, and force Docker into the
  scorer path. `result.json` as the decoupling artifact keeps both phases simple.

## Relationships

- Amends ADR generic-containerized-eval-framework (the run phase no longer shapes `prediction.jsonl`; `Suite`
  drops `prediction_record`; scoring gains an optional in-framework seam).
- Builds on ADR oracle-run-mode-for-suite-self-check (the `reuse` `evaluate` hook reuses the in-process-capable
  generic-runner precedent set by the oracle hook).
- Relates to ADR collect-training-trajectories-across-design-versions / ADR keep-benchmark-suites-as-eval-adapters (keeps raw trajectories, eval scores, and
  training labels separated; the scorer emits eval-result rows only).
