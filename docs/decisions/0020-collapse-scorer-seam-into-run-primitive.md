# ADR 0020: Collapse the Scorer Seam into the Run Primitive

## Status

Accepted

## Context

ADR 0019 added a `Scorer` seam: a `Scorer` protocol, a `ScoreRequest` value, a
`Suite.scorer()` method, and a `score_dataset` host driver that read each run's
`result.json` back, built `ScoreRequest`s, and called the suite's scorer once.
Two scoring topologies hung off it — `separate` (the official harness in fresh
containers) and `reuse` (the container-half `evaluate` hook, with the host scorer
a passthrough).

In use, that seam turned out to carry two-and-a-half different jobs under one
abstraction, which is what made it feel heavy:

- **In-environment scoring** — the thing we actually wanted ("score where the run
  ran") — was already the `evaluate` hook, which rides on `run_suite_instance`
  and so already inherits retry, `submit`/`reconcile`, and backend portability.
  For this case the host `Scorer` was a near-pure passthrough over a verdict
  already in `result.json`.
- **Host interpretation of a raw product** — e.g. SWE-bench grading a captured
  official eval log — is a small host step, not a framework-shaped batch seam.
- **A fresh/official scoring environment** — the official harness — is a parity
  oracle that must run the official tool verbatim.

Meanwhile the only genuinely framework-native "score elsewhere" case, an
agent-judge (today's example; a rubric judge tomorrow), is itself *another run* —
not a `Scorer` at all. So the `Scorer`/`score_dataset` machinery mostly added a
parallel control path that duplicated the run primitive's concerns without its
fault tolerance.

## Decision

Collapse scoring into the run primitive. There is **one** way to run things
(`run_suite_instance` / `run_dataset`), and scoring is expressed through it:

- **In-environment scoring is the `evaluate` hook**, gated on staged gold. A
  suite that scores where the run ran exposes the container-half
  `evaluate(workspace, instance, *, context)` and stages gold via `eval_inputs`
  (host-written under EVAL_KEY, never shown to the agent). Staged gold is the
  toggle: present → the generic runner calls `evaluate` and merges its verdict
  into `result.json`; absent → the hook is skipped. This rides on
  `run_suite_instance`, so it reuses the run's retry / `submit`-`reconcile` /
  backend portability and runs wherever the run ran (in-process or in-container).
- **Scoring elsewhere is a follow-up run.** An agent judge (or rubric judge) is
  just another `Suite` run, wired to the candidate's `result.json` through the
  shared `ArtifactStore` — the same primitive, not a bespoke scoring API.
- **Official-harness parity stays a standalone CLI.** `evaluate_predictions.py`
  runs the official SWE-bench harness verbatim; that is its job and it is not
  dressed up as a framework seam. For the SWE-bench in-environment path, the host
  step that turns the captured official eval log into a verdict row is a small
  helper, `evaluate_predictions.reuse_eval_row` (it needs the `swebench` grader +
  the gold test spec, which live host-side). Both the official CLI path and
  `reuse_eval_row` route through the same `eval_result_from_official` mapping, so
  rows are interchangeable and the parity gate (`parity_mismatches` /
  `--verify-parity`) still cross-checks reuse against the official harness.

**Removed:** the `Scorer` protocol, `ScoreRequest`, `Suite.scorer()`, the
`score_dataset` / `suite_scorer` driver (`scoring.py`), and the
`SeparateScorer` / `ReuseScorer` classes (`scorer.py`). The
SWE-bench suite's `score_mode` becomes a single `in_env_scoring: bool` that just
decides whether `eval_inputs` stages the official eval script (turning the hook
on).

**Kept from ADR 0019:** `result.json` as the single decoupling artifact, the
`evaluate` hook, private gold staging (`eval_inputs` → EVAL_KEY, distinct from
the agent-visible `task_input`), and the official-parity requirement + gate.

## Consequences

- The `Suite` host surface shrinks again: `name`, `container_module`,
  `launch_spec`, `task_input`, `eval_inputs`. `eval_inputs` is the only
  scoring-related method, and it doubles as the in-environment-scoring toggle.
- One control path. In-environment scoring inherits the run's fault tolerance for
  free; there is no second driver to keep in sync with the runner.
- Aggregating a batch's verdicts is a few lines reading `result.json` (or the
  follow-up judge run's `result.json`), not a protocol — kept out of the
  framework deliberately.
- The reference example (`examples/bench_suite`) now scores in-environment via
  `evaluate` + `eval_inputs`, which demonstrates the real gold → stage →
  score-in-place mechanism rather than a host-side scorer.
- SWE-bench keeps both paths: in-environment (`in_env_scoring=True` →
  `reuse_eval_row`) and separate official harness (`evaluate_predictions.py`).
  The host log-grading step is now a plain function, not a `Scorer` class.

## Alternatives Considered

- **Keep a thin `score_dataset` reader (no `Scorer` protocol).** Rejected — once
  scoring is the `evaluate` hook or a follow-up run, a batch reader is a trivial
  `result.json` loop; a named driver re-introduces a parallel path for no gain.
- **Model separate/official scoring as a `run_suite_instance` "scoring run".**
  Rejected for the official harness: it orchestrates its own Docker containers
  and must run verbatim for parity, so wrapping it as an agent-loop run is an
  impedance mismatch. A standalone CLI is the honest shape. (Agent-judge scoring,
  which *is* an agent run, naturally uses the run primitive.)
- **Keep the ADR 0019 `Scorer` seam.** Rejected — it conflated three jobs and
  duplicated the run primitive without its fault tolerance; the owner asked to
  simplify toward "everything is a run."

## Relationships

- Amends [ADR 0019](0019-scorer-seam-and-scoring-topology.md): removes the
  `Scorer` seam and the separate score phase while keeping its durable parts
  (`result.json` decoupling, the `evaluate` hook, `eval_inputs`/EVAL_KEY gold
  staging, official-parity gate).
- Amends [ADR 0017](0017-generic-containerized-eval-framework.md): the suite
  surface no longer includes a scoring method; scoring is the `evaluate` hook or
  a follow-up run.
- Builds on [ADR 0018](0018-oracle-run-mode-for-suite-self-check.md): the
  `evaluate` hook reuses the in-process-capable generic-runner precedent set by
  the oracle hook.
