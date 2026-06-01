# ADR 0018: Oracle Run Mode for Suite Self-Check

## Status

Proposed

## Context

ADR 0017 made a containerized eval "implement a `Suite` + a container module,"
and kept **scoring a separate, official step**: the framework produces
`prediction.jsonl`; it never decides correctness. That separation is a
deliberate strength — it avoids the harness's pass/fail logic drifting from a
benchmark's authoritative scorer.

It also leaves a gap. The Docker-free smoke tests today drive a suite with
`provider="fake"` (the agent does not actually solve anything), which proves the
*pipeline is connected* but not that `extract_result` can recover the expected
product from a genuinely-solved instance. A subtly wrong `extract_result`,
`prepare`, or `prediction_record` — or a benchmark instance that is not actually
solvable — passes the fake-provider smoke test and only surfaces during a real,
expensive model run.

Common benchmarks (e.g. terminal-bench) ship a reference ("oracle") solution per
task and use it to self-validate the task/harness. We want the same assurance,
but without importing their harness-bundled scoring — the check must stay on the
"produce a product" side of our scoring split.

## Decision

Add an **oracle run mode**: a deterministic, model-free way to run a suite
end-to-end where the agent loop is replaced by applying the suite's reference
solution.

- The mode is carried by the existing `provider` string as `provider="oracle"`,
  so it threads through `RunSpec`, `run_suite_instance`, `run_dataset`, and every
  backend with no new parameter on the public entry points.
- `run_in_container(..., oracle=True)` skips agent construction and the turn
  loop, calls the container half's optional `apply_oracle(workspace, instance)`,
  and then runs the **unchanged** `extract_result`. A suite without
  `apply_oracle` raises (a wiring error, not a silent no-op). `provider` may be
  `None` in this mode.
- `run_suite_instance` writes the **unsanitized** instance to the store when
  `provider="oracle"`. Sanitization exists to hide gold/private fields *from the
  agent*; the oracle is the trusted exception that applies them. This is the only
  place gold reaches the store.
- The convention (documented in `docs/human/integrating-a-bench.html`) is: each suite
  ships a tiny self-contained **oracle instance** (no dataset download) and an
  `apply_oracle`, plus a Docker-free unit test that runs it with
  `provider="oracle"` and asserts the product equals the gold. SWE-bench is the
  reference: `apply_oracle` `git apply`s the gold `patch` (never `test_patch`),
  and the test fabricates a hermetic git repo + gold patch.

## Consequences

- Suites gain a deterministic, model-free, Docker-free self-check of the whole
  pipeline (`build_task → prepare → apply_oracle → extract_result →
  prediction_record`) and of instance solvability.
- The check is explicitly **not scoring**: it validates wiring and solvability,
  not model quality, so ADR 0017's scoring-separation is preserved.
- `apply_oracle` is **optional**, so the required `Suite` / container-half surface
  is unchanged; suites that do not add it simply cannot be oracle-checked.
- Oracle mode works on every backend. Through Docker it also validates the real
  image path (the gold patch applies inside the upstream image); in-process it is
  the fast default for unit tests.
- One trusted gold-exposure path now exists (`provider="oracle"` writes the
  unsanitized record). It is narrow and explicit; the agent path is unchanged.

## Alternatives Considered

- **Add an `oracle` `Provider`/LLM.** Rejected — oracle is a *run mode*, not a
  model; applying a gold patch is suite-specific work, which belongs in the
  container half (`apply_oracle`), not in a provider.
- **A separate `run_oracle(...)` entry point.** Rejected — it would duplicate the
  `run_suite_instance` / `run_dataset` wiring; piggybacking on the `provider`
  string reuses the whole path for free.
- **Keep using `provider="fake"` for smoke tests.** Rejected — fake proves
  connectivity only; it cannot catch a wrong `extract_result` or an unsolvable
  instance.
- **Bundle scoring like terminal-bench's oracle.** Rejected — it would entangle
  the harness with correctness judgments that ADR 0017 deliberately keeps as a
  separate official step.
