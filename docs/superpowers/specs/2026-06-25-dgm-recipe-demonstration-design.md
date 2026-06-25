# DGM Recipe Demonstration Design

Date: 2026-06-25
Status: Proposed

## Purpose

The DGM SWE-bench recipe should demonstrate the functionality and flexibility of
Simple Agent Lab's self-evolution framework. It should be relatively faithful to
DGM mechanics and produce decent performance, but it must remain a recipe-level
composition of the framework rather than a second hidden evolution framework.

The guiding principle is:

> DGM should be the best demonstration recipe for the framework, not a fork of
> the framework.

## Current Problem

The current run shape has the right outline but weak experimental behavior.

- Generated candidates mostly append SWE-bench workflow text to
  `agent/agent_program.py`; the evolvable surface is too narrow to show
  meaningful agent-design flexibility.
- DGM admission wraps `valid_when("reward")`, but rejects a whole candidate when
  any run receives a negative reward.
- Negative reward is currently used to encode `agent_package.used_fallback`.
  That fallback field mixes agent-package failures with runner/container
  failures such as "container exited before writing a result".
- The baseline can also contain fallback/container failures, but candidate
  fallback failures alone invalidate admission.
- Reporting does not clearly separate valid children, invalid children,
  improved children, tied children, regressed children, agent-package failures,
  container failures, and scoring failures.

This makes the recipe look like DGM while preventing the archive from behaving
like a useful open-ended search.

## Non-Goals

- Do not redesign the whole `src/simple_agent_lab/evolution/` substrate.
- Do not move DGM-specific policies such as `score_child_prop` into the
  framework.
- Do not let DGM edit broad repository/runtime code in this iteration.
- Do not chase state-of-the-art SWE-bench performance. The goal is a credible
  demonstration recipe with inspectable mechanics.

## Design Summary

Implement Track A and Track B together:

- Track A: make DGM admission and reporting trustworthy.
- Track B: make the bounded evolved `agent/` package worth evolving.

The framework remains the visible substrate: versions, runs, slices, proposals,
contexts, rollout, reward, criterion, store, and decision log. DGM supplies the
recipe policy: branch scheduling, archive parent selection, SWE-bench
diagnostics, reporting, and the default evolvable scaffold.

## Track A: Admission, Diagnostics, Reporting

### Valid Child Semantics

A DGM child should be admitted to the archive when it is a valid, gradable agent
candidate, even if its reward is tied or worse than its parent.

For this recipe, a candidate is valid when:

- the staged `agent/` package is present and importable,
- `build_agent` is callable and does not fail systemically,
- enough instance runs produce result artifacts to avoid systemic rollout loss,
- scoring can produce per-run rewards for completed runs.

Runner/container failures should be recorded as diagnostics and counted in
metrics, but they should not automatically make the whole candidate invalid
unless they are systemic.

### Diagnostic Categories

DGM should derive recipe-local diagnostics from each `Run.result` and any
`failure.json` files. The report should distinguish:

- `agent_load_failed`: staged package could not be loaded or imported.
- `agent_build_failed`: `build_agent` loaded but raised while constructing the
  agent.
- `container_failed`: container exited before writing a result.
- `missing_result`: expected `out/result.json` is missing.
- `scoring_failed`: reward/scoring logic failed after a result was produced.
- `completed`: run produced a result and was scored.

These categories are DGM/SWE-bench recipe diagnostics. They should be computed
in the recipe layer, not added as new framework concepts unless another recipe
needs the same abstraction later.

### Reward Policy

Keep `reward: Run -> float` as the framework-level scoring interface.

For DGM, stop using `-1.0` as the only way to carry validity. Instead:

- reward should primarily reflect SWE-bench resolution outcome;
- diagnostics should carry validity/error information;
- candidate validity should be judged from diagnostics plus completion
  thresholds;
- optional adjusted reward may be reported separately, but should not be the
  only validity signal.

This keeps the substrate simple while avoiding the current negative-reward
overload.

### Admission Criterion

Replace the current candidate-wide "any negative reward invalidates the child"
guard with a DGM-local criterion that:

- admits valid children even when reward regresses,
- rejects candidates with systemic agent-package failure,
- rejects candidates with systemic rollout loss below the existing completion
  floor,
- records `valid_parent` and diagnostic counts in the decision record.

The archive parent selector should continue to consider only valid parents.

### Reporting

`recipes/dgm/ops/report.py` should headline:

- decision count,
- valid children,
- invalid children,
- improved children,
- tied children,
- regressed children,
- best train version,
- current promoted version,
- best held-out result when available.

It should also expose diagnostic counts:

- agent load failures,
- agent build failures,
- container failures,
- missing results,
- scoring failures,
- fallback count.

Official performance claims should still use the held-out official scoring
artifacts.

## Track B: Moderate Evolvable Agent Surface

The DGM recipe should keep a bounded editable `agent/` package. The meta-agent
may edit the whole `agent/` package, but not broader framework, recipe, or eval
runtime code.

The default package should be expanded from a single prompt-heavy
`agent_program.py` into a small readable scaffold:

```text
agent/
  agent_program.py
  prompts.py
  workflow.py
  review.py
  tool_policy.py
```

Suggested responsibilities:

- `agent_program.py`: entry point; assembles the Agent using framework helpers.
- `prompts.py`: system prompt fragments and SWE-bench task strategy text.
- `workflow.py`: lightweight workflow knobs, such as when to inspect, reproduce,
  edit, validate, and review.
- `review.py`: optional self-review guidance before finalizing a patch.
- `tool_policy.py`: shell/test/diff guidance and retry policy text.

This is intentionally modest. It gives evolution more useful design freedom
than prompt paraphrasing while keeping the generated code easy to inspect and
safe to stage.

### Validation

Before rollout, the recipe should validate candidate package files:

- paths stay under `agent/`,
- Python files parse with `ast.parse`,
- `agent/agent_program.py` exists,
- package import/load succeeds,
- `build_agent` is callable.

Invalid candidates should be logged as invalid without spending Docker rollout
budget when the invalidity is known before evaluation.

## Framework Boundary

No broad framework redesign is needed for this iteration.

The framework's current shape is plausible and useful:

- `Version` captures immutable agent state.
- `Slice` identifies the frozen task set.
- `Run` reads per-instance artifacts.
- `rollout`, `reward`, `criterion`, and `strategy` remain swappable components.
- `Context` and `Proposal` give the strategy a clean interface.
- `store` and `log` provide durable promotion and auditability.

The DGM recipe should demonstrate those pieces by composing them, not hiding
them behind a separate recipe-private abstraction.

One framework limitation was exposed: scalar reward alone is not enough to
express validity, diagnostics, and infrastructure health. For now, keep the
diagnostic channel recipe-local by enriching decision candidate metadata. If
multiple recipes need the same pattern later, promote a generic diagnostic
interface to the substrate in a separate design.

## Implementation Shape

Likely code changes:

- Update DGM reward/admission helpers in `recipes/dgm/evolve.py` and
  `recipes/dgm/swebench.py`.
- Add DGM-local diagnostic extraction helpers and tests.
- Update `recipes/dgm/ops/report.py` to summarize validity and diagnostic
  counts.
- Add a DGM-specific default package factory that reuses framework package
  loading/staging primitives.
- Update DGM meta-agent prompt to encourage meaningful bounded `agent/` package
  edits.
- Add tests for valid-but-regressed admission, container failure diagnostics,
  pre-rollout invalid package rejection, and report summaries.

Keep the richer scaffold recipe-local. If another recipe later needs the same
shape, promote it deliberately in a separate design instead of broadening the
framework package default now.

## Validation Plan

Use narrow deterministic tests first:

- DGM admission accepts valid tied and regressed children.
- DGM admission rejects package import/build failure.
- DGM admission does not reject a whole candidate for isolated container
  failures when the completion threshold holds.
- Report separates valid/invalid/improved/tied/regressed and diagnostic counts.
- The default DGM agent package loads through the existing agent-package loader.
- The meta-strategy still rejects edits outside `agent/`.

Then run a small dry or fake rollout smoke test before any expensive SWE-bench
run.

## Success Criteria

The next DGM run should show:

- generated candidates with meaningful bounded `agent/` package differences,
- valid children admitted into the archive even when tied or worse,
- invalid children explained by precise diagnostic categories,
- best-in-archive reporting on train and held-out artifacts,
- no hidden DGM framework replacing the shared evolution substrate.
