# Source-Tree Evolution Design

Date: 2026-06-25

## Status

Design draft approved in conversation for spec writing. Implementation has not
started.

## Problem

The current self-evolving SWE-bench recipes expose a staged wrapper package
under `agent/` as the editable target. In configs this is called
`editable_components: [everything]`, but "everything" only means everything in
that small wrapper package, not the real Simple Agent Lab implementation.

That mismatch makes experiment results misleading. A run can appear to evolve
the whole agent while only changing a prompt wrapper around
`make_bash_agent`. The observed accepted candidates were prompt-only edits
because the real framework source under `src/simple_agent_lab/` was not the
evolution target.

## Goals

- Make real DGM-style source evolution the only user-facing self-evolution path.
- Evolve the actual Simple Agent Lab source used by task agents:
  `src/simple_agent_lab/**`.
- Remove or demote the lightweight wrapper-package surface from real recipes.
- Provide one framework-level source-tree surface that simple, DGM, AHE, and
  future recipes can reuse.
- Use git-backed candidate worktrees or equivalent isolated source trees so
  candidate edits are inspectable and never mutate the seed checkout.
- Run a cheap local validation gate before spending SWE-bench Docker and model
  budget.
- Keep the outer DGM/simple/AHE recipe machinery fixed for this phase. This is
  DGM-style source evolution, not Hyperagent-style self-modification of the
  improver itself.

## Non-Goals

- Do not allow candidate edits to `recipes/**`, `configs/**`, `evals/**`,
  `tests/**`, or docs in this phase.
- Do not make the meta-improvement loop itself editable yet.
- Do not require official SWE-bench scoring for every candidate.
- Do not mutate the user's main working tree while generating or evaluating
  candidates.
- Do not preserve the wrapper-package evolution path as a supported recipe
  mode.

## Editable Scope

The source-tree surface exposes exactly:

```text
src/simple_agent_lab/**
```

The surface rejects edits outside that root. It also excludes generated,
runtime, and sensitive paths by construction because they are not under the
editable root.

The candidate may change agent runtime, starter factories, tools, context
projection, trace handling, memory, LLM bridge behavior, and other framework
code under `src/simple_agent_lab/`. It may not change recipe policy, benchmark
fixtures, configs, tests, output artifacts, or secrets.

## Architecture

Add a reusable source-tree evolution surface to the framework:

```text
SourceTreeSurface
  editable_root = src/simple_agent_lab
  validates paths and Python syntax
  creates candidate worktrees/source trees
  packages candidate source for rollout
  exposes a prompt brief for meta-agents
```

Recipes consume the same surface:

```text
simple recipe -> sequential source evolution
DGM recipe    -> archive/open-ended source evolution
AHE recipe    -> observability around source-evolution candidates
future recipe -> opt into source_tree surface
```

The evolution kernel remains benchmark-agnostic. It still records immutable
versions, decision logs, fair same-slice comparisons, and promotions. The new
surface and rollout staging decide how source-tree candidates become executable
inside a benchmark container.

## Candidate Flow

For each candidate:

```text
current accepted version
  -> create isolated git worktree or candidate source tree
  -> run meta-agent in that tree
  -> meta-agent edits only src/simple_agent_lab/**
  -> run cheap local validation
  -> capture git diff / changed tree
  -> stage candidate version
  -> rollout mounts or installs candidate source
  -> SWE-bench container imports candidate simple_agent_lab
  -> reward, decision log, archive/promotion
```

Candidate generation and candidate evaluation are separate. A candidate that
fails cheap validation is still recordable as an invalid candidate, but it does
not spend SWE-bench rollout budget.

## Meta-Agent Strategy

The default source-tree strategy should be an agentic coding agent, not a
single JSON LLM call.

The meta-agent receives:

- the candidate worktree rooted at the repo;
- a clear editable-scope instruction: only `src/simple_agent_lab/**`;
- recent failure summaries and decision context;
- shell/read/search tools;
- the cheap validation command it must run before finalizing.

The meta-agent edits files directly. Its output is the resulting worktree diff
plus a short final summary, not hand-written JSON full-file edits.

The existing one-shot `model_program_strategy` may remain as a low-level test
strategy if useful, but it should not be the default for real source-tree
recipes.

## Git and Version Storage

Use git for candidate editing and diff capture:

- seed state is a commit or tree reference;
- each candidate gets an isolated worktree or copied repo tree;
- candidate changes are captured with `git diff -- src/simple_agent_lab`;
- the evolution version records the patch/tree metadata and provenance;
- the decision log remains the experiment source of truth.

The old seed is safe because candidates are isolated. Git is not the safety
boundary for executing candidate code; Docker/process isolation and resource
limits still matter when candidate code runs.

## Rollout Staging

Rollout must ensure the SWE-bench container imports the candidate source tree,
not the installed baseline wheel.

The staging contract should make this explicit. Acceptable mechanisms include:

- mount the candidate repo/source tree into the container and put its `src/`
  first on `PYTHONPATH`;
- or install the candidate package editable inside the container before running
  the task agent.

The contract must be testable. A candidate that changes a small identifiable
constant or version marker under `src/simple_agent_lab` should be observable
inside the container.

## Cheap Local Validation

Before SWE-bench rollout, run cheap validation in the candidate tree:

```bash
python -m compileall -q src/simple_agent_lab
python - <<'PY'
import simple_agent_lab
from simple_agent_lab.agents.starter import make_bash_agent
PY
```

If validation fails:

- record the candidate as invalid;
- attach the validation output to candidate diagnostics;
- score the candidate as zero or invalid according to the recipe criterion;
- skip SWE-bench rollout for that candidate.

This avoids spending Docker/provider budget on syntax or import failures while
still preserving evidence that the meta-agent generated a bad candidate.

## Recipe Changes

### Simple

`simple` becomes the minimal sequential source-evolution recipe. It should no
longer register the wrapper-package surface as the default real surface.

### DGM

`dgm` becomes the archive/open-ended source-evolution recipe. It should reuse
the same source-tree surface and candidate validation gate, while keeping its
archive parent-selection and branch scheduling recipe-local.

### AHE

`ahe` should observe and analyze source-evolution runs. If it proposes edits,
those edits should use the same source-tree surface rather than a separate
wrapper scaffold.

### Future Recipes

Future recipes should be able to choose the framework-provided source-tree
surface by name and avoid writing custom staging code unless their benchmark
requires a new rollout adapter.

## Documentation Changes

Update docs and examples to remove ambiguous language:

- Avoid saying "whole agent" when the target is only a wrapper package.
- Replace `everything under agent/` examples in real recipes with
  `src/simple_agent_lab/**` source-tree evolution.
- Explain that wrapper-package evolution was removed from user-facing recipes
  because it did not represent true framework self-evolution.
- Keep any tiny wrapper fixture only in tests, named as a toy fixture if it
  remains at all.

## Error Handling

- Invalid meta-agent output or no source changes: record proposal failure and
  do not stage a candidate.
- Edits outside `src/simple_agent_lab/**`: reject before validation.
- Cheap validation failure: record invalid candidate diagnostics and skip
  SWE-bench rollout.
- Rollout missing `out/result.json`: treat as a bad candidate result that scores
  zero, not as a process-killing error.
- Docker/provider failure outside candidate behavior: surface as operational
  failure so the run can be retried or monitored separately.

## Testing Plan

Unit tests:

- source-tree surface accepts `src/simple_agent_lab/**` and rejects all other
  paths;
- candidate diff capture only includes allowed paths;
- cheap validation failure produces invalid candidate diagnostics;
- wrapper-package surface is not registered by user-facing simple/DGM recipes;
- rollout staging gives candidate source precedence over installed package.

Integration smoke:

- create a synthetic candidate changing a visible marker under
  `src/simple_agent_lab`;
- run a tiny container/eval smoke;
- assert the container sees the candidate marker.

Real smoke:

- run simple source evolution on a tiny train slice;
- run DGM source evolution with one round and one or two branches;
- verify candidates are source-tree diffs, not wrapper prompt edits.

## Migration Plan

1. Introduce source-tree surface and candidate worktree helpers.
2. Add cheap validation and diagnostics.
3. Add rollout staging for candidate `src/simple_agent_lab`.
4. Convert simple recipe to source-tree evolution.
5. Convert DGM recipe to source-tree evolution.
6. Convert or clarify AHE behavior.
7. Remove wrapper-package evolution from user-facing configs/docs.
8. Run focused unit tests, synthetic staging smoke, and real tiny SWE-bench
   smoke.

## Open Questions

- Should source-tree versions store full file snapshots, git patches, or both?
- Should a cheap unit-test subset be configurable per recipe after compile/import
  checks?
- How much recent trajectory evidence should the meta-agent receive by default?
- Should failed cheap-validation candidates enter the DGM archive as invalid
  nodes or only the decision log?
