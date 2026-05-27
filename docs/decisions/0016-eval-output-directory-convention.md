# ADR 0016: Eval Output Directory Convention

## Status

Accepted

## Context

Eval runs produce local artifacts (trajectories, predictions, instance records,
pre-built wheels) under `evals/out/`. This directory was fully gitignored, so
new contributors cloning the repo had no visibility into the expected output
structure. When they ran an eval for the first time, there was no guidance on
where files should land or how to organize outputs from different benchmark
suites.

As we add more benchmarks beyond SWE-bench — each with potentially different
setup requirements (docker images, git-cloned repos, datasets) — a consistent
convention becomes necessary to keep the output tree navigable and reproducible
across machines.

## Decision

1. **One subdirectory per benchmark suite.** Each suite's outputs live under
   `evals/out/<suite>/`, matching the adapter directory name `evals/<suite>/`.
   All suite-specific artifacts (fetched instances, wheelhouses, run outputs)
   stay inside that subdirectory.

2. **Committed README skeleton.** Each `evals/out/` and
   `evals/out/<suite>/` directory contains a committed `README.md` that
   documents the expected layout, reproduction commands, and file-size
   expectations. Gitignore uses negation rules (`!…/README.md`) to allow
   only these files through.

3. **Standard checklist for new benchmarks.** Adding a new suite requires:
   - `evals/<suite>/` — adapter code and README
   - `evals/out/<suite>/README.md` — output layout docs
   - `.gitignore` negation entry for the new README
   - `runs/run_<suite>.sh` — convenience runner
   - `runs/setup_<suite>.sh` — if external resources are needed

## Consequences

- Users see the full directory skeleton after cloning, before running anything.
- Large artifacts (JSONL traces, wheels, cloned repos) stay gitignored.
- Adding a new benchmark follows a repeatable pattern instead of ad-hoc
  directory creation.
- Each README doubles as living documentation for the output format, reducing
  the chance of silent layout drift.

## Alternatives Considered

- **Commit sample output files.** Rejected — even small JSONL files grow stale
  and add noise to diffs. READMEs with tree diagrams are cheaper to maintain.
- **Generate the skeleton via a script.** Considered but deferred — a script
  adds a step users can skip; committed READMEs are zero-effort and always
  present.
- **Use a monorepo artifact store (e.g. DVC, Git LFS).** Out of scope for the
  current project size; can be layered on later without changing the directory
  convention.
