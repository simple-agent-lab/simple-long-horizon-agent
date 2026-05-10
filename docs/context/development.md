# Development

Day-to-day commands for working on this repo, plus the quality contract that
remote and local CI both enforce.

## Setup

The runtime itself is stdlib-only, but `ty` (the type checker) is a dev tool
and lives in the `dev` dependency group. Use `uv` to manage the environment:

```bash
uv sync --group dev
```

`--group dev` installs the dev tools (currently just `ty`). Without the flag
you get the runtime install only — fine for a quick demo, not enough to run
the type check.

## The quality gate

Two checks must pass before a change ships. They're cheap; run them often.

| Check | Command | Scope |
| --- | --- | --- |
| Type check | `uv run ty check src` | Every module under `src/` |
| Unit tests | `uv run python -m unittest discover -s tests` | Every test under `tests/` |

Both checks must exit `0`. There are no warn-only or skip lists — if a
diagnostic is wrong, fix the code or fix the type, do not silence the
checker. The only sanctioned suppression is `cast(...)` for known type-checker
limitations (e.g. the `Mapping` parametrized-isinstance narrowing in
`core.py:resolve_request_extra`), and only with a one-line comment explaining
why.

## Local CI

`runs/run_ci.sh` is the canonical local pre-push gate. It mirrors the remote
workflow exactly:

```bash
bash runs/run_ci.sh
```

The script:

1. Verifies `uv` is installed (fails fast if not).
2. Runs `uv sync --group dev` so the dev tools are present.
3. Runs `uv run ty check src`.
4. Runs `uv run python -m unittest discover -s tests`.
5. Prints `All CI checks passed.` only if every step exited `0`.

If you want the same checks individually (e.g. while iterating on one of
them), run the commands from the table above directly. `run_ci.sh` is the
"single command, both gates" entry point.

## Remote CI

GitHub Actions runs the same gate on every push to `main` and every pull
request. The workflow is at [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml).

Two jobs run in parallel:

- **`tests / py3.10`** and **`tests / py3.13`** — the unittest suite on the
  oldest and newest supported interpreters. The matrix is intentionally just
  the two endpoints: enough to catch syntax features that 3.10 doesn't
  support and behavior changes in newer stdlib, without paying for every
  version in between.
- **`ty / src`** — the type check, on Python 3.13 only. ty's diagnostics
  don't depend on the runtime Python version, so a single job is enough.

A pull request is mergeable only when all three jobs pass. There is no
auto-merge or required-reviewer config in this repo yet; the gate is
advisory but expected.

## Adding a new check

When extending the gate (e.g. a linter, a smoke test, a doc check):

1. Make it runnable via `uv run` so it picks up the dev environment.
2. Add it to `runs/run_ci.sh` first, exactly as you'd want it to run in CI.
3. Add a matching step to `.github/workflows/ci.yml`. Keep the local script
   and the workflow in lockstep — divergence is the surest way for "passes
   locally, fails in CI" surprises to creep in.
4. Update the table above so contributors know what runs.

## Adding a new dev dependency

Add it to `[dependency-groups] dev` in `pyproject.toml`, then `uv sync
--group dev`. Do **not** add it to `[project] dependencies` unless the
runtime needs it at import time — the package is supposed to install with
zero third-party deps for the simple-demo path.

## When CI is wrong

If `ty` flags something you believe is correct, reproduce locally first
(`uv run ty check src`). Common false-positive shapes:

- **Closure narrowing**: ty doesn't propagate `is None` / `isinstance` checks
  into nested function bodies. Fix by binding the narrowed value to a fresh
  local variable before defining the closure.
- **Parametrized isinstance**: `isinstance(x, Mapping)` narrows to
  `Mapping[Unknown, Unknown]`, dropping the `[str, Any]` parameters. Fix
  with `cast("dict[str, Any]", ...)` immediately after the check, with a
  short comment explaining the limitation.
- **Forward references to types in the same package**: use
  `if TYPE_CHECKING: from .other import Thing` rather than a bare string
  annotation; ty can't always resolve forward refs through `__future__
  annotations` alone.

If `unittest` flakes (it shouldn't — there's no nondeterminism in the
suite), open the failing test, re-read the assertion, and fix the test or
the code. Do not retry-loop in CI.
