# Development

Day-to-day commands for working on this repo, plus the quality contract that
remote and local CI both enforce.

## Setup

The base package includes the supported model-provider SDKs. Heavier benchmark
tooling, such as SWE-bench, stays behind optional extras. `ruff` (formatter) and
`ty` (type checker) are dev tools and live in the `dev` dependency group. Use
`uv` to manage the environment:

```bash
uv sync --group dev
```

`--group dev` installs the dev tools. Without the flag you get the runtime
install only — fine for a quick demo, not enough to run the formatter or type
check. Add `--extra swebench` only when working on the SWE-bench adapter.

The MCP integration lives under `src/` (so it is type-checked and unit-tested)
but stays behind the optional `mcp` extra. The gate therefore syncs
`--group dev --extra mcp`; sync the same way locally when touching
`src/simple_agent_lab/mcp/`.

## The quality gate

Five checks must pass before a change ships. They're cheap; run them often.

| Check | Command | Scope |
| --- | --- | --- |
| Format | `uv run ruff format --check .` | Python code in the repo |
| Lint | `uv run ruff check .` | Python code in the repo |
| Docs lint | `uv run python scripts/lint_docs.py` | Local Markdown links and backticked path references |
| Type check | `uv run ty check src` | Every module under `src/` |
| Unit tests | `uv run python -m unittest discover -s tests/unit` | Every unit test under `tests/unit/` |

All checks must exit `0`. Use `uv run ruff format .` to format Python code
before running the check. There are no warn-only or skip lists — if a diagnostic
is wrong, fix the code or fix the type, do not silence the checker. The only
sanctioned suppression is a narrow `cast(...)` or `type: ignore[...]` for known
type-checker limitations, and only with a one-line comment explaining why.

## Local CI

`runs/dev/run_ci.sh` is the canonical local pre-push gate. It mirrors the remote
workflow exactly:

```bash
bash runs/dev/run_ci.sh
```

The script:

1. Verifies `uv` is installed (fails fast if not).
2. Runs `uv sync --group dev --extra mcp` so the dev tools and the
   type-checked/tested MCP extra are present.
3. Runs `uv run ruff format --check .`.
4. Runs `uv run ruff check .`.
5. Runs `uv run python scripts/lint_docs.py`.
6. Runs `uv run ty check src`.
7. Runs `uv run python -m unittest discover -s tests/unit`.
8. Runs `bash runs/demos/run_bash_agent_demo.sh` so the public teaching demo stays runnable.
9. Prints `All CI checks passed.` only if every step exited `0`.

If you want the same checks individually (e.g. while iterating on one of
them), run the commands from the table above directly. `run_ci.sh` is the
"single command, all gates" entry point.

## Remote CI

GitHub Actions runs the same gate on every push to `main` and every pull
request. The workflow is at [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml).

Three job groups run in parallel:

- **`tests / py3.10`** through **`tests / py3.13`** — the unittest suite on
  supported interpreters, plus the deterministic bash-agent demo smoke.
- **`ty / src`** — the type check, on Python 3.13 only. ty's diagnostics
  don't depend on the runtime Python version, so a single job is enough.
- **`docs lint`** — local Markdown link and backticked path-reference checks,
  on Python 3.13 only.
- **`ruff format / check`** — the formatter check (`ruff format --check .`)
  plus the linter (`ruff check .`), on Python 3.13 only. Run
  `uv run ruff format .` (and `uv run ruff check --fix .`) locally when this
  fails.

A pull request is mergeable only when all three jobs pass. There is no
auto-merge or required-reviewer config in this repo yet; the gate is
advisory but expected.

## Adding a new check

When extending the gate (e.g. a linter, a smoke test, a doc check):

1. Make it runnable via `uv run` so it picks up the dev environment.
2. Add it to `runs/dev/run_ci.sh` first, exactly as you'd want it to run in CI.
3. Add a matching step to `.github/workflows/ci.yml`. Keep the local script
   and the workflow in lockstep — divergence is the surest way for "passes
   locally, fails in CI" surprises to creep in.
4. Update the table above so contributors know what runs.

## Adding a new dev dependency

Add it to `[dependency-groups] dev` in `pyproject.toml`, then `uv sync
--group dev`. Do **not** add it to `[project] dependencies` unless the runtime
or supported provider adapters need it. Heavy eval-only dependencies should live
under `[project.optional-dependencies]`.

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
