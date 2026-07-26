# Development

The quality contract local and remote CI both enforce, plus the things that are
easy to get wrong.

## Setup

```bash
uv sync --group dev --extra mcp
```

`--group dev` brings in `ruff` and `ty`. The `mcp` extra matters because the MCP
integration lives under `src/` and is both type-checked and unit-tested — sync
it or those checks silently skip. Add `--extra swebench` only when working on
that adapter.

## The gate

```bash
bash runs/dev/run_ci.sh     # single command, all gates — the pre-push check
```

It mirrors `.github/workflows/ci.yml` exactly; keep the two in lockstep when
adding a check, or you get "passes locally, fails in CI". The individual
commands, when iterating on one of them:

| Check | Command |
| --- | --- |
| Format | `uv run ruff format --check .` |
| Lint | `uv run ruff check .` |
| Docs lint | `uv run python -m scripts.lint_docs` |
| Config reference | `uv run python -m scripts.build_config_reference --check` |
| Architecture lint | `uv run python -m scripts.arch_lint` |
| Environment lint | `uv run python -m scripts.env_lint` |
| Type check | `uv run ty check src` |
| Unit tests | `uv run python -m unittest discover -s tests/unit` |

All must exit `0`. There are no warn-only or skip lists — if a diagnostic is
wrong, fix the code or the type, do not silence the checker. The only sanctioned
suppression is a narrow `cast(...)` or `type: ignore[...]` with a one-line
comment explaining why.

Tests use the stdlib `unittest` runner; there is no `pytest` dependency.

## Known `ty` false positives

Reproduce locally first (`uv run ty check src`). Three recurring shapes:

- **Closure narrowing** — `ty` does not propagate `is None` / `isinstance`
  checks into nested function bodies. Bind the narrowed value to a fresh local
  before defining the closure.
- **Parametrized isinstance** — `isinstance(x, Mapping)` narrows to
  `Mapping[Unknown, Unknown]`, dropping `[str, Any]`. Use
  `cast("dict[str, Any]", ...)` right after the check, with a short comment.
- **Forward references within a package** — use `if TYPE_CHECKING: from .other
  import Thing` rather than a bare string annotation.

If `unittest` flakes (it should not — the suite has no nondeterminism), fix the
test or the code. Do not retry-loop in CI.

## Dependencies

Dev tools go in `[dependency-groups] dev`. Heavy eval-only dependencies go in
`[project.optional-dependencies]`. Only add to `[project] dependencies` when the
runtime or a supported provider adapter needs it. Always `uv add` / `uv sync`,
never bare `pip`.

## Releases

Do not tag, publish, or change package metadata without explicit owner
instruction.
