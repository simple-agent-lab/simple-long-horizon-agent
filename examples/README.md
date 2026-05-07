# Examples

The active runnable examples are the three focused runtime designs in
[`design_versions/`](design_versions/README.md). ADR 0005 makes
`02_balanced_runtime` the lead core candidate, and ADR 0009 promotes it into
`src/simple_agent_lab/core.py`; `01` and `03` remain side by side as teaching
and graph/observability references.

## Run

```bash
uv run python examples/design_versions/01_functional_loop/demo.py
uv run python examples/design_versions/02_balanced_runtime/demo.py
uv run python examples/design_versions/03_event_runtime/demo.py
```

Plain `python3` works too — the demos are stdlib-only — but `uv run` is the
recommended path so the same commands keep working when dependencies are
added. From a source checkout, prefer the run script because it sets
`PYTHONPATH=src`:

```bash
bash runs/run_design_versions.sh
```

## What you are looking at

Each version is a runnable sketch of a different runtime shape:

| Version | Shape | When to read it |
| --- | --- | --- |
| `01_functional_loop` | Simple: one function, one file | Smallest readable agent loop |
| `02_balanced_runtime` | Moderate: promoted `src` core, generator-based, request/response events, tools, agent-as-tool delegation | Lead core for self-evolution work |
| `03_event_runtime` | Complex: graph nodes, edges, handoffs, event sourcing, observers, replay, reports, provider boundary | Graph orchestration and observability reference |

See [`design_versions/README.md`](design_versions/README.md) for the
side-by-side comparison and the current selection criteria.

## Earlier sketches

- `../scripts/run_tiny_demo.py`: a recipe demo built on the promoted
  `src/simple_agent_lab/core.py`. It shows debate / pipeline / parallel
  patterns over the same runtime used by 02.
