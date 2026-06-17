# Recipes

A **recipe** is a small, runnable example that wires the benchmark-agnostic
evolution substrate to a concrete benchmark. Recipes are where keys, Docker, and
benchmark glue live; the substrate itself stays generic.

| Recipe | Role | Loop | When to read |
| --- | --- | --- | --- |
| [`simple/`](simple/README.md) | Ergonomics showcase — minimal user code, maximal agent freedom | Sequential `Experiment.run` | "How little does it take to start a real self-evolving run?" |
| [`dgm/`](dgm/README.md) | Faithfulness showcase — a Darwin Gödel Machine reproduction with every knob exposed | Parallel open-ended archive-admission loop | "How do I reproduce DGM and tune it?" |

Both recipes evolve the **whole agent program** under `agent/`: a model rewrites
the agent's own Python files, each candidate is graded on a train slice inside a
SWE-bench Docker sandbox, and the best valid agent is scored on a held-out test
slice.

## How the layers fit together

```text
recipes/                                  # this directory — examples + ops scripts
  _shared.py                              # Docker/env helpers (kept out of src/)
  simple/evolve.py                        # the simple recipe
  dgm/evolve.py, baseline.py, report.py   # the DGM recipe + its ops scripts

src/simple_agent_lab/evals/suites/swebench/
  evolving_rollout.py                     # SWE-bench adapter (benchmark glue, docker-free)

src/simple_agent_lab/evolution/           # benchmark-agnostic substrate
  kernel/        # store, log, loop, experiment, types
  components/    # reward, criterion, rollout, strategy
  archive.py, open_ended.py               # archive + parallel admission loop
```

The substrate never imports a benchmark; the SWE-bench adapter never imports
Docker (that boundary is enforced by `scripts/arch_lint.py`); anything that has
to touch Docker or the host environment lives here in `recipes/`.

For the concepts behind self-evolution (substrate vs. recipe, the archive, parent
selection, criteria), see the self-evolving concept guide under `docs/`.

## Common prerequisites

Both recipes are real — they call a model and run Docker:

- A `.env` file with your provider credentials (`OPENAI_API_KEY`, optionally
  `OPENAI_MODEL` and an `OPENAI_BASE_URL`).
- A reachable Docker daemon (Docker Desktop or Colima).
- A SWE-bench wheelhouse (the run wrappers under `runs/` prepare this for you).
- Train/test SWE-bench splits as JSONL files (see the DGM recipe's
  `baseline.py` for building a balanced "headroom" split).

Every recipe is a **dry plan by default**; pass `--execute` to run the real model
and Docker.
