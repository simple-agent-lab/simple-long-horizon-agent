# Recipes

A **recipe** is a small, runnable example that wires the benchmark-agnostic
evolution substrate to a concrete benchmark. Recipes are where keys, Docker, and
benchmark glue live; the substrate itself stays generic.

| Recipe | Role | Loop | When to read |
| --- | --- | --- | --- |
| [`simple/`](simple/README.md) | Ergonomics showcase — minimal user code, maximal agent freedom | Sequential step loop | "How little does it take to start a real self-evolving run?" |
| [`dgm/`](dgm/README.md) | Faithfulness showcase — a config-backed Darwin Gödel Machine reproduction with every knob exposed | Parallel open-ended archive-admission loop | "How do I reproduce DGM and tune it?" |

Both recipes evolve the **whole agent program** under `agent/`: a model rewrites
the agent's own Python files and each candidate is graded on a train slice inside
a SWE-bench Docker sandbox. The simple recipe is the config-backed generic
runner path with optional suite-scored heldout before/final reporting; the DGM
recipe is also YAML-backed, but owns its archive-specific official scoring
workflow and before/after delta used by the faithful reproduction.

## How the layers fit together

```text
recipes/                                  # this directory — examples + ops scripts
  runtime.py                              # Docker/env helpers (kept out of src/)
  simple/evolve.py                        # sequential not-worse recipe
  dgm/config.py                           # DGM YAML schema and CLI overrides
  dgm/evolve.py                           # the DGM recipe entrypoint
  dgm/swebench.py                         # DGM-specific SWE-bench run/scoring support
  dgm/algorithm/archive.py, open_ended.py  # DGM archive + parallel admission loop
  dgm/algorithm/repo_edits.py              # DGM proposal helpers
  dgm/ops/baseline.py, report.py           # DGM operational commands

evals/swebench/
  suite.py                                # SWE-bench benchmark interface

src/simple_agent_lab/evolution/           # benchmark-agnostic substrate
  kernel/        # store, log, loop, experiment, types
  components/    # reward, criterion, rollout, strategy
```

The substrate never imports a benchmark or the host-side `evals/` tree; the
benchmark interface stays in `evals/swebench/suite.py`; DGM-specific SWE-bench
support lives with the DGM recipe and never imports Docker (that boundary is
enforced by `scripts/arch_lint.py`). Anything that has to touch Docker or the
host environment lives here in `recipes/`.

For the concepts behind self-evolution (substrate vs. recipe, the archive,
parent selection, criteria), see
[src/simple_agent_lab/evolution/README.md](../src/simple_agent_lab/evolution/README.md)
and [docs/agent-native/self-evolving.md](../docs/agent-native/self-evolving.md).

## Common prerequisites

Both recipes are real — they call a model and run Docker:

- A `.env` file with your provider credentials (`OPENAI_API_KEY`, optionally
  `OPENAI_MODEL` and an `OPENAI_BASE_URL`).
- A reachable Docker daemon (Docker Desktop or Colima).
- A SWE-bench wheelhouse (the run wrappers under `runs/` prepare this for you).
- Train/test SWE-bench splits as JSONL files. The checked-in defaults live under
  `configs/swebench/`; see the DGM recipe's `ops/baseline.py` when you want to
  build a larger repo-balanced "headroom" split.

Every recipe is a **dry plan by default**; pass `--execute` to run the real model
and Docker.
