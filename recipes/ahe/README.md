# AHE SWE-bench recipe

This recipe demonstrates a self-evolving SWE-bench run that keeps AHE behavior
visible instead of hiding it behind a larger framework. The current
implementation reproduces the AHE component, experience, and decision
observability story on top of the SAL evolution framework, using
`ahe_harness_surface` as the editable surface. Each proposal round runs a
role-separated SAL evolve agent over a materialized harness workspace; the
evaluated code agent and the evolve agent are separate roles, matching AHE's
shape without adding a new framework runtime.

## What it maps to

- Component observability -> `ahe_harness_surface`
- Experience observability -> the model-backed analyzer plus its analysis
  artifacts staged into the evolve-agent workspace
- Decision observability -> `change_manifest.json`,
  `change_evaluation.json`, and the `ahe/` ledger files

The recipe is explicit about the boundary choices:

- It uses SAL's evolution substrate and `AgentSurface`.
- It runs a model-backed analyzer before each proposal.
- It runs a SAL evolve agent to inspect `analysis/`, edit `harness/`, and write
  `change_manifest.json`.
- It stores the AHE ledger under the run root's `ahe/` directory.
- It does not add Best-of-N search, live external exploration, a
  Terminal-Bench adapter, or a NexAU runtime dependency.

## Run it

Dry run the configured plan:

```bash
bash runs/run_self_evolving_ahe.sh --run-id ahe-smoke
```

Run the configured model and Docker loop:

```bash
bash runs/run_self_evolving_ahe.sh --run-id ahe-real --execute
```

The default config is [`../../configs/ahe_swebench.yaml`](../../configs/ahe_swebench.yaml).
If you need a different model, split, or output root, copy that file and edit
the YAML before running the wrapper.

## What it writes

Executed runs write the normal evolution log plus AHE-specific analysis and
ledger artifacts.

Expected paths include:

- `evolution/decisions.jsonl`
- `ahe/history.md`
- `ahe/task_history.json`
- `ahe/best_ever.json`
- `ahe/rounds/round_001/analysis/overview.md`
- `ahe/rounds/round_001/change_manifest.json`
- `ahe/rounds/round_001/change_evaluation.json`
- `evaluation/summary.json` when heldout evaluation is enabled

## Research claim checklist

Treat a run as research evidence only when the report names all of the
following:

- Config path
- Train and heldout instance paths
- Provider and model
- Number of rounds
- Train deltas
- Heldout baseline and heldout final
- Missing-result counts and fallback counts
- Paths to the decision log and the AHE ledger

A dry run is useful for checking wiring, but it does not establish a
performance claim.

## Out of scope

This recipe does not include:

- Best-of-N proposal search
- Live web exploration
- Terminal-Bench support
- NexAU runtime dependency
