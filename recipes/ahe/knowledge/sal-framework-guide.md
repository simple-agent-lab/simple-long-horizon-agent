# SAL Framework Guide for AHE

Use this note as a local working guide when editing the AHE recipe or related
surface, analyzer, and ledger code.

## Purpose

The AHE recipe is a SAL-backed SWE-bench experiment.
It uses the evolution substrate for versioning, scoring, and promotion, while
keeping AHE-specific observability in recipe-local code and artifacts.

## Core pieces

- `recipes/ahe/surface.py` defines `ahe_harness_surface`.
- `recipes/ahe/strategy.py` builds the model strategy.
- `recipes/ahe/analyzer.py` creates the pre-proposal analysis artifacts.
- `recipes/ahe/ledger.py` writes the AHE run ledger.
- `recipes/ahe/evolve.py` wires config, registry setup, and execution.

## Surface guidance

- Treat `ahe_harness_surface` as the editable contract.
- Keep the component names stable unless the code changes first.
- Prefer component-level edits over whole-tree churn when the surface can
  express the change.
- When adding files, make sure they land inside the surface's allowed harness
  root.
- Preserve the entrypoint check and the syntax and path validators.

## Analyzer guidance

- The analyzer runs before each proposal step.
- It should summarize observed failures and patterns, not invent new
  capabilities.
- Keep prompts short enough that the strategy can reason over them.
- Write analysis artifacts to the round analysis directory, not to the root
  ledger.
- If the model does not return a usable payload, keep the fallback behavior
  explicit and legible.

## Ledger guidance

- The AHE ledger lives under `<run-root>/ahe/`.
- Round data belongs under `ahe/rounds/round_###/`.
- `change_manifest.json` describes the proposed change.
- `change_evaluation.json` records how the change lined up with measured task
  outcomes.
- `history.md` is the human-readable append-only log.
- `task_history.json` accumulates per-instance round history.
- `best_ever.json` tracks the best reward mean seen so far.

## Safe editing rules

- Read the current version and prior decisions before proposing a change.
- Prefer the smallest component that can explain the edit.
- Keep the manifest and evaluation data deterministic where possible.
- If a field is unused by the current flow, do not promise it in docs or code.
- Do not add new orchestration layers unless the recipe needs them.

## What to check after edits

- The recipe still registers `swebench`, `ahe_harness_surface`, `local_docker`,
  `local_dir`, and `ahe_model`.
- The analysis artifacts still land under the round analysis directory.
- The ledger still writes into `ahe/`.
- The dry-run path still reports the plan without claiming improved results.

## Good change shape

Favorable changes usually look like one of these:

- tighten the analyzer prompt
- make the manifest clearer
- improve the evidence record
- reduce unnecessary file churn on the editable surface
- add a small, explainable artifact to the ledger

Less favorable changes usually look like:

- broadening the editable surface without a reason
- making the strategy depend on hidden state
- moving recipe-local AHE data into the substrate
- adding benchmark-specific assumptions to SAL core code

## Remember

AHE here is a teaching and research recipe.
Keep the documentation honest, the artifacts inspectable, and the component
boundaries visible.

