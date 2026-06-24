# AHE SWE-bench Recipe Design

Date: 2026-06-24
Status: Draft for owner review

## Goal

Create one executable self-evolving recipe that reproduces the idea and effect
of Agentic Harness Engineering (AHE) on SWE-bench while demonstrating the
flexibility of Simple Agent Lab's existing evolution framework.

The recipe should let a user run a real coding benchmark loop and inspect
whether harness edits improve performance. It should stay simple enough that the
framework boundary is visible: the recipe supplies AHE policy and observability;
the evolution substrate supplies versioning, fair comparison, decision logging,
and promotion.

## Reference Interpretation

AHE's core contribution is observability-driven harness evolution, not a separate
evolution kernel. The recipe should reproduce three AHE pillars:

- Component observability: the editable coding-agent harness is exposed as
  file-level components.
- Experience observability: a model-backed analyzer distills rollout traces into
  layered reports before the meta-agent proposes edits.
- Decision observability: each edit carries evidence, root cause, predicted
  fixes, risk tasks, and later task-delta attribution.

The recipe should not import or depend on NexAU. NexAU's role in AHE is the
component substrate. In Simple Agent Lab, the existing `AgentSurface` abstraction
can provide the same action-space shape with AHE-specific components.

The recipe should not implement Best-of-N in this version. Best-of-N appears in
the local AHE implementation as a practical extension, but it is not one of the
paper's three observability pillars and would add substantial orchestration and
benchmark cost.

## Architecture Boundary

The AHE recipe lives under `recipes/ahe/` and composes the existing configured
evolution runner with recipe-specific factories.

The evolution kernel remains authoritative for:

- immutable versions under `<run-root>/evolution/versions/`
- the current pointer
- fair baseline/candidate rollout on the same frozen SWE-bench train slice
- criterion evaluation
- `decisions.jsonl`
- promotion

The AHE recipe owns:

- the AHE-shaped `AgentSurface`
- seed harness files
- the model-backed analyzer
- the meta-strategy prompt and manifest parsing
- the AHE evidence ledger
- change attribution reports
- recipe docs, run wrapper, default config, and optional local knowledge pack

One round should behave conceptually as:

```text
current harness version
  -> rollout on SWE-bench train slice
  -> model-backed analyzer writes evidence corpus
  -> meta-agent proposes harness edits plus manifest
  -> kernel stages candidate
  -> candidate rollout on the same train slice
  -> kernel criterion accepts or rejects
  -> AHE ledger records manifest, task deltas, attribution, and best-ever
```

## Editable Harness Surface

Define a concrete `AgentSurface` named `ahe_harness_surface`, with files under
`harness/`.

Seed files:

```text
harness/
  agent_program.py
  code_agent.yaml
  systemprompt.md
  LongTermMEMORY.md
  ShortTermMEMORY.md
  tool_descriptions/
    bash.tool.md
  tools/
    bash.py
  middleware/
    README.md
  skills/
    README.md
  sub_agents/
    README.md
```

`harness/agent_program.py:build_agent` is the adapter that builds a Simple Agent
Lab `Agent` from the harness files. The initial seed should stay intentionally
minimal: one bash-style tool, a short system prompt, empty long-term memory, no
middleware, no skills, and no sub-agents. This mirrors AHE's minimal seed
principle while keeping the runtime inside Simple Agent Lab.

Surface components:

| Component id | Paths | Purpose |
| --- | --- | --- |
| `agent_program` | `harness/agent_program.py` | Adapter and agent assembly. |
| `system_prompt` | `harness/systemprompt.md` | Global behavioral guidance. |
| `tool_descriptions` | `harness/tool_descriptions/**` | Model-facing tool guidance and examples. |
| `tool_implementations` | `harness/tools/**` | Executable tool behavior. |
| `middleware` | `harness/middleware/**` | Hook-like policies implemented through the adapter. |
| `skills` | `harness/skills/**` | Reusable workflow guidance loaded by the agent. |
| `sub_agents` | `harness/sub_agents/**` | Delegated worker definitions or prompts. |
| `long_term_memory` | `harness/LongTermMEMORY.md` | Persistent cross-run lessons. |
| `everything` | `harness/**` | Whole harness edits when enabled by config. |

Validators should reject unsafe paths, invalid Python syntax, deletion of the
required `build_agent` entrypoint, and any edit outside the selected components.
Provider configuration, benchmark fixtures, verifier code, and scoring remain
outside the surface.

The meta-agent prompt should use the surface's component brief to make the AHE
action space explicit. It should ask the model to choose the component level
that best matches the root cause, rather than defaulting to prompt edits.

## Model-Backed Analyzer

The analyzer is required. It runs after the current version's train rollout and
before the meta-strategy proposes edits.

Inputs:

- current `Version`
- baseline `Run` objects from the train slice
- previous kernel decisions
- the AHE ledger, including history and prior change evaluations
- local knowledge pack files when enabled

Outputs:

```text
<run-root>/ahe/rounds/round_XXX/analysis/
  overview.md
  detail/<instance_id>.md
  index.json
```

The analyzer should summarize failures and partial successes into component-level
hypotheses:

```text
failure pattern -> root cause -> likely component -> evidence -> suggested fix shape
```

It should prefer progressive disclosure. `overview.md` is the main entry point
for the meta-agent. Per-instance details hold drill-down evidence and pointers
to raw run artifacts. `index.json` gives structured data for tests and later
attribution.

The analyzer may use a fake completion function in tests. Real execution uses
the configured provider.

## Local Knowledge Pack

Include a small checked-in knowledge pack under `recipes/ahe/knowledge/`.

Initial files:

```text
recipes/ahe/knowledge/
  sal-framework-guide.md
  coding-agent-design-patterns.md
```

The default recipe consumes these local files for reproducibility. A future
explore step may refresh them from web and source-code research, but live
external exploration is not part of the default executable recipe.

The knowledge pack should give the analyzer and meta-agent concrete guidance on
Simple Agent Lab extension points and frontier coding-agent design patterns
without requiring network access during benchmark runs.

## Decision Observability And Ledger

Add an AHE recipe ledger beside, not inside, the kernel state:

```text
<run-root>/ahe/
  history.md
  task_history.json
  best_ever.json
  rounds/
    round_001/
      analysis/
        overview.md
        detail/<instance_id>.md
        index.json
      change_manifest.json
      change_evaluation.json
      meta_trace.jsonl
```

The ledger answers why a change was proposed and whether the meta-agent's
predictions matched the measured task deltas. It does not replace
`evolution/decisions.jsonl`, which remains the source of truth for promotion.

The meta-strategy should produce a structured manifest alongside full-file
edits:

```json
{
  "round": 1,
  "base_version": "...",
  "changes": [
    {
      "id": "chg-1",
      "type": "new|improvement|rollback",
      "component": "system_prompt|tool_description|tool_implementation|middleware|skill|sub_agent|long_term_memory|agent_program",
      "files": ["harness/systemprompt.md"],
      "failure_pattern": "...",
      "root_cause": "...",
      "targeted_fix": "...",
      "predicted_fixes": ["instance-id"],
      "risk_tasks": ["instance-id"],
      "why_this_component": "..."
    }
  ]
}
```

After candidate evaluation, the recipe writes `change_evaluation.json` by
comparing baseline and candidate per-instance scores. Prediction verification is
reporting and future context, not a second promotion gate. The kernel criterion
accepts or rejects candidates.

## Loop Shape

The default loop is sequential and uses one candidate proposal per round.

Best-of-N is acknowledged as a useful future extension and as a feature present
in the local AHE implementation, but it is out of scope for this recipe version.
Leaving it out keeps the implementation focused on the AHE observability pillars
and keeps real SWE-bench runs cheaper.

## Config And Runner

Add:

```text
recipes/ahe/
  README.md
  evolve.py
  analyzer.py
  strategy.py
  surface.py
  ledger.py
  knowledge/
configs/ahe_swebench.yaml
runs/run_self_evolving_ahe.sh
```

`recipes/ahe/evolve.py` should follow the same factory-registration style as
`recipes/simple/evolve.py`, then delegate to `simple_agent_lab.evolution.run`.
The default config should be a dry-run plan unless `--execute` is passed.

The config should reuse the existing SWE-bench suite, local Docker backend,
local artifact store, model provider settings, heldout before/final evaluation,
and `promote_not_worse` criterion unless a task requires otherwise.

## Validation

Dry-run validation:

```bash
bash runs/run_self_evolving_ahe.sh --run-id ahe-smoke
```

This should require no model key or Docker execution. It should prove config
loading, factory registration, train/heldout count resolution, surface
construction, and run-root planning.

Focused tests:

- `ahe_harness_surface` exposes AHE component paths and rejects unsafe edits.
- seed harness files include a valid `harness/agent_program.py:build_agent`.
- analyzer writes `overview.md`, `detail/*.md`, and `index.json` using fake
  runs and fake model completion.
- meta-strategy parses JSON into a `Proposal` plus `change_manifest.json`.
- change attribution computes expected fixes, false predictions, unexpected
  fixes, and regressions from baseline/candidate scores.
- `recipes/ahe/evolve.py` registers factories and delegates to the configured
  run path.

Real-run validation:

```bash
bash runs/run_self_evolving_ahe.sh \
  --run-id ahe-real \
  --execute
```

For any performance claim, report:

- config path
- train and heldout instance files
- model/provider
- number of rounds
- train score deltas and accepted/rejected decisions
- heldout baseline/final score
- missing-result count and fallback count
- paths to `evolution/decisions.jsonl` and the AHE ledger

The recipe should make real improvement possible and inspectable; a dry-run
smoke check is not itself a performance claim.

## Out Of Scope

- NexAU runtime dependency.
- Terminal-Bench adapter.
- Best-of-N candidate orchestration.
- Live web exploration in the default run.
- Model-weight training or changes to provider configuration.
- A second evolution kernel or promotion mechanism.
