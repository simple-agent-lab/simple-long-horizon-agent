# OpenClaw Harness Experiment Plan

This note turns the current OpenClaw benchmark adapters into a harness study.
The goal is not just to report benchmark averages, but to measure which harness
choices repair which agent failure modes, and which choices introduce negative
interference.

## Design Principle

The experiment should follow the clawRecipe philosophy:

- Treat each score as a property of a model + harness + benchmark configuration,
  not the model alone.
- Keep task instances, model, decoding, budgets, and scoring fixed while varying
  one harness dimension at a time.
- Log final artifacts, trace, token usage, tool calls, wall time, validator
  outputs, and failure labels.
- Analyze both repaired failures and broken successes.

This matches clawRecipe's control-loop diagnosis:

- A. Goal grounding: task interpretation, success criteria, plan drift.
- B. Action instantiation: tool choice, paths, arguments, output format.
- C. State representation: durable facts, progress, artifacts, unresolved issues.
- D. Outcome verification: checking outputs, reacting to tool errors, recovery.
- E. Skill activation/composition: choosing and sequencing procedural knowledge.
- F. External/infrastructure: dependency, sandbox, evaluator, permission failures.

It also matches recent harness-aware evaluation work. Harness-Bench argues that
agent capability should be reported at the model-harness configuration level and
logs artifacts, traces, usage statistics, and validator outputs. ToolSandbox and
tau-bench emphasize stateful tool execution, user/tool interaction, and final
state evaluation. OSWorld emphasizes reproducible setup, post-processing, and
execution-based evaluation. Agent Harness Engineering treats sandboxing as a
first-class harness layer for security, reproducibility, and liveness.

## Experimental Unit

Each run is one tuple:

```text
(benchmark, task_id, model, harness_variant, backend, budget, seed_or_trial)
```

Use the existing suite boundary:

```text
Suite host half -> LaunchSpec/backend/store -> container half -> result.json + trajectory.jsonl
```

The harness variant should be configured outside benchmark task content. Do not
edit task prompts or gold verifiers while testing harness effects.

## Bench Coverage

Run all integrated suites, but separate scoring maturity:

| Bench | Current scoring status | Use in harness study |
| --- | --- | --- |
| clawbench_tribe | Host trajectory checks | Fast sanity and regression signal |
| pinchbench | In-run grade() score | Good for trace-sensitive evaluation |
| clawbench_official | Pytest/verifier score | Strong artifact-based signal |
| skillsbench | Pytest/verifier score | Strong skill/procedure signal |
| agentbench | Layer-0 structural score only | Useful, but label as partial score |
| zclawbench | Missing host judge | Run for trace collection, exclude from CRR until judged |
| claweval | Missing user-agent/LLM judge | Run for trace collection, exclude from CRR until judged |

## Harness Variants

Start with variants that map directly to clawRecipe breakpoints and can be
implemented in this framework without changing benchmark data.

| Variant | Harness dimension | Target label | Implementation hook |
| --- | --- | --- | --- |
| BL | Current adapter behavior | Baseline | Existing agent_spec/build_agent |
| H1 Plan+Criteria | Control-loop prompt requires plan and success criteria before tools | A/B | `AgentSpec.system_prompt` |
| H2 Preflight | Before each tool call, check path/inputs/format | B | Tool wrapper or system prompt |
| H3 Structured State | Maintain `state.md` with constraints, facts, files, pending items | C | Tool wrapper plus post-tool state update |
| H4 Verifier Pass | After final answer or key artifact, run verifier-style self-check | D | Dedicated verifier agent or post-turn hook |
| H5 Skill Cards | Inject top-k procedural cards selected from task text | E | Skill/card retrieval before run |
| H6 Combo | H1 + H3 + H4 | A/B/C/D | Stacked harness config |
| H7 Negative Control | Commander/executor split without verification | Expected interference | Multi-agent prompt split |
| H8 Sandbox Strict | Docker backend, no network, clean workspace | F and reproducibility | `LocalDockerBackend` + launch options |

Keep each variant frozen before running the evaluation subset.

## Metrics

Report aggregate and diagnostic metrics:

- `score`: benchmark-native 0-100 score.
- `pass`: benchmark thresholded pass/fail.
- `CRR_k`: repaired baseline failures in category `k`.
- `NIR`: baseline passes that the variant breaks.
- `net_gain`: `CRR - NIR`.
- `cost`: input/output tokens, tool calls, wall time.
- `liveness`: completed without timeout or infra error.
- `process_quality`: tool-error recovery rate, repeated-action loops, evidence use.

Use paired analysis. For a fixed task set, compare variant outcomes against BL
on the same task IDs. For binary pass/fail, use McNemar or paired bootstrap.

## Failure Labeling Workflow

1. Run BL on a fixed subset across scored benches.
2. For failed BL trajectories, assign A-F labels using earliest actionable
   breakpoint.
3. Run harness variants on the same task IDs.
4. Compute CRR by label and NIR on BL passes.
5. Inspect disagreements and relabel only if the original label violated the
   written decision rules.

The label is a diagnostic hypothesis, not a causal truth. Cascading failures
should be labeled by the earliest breakpoint whose repair would plausibly
prevent downstream errors.

## Recommended First Sweep

Use a two-stage sweep so the study stays cheap:

1. Smoke subset: 1 task per scored bench, max 3 turns.
2. Diagnostic subset: 10 tasks per scored bench, max 10 turns.
3. Full scored subset: all `clawbench_tribe`, all `pinchbench`, 30 each from
   `clawbench_official`, `skillsbench`, and `agentbench`.
4. Full run after the judge gaps for `zclawbench` and `claweval` are closed.

Current smoke command shape:

```bash
uv run python run_benches.py \
  --bench pinchbench \
  --model gpt-4o-2024-11-20 \
  --sample 1 \
  --max-turns 3 \
  --backend process \
  --run-root .tmp/openclaw_smoke_pinchbench
```

## Implementation Next Steps

1. Add a `--harness` flag to `run_benches.py`.
2. Define harness profiles in a small registry, not in each benchmark suite.
3. Let profiles patch `AgentSpec.system_prompt`, tool wrappers, and optional
   post-run hooks while leaving task inputs unchanged.
4. Extend `summary.json` with `harness`, `scoring_status`, `score_source`,
   token/tool/wall-time metadata, and `failure_label`.
5. Add a label file format:

```json
{
  "task_id": "acad-001-citation-network",
  "baseline_score": 0,
  "failure_label": "A",
  "evidence": "Agent created unrelated intermediate files and never grounded required output schema.",
  "earliest_breakpoint": "Goal requirements not converted into success criteria."
}
```

## Sources

- Local reference: `BrainHao/🤠 Self/paper/agentic/clawRecipe`.
- Harness-Bench: https://arxiv.org/abs/2605.27922
- ToolSandbox: https://machinelearning.apple.com/research/toolsandbox-stateful-conversational-llm-benchmark
- tau-bench: https://huggingface.co/papers/2406.12045
- OSWorld: https://os-world.github.io/
- Agent Harness Engineering survey: https://openreview.net/pdf/f358711a95aaaf61fdeffd4ef3fc60fba9b8da57.pdf
