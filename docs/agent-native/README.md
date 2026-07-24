# Agent Native Context: Simple Agent Lab

This is the single loading map for future agents. Read this file after
`AGENTS.md`, then load only the docs relevant to the task.

Sources:

- Current code snapshot: the working tree at the first-parent commits below.
- Research north star: *Building Reliable Long-Horizon Agents: A Survey*.
- Recent first-parent history: `c9c979b`, `882db33`, `4904e9c`, `c52d57d`.
- Primary anchors: `README.md`, `CONTEXT.md`, `docs/agent-native/`,
  `docs/decisions/`, `runs/README.md`, `tests/README.md`,
  `evals/README.md`, `src/simple_agent_lab/`, and
  `evals/swebench/README.md`.
- Owner interview: not yet completed. Open prompts live in
  `docs/agent-native/owner-questions.md`.

## Documentation Roots

- `AGENTS.md`: primary collaboration contract and first-hop instructions.
- `CONTEXT.md`: root vocabulary and resolved terminology boundaries for
  `Message`, `LLMMessage`, provider adapters, content blocks, and sidecars.
- `docs/agent-native/`: living agent context, loading map, project intent, code
  style, harness workflow, development commands, source-of-truth routing,
  owner questions, stop conditions, and validation map.
- `docs/decisions/`: accepted ADRs. This repo uses `docs/decisions/` instead
  of a parallel ADR tree.
- `docs/reference-architectures/`: local workspace for reference-architecture
  research notes. The directory's contents are gitignored except for the
  README and template; the convention is shared, but individual notes stay
  on the contributor's local disk.

## Repo Map

Simple Agent Lab is the companion research repository for *Building Reliable
Long-Horizon Agents: A Survey*. The canonical Python package under
`src/simple_agent_lab/` is its inspectable reference harness; `evals/` and
`runs/` provide the executable environment, trace, artifact, and benchmark
edges needed for controlled reliability studies. The small teaching core is a
means to make model--harness attribution possible, not the public positioning
by itself.

Treat the root `README.md` as the public paper and artifact map. It must
distinguish available infrastructure from planned six-axis annotations,
reliability-surface analysis, and paper-scale repeated evaluations.

For normal work, this checkout is self-contained. Future agents should treat
the repo root as the only required workspace unless a task explicitly names an
external repository, service, benchmark fixture source, or release automation
system.

The current source-of-truth layers are:

- `src/simple_agent_lab/core.py`: canonical message-first run loop.
- `src/simple_agent_lab/messages.py`: runtime and provider-neutral message
  protocol.
- `src/simple_agent_lab/context_view.py`: model-visible context projection.
- `src/simple_agent_lab/compression/`: context-compression strategies applied
  before each model request (visibility shaping lives behind `ContextPolicy`).
  Summaries cite the transcript indices they folded; the `recall` tool
  retrieves originals and `make_compact_control` adds agent-controlled
  compaction. See ADR recoverable-compression-and-agent-compaction.
- `src/simple_agent_lab/tools/`: shared tool/result values plus concrete tool
  implementations such as bash and the sub-agent `task` tool.
- `src/simple_agent_lab/agents/`: the general agent starter — one
  `AgentSession` runner plus `Toolset`s (`toolsets.py`) and one composable
  `agent_session()` front door (`starter.py`) that turns on bash, read, a
  general-purpose sub-agent, skills, and MCP servers in any combination. Bare-`Agent`
  factories (`make_agent`, `make_bash_agent`, `make_skill_agent`) cover
  resource-free capabilities; `AgentSession`/`mcp_session` are only for MCP's
  live connection. Skills ride on the core `Agent.init_state` hook
  (`pluggable-state-init-hook`),
  not a session.
- `src/simple_agent_lab/llm/`: provider-agnostic model access layer. Building a
  `Provider` from environment variables (env-var names, `.env` loading,
  `provider_from_env`, `FAKE_PROVIDER`, adapter key resolution) is owned by
  `llm/env.py` — the single source of truth (ADR consolidate-provider-env).
- `src/simple_agent_lab/mcp/`: optional Model Context Protocol integration —
  connect to MCP servers and wrap their tools (including multimodal results)
  as `AgentTool`s, behind the `mcp` extra. See ADR mcp-as-tool-source.
- `src/simple_agent_lab/memory/`: optional memory boundary. Filesystem memory
  binds through core lifecycle hooks, injects model-visible recall context at
  session start, and persists Markdown evidence plus distilled handbooks after
  the run. See `docs/agent-native/memory.md`.
- `src/simple_agent_lab/trace/`: the three-layer trace (Event → Span →
  Training) split by concern — `spans.py`/`training.py` (pure event→span/turn
  transforms), `run_trace.py` (record schema), `jsonl.py` (atomic JSONL IO),
  `render.py` (console `print_trace`), `openai_export.py` (OpenAI Chat
  fine-tuning JSONL export), and `live.py` (the incremental live-trace
  session/writer edge).
- `evals/swebench/`, `evals/programbench/`: optional benchmark adapters, outside
  the core runtime (SWE-bench bug-fixing; ProgramBench reverse-engineering).

## Core Mental Model

The repo is intentionally docs-first and harness-first. A good change should
make the next human or agent better able to inspect, modify, verify, and explain
the system.

The canonical runtime direction is:

```text
Agent + Message + State + build_context_view() + run()
```

Do not treat historical architecture options as live runtime copies. Use them
as rationale and teaching context, then verify current behavior against `src/`
and accepted ADRs.

## Global Stop Conditions

Stop and collect more evidence before changing behavior when:

- A change would add a framework-style abstraction to the core runtime. Read
  ADR use-tiny-message-runtime and ADR promote-balanced-runtime-to-src-core first.
- A change modifies `Message`, `LLMMessage`, role names, tool-call or
  tool-result blocks, or provider-boundary conversion. Read ADR use-role-specific-message-protocol,
  ADR unify-message-protocol-on-content-blocks, ADR tool-result-as-content-block, `src/simple_agent_lab/messages.py`, and
  `src/simple_agent_lab/llm/README.md`.
- A change alters context trimming, token estimates, or tool-call/tool-result
  grouping. Read ADR make-context-view-an-explicit-projection and `tests/unit/test_core.py`.
- A change mixes raw trajectories, eval scores, or training labels. Read ADR
  0008 and ADR keep-benchmark-suites-as-eval-adapters.
- A change requires a live model provider or external benchmark dependency.
  Preserve deterministic local smoke paths unless the owner explicitly accepts
  the extra setup cost.
- A change adds or changes a live provider adapter. Keep live-provider smoke
  opt-in and outside required CI unless the owner changes that policy.
- A new external reference architecture starts to drive implementation. Add or
  update a local note under `docs/reference-architectures/` (gitignored
  workspace) and capture the durable commitment in an ADR before changing code.

## Loading Map

| If the task touches... | Read next | Why |
| --- | --- | --- |
| Current status or repo tour | `README.md`, then this loading map | Public map plus task-specific routing. |
| Product direction, audience, or teaching taste | `docs/agent-native/project-intent.md` | Mission, audience, design principles, and current phase. |
| Day-to-day implementation | `docs/agent-native/development.md`, `docs/agent-native/code-style.md`, `runs/README.md` | Commands, quality gate, and style constraints. |
| Harness workflow or docs-first process | `docs/agent-native/harness-engineering.md`, ADR adopt-harness-engineering-workflow, ADR make-testing-and-feedback-first-priority | Feedback signal and repository-as-harness rules. |
| Core runtime shape | ADR use-tiny-message-runtime, ADR make-balanced-runtime-the-lead-core-candidate, ADR promote-balanced-runtime-to-src-core, `src/simple_agent_lab/core.py` | Canonical runtime boundary and stateful run-loop rationale. |
| Message protocol or provider conversion | `CONTEXT.md`, ADR use-role-specific-message-protocol, ADR unify-message-protocol-on-content-blocks, ADR tool-result-as-content-block, `src/simple_agent_lab/messages.py`, `src/simple_agent_lab/llm/README.md` | Runtime-vs-model message boundary and vocabulary. |
| Any environment variable (full catalog) | `docs/agent-native/configuration.md` | Single discoverability table of every env var grouped by owning layer, plus the rule for where to declare a new one. Start here when you can't find a config knob. |
| Launching a run from one file (`--profile`) | ADR run-profile-file, `docs/agent-native/configuration.md`, `src/simple_agent_lab/evals/profile.py`, `runs/profiles/*.example.json` (one per benchmark), `tests/unit/test_run_profile.py` | JSON run-profile bundles the two launch surfaces (`env` fill-gaps + `run` CLI defaults); a thin bundle, not a second schema. |
| One entry point over every bench (dashboard) | `docs/agent-native/configuration.md`, `runs/run_bench.py`, `runs/bench-manifest.example.json`, `tests/unit/test_run_bench.py` | `run_bench.py list/setup/<bench>/all` — JSON-friendly dispatcher; `setup` probes the environment (+ oracle smoke); `all` runs a manifest of benches as isolated subprocesses. Per-bench `runs/run_<bench>.py` stay runnable and are imported by it. |
| Provider config / env vars (`OPENAI_*`, `.env`, `Provider`) | ADR consolidate-provider-env, `src/simple_agent_lab/llm/env.py`, `tests/unit/test_evals_framework.py`, `tests/unit/test_onemillion_container.py` | Single source of truth for env-var names, `load_dotenv`, `provider_from_env`, `FAKE_PROVIDER`, and adapter key resolution. Build providers from env here; don't re-declare the names. |
| Named model aliases (`strong` / `fast`) | ADR model-alias-registry, `src/simple_agent_lab/llm/registry.py`, `tests/unit/test_model_registry.py`, `scripts/run_model_registry_demo.py` | `ModelRegistry.from_env` maps aliases to providers via `<ALIAS>_*` env (fallback `OPENAI_*`); ask for models by role name. Opt-in; single-model setups collapse onto `OPENAI_*`. |
| Context visibility or budgeting | ADR make-context-view-an-explicit-projection, `src/simple_agent_lab/context_view.py`, `tests/unit/test_core.py`, `tests/unit/test_token_usage.py` | Projection behavior and token-estimate constraints. |
| Context compression (strategies, recall, `compact`, metrics) | ADR recoverable-compression-and-agent-compaction, `src/simple_agent_lab/compression/`, `src/simple_agent_lab/tools/recall.py`, `tests/unit/test_compression_control.py`, `tests/unit/test_compression_effectiveness.py`, `tests/unit/test_core.py` | Summaries cite folded transcript indices; `recall` retrieves originals; `make_compact_control` pairs a `compact` tool with the strategy applying it at the next turn start. Each fold names its strategy on `ContextCompressionEvent.strategy` (a `TieredStrategy` stage rides through); `summarize_compression(events)` measures effectiveness and attributes folds per strategy. |
| Tool execution or bash demo | `src/simple_agent_lab/tools/`, `src/simple_agent_lab/agents/starter.py` (`agent_session` / `make_bash_agent`), `tests/unit/test_bash_agent.py`, `tests/unit/test_agent_starter.py` | Tool result semantics and deterministic demo checks. |
| Multi-agent delegation (`task` tool) | `src/simple_agent_lab/tools/task.py`, `src/simple_agent_lab/agents/starter.py` (`agent_session(general_purpose=True)` parent + general-purpose worker), `tests/unit/test_bash_task_agent.py`, `src/simple_agent_lab/core.py` docstring | Sub-agent delegation shape: a parent picks one worker via `subagent_type` and gets its final message back as the tool result. |
| MCP tools (incl. multimodal) | ADR mcp-as-tool-source, `src/simple_agent_lab/mcp/README.md`, `tests/unit/test_mcp.py`, `scripts/run_mcp_agent_demo.py` | MCP servers wrapped as `AgentTool`s at the tool boundary; image results map straight to `ImageBlock`. Optional `mcp` extra. |
| Agent skills (discover/advertise/load `SKILL.md`) | ADR add-agent-skills, ADR pluggable-state-init-hook, `src/simple_agent_lab/skills/`, `src/simple_agent_lab/agents/starter.py` (`make_skill_agent`), `src/simple_agent_lab/tools/read.py`, `tests/unit/test_skills.py`, `tests/unit/test_read_tool.py` | Read-based skills: a prompt menu plus model-driven `read`/`bash`; on by default with `/no-skills`. Installed via the core `Agent.init_state` hook so a bare `agent.run` is skills-aware (`make_skill_agent`). Benchmark `bash_skills` flavor folds the menu into the system prompt. |
| Filesystem memory / distillation | `docs/agent-native/memory.md`, ADRs serialize-filesystem-memory-consolidation and bound-filesystem-memory-growth, `src/simple_agent_lab/memory/`, `src/simple_agent_lab/hooks.py`, `src/simple_agent_lab/evals/in_container.py`, `tests/unit/test_memory.py` | Memory stays outside core: recall and finish are lifecycle hooks; filesystem consolidation is one root-scoped read-distill-commit critical section with bounded retained evidence; evals opt in through `SAL_MEMORY_*` env and optional suite `memory_artifacts`. |
| Trace printing or OpenAI Chat JSONL export | ADR extra-channel-and-two-layer-trace, ADR three-layer-trace-event-span-training, `src/simple_agent_lab/trace/render.py`, `src/simple_agent_lab/trace/openai_export.py`, `tests/unit/test_openai_training.py` | Trace rendering and provider-shaped transcript export. |
| Trajectories, spans, or training data | ADR collect-training-trajectories-across-design-versions, ADR keep-benchmark-suites-as-eval-adapters, ADR three-layer-trace-event-span-training, `src/simple_agent_lab/trace/` (`spans.py`, `training.py`, `run_trace.py`), `evals/README.md`, `evals/swebench/README.md` | Three-layer trace: Event → Span → Training. |
| Docker incremental trace / host viewer | `docs/agent-native/docker-live-trace.md`, `src/simple_agent_lab/trace/live.py` (`LiveTraceSession`), `scripts/run_live_trace_demo.py` | Bind-mount contract and reusable live export API. |
| Containerized eval framework / suites | ADR generic-containerized-eval-framework, `evals/README.md`, `src/simple_agent_lab/evals/` | Suite x ContainerBackend x ArtifactStore seams; `run_suite_instance` / `run_dataset` entry points. |
| SWE-bench Pro repo-chain experiments | `docs/agent-native/swebench-pro-repo-chain-experiment.md`, `runs/swebench/run_swebench_pro_repo_chains.py`, `src/simple_agent_lab/evals/chain.py`, `src/simple_agent_lab/evals/suites/swebench/container.py`, `evals/swebench/pro_repo_chain.py` | Long repo chains run the selected `--agent-flavor` inside each Pro instance container; the single host runner stages chain config/state artifacts and varies agent flavor, `--task-tool`, and `summarize`/`none` compression. No-compression chains stay under the context window with boundary handoff (default on; `--handoff`/`--no-handoff`, `--context-window-tokens`). |
| Scoring: how a suite scores / parity | ADR collapse-scorer-seam-into-run-primitive (amends ADR scorer-seam-and-scoring-topology), `src/simple_agent_lab/evals/in_container.py` (`evaluate` hook), `evals/swebench/evaluate_predictions.py` (`reuse_eval_row`, parity) | No scorer seam: in-env scoring is the `evaluate` hook (gated on `eval_inputs`); scoring elsewhere is a follow-up run; official harness is a standalone CLI; `result.json` decoupling; official-parity gate. |
| Integrating a new Docker eval suite (step-by-step) | `docs/agent-native/integrating-a-docker-eval-suite.md`, ADR generic-containerized-eval-framework, ADR collapse-scorer-seam-into-run-primitive, `evals/swebench/suite.py`, `src/simple_agent_lab/evals/suites/swebench/container.py` | Two halves + registration; the developer/agent how-to with a checklist. |
| ProgramBench (reverse-engineering) suite | ADR `programbench-reverse-engineering-adapter`, `evals/programbench/README.md`, `evals/programbench/suite.py`, `src/simple_agent_lab/evals/suites/programbench/container.py`, `tests/unit/test_programbench_suite.py` | Peer of SWE-bench with two twists: workspace-as-product (base64 tar in `result.json`) and per-command network isolation (`unshare --net` via the bash `exec_prefix`); scored by the official ProgramBench evaluator. |
| Harbor eval integration | ADR `harbor-as-eval-harness`, `evals/harbor/README.md`, `runs/_benches/harbor.py`, `src/simple_agent_lab/evals/harbor/`, `tests/unit/test_harbor_bench.py`, `tests/unit/test_harbor_runner.py`, `tests/unit/test_harbor_agent.py` | One bench entry over Harbor datasets. Harbor owns dataset/task/environment/verifier/result aggregation; SAL supplies a Harbor installed agent whose runner and tools execute inside the Harbor task container. |
| Multi-machine eval deployment / workers / k8s | `docs/agent-native/multi-machine-deployment.md`, ADR generic-containerized-eval-framework | Worker setup, image distribution, online/offline, store-by-topology; runtime injection. |
| External architecture borrowing | `docs/reference-architectures/README.md` (local notes workspace, gitignored) plus your own reference note | Capture rationale locally; record durable commitments in an ADR. |
| Agent-native doc maintenance | This loading map, `docs/agent-native/operating-rules.md` | Canonical routing and stop conditions. |

## Maintenance Workflow

1. Start from the loading map above.
2. Update the canonical topic doc first.
3. Update this loading map if doc roles, freshness, loading triggers, or
   first-read choices change.
4. Move unresolved owner or external-system facts to `owner-questions.md`.
5. Create or update an ADR under `docs/decisions/` only for hard-to-reverse
   decisions with real tradeoffs.

## ADR Index

Use `docs/decisions/README.md` as the canonical ADR index. The loading map
above names only the ADRs most relevant to each task trigger.

## Open Questions

See `docs/agent-native/owner-questions.md`.
