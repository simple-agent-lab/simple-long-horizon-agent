# Agent Native Context: Simple Agent Lab

This is the single loading map for future agents. Read this file after
`AGENTS.md`, then load only the docs relevant to the task.

Sources:

- Current code snapshot: the working tree at the first-parent commits below.
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

Simple Agent Lab is a small Python package for teaching and experimenting with
agent runtimes. The canonical package lives under `src/simple_agent_lab/`.

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
- `src/simple_agent_lab/memory/`: small memory boundary plus
  `FilesystemMemory` (scoped Markdown directory + run-end evidence writes).
  Keep the starter filesystem mechanism in one implementation file:
  `filesystem.py`.
- `src/simple_agent_lab/tools/`: shared tool/result values plus concrete tool
  implementations such as bash and the sub-agent `task` tool.
- `src/simple_agent_lab/agents/`: preset agents built on the core layers
  (`bash/` single bash-use agent, `bash_task/` parent that delegates to a bash
  worker via the `task` tool).
- `src/simple_agent_lab/llm/`: provider-agnostic model access layer.
- `src/simple_agent_lab/mcp/`: optional Model Context Protocol integration —
  connect to MCP servers and wrap their tools (including multimodal results)
  as `AgentTool`s, behind the `mcp` extra. See ADR mcp-as-tool-source.
- `src/simple_agent_lab/trace/`: the three-layer trace (Event → Span →
  Training) split by concern — `spans.py`/`training.py` (pure event→span/turn
  transforms), `run_trace.py` (record schema), `jsonl.py` (atomic JSONL IO),
  `render.py` (console `print_trace`), `openai_export.py` (OpenAI Chat
  fine-tuning JSONL export), and `live.py` (the incremental live-trace
  session/writer edge).
- `evals/swebench/`: optional benchmark adapter, outside the core runtime.

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
| Context visibility or budgeting | ADR make-context-view-an-explicit-projection, `src/simple_agent_lab/context_view.py`, `tests/unit/test_core.py`, `tests/unit/test_token_usage.py` | Projection behavior and token-estimate constraints. |
| Memory, recall, or persistent context | `docs/agent-native/memory.md`, `src/simple_agent_lab/memory/`, ADR use-tiny-message-runtime, ADR make-context-view-an-explicit-projection, ADR three-layer-trace-event-span-training | Memory API and implementations, plus the runtime, context-view, and trace boundaries they must preserve. |
| Tool execution or bash demo | `src/simple_agent_lab/tools/`, `src/simple_agent_lab/agents/bash/` (preset agent), `tests/unit/test_bash_agent.py` | Tool result semantics and deterministic demo checks. |
| Multi-agent delegation (`task` tool) | `src/simple_agent_lab/tools/task.py`, `src/simple_agent_lab/agents/bash_task/` (parent + explorer worker), `tests/unit/test_bash_task_agent.py`, `src/simple_agent_lab/core.py` docstring | Sub-agent delegation shape: a parent picks one worker via `subagent_type` and gets its final message back as the tool result. |
| MCP tools (incl. multimodal) | ADR mcp-as-tool-source, `src/simple_agent_lab/mcp/README.md`, `tests/unit/test_mcp.py`, `scripts/run_mcp_agent_demo.py` | MCP servers wrapped as `AgentTool`s at the tool boundary; image results map straight to `ImageBlock`. Optional `mcp` extra. |
| Agent skills (discover/advertise/load `SKILL.md`) | ADR add-agent-skills, `src/simple_agent_lab/skills/`, `src/simple_agent_lab/tools/read.py`, `tests/unit/test_skills.py`, `tests/unit/test_read_tool.py` | Read-based skills: a prompt menu plus model-driven `read`/`bash`; on by default with `/no-skills`. Benchmark `bash_skills` flavor folds the menu into the system prompt. |
| Trace printing or OpenAI Chat JSONL export | ADR extra-channel-and-two-layer-trace, ADR three-layer-trace-event-span-training, `src/simple_agent_lab/trace/render.py`, `src/simple_agent_lab/trace/openai_export.py`, `tests/unit/test_openai_training.py` | Trace rendering and provider-shaped transcript export. |
| Trajectories, spans, or training data | ADR collect-training-trajectories-across-design-versions, ADR keep-benchmark-suites-as-eval-adapters, ADR three-layer-trace-event-span-training, `src/simple_agent_lab/trace/` (`spans.py`, `training.py`, `run_trace.py`), `evals/README.md`, `evals/swebench/README.md` | Three-layer trace: Event → Span → Training. |
| Docker incremental trace / host viewer | `docs/agent-native/docker-live-trace.md`, `src/simple_agent_lab/trace/live.py` (`LiveTraceSession`), `scripts/run_live_trace_demo.py` | Bind-mount contract and reusable live export API. |
| Containerized eval framework / suites | ADR generic-containerized-eval-framework, `evals/README.md`, `src/simple_agent_lab/evals/` | Suite x ContainerBackend x ArtifactStore seams; `run_suite_instance` / `run_dataset` entry points. |
| Scoring: how a suite scores / parity | ADR collapse-scorer-seam-into-run-primitive (amends ADR scorer-seam-and-scoring-topology), `src/simple_agent_lab/evals/in_container.py` (`evaluate` hook), `evals/swebench/evaluate_predictions.py` (`reuse_eval_row`, parity) | No scorer seam: in-env scoring is the `evaluate` hook (gated on `eval_inputs`); scoring elsewhere is a follow-up run; official harness is a standalone CLI; `result.json` decoupling; official-parity gate. |
| Integrating a new Docker eval suite (step-by-step) | `docs/agent-native/integrating-a-docker-eval-suite.md`, ADR generic-containerized-eval-framework, ADR collapse-scorer-seam-into-run-primitive, `evals/swebench/suite.py`, `src/simple_agent_lab/evals/suites/swebench/container.py` | Two halves + registration; the developer/agent how-to with a checklist. |
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
