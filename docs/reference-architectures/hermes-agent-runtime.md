# Reference Architecture: Hermes Agent Runtime

## Source

- Project docs: [Architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture), [Agent Loop Internals](https://hermes-agent.nousresearch.com/docs/developer-guide/agent-loop), [Tools Runtime](https://hermes-agent.nousresearch.com/docs/developer-guide/tools-runtime), [Subagent Delegation](https://hermes-agent.nousresearch.com/docs/user-guide/features/delegation/), [Persistent Memory](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory), [Session Storage](https://hermes-agent.nousresearch.com/docs/developer-guide/session-storage), [Memory Providers](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory-providers), [Prompt Assembly](https://hermes-agent.nousresearch.com/docs/developer-guide/prompt-assembly), [Context Files](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files), [Context References](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-references), [Context Compression and Caching](https://hermes-agent.nousresearch.com/docs/developer-guide/context-compression-and-caching), [Skills System](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)
- Related issue: [Acceptance Criteria & Independent Judge for Sub-agent Delegation](https://github.com/NousResearch/hermes-agent/issues/356)
- Date reviewed: 2026-04-26
- Reviewer note: Based on static architecture notes provided to this repository. The project was not run locally during this review.

## Summary

Hermes Agent is useful as a reference because it centers the system around one `AIAgent` runtime that multiple entrypoints reuse. CLI, gateway, ACP, batch runner, API server, and Python library flows all converge on the same runtime loop.

The main lesson for Simple Agent Lab is that a capable agent system should be composed from replaceable layers:

```text
User / Gateway / Cron / ACP
        -> AIAgent.run_conversation()
        -> PromptBuilder + Memory + Skills + Context Files
        -> Provider Adapter
        -> LLM response
        -> Tool dispatch loop, if tool calls exist
        -> Session persistence + memory flush + return
```

Hermes is not just an agent plus tools. It separates tool capability, toolsets, availability checks, memory, skills, context references, context compression, subagents, provider adapters, and plugin-backed extensions.

## Core Ideas

- Use one central agent runtime across many platform entrypoints.
- Keep provider-specific message formats at adapter boundaries.
- Treat tools as registered capabilities with schema, handler, toolset, availability check, and metadata.
- Use toolsets as capability bundles for sessions, platforms, and subagents.
- Hide unavailable tools from the model instead of exposing broken schemas.
- Treat skills as procedural memory: reusable workflow documents, not executable plugins.
- Treat memory as curated long-term facts, distinct from session history, skills, and context files.
- Separate stable prompt snapshots from ephemeral per-call overlays.
- Use subagents as isolated execution units with explicit context and restricted capability policies.
- Keep context management as a first-class runtime layer.

## Runtime Boundaries

Hermes has a central `AIAgent` that handles:

- System prompt and tool schema construction.
- Provider and API mode resolution.
- Interruptible model calls.
- Sequential or concurrent tool execution.
- OpenAI-style internal message history.
- Compression, retry, fallback, and iteration budget.
- Session persistence.
- Memory flushing before context is lost.

The architecture has these broad layers:

| Layer | Responsibility |
| --- | --- |
| Platform entrypoints | CLI, gateway, ACP, batch runner, API server, library |
| Agent runtime | Conversation loop, provider call, tool loop, session persistence |
| Provider adapters | OpenAI-compatible, Anthropic-style, Codex responses, fallback |
| Prompt assembly | Stable prompt snapshot and per-call overlays |
| Tool runtime | Registry, toolsets, dispatch, availability checks, runtime tools |
| Memory | Curated files, session database, external providers |
| Skills | Progressive disclosure for procedural knowledge |
| Context engine | Token measurement, compression, caching, context references |
| Delegation | Child agents with isolated context and restricted tools |

## Agent Loop

Hermes supports multiple provider API modes, but keeps an internal OpenAI-style message model:

```json
{"role": "system", "content": "..."}
{"role": "user", "content": "..."}
{"role": "assistant", "content": "...", "tool_calls": []}
{"role": "tool", "tool_call_id": "...", "content": "..."}
```

This is worth borrowing because provider adapters should translate at the edge. Provider-specific wire shapes should not leak into the agent runtime's durable state.

A simplified turn flow:

1. Append the user message.
2. Build or reuse system prompt state.
3. Run preflight compression when needed.
4. Convert internal messages to the selected provider format.
5. Add ephemeral prompt layers for this API call.
6. Call the model.
7. Parse text or tool calls.
8. Dispatch tool calls and append structured tool results.
9. Continue until final response or budget stop.
10. Persist session state and flush memory.

## Tool Model

Hermes tools self-register through a singleton registry. A tool registration includes:

```text
name
toolset
schema
handler
check_fn
requires_env
is_async
description
metadata
```

Tool modules are discovered automatically. The discovery process scans tool files and imports only modules that register tools. This removes the need for a central import list while still keeping tool registration explicit.

For Simple Agent Lab, a future typed tool spec could look like:

```text
ToolSpec
  - name
  - description
  - schema
  - toolset
  - handler
  - availability
  - risk_level
  - side_effects
```

The `side_effects` field is an important improvement to consider. Tools should be able to declare effects such as file reads, file writes, network calls, shell execution, browser control, secrets access, or message sending.

### Toolsets

Hermes treats toolsets as named capability bundles. This is a useful distinction:

```text
tool = one callable capability
toolset = a capability bundle exposed to a session, platform, or agent profile
```

Example shape:

```yaml
toolsets:
  research:
    - web_search
    - web_extract
    - read_file

  coding:
    - read_file
    - write_file
    - patch
    - terminal

  safe_review:
    - read_file
    - search_files
```

Toolsets give the runtime a clean policy boundary for primary agents, subagents, gateways, and automation jobs.

### Availability Checks

Hermes tools can define `check_fn` availability checks. If a tool is unavailable because credentials, binaries, services, or environment variables are missing, the tool schema is not exposed to the model.

This is worth borrowing directly:

```text
Do not expose unavailable tools to the model.
```

The model should see only capabilities that can actually run.

### Dispatch Flow

The dispatch path can be summarized as:

```text
model tool_call
  -> agent loop
  -> handle_function_call(name, args, task_id, user_task)
  -> runtime-special tool handling, if needed
  -> plugin pre-hook
  -> registry.dispatch()
  -> sync or async handler
  -> structured JSON result
  -> plugin post-hook
```

Hermes wraps tool errors into structured results. This matters because tool failures are part of the model's next context and should be readable, recoverable, and traceable.

### Runtime Tools

Some Hermes tools are registered like normal tools but executed inside the agent loop because they need agent-local state:

| Tool | Runtime state needed |
| --- | --- |
| `todo` | Agent-local task state |
| `memory` | Persistent memory writes and capacity checks |
| `session_search` | Session database access |
| `delegate_task` | Child agent creation |

For Simple Agent Lab, this suggests distinguishing:

```text
ExternalTool    -> ordinary registry dispatch
RuntimeTool     -> needs agent runtime state
PlatformTool    -> needs gateway callbacks
DangerousTool   -> needs approval or sandbox
```

### Terminal Safety

Hermes treats terminal execution as a high-risk capability. It supports different backends and has approval flows for dangerous command patterns.

For Simple Agent Lab, shell tools should be designed with:

- Risk classification.
- Approval gates.
- Optional sandbox backends.
- Auditable command history.
- Deny patterns for destructive operations.

## Subagent Model

Hermes subagents are isolated child `AIAgent` instances. A child agent gets a fresh conversation, restricted toolsets, its own terminal session, and only explicit task context. The parent receives the final summary, not the whole child transcript.

This is the key pattern:

```text
Subagent = AgentRuntime + FreshContext + CapabilityPolicy + Budget + IsolatedWorkspace
```

A future task object could look like:

```text
SubAgentTask
  - goal
  - context
  - toolsets
  - max_iterations
  - model
  - role: leaf | orchestrator
```

### Explicit Context Only

The child agent does not inherit parent history. The parent must pass clear paths, errors, constraints, and task context. This avoids token pollution and makes delegation easier to inspect.

### Batch Delegation

Hermes supports parallel child tasks with a concurrency limit. Useful details to borrow:

- Maximum concurrent children.
- Results returned in input order.
- Interrupt propagation from parent to children.
- Per-child progress events.
- Per-child terminal and session isolation.

### Restricted Capability Policy

Child agents should not automatically inherit all parent tools. Hermes blocks or restricts sensitive capabilities such as asking the user, writing memory, sending messages, and recursive delegation unless explicitly configured.

For Simple Agent Lab, default subagents should probably be leaf agents. Recursive orchestration should require an explicit max-depth policy.

### Acceptance Criteria and Judge

The related Hermes issue about acceptance criteria and independent judging is important. Simple Agent Lab should consider making this part of the subagent design from the beginning.

Potential result shape:

```text
SubAgentResult
  - status: success | partial | failed
  - summary
  - artifacts
  - files_changed
  - commands_run
  - judge_verdict: pass | fail | not_run
  - judge_feedback
```

Subagent completion should be evaluated against acceptance criteria, not just accepted because the subagent produced a confident summary.

## State and Memory

Hermes separates memory from session history and procedural skills.

### Curated Memory

Built-in memory is represented by small files:

| File | Purpose |
| --- | --- |
| `MEMORY.md` | Agent notes such as environment facts, project conventions, tool issues, completed work |
| `USER.md` | User profile, preferences, communication style, expectations |

These files are injected as frozen snapshots at session start. Writes through the memory tool do not mutate the current prompt immediately; they affect future sessions or later snapshots.

This is useful because it keeps memory small, curated, and stable during a session.

### Session History

Hermes stores session metadata, message history, tool calls, reasoning, and model config in SQLite, with full-text search support.

This suggests a clear distinction:

| Kind | Purpose |
| --- | --- |
| Curated memory | Stable high-value facts and preferences |
| Episodic history | What happened in previous sessions |
| Procedural memory | Reusable workflows and practices, stored as skills |
| Context files | Project-specific rules and conventions |

### Memory Providers

Hermes supports external memory providers as plugins. A provider can inject context, prefetch memories, sync turns, extract memories at session end, mirror built-in writes, and expose provider-specific tools.

For Simple Agent Lab, memory should eventually be pluggable, but the first design should keep a strict boundary:

```text
MemoryProvider
  - snapshot(user_id, budget)
  - search(query, budget)
  - propose_writes(turn)
  - write(item)
```

Useful metadata for future memory items:

- Confidence.
- Source session.
- Created and updated timestamps.
- Expiry or stale-after policy.
- Tags.
- Conflict group.
- Provenance.
- Trust level.

## Context Management

Hermes treats prompt assembly and compression as a core architecture concern.

### Stable Snapshot and Ephemeral Overlays

Prompt assembly separates cached system prompt state from additions that are injected only for one API call.

A stable prompt snapshot may include:

1. Identity.
2. Tool-aware behavior guidance.
3. Static platform or policy blocks.
4. Optional system message.
5. Frozen `MEMORY.md`.
6. Frozen `USER.md`.
7. Skills index.
8. Context files.
9. Timestamp and session ID.
10. Platform hint.

The important design rule:

```text
Stable layers should be cacheable.
Ephemeral overlays should not pollute durable prompt state.
```

Potential Simple Agent Lab shape:

```text
PromptAssembler
  - build_snapshot(session_start_context)
  - build_call_messages(snapshot, history, overlays)
```

### Context Files

Hermes loads project context files with explicit priority. Supported files include `.hermes.md`, `HERMES.md`, `AGENTS.md`, `CLAUDE.md`, `SOUL.md`, Cursor rule files, and related compatibility files.

It also supports progressive discovery of subdirectory instructions. Rules near files discovered during reading or search can be injected later instead of loading every possible rule at startup.

For Simple Agent Lab, context chunks should eventually carry provenance:

```json
{
  "kind": "project_context",
  "source": "/repo/backend/AGENTS.md",
  "scope": "/repo/backend",
  "priority": 80,
  "injected_at": "turn_12"
}
```

### Context References

Hermes supports explicit context references such as files, folders, diffs, staged changes, git references, and URLs.

This suggests two separate mechanisms:

```text
@file      -> user-explicit context
read_file  -> agent-discovered context
```

Both should have provenance and token budgeting.

### Context Engine and Compression

Hermes abstracts context management behind a `ContextEngine`. The default compressor can be replaced by plugins.

Compression protects important regions:

- Stable system and first exchange.
- Recent tail messages.
- Tool call and tool result groups.
- Structured summaries of the middle history.

The summary format preserves:

```text
Goal
Constraints
Progress
Key Decisions
Relevant Files
Next Steps
Critical Context
```

For Simple Agent Lab, a future interface could be:

```text
ContextEngine
  - measure(messages)
  - should_compact(state)
  - compact(messages, policy)
  - retrieve(query, budget)
```

Budgets should be explicit by source:

```yaml
context_budget:
  reserved:
    system: 8000
    tools: 12000
    memory: 2000
    skills_index: 3000
    project_context: 10000
    recent_history: 50000
    retrieved_history: 8000
    scratch: 5000
```

## Skill Model

Hermes skills are on-demand knowledge documents. They use progressive disclosure:

```text
Level 0: skills_list()          -> name, description, category
Level 1: skill_view(name)       -> full skill content and metadata
Level 2: skill_view(name, path) -> specific reference file
```

This creates a clear separation:

| Concept | Purpose |
| --- | --- |
| Context file | Project or directory rules |
| Memory | Long-term facts and preferences |
| Skill | Reusable workflow, procedure, best practice, manual |
| Tool | Executable capability |
| Plugin | Runtime, backend, or tool extension |

Skills should teach the agent how to do work. Tools should execute actions.

### Skill Format

A useful future `SKILL.md` shape:

```markdown
---
name: github-pr-review
description: Review a GitHub PR for security, tests, and maintainability.
version: 1.0.0
category: software-engineering
requires_toolsets: [web, file]
risk_level: low
---

# GitHub PR Review

## When to Use

## Inputs

## Procedure

## Tools

## Verification

## Failure Modes

## Output Contract
```

The output contract is especially important for making skills testable and reusable.

### Conditional Activation

Hermes can hide or show skills based on platform and tool availability. Simple Agent Lab should consider activation metadata such as:

```yaml
activation:
  platforms: [linux, macos]
  requires_tools: [terminal]
  fallback_for_tools: [web_search]
  min_model_capability: tool_calling
  risk_level_allowed: medium
```

### Agent-Managed Skills

Hermes lets agents create, patch, edit, and delete skills. This is powerful because it turns repeated procedures into external procedural memory.

For Simple Agent Lab, agent-created skills should require governance:

- Pending review before use by default.
- Provenance from source session and task.
- Version and changelog.
- Usage success and failure stats.
- Automatic downgrade for low-success skills.
- Quarantine for risky or suspicious skills.
- Periodic pruning for duplicates and stale procedures.

## Concept Map

```text
Tools            = what the runtime can do
Toolsets         = what this session or agent may do
Memory           = stable facts worth remembering
Session DB       = what happened before
Context files    = what this project or directory requires
Skills           = how to perform recurring workflows
Context engine   = how to fit important context into the window
Subagents        = how to isolate or parallelize complex work
```

## What We Might Borrow

- One core runtime used by CLI, API, gateway, and automation entrypoints.
- Internal provider-neutral message model.
- Tool self-registration with schema, handler, availability check, and metadata.
- Toolsets as first-class capability bundles.
- Hiding unavailable tools from the model.
- Separate runtime tools for memory, session search, todo state, and delegation.
- Frozen memory snapshots for stable prompt behavior.
- Clear separation between curated memory, session history, skills, and context files.
- Prompt snapshots separated from ephemeral overlays.
- Project context files with priority, scope, and provenance.
- Explicit context references such as `@file`, `@diff`, and `@url`.
- Replaceable context engine for compression and retrieval.
- Progressive skill disclosure.
- Isolated subagents with restricted toolsets and final-summary-only return.
- Acceptance criteria and judge verification for delegated tasks.

## What We Should Avoid

- Do not put too much runtime behavior in one giant agent loop file.
- Do not let provider wire formats leak into stored session state.
- Do not expose unavailable or unauthorized tools to the model.
- Do not treat memory, session history, skills, and context files as one undifferentiated store.
- Do not allow agent-created skills to become trusted without review, provenance, or eval feedback.
- Do not let subagents inherit parent context by default.
- Do not allow unrestricted recursive delegation.
- Do not build terminal execution without approval, risk classification, or sandbox strategy.
- Do not let gateway/platform concerns pollute the core runtime.

## Suggested Future Shape for Simple Agent Lab

This is not an accepted design yet. It is a candidate direction to compare against other reference architectures.

```text
agent_core/
  runtime/
  providers/
  prompts/
  context/
  tools/
  memory/
  skills/
  delegation/
  storage/
  plugins/

gateways/
  cli/
  api/

skills/
  software-engineering/
  research/
  devops/

tests/
  golden_prompts/
  tool_contracts/
  memory/
  context/
  delegation/
  security/

docs/
  architecture.md
  tool-authoring.md
  skill-authoring.md
  memory-model.md
```

Potential future interfaces:

```text
AgentRuntime
ProviderAdapter
PromptAssembler
ToolRegistry
ToolSpec
CapabilityPolicy
MemoryProvider
SkillStore
ContextEngine
SubagentExecutor
SessionStore
PluginManager
```

Potential staged implementation order:

1. Minimal runtime: provider adapter, internal message model, tool registry, safe read/search tools, basic terminal with approval, SQLite session store, prompt assembler.
2. Context system: stable prompt snapshots, ephemeral overlays, project context discovery, context references, token budgeting, basic compression.
3. Memory: curated memory store, user profile store, memory tool, session search, provenance.
4. Skills: `SKILL.md` schema, skills index, skill view, managed skill lifecycle, security scan.
5. Subagents: isolated child runtime, explicit task context, restricted policy, batch execution, timeout, structured summary, acceptance judge.
6. Plugins and gateways: hook manager, MCP bridge, CLI gateway, optional external messaging gateway.

## Open Questions for Simple Agent Lab

- Should the first runtime be Python, TypeScript, or a language-neutral spec with one reference implementation?
- Should toolsets and permission policy exist in v0, or should they be introduced immediately after basic tools?
- Should `AGENTS.md` be treated as project context in the runtime from the first implementation?
- Should memory start as tiny curated Markdown files, SQLite records, or both?
- Should subagent judging be part of the first delegation implementation or a later eval layer?
- How much plugin surface is useful before the first stable core runtime exists?

