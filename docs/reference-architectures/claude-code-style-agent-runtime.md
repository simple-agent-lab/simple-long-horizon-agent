# Reference Architecture: Claude Code-Style Single Loop Runtime

## Source

- Local notes: `/Users/bytedance/Downloads/package 3/AGENT_ARCHITECTURE_DESIGN_NOTES.md`
- Date reviewed: 2026-04-26
- Reviewer note: Based on architecture notes from a sourcemap-recovered Claude Code-like source shape. Some modules were incomplete in the source notes and are treated as inferred from call sites, types, and comments rather than complete implementation evidence.

## Summary

This reference is useful because it frames agent architecture around one reusable execution kernel:

```text
messages
  -> context pipeline
  -> model streaming
  -> collect assistant tool_use blocks
  -> execute tools
  -> append tool_result as user messages
  -> continue same loop until no tool_use
```

The main lesson for Simple Agent Lab is that subagents, skills, memory extraction, session summaries, forked agents, and multiagent teammates can all reuse the same agent loop. They differ by runtime context, toolset, permissions, lifecycle, and transcript handling, not by having separate execution engines.

Core idea:

```text
Agent loop = the only execution kernel.
Everything else = runtime context + definitions + tools + memory + lifecycle policy.
```

## Core Ideas

- Use a single agent loop for primary agents, subagents, forked agents, background agents, and maintenance tasks.
- Treat runtime context as an explicit capability bus, not global state.
- Treat tools as a full protocol with schema, permissions, safety metadata, concurrency metadata, rendering, and result budgeting.
- Treat agent definitions as declarative runtime profiles with tools, skills, memory, model, permission mode, and isolation policy.
- Keep subagents isolated by default; share context only when explicitly requested.
- Use forked agents for maintenance tasks such as memory extraction, summarization, compaction, and prompt suggestions.
- Model multiagent teammates as long-lived actors with mailboxes, not as recursive subagent calls.
- Treat memory as layered: project rules, file-based durable memory, relevant retrieval, session memory, and background extraction.
- Treat context management as a pipeline with low-loss transforms before lossy summaries.
- Treat skills as prompt packages that can be inlined or executed in a forked context.
- Use tool discovery when the tool surface becomes too large for the initial prompt.

## Runtime Boundaries

The notes identify these major boundaries:

| Concept | Source-shape location | Responsibility |
| --- | --- | --- |
| Agent loop | `src/query.ts` | Model calls, tool calls, compaction, error recovery |
| Runtime context | `ToolUseContext` | Capability bus for tools, permissions, memory, skills, state, abort, MCP |
| Tool protocol | `Tool` / `buildTool()` | Schema, execution, permission, safety, UI mapping, result mapping |
| Tool execution | `services/tools/*` | Scheduling, validation, permission, progress, result handling |
| Agent definition | `AgentTool/loadAgentsDir.ts` | Agent profiles, prompts, tools, skills, memory, model, hooks |
| Agent runner | `AgentTool/runAgent.ts` | Build isolated context and call the shared loop |
| Forked agent | `forkedAgent.ts` | Side work with prompt-cache-aware context |
| Multiagent teammate | `spawnMultiAgent.ts`, `swarm/*` | Long-lived actor, mailbox, team state, permission bridge |
| Memory | `memdir/*`, `SessionMemory/*`, `extractMemories/*` | Long-term files, session memory, retrieval, extraction |
| Context management | `compact/*`, `toolResultStorage.ts` | Budgeting, microcompaction, autocompaction, recovery |
| Skill | `skills/loadSkillsDir.ts`, `SkillTool/*` | Frontmatter-based prompt package and optional fork |
| Tool discovery | `ToolSearchTool/*`, `toolSearch.ts` | Deferred tool loading and search |

## Agent Loop

The loop is the central abstraction. Each turn:

1. Slices history after the current compact boundary.
2. Applies context transforms such as result budgeting, history snipping, microcompaction, collapse, and autocompaction.
3. Calls the model streaming API.
4. Collects `tool_use` blocks.
5. Executes tools.
6. Converts results into `tool_result` user messages.
7. Continues until there are no more tool calls or a budget is reached.

Important implications:

- Tool results re-enter message history instead of living in side-channel state.
- Model calls and tool calls share one transcript.
- Memory, skills, attachments, and compaction all enter through messages or context attachments.
- Subagents and forked agents do not need a separate execution engine.

Potential Simple Agent Lab shape:

```text
runAgentLoop(input)
  - messages
  - systemPrompt
  - runContext
  - modelClient
  - permission function
  - maxTurns
  -> stream AgentEvent
```

## Runtime Context

The runtime context acts as the capability bus for the whole system. In the source notes, `ToolUseContext` carries:

- Available tools, commands, MCP clients, and MCP resources.
- Agent definitions.
- Model and reasoning configuration.
- Abort controller.
- File read cache.
- Runtime app state access.
- Permission context.
- Memory and skill trigger sets.
- Agent ID and agent type.
- Query tracking depth and chain ID.
- Content replacement state.
- UI, progress, and notification callbacks.

For Simple Agent Lab, this suggests:

```text
RunContext
  - tools
  - commands
  - agents
  - memory
  - permissions
  - state
  - abort
  - cwd
  - model
  - agentId
  - agentType
  - queryTracking
  - contentState
```

Design rules to borrow:

- Pass context explicitly instead of letting tools read global state.
- Clone context for subagents and forked agents.
- Share only selected capabilities.
- Keep UI callbacks separate from core execution.
- Let child contexts override permission behavior.

## Tool Model

The tool abstraction is deliberately larger than a function. The reviewed source shape includes:

- Name, aliases, and search hint.
- Input and output schemas.
- Prompt and description methods.
- Call method.
- Input validation.
- Permission checks.
- Concurrency safety.
- Read-only, destructive, and open-world flags.
- Interrupt behavior.
- Tool result mapping.
- UI rendering hooks.
- Result size limits.

Simple Agent Lab should not copy every field immediately, but should preserve these boundaries:

```text
Tool = callable capability + schema + policy metadata + execution contract + result contract.
```

### Safety Metadata

Tools should declare safety properties rather than forcing the permission system to infer from names:

```text
ToolPolicy
  - readOnly
  - destructive
  - openWorld
  - requiresUserInteraction
```

### Concurrency Metadata

The notes describe batching consecutive concurrency-safe tools while serializing writes and destructive operations.

Useful scheduling model:

```text
safe, safe, safe -> concurrent batch
write            -> serial
safe, safe       -> concurrent batch
destructive      -> serial with permission
```

### Result Budgeting

Tool results should be structured and budget-aware:

- Small results enter model context directly.
- Large results are persisted.
- The model sees a stable preview and a path or reference.
- UI rendering may differ from model-facing content.
- Replacement decisions should be stable to protect prompt cache behavior.

Potential result shape:

```text
ToolResult
  - data
  - modelContent
  - displayContent
  - newMessages
  - contextPatch
  - metadata
```

## Agent Definitions

The agent definition layer makes agent behavior declarative. The source notes include fields such as:

```text
agentType
whenToUse
tools
disallowedTools
skills
mcpServers
hooks
color
model
effort
permissionMode
maxTurns
initialPrompt
memory
background
isolation
omitClaudeMd
```

For Simple Agent Lab, a future agent definition could include:

```text
AgentDefinition
  - name
  - description
  - systemPrompt
  - tools
  - disallowedTools
  - skills
  - mcpServers
  - memory
  - model
  - effort
  - permissionMode
  - maxTurns
  - background
  - isolation
  - hooks
```

Suggested loading precedence:

```text
built-in
plugin
user
project
managed policy
runtime flag
```

The important rule is to keep override behavior explicit and inspectable.

## Subagent Model

In this architecture, a subagent is not a new runtime. The agent runner resolves an agent definition, builds a child context, and calls the same query loop.

Child context behavior:

- Clone file read state.
- Create a child abort controller when needed.
- Create new memory and skill trigger sets.
- Increase query tracking depth.
- Avoid sharing app state by default.
- Prevent async subagents from showing permission UI by default.
- Clone content replacement state to preserve cache-friendly behavior.

Design rule:

```text
Subagent = same loop + child context + narrowed tools + adjusted permissions + side transcript.
```

For Simple Agent Lab:

- Child agents should be isolated by default.
- Temporary parent approvals should not leak into child agents.
- Child transcripts should be saved separately for resume and audit.
- Synchronous child agents may share abort behavior.
- Background child agents should have independent abort behavior.

## Forked Agent

Forked agents are side workers for maintenance tasks. They reuse the same loop but are not always user-facing.

Good use cases:

- Session summary.
- Durable memory extraction.
- Prompt suggestions.
- Compaction.
- Progress summarization.
- Background verification.
- Failure recovery analysis.

Not good use cases:

- The user's primary task.
- High-permission write actions.
- Workflows that require ongoing user interaction.

The notes highlight `CacheSafeParams`: a bundle of inputs that affect prompt caching, such as system prompt, user context, system context, tool context, model, reasoning configuration, and parent message prefix.

Simple Agent Lab can borrow this by making forked-agent inputs explicit and hashable.

## Background Agents

Background agents need lifecycle management. They should not be fire-and-forget.

A background task should track:

```text
BackgroundTask
  - id
  - kind
  - description
  - status
  - agentId
  - startedAt
  - updatedAt
  - outputPath
  - abort controller
  - progress
  - result
```

Rules to borrow:

- Background work must have queryable state.
- Output should be persisted.
- Transcripts should be auditable.
- Permission prompts should either bubble to a visible owner or be denied.
- Recursive spawning should be limited.

## Multiagent Teammates

The notes distinguish subagents from teammates.

Subagent:

- One-off task.
- Result returns to parent.
- Lifecycle belongs to one tool call.
- No stable mailbox.

Teammate:

- Stable name and team name.
- Mailbox.
- Task state.
- Can idle and receive later messages.
- Can receive shutdown messages.
- Can bridge permission requests to a leader.
- Has team context and roster.

Potential future shape:

```text
Teammate
  - id
  - name
  - teamName
  - agentDefinition
  - state
  - inbox
  - outbox
  - taskState
  - permissionMode
```

Simple Agent Lab should not implement this early, but the distinction is valuable:

```text
Subagent = isolated task.
Teammate = long-lived actor.
```

## State and Memory

The notes identify several memory-like layers.

### Instruction Memory

Files such as `CLAUDE.md` or project rules act as instruction memory. They are suitable for:

- Project conventions.
- Test commands.
- Review standards.
- User preferences.

They are not suitable for:

- Current task scratch state.
- One-off intermediate results.
- Large logs.

For Simple Agent Lab, `AGENTS.md` can serve a similar role once runtime support exists.

### File-Based Durable Memory

The notes describe a memory directory where `MEMORY.md` is an index, not the full memory body. Topic files hold the detailed entries.

Potential shape:

```text
memory/
  MEMORY.md
  user/
  project/
  reference/
  feedback/
  logs/
```

`MEMORY.md` should stay short and navigational:

```text
- Python E2E workflow -> project/python-e2e.md - local smoke test command and failure boundary
- User response preference -> feedback/terse-answers.md - prefers concise operational answers
```

### Relevant Memory Retrieval

The notes describe retrieval from a memory manifest:

- Select a small number of relevant memories.
- Exclude already-surfaced memory.
- Track modification time.
- Apply line and byte limits.
- Deduplicate within a session.
- Prefetch asynchronously.

Potential interface:

```text
MemoryStore
  - scan()
  - selectRelevant(query, limit)
  - read(ref, budget)
  - write(scope, content)
```

### Memory Extraction Agent

A forked agent can extract durable memory after a turn.

Rules to borrow:

- Skip extraction if the main agent already wrote memory.
- Limit extraction tool permissions to the memory directory.
- Enforce max turns.
- Treat index updates separately from true memory writes.

This keeps memory maintenance out of the main user response path.

## Context Management

The context system is a pipeline, not a single summary step.

The reviewed layers:

1. Tool result budget.
2. History snipping.
3. Microcompaction.
4. Context collapse or projection.
5. Autocompaction.
6. Reactive compaction after prompt-too-long failures.
7. Max-output recovery.

Potential future interface:

```text
ContextTransform
  - name
  - apply(messages, context, model)
  -> messages, events, statePatch
```

Recommended order:

```text
compact boundary slicing
  -> tool result budget
  -> memory attachment dedupe
  -> microcompact
  -> context collapse / projection
  -> autocompact
  -> API call
  -> reactive recovery on overflow
```

Key principles:

- Prefer low-loss transforms before lossy summaries.
- Persist large tool results instead of repeatedly truncating them differently.
- Preserve restored attachments after compaction.
- Make the pipeline inspectable through debug or context commands.

## Skill Model

Skills are frontmatter-based prompt packages. The notes describe fields such as:

```text
name
description
when_to_use
version
allowed-tools
arguments
model
effort
disable-model-invocation
user-invocable
hooks
context
agent
paths
shell
```

Two execution modes matter:

### Inline Skill

The skill content is expanded into new messages in the main loop.

Good for:

- Standard operating procedures.
- Prompt templates.
- Small workflows.
- Parameterized instructions.

### Forked Skill

The skill runs in a forked agent context.

Good for:

- Long workflows.
- Skills that need independent tool calls.
- Work that should not pollute the main context.
- Large operations with their own lifecycle.

Potential shape:

```text
SkillDefinition
  - name
  - description
  - whenToUse
  - allowedTools
  - arguments
  - model
  - effort
  - context: inline | fork
  - agent
  - paths
  - hooks
  - rootDir
  - content
```

Discovery ideas:

- Load user, project, and bundled common skills first.
- Discover nested skills when relevant paths are read or searched.
- Activate path-scoped skills through frontmatter.
- Avoid loading skills from ignored directories by default.

## MCP and Tool Search

MCP tools can be wrapped into the same tool protocol, preserving JSON schema and safety annotations such as read-only, destructive, and open-world behavior.

When the tool surface grows too large, deferred tool search becomes useful:

- Mark some tools as deferred.
- Keep only essential tools and `ToolSearch` in the initial prompt.
- Let the model search for relevant tools.
- Return tool references.
- Expand real schemas only for selected tools.

Potential interface:

```text
DeferredToolIndex
  - add(tool)
  - search(query, limit)
  - load(names)
```

This is only necessary once the tool count or schema size is high. It is unnecessary for a small v0 with fewer than roughly 20 simple tools.

## Permission Model

Permission is layered:

- Tool input validation.
- Tool-level permission checks.
- Shared permission rules.
- Hooks.
- Automatic classifier.
- Async-agent no-prompt strategy.
- Teammate permission bridge.

Potential interface:

```text
PermissionRuntime
  - mode: ask | auto | read-only | bypass
  - rules
  - decide(tool, input, ctx)
```

Design rules:

- Tools declare risk.
- Permission runtime makes centralized decisions.
- Subagents do not inherit temporary parent approvals by default.
- Background agents cannot wait on invisible UI prompts.
- Multiagent workers route permission requests to a visible leader.

## What We Might Borrow

- A single agent loop reused by all agent-like work.
- Explicit `RunContext` as a capability bus.
- Tool protocol with schema, permissions, safety metadata, concurrency metadata, result budgeting, and model/UI result mapping.
- Declarative agent definitions with tools, skills, memory, model, permission mode, and isolation policy.
- Child context construction for subagents.
- Forked agents for summarization, compaction, memory extraction, and background maintenance.
- Query tracking for depth and chain identity.
- Background task registry with output files, status, progress, and abort handling.
- Clear distinction between subagents and long-lived teammates.
- File-based memory with a short index and topic-specific memory files.
- Relevant memory retrieval with session-level dedupe.
- Layered context pipeline with low-loss transforms before lossy summaries.
- Inline and forked skill execution modes.
- Deferred tool search for large tool surfaces.
- Permission bridges for async or multiagent workflows.

## What We Should Avoid

- Do not let a v0 inherit the full complexity of a mature coding agent.
- Do not couple terminal or UI rendering deeply into the core tool protocol at the start.
- Do not build remote agents, split panes, or teammate infrastructure before subagents and background tasks are stable.
- Do not implement complex context collapse before basic result budgeting and compaction are proven.
- Do not add marketplace or remote skill systems before local skills are useful.
- Do not put all telemetry, feature flags, and runtime modes into the first implementation.
- Do not let permissions live only inside individual tools.

## Suggested Future Shape for Simple Agent Lab

This is not an accepted design yet. It is a candidate direction to compare against other reference architectures.

```text
src/
  core/
    agentLoop
    messages
    runContext
    events
    modelClient

  tools/
    protocol
    registry
    executor
    orchestration
    resultBudget
    builtin

  agents/
    definitions
    loading
    runner
    childContext
    forkedAgent
    backgroundTasks

  memory/
    store
    index
    retrieval
    extractionAgent
    sessionMemory

  context/
    pipeline
    toolResultBudget
    microcompact
    autocompact
    recover

  skills/
    definitions
    loading
    skillTool
    discovery

  integrations/
    mcp
    toolSearch

  permissions/
    runtime
    rules
    classifier
```

Potential staged implementation order:

1. Single agent loop: messages, tool protocol, executor, permission runtime, basic model client, transcript.
2. Subagents: agent definitions, built-in/project loading, child contexts, side transcripts.
3. Context management: result budgets, large result persistence, compact boundary, manual and automatic compaction.
4. Memory: file-based memory directory, short index, relevant retrieval, dedupe, extraction agent.
5. Skills: Markdown loader, frontmatter schema, inline skill tool, forked skill mode, path-based discovery.
6. Background and multiagent: task registry, output files, in-process teammate, mailbox, permission bridge.
7. MCP and tool search: MCP wrapper, deferred tool index, tool search, plugin loading.

## Open Questions for Simple Agent Lab

- Should v0 optimize for a single clean loop before introducing subagents?
- Should tool result persistence exist before memory and skills?
- Should skills support both inline and forked execution from the start?
- Should multiagent teammates be deferred until background tasks are proven?
- Should `RunContext` be part of the public extension API or treated as internal?
- What is the smallest permission runtime that still supports safe subagent and background execution?
