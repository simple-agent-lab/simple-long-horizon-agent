# Reference Architecture: opencode Agent Runtime

## Source

- Project: [opencode](https://github.com/anomalyco/opencode)
- Official docs: [Agents](https://opencode.ai/docs/agents/), [Tools](https://opencode.ai/docs/zh-cn/tools/), [Custom Tools](https://opencode.ai/docs/custom-tools/), [Permissions](https://opencode.ai/docs/zh-cn/permissions/), [Skills](https://opencode.ai/docs/skills), [Rules](https://opencode.ai/docs/rules/), [Plugins](https://opencode.ai/docs/plugins/)
- Source areas reviewed: `packages/opencode/src/agent`, `tool`, `skill`, `permission`, `session`, `provider`, `plugin`, `server`, `lsp`, `mcp`, `storage`, `snapshot`
- Date reviewed: 2026-04-26
- Reviewer note: Based on source-level architecture notes provided to this repository. The project was not run locally during this review.

## Summary

opencode is useful as a reference because it treats agents as a runtime system, not just as prompt files. The most relevant idea for Simple Agent Lab is the separation between agent configuration, tool capability, permissions, skills, sessions, context management, and plugin hooks.

Its core runtime can be summarized as:

```text
Client / TUI / API
  -> Server
  -> Session runtime
  -> Agent resolution
  -> LLM stream
  -> Tool registry
  -> Permission gate
  -> Tool execution
  -> Message parts / snapshot / summary / compaction
  -> Event bus / plugin hooks
```

The important lesson is that an agent is a policy-bearing runtime role:

```text
Agent = prompt + mode + model + tools + permissions + session context + step budget
```

## Core Ideas

- Agents are runtime specs, not just prompts.
- Subagents run as isolated child sessions, with resumable task IDs and their own policy boundaries.
- Tools are structured capabilities with schemas, metadata, attachments, abort handling, and permission checks.
- Skills are lazily loaded context packages, not executable tools and not agents.
- Memory is split by lifecycle instead of centralized into one generic memory service.
- Context management is a runtime responsibility with budgeting, pruning, compaction, and auto-continue behavior.
- Plugin hooks extend behavior around LLM calls, tool calls, permissions, skill loading, compaction, and memory.

## Runtime Boundaries

The architecture has clear module boundaries:

| Layer | Responsibility | opencode areas |
| --- | --- | --- |
| UI / Client | User interaction, session switching, display | `app`, `web`, `desktop`, `console` |
| Server | API and session entrypoint | `server` |
| Agent Runtime | Agent config, mode, prompt, model, permission | `agent` |
| LLM Runtime | System prompt assembly, messages, provider calls | `session/llm.ts` |
| Tool Runtime | Built-in, custom, plugin, MCP tools | `tool`, `mcp` |
| Permission | `allow`, `ask`, `deny` policy | `permission` |
| Context Manager | History, overflow, compaction, summary | `session/*` |
| Skill Loader | `SKILL.md` discovery and lazy loading | `skill`, `tool/skill.ts` |
| Extension | Hooks, custom tools, external services | `plugin` |

## Agent and Subagent Model

opencode distinguishes primary agents, subagents, and hidden system agents.

Examples from the reviewed design:

- Primary agents: `build`, `plan`
- Subagents: `general`, `explore`
- Hidden system agents: `compaction`, `title`, `summary`

The agent schema includes fields such as:

```text
name
description
mode
native
hidden
topP
temperature
color
permission
model
variant
prompt
options
steps
```

This is worth borrowing because it prevents the agent system from becoming a set of loosely related prompt templates. Each agent can express role, visibility, model choice, runtime strategy, permissions, and step budget.

### Subagent Mechanism

The key design is that subagents are invoked through a task tool and run inside child sessions.

The task tool accepts inputs similar to:

```text
description
prompt
subagent_type
task_id
command
```

The runtime flow is:

1. Ask for permission to run the task.
2. Resolve the target subagent from `subagent_type`.
3. Reuse an existing child session when `task_id` is provided.
4. Otherwise create a new child session with `parentID`.
5. Apply child-session permission boundaries.
6. Select the subagent model, or inherit the parent model.
7. Run the target agent inside the child session.
8. Return a `task_id` so the subagent work can be resumed.

For Simple Agent Lab, this suggests that future subagents should be represented as:

```text
runSubagent({
  parentSessionId,
  agentId,
  task,
  resumeTaskId?,
  permissionsOverride?
})
```

not as:

```text
runAgent("reviewer", prompt)
```

## Tool Model

opencode tools are not bare functions. A tool definition includes:

- Stable ID.
- Description.
- Parameter schema.
- Execute function.
- Structured result with title, metadata, output, and attachments.
- Optional validation error formatting.

The tool execution context includes runtime information such as:

```text
sessionID
messageID
agent
abort
callID
messages
metadata()
ask()
```

This is important because a tool needs to know where it is being executed, how to request permission, how to respond to cancellation, and how to record metadata.

The registry unifies built-in tools, custom tools, plugin tools, MCP tools, and dynamic tools like `task` and `skill`. Tool descriptions can be generated at runtime, for example by listing available subagents or skills.

## Permission Model

opencode uses three permission actions:

```text
allow
ask
deny
```

Rules can be broad or specific. More specific rules can override broader wildcard rules.

The architecture pattern to borrow is:

```text
Agent policy
  -> Tool registry
  -> Permission engine
  -> Human approval, if needed
  -> Tool execute
```

For Simple Agent Lab, any real-world side effect should eventually go through a tool and a permission check:

- File writes.
- Shell commands.
- Browser actions.
- Database calls.
- Memory writes.
- Network calls.
- Deployment actions.

## Skill Model

opencode skills are reusable context packages defined by `SKILL.md`. The agent initially sees only name and description. Full skill content is loaded later through the native skill tool.

This keeps the active context smaller and avoids loading every workflow into every task.

Useful distinction:

| Concept | Meaning | Lifecycle | Actively executes |
| --- | --- | --- | --- |
| Agent | Runnable role | Session or task | Yes |
| Subagent | Agent in child session | Child session | Yes |
| Tool | Callable capability | Tool call | Yes |
| Skill | Workflow or context package | Context injection | No |
| Memory | Preserved state or facts | Multiple lifecycles | Sometimes |

For Simple Agent Lab, skills should be treated as manuals and resource bundles. Execution should remain in tools.

Potential future skills:

```text
skills/
  pr-review/SKILL.md
  repo-onboarding/SKILL.md
  database-migration/SKILL.md
  incident-debugging/SKILL.md
  performance-review/SKILL.md
```

## State and Memory

opencode does not appear to center its runtime around one default `MemoryService` or vector database. Instead, it splits memory-like behavior by lifecycle.

### Project Memory

Project instructions are loaded from files such as:

```text
AGENTS.md
CLAUDE.md
CONTEXT.md
```

The reviewed notes describe support for both project-level and global instructions. There is also a directory-aware behavior: when a file is read, nearby instruction files can be resolved from the file path upward. This acts like local project memory.

### Session Memory

Session state is represented through structured messages and parts:

- Reasoning.
- Tool calls.
- Tool results.
- Step start and finish events.
- Patches.
- Usage.
- Snapshots.

This keeps operational facts in runtime state instead of relying on the model to remember them.

### Summary and Diff Memory

Session summaries and snapshot diffs capture what changed during a session. These can be used as durable context for future continuation.

### Long-Term Memory

The best long-term memory shape for Simple Agent Lab should likely be plugin-managed or service-managed instead of built into every agent.

Possible future service boundary:

```text
MemoryService
  - project facts
  - user preferences
  - repo conventions
  - prior decisions
  - embeddings / search index
  - retention policy
  - redaction / privacy policy
```

## Context Management

opencode treats LLM input as a structured runtime object rather than one large prompt string.

Important fields include:

```text
user
sessionID
parentSessionID
model
agent
permission
system
messages
tools
retries
toolChoice
```

System prompt construction is layered:

- Agent prompt.
- Provider prompt.
- Custom system blocks.
- User system blocks.
- Plugin transforms.

Tools are filtered before they enter the model context. The final tool list is a runtime decision based on permissions, agent, model, and user constraints.

### Processor

The session processor acts as a streaming state machine. It handles:

- Tool input.
- Tool calls.
- Tool results.
- Tool errors.
- Reasoning.
- Step start and finish.
- Patch events.
- Repeated tool-call detection.
- Overflow and compaction triggers.
- Continue, stop, and compact states.

### Overflow and Compaction

The compaction system reserves output and compaction budget before context overflow occurs. It keeps recent turns while summarizing older history.

The summary format is structured around:

```text
Goal
Constraints
Progress
Key Decisions
Next Steps
Critical Context
Relevant Files
```

The compaction strategy also prunes old tool outputs while preserving recent context and important identifiers such as paths, commands, error strings, and names.

For Simple Agent Lab, this suggests three future modules:

```text
ContextBuilder
  - system blocks
  - agent prompt
  - project rules
  - loaded skills
  - selected memories
  - recent messages
  - retrieved files
  - active tool definitions

ContextBudgeter
  - token estimate
  - model context limit
  - output reserve
  - per-tool output cap
  - media handling

Compactor
  - summary schema
  - recent-tail preservation
  - tool-output pruning
  - replay / auto-continue
```

## What We Might Borrow

- Treat `AgentSpec` as a runtime policy object, not a prompt file.
- Model subagents as child sessions with resumable task IDs.
- Put every external capability behind `ToolDef`, `ToolContext`, and permission checks.
- Add a permission engine with `allow`, `ask`, and `deny`.
- Treat skills as lazy context packages loaded through a tool.
- Keep context building, budgeting, and compaction outside the agent prompt.
- Preserve structured session facts: messages, tool calls, tool results, snapshots, summaries, and diffs.
- Leave plugin hook points around LLM calls, tool calls, permission requests, skill loading, compaction, and memory writes.
- Add evals for routing, tool arguments, permission behavior, skill loading, compaction retention, and seeded review failures.

## What We Should Avoid

- Do not copy opencode's full scale too early.
- Do not adopt a heavy service/layer framework before the educational path is clear.
- Do not create many agents that only differ by prompt but share the same capabilities and permissions.
- Do not make memory a vague global bucket.
- Do not put permissions only in prompt instructions.
- Do not load every skill into every context.
- Do not start with production-oriented UI complexity.

## Suggested Future Shape for Simple Agent Lab

This is not an accepted design yet. It is a candidate direction to compare against other reference architectures.

```text
packages/
  core/
    agent/
    runtime/
    tool/
    permission/
    skill/
    memory/
    context/
    storage/
    eval/

agents/
  build.md
  plan.md
  explore.md
  reviewer.md
  tester.md
  researcher.md
  compactor.md
  summarizer.md

skills/
  pr-review/SKILL.md
  repo-onboarding/SKILL.md
  database-migration/SKILL.md
  incident-debugging/SKILL.md
  performance-review/SKILL.md

tools/
  read.ts
  write.ts
  edit.ts
  grep.ts
  shell.ts
  test.ts
  browser.ts
  memory-search.ts

evals/
  agents/
  skills/
  tools/

docs/
  architecture.md
  permission-model.md
  skill-authoring.md
```

Potential future core interfaces:

```text
AgentSpec
ToolDef
ToolContext
PermissionRule
SkillManifest
MemoryService
ContextBuildInput
ContextPack
CompactionSummary
PluginHook
```

Potential staged implementation order:

1. Core runtime: `AgentSpec` loader, `ToolDef`, `ToolRegistry`, `PermissionEngine`, `SessionStore`, `LLMRunner`, `ContextBuilder`.
2. Subagents: `TaskTool`, child sessions, task resume, subagent summaries.
3. Skills: `SKILL.md` discovery, skill tool, permission filtering, resource listing.
4. Context and compaction: token estimates, tool output truncation, recent-tail preservation, structured summaries.
5. Memory: project rules, session summaries, decision logs, diff summaries, optional long-term memory plugin.
6. Evals: routing, tool calls, permissions, skills, compaction, review quality.

## Open Questions for Simple Agent Lab

- Should the first implementation be TypeScript, Python, or both?
- Should the repository start as a monorepo, or stay single-package until the runtime boundaries become real?
- Which concepts are essential for v0: permissions, subagents, skills, context budgeting, or evals?
- Should `AGENTS.md` be treated as runtime project memory in v0?
- How much plugin surface should exist before the first usable agent loop?

