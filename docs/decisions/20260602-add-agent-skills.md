---
title: "Add Agent Skills (read-based, on by default)"
status: Accepted
date: 2026-06-02
slug: add-agent-skills
---

# Add Agent Skills (read-based, on by default)

## Status

Accepted

## Context

Agents benefit from reusable, task-specific "skills" — packaged instructions
(plus optional scripts and references) that the model can follow on demand.
The reference design (`.idea/SKILLS_IMPLEMENTATION_GUIDE.md`) and the Codex
source (`.idea/codex/codex-rs/core-skills`) converge on a model where a skill
is *context, not a tool*: the framework discovers `SKILL.md` files, advertises
them in a prompt menu, and the model loads a skill by reading it and runs its
scripts with ordinary tool calls.

We needed a shape that fits Simple Agent Lab's small, inspectable runtime
(ADR use-tiny-message-runtime, ADR promote-balanced-runtime-to-src-core): no new core abstraction, everything visible in
`state.events`, deterministic local tests.

## Decision

1. **Read-based, not a skill tool.** Loading a skill is reading a file. The
   model opens `SKILL.md` and `references/*` with a new small `read` tool
   (`src/simple_agent_lab/tools/read.py`) and runs `scripts/*` with the
   existing `bash` tool. There is no skill-execution engine and no `skill()`
   tool. The `read` tool exists because `bash` truncates output at 4000 chars,
   which would mangle a real `SKILL.md`.

2. **Skills live at a visible edge, not in core.** For interactive/CLI use,
   `run_with_skills` (`src/simple_agent_lab/skills/runtime.py`) builds the
   `State`, records the `<skills_instructions>` menu (a system message) and any
   `/mention` / `preload` bodies (context messages) before the task, then calls
   the unchanged `core.run`. This mirrors how `task_tool` injects context.

3. **On by default, `/no-skills` to disable.** Discovery + menu + injection run
   automatically. The only "/" surface is a directive parser handling
   `/no-skills` (disable) and `/name` (explicit mention) — not a general
   slash-command framework.

4. **Bundled library ships empty.** A `skills/library/` directory ships in the
   wheel and is scanned by default, but contains no skills, so a fresh
   benchmark run is unchanged until a skill is added. A `bash_skills` SWE-bench
   flavor opts a run into skills; `bash`/`bash_task` baselines are untouched.

5. **Scope-based directories, never per-agent.** Skills live in one shared tree
   scoped by location (bundled, project `.agents/skills` + native, user),
   matching the open Agent Skills convention — not in per-agent directories.
   Per-agent scoping is data, not directories: callers pass a filtered
   `skills=` list and/or `preload=[...]` of names to `run_with_skills` (the
   analogue of Claude Code's subagent `skills:` field). Subagents
   (`task_tool`) do not inherit the parent's menu — the menu is per-agent and
   only the agent run through `run_with_skills` sees it; subagents get skills
   deliberately via `preload` or by being handed a skill path/body in their
   delegated task. The skill-selection prose is adapted verbatim from Codex's
   standard `SKILLS_HOW_TO_USE` prompt.

6. **`read` loads, `bash` executes; loading is progressive, not layered.** The
   `read` tool loads any text content (`SKILL.md`, `references/*`, schemas) and,
   given a directory, lists its files; `bash` runs `scripts/*`. Injected bodies
   carry a short `<files>` manifest so the model knows what else it can load.
   The model navigates a skill freely (re-read, load another reference, run a
   script) rather than following a fixed metadata→body→done flow.

7. **Benchmark path folds the menu into the system prompt.** The generic
   containerized runner (ADR generic-containerized-eval-framework) drives the agent through `agent.run` and is
   suite-agnostic, so it has no per-turn seam to record a menu into. The
   `bash_skills` flavor therefore builds a `bash` + `read` agent and folds the
   discovered `<skills_instructions>` menu into the agent's system prompt
   (`system_prompt_with_skills` in `src/simple_agent_lab/skills/prompt.py`,
   called from `build_agent` in `src/simple_agent_lab/evals/in_container.py`) —
   the same content
   `run_with_skills` records as a system message in the interactive edge. The
   core run loop and the generic runner stay unchanged.

## Consequences

- Skills add no core-runtime abstraction; the loop stays inspectable.
- The menu and any loaded bodies are ordinary, visible messages in
  `state.events` and traces (interactive edge), or ride the system prompt
  (benchmark path).
- Adding a skill is "drop a directory with a `SKILL.md`" — into the bundled
  library or a project/user root.
- The Codex 2%-context budget, description truncation, and path aliasing are
  deliberately deferred (not needed at this scale).

## Alternatives Considered

- **Dedicated `skill()` tool (OpenCode style).** Mirrors `task_tool` and gives
  an auditable activation, but adds a tool and diverges from the Codex
  read-based baseline; the `read` tool already makes loading auditable.
- **Reuse `bash cat` for loading (no read tool).** Simplest, but bash's 4000-
  char truncation would mangle real skill bodies.
- **Menu as a recorded message on the benchmark path too.** Works for the
  interactive edge, but the generic runner calls `agent.run` and is
  suite-agnostic; recording a menu there would couple the runner to skills, so
  the benchmark path folds the menu into the system prompt instead.
- **Per-agent skill directories (`.agent1/`, `.agent2/`).** Rejected — no
  reference agent does this; it fragments and duplicates skills and breaks the
  shared `.agents/skills` convention. Per-agent scoping is a filtered list /
  preload instead.
- **`read`-only (no bash) or `bash`-only (no read) for skills.** Rejected —
  skills mix loadable content (md, schemas, references) and executable scripts;
  the run needs both. `read` also lists directories so the model can discover a
  skill's files without a separate listing tool.
