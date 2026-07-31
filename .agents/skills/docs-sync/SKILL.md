---
name: docs-sync
description: Keep Simple Long Horizon Agent documentation in sync with the current repo. Use when the user asks to check whether docs match code, update docs after source/example/run changes, find missing docs, or propose small doc structure fixes. Start with a short evidence-backed report, then edit docs only when the user has asked or approved.
---

# Docs Sync

## Overview

Keep the docs aligned with the current worktree. This repo is docs-first, so
doc sync should preserve the source of truth for future humans and agents.

## Workflow

1. Read the local contract first: `AGENTS.md`, `README.md`, and the relevant
   docs or source files for the requested scope.
2. Check `git status --short --branch`. Work with the current dirty worktree;
   do not switch branches, revert files, or overwrite user edits.
3. Compare the source of truth with the docs:
   - source: `src/simple_long_horizon_agent/`, `scripts/`, `runs/`, `tests/`,
     `evals/`, `pyproject.toml`, `.github/workflows/ci.yml`
   - docs: `README.md`, `AGENTS.md`, `CONTEXT.md`, `CONTRIBUTING.md`,
     `docs/README.md` and the guides it indexes, `runs/README.md`,
     `tests/README.md`, `evals/README.md`, `evals/swebench/README.md`,
     `src/simple_long_horizon_agent/llm/README.md`
4. Look for only practical mismatches: missing behavior, outdated paths,
   stale commands, conflicting architecture/status claims, or docs that promise
   more than the repo can verify.
5. Produce a short Docs Sync Report with evidence and proposed edits before
   changing files, unless the user already asked you to apply the fixes.
6. If editing, make small section-level changes. Keep durable architectural
   guidance in the relevant topic doc, reference notes only for external
   architecture sources, and task specs only when future handoff needs them.
7. Run the narrowest useful check and report it. If no command applies, use an
   explicit review checklist.

## Useful commands

Start with targeted searches:

```bash
git status --short --branch
rg -n "bash runs/|python3|uv run|PYTHONPATH|Current Status|TODO|TBD" README.md CONTEXT.md docs runs tests evals
rg -n "Message|State|Agent|Tool|context_view|run_agent|trajectory|evaluation|training" src docs tests evals runs
```

Common checks:

```bash
bash runs/dev/run_ci.sh
uv run python -m unittest discover -s tests/unit
uv run python -m scripts.lint_docs
bash runs/demos/run_bash_agent_demo.sh
```

## Report Format

```text
Docs Sync Report

Scope and baseline
- Branch / dirty state
- Files reviewed

Findings
- Mismatch -> evidence -> proposed doc location

Proposed edits
- File -> change summary

Verification
- Command or review checklist

Open questions
- Decisions needed before editing
```
