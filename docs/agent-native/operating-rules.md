# Operating Rules

Read when:

- You are changing behavior, public contracts, validation gates, eval records,
  generated artifacts, or doc routing.
- You are unsure whether a change needs more evidence, owner input, or an ADR.

Do not read for:

- Small typo fixes that do not change loading paths, commands, or policy.
- Purely mechanical formatting changes that do not alter behavior.

This file is a maintenance policy, not a loading map. Use the agent-native
README to choose task-specific code, docs, tests, runbooks, and ADRs.

## Source-Of-Truth Principles

- Code wins for behavior that is visible in the implementation.
- Tests, smoke scripts, and CI config win for what is actually verified.
- The public README wins for the user-facing project tour and setup path.
- The agent loading map wins for what future agents should read first.
- ADRs win for durable architecture decisions and rejected alternatives.
- Topic docs should explain intent, constraints, and maintenance preferences
  that code does not make obvious.
- Generated artifacts, local eval outputs, traces, and scratch notes are not
  source of truth.

When sources disagree, update the stale source or record the unresolved owner
question. Do not make a second doc repeat the same detailed fact unless it is
needed as a routing hint.

## Maintenance Flow

1. Start from the loading map and inspect the current source of truth.
2. Decide the feedback signal before editing: unit test, smoke run, eval,
   trace, type check, format check, or explicit review checklist.
3. Make the smallest change that resolves the mismatch.
4. Keep public examples small, runnable, and free of external services by
   default.
5. Update docs only where the reader needs non-obvious context or a changed
   command/path.
6. If a change creates a hard-to-reverse architectural commitment, add or
   update an ADR instead of burying the decision in a topic doc.

## Usually Safe To Edit

Agents can usually edit these with code evidence and a narrow check:

- Small behavior covered by focused tests.
- Deterministic examples and smoke scripts.
- Stale commands, stale links, and doc routing.
- Local reference notes before implementation begins.
- Handoff notes that are clearly temporary and not architectural decisions.

## Needs More Evidence

Collect code, doc, test, or config evidence before editing:

- Public API surface and package exports.
- Message roles, content blocks, tool-call fields, and token usage semantics.
- Context budgeting, trimming, or grouping behavior.
- Validation gate changes across local and remote checks.
- Eval output schemas, trajectory records, and training/export formats.
- Optional dependency policy, especially when external services or containers
  are involved.

## Needs Owner Confirmation

Ask or record a question before changing:

- Required-vs-optional status for live providers, Docker, or benchmark suites.
- Release ownership, reviewer expectations, tagging, or publishing flow.
- Product positioning, teaching audience, or scope boundaries that are not
  already reflected in the public README, loading map, or ADRs.

## Validation Selection

Choose the narrowest useful check first, then broaden when the blast radius
crosses module boundaries.

| Change type | Narrow check | Broader check |
| --- | --- | --- |
| Formatting-only or broad Python edits | Formatter check | Full local CI gate |
| Core runtime, messages, tools, or context | Focused unit tests | Full local CI gate |
| Type or public API surface | Type check | Full local CI gate |
| Public examples or run scripts | Affected smoke script | Full local CI gate |
| Eval or benchmark adapter plumbing | Local deterministic smoke | Optional external suite only when configured |
| Docs-only routing change | Link/loading-map review | Full local CI only if commands changed |

## Boundaries Worth Preserving

- Beginner readability is a product constraint, not only a style preference.
- Keep durable history separate from model-visible context projections.
- Keep raw trajectory facts separate from eval scores and training labels.
- Keep provider-specific details at provider boundaries.
- Keep heavy benchmark dependencies outside the minimal core runtime unless
  the owner explicitly changes that policy.
- Keep generated outputs in ignored artifact directories, not in source docs.

## ADR Boundary

Create or update an ADR only when all of these are true:

- The decision is hard or expensive to reverse.
- Future agents or engineers would otherwise wonder why the system is shaped
  this way.
- There is a real tradeoff between credible alternatives.

Otherwise, update the relevant code, test, topic doc, run script, reference
note, or loading-map entry.
