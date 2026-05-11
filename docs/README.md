# Docs

Human-facing entry point to the doc tree. Future agents should start at
[`agent-native/README.md`](agent-native/README.md) instead — it routes
based on the task.

## Three doc roots

- [`agent-native/`](agent-native/README.md) — agent-native context tree:
  project intent, code style, harness workflow, development commands,
  doc inventory, operating rules, and unresolved owner questions.
- [`decisions/`](decisions/README.md) — accepted architecture decision
  records (ADRs). New hard-to-reverse choices land here.
- [`reference-architectures/`](reference-architectures/README.md) —
  notes on external agent architectures we want to learn from. Load
  individually when a task names that reference.

## Supporting

- [`glossary.md`](glossary.md) — shared vocabulary. The repo-root
  [`CONTEXT.md`](../CONTEXT.md) holds resolved terminology boundaries
  (message protocol, provider adapter, content blocks).
