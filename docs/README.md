# Docs

Human-facing entry point to the doc tree. Future agents should start at
[`agent-native/README.md`](agent-native/README.md) instead — it routes
based on the task.

## Documentation

- [`agent-native/`](agent-native/README.md) — agent-native context tree:
  project intent, code style, harness workflow, development commands,
  operating rules, and unresolved owner questions.

## Supporting

- [`agent-native/integrating-a-docker-eval-suite.md`](agent-native/integrating-a-docker-eval-suite.md)
  — the maintained step-by-step guide for adding a containerized benchmark.
- [`glossary.md`](glossary.md) — general vocabulary (agent, tool, state,
  event, evaluation). The repo-root [`CONTEXT.md`](../CONTEXT.md) is the
  single source for message-protocol terminology (message protocol, provider
  adapter, content blocks).
- [`reference-architectures/`](reference-architectures/README.md) — local
  workspace for reference-architecture research notes. The directory's
  contents are gitignored except for `README.md` and `template.md`, so the
  convention is shared but individual notes stay on the contributor's
  local disk.
