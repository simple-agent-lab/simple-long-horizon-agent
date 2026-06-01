# Docs

Human-facing entry point to the doc tree. Future agents should start at
[`agent-native/README.md`](agent-native/README.md) instead — it routes
based on the task.

## Two doc roots

- [`agent-native/`](agent-native/README.md) — agent-native context tree:
  project intent, code style, harness workflow, development commands,
  operating rules, and unresolved owner questions.
- [`decisions/`](decisions/README.md) — accepted architecture decision
  records (ADRs). New hard-to-reverse choices land here.

## Supporting

- [`human/`](human/README.md) — plain-language, narrative explainers for people
  (start here for the big picture). Includes
  [`how-swebench-runs.md`](human/how-swebench-runs.md): an end-to-end walkthrough
  of a full SWE-bench run (prepare → generate → score) with sequence diagrams.
- [`glossary.md`](glossary.md) — shared vocabulary. The repo-root
  [`CONTEXT.md`](../CONTEXT.md) holds resolved terminology boundaries
  (message protocol, provider adapter, content blocks).
- [`reference-architectures/`](reference-architectures/README.md) — local
  workspace for reference-architecture research notes. The directory's
  contents are gitignored except for `README.md` and `template.md`, so the
  convention is shared but individual notes stay on the contributor's
  local disk.
