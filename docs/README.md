# Docs

One guide per subsystem. The working contract for contributors and agents is
[`AGENTS.md`](../AGENTS.md); code and `tests/` are the source of truth for
behavior.

## Guides

- [`development.md`](development.md) — the quality gate, and the `ty` false
  positives worth recognizing.
- [`configuration.md`](configuration.md) — every environment variable, grouped
  by owning layer. Start here when you cannot find a config knob.
- [`memory.md`](memory.md) — the memory boundary and `FilesystemMemory`.
- [`adding-an-eval-suite.md`](adding-an-eval-suite.md) — adding a containerized
  benchmark: the two halves plus registration.
- [`multi-machine-eval.md`](multi-machine-eval.md) — running evals across
  several machines: daemons, image distribution, online vs offline.
- [`swebench-pro-chains.md`](swebench-pro-chains.md) — operator contract for
  long SWE-bench Pro chain runs.
- [`docker-live-trace.md`](docker-live-trace.md) — the mount + env contract for
  tailing a containerized run's trace.

## Vocabulary

- [`glossary.md`](glossary.md) — general terms (agent, tool, state, event).
- [`../CONTEXT.md`](../CONTEXT.md) — the single source for message-protocol
  terminology: `Message`, `LLMMessage`, content blocks, provider adapters.

## Local workspace

- [`reference-architectures/`](reference-architectures/README.md) — notes on
  external agent systems, captured before borrowing a pattern. Contents are
  gitignored except the README and template, so the convention is shared but
  individual notes stay on your disk.
