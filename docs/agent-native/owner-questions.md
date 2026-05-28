# Owner Questions

Read when:

- Code and existing docs cannot answer a repo-boundary, release, external
  dependency, provider, or owner-preference question.
- You are preparing a future interview or moving confirmed answers into
  canonical docs.

Do not read for:

- Routine code changes that are already covered by `README.md`, `docs/agent-native/`,
  ADRs, tests, and run scripts.

Sources:

- Open questions identified during the 2026-05-11 agent-native cold-start pass.

Freshness:

- Questions here are unresolved unless a dated answer is added and moved into
  `README.md`, `operating-rules.md`, another canonical doc, or an ADR.

## Current Questions

1. Should SWE-bench and Docker remain optional local validation, or should any
   part of that setup become required CI?

   Evidence: ADR 0011 and `evals/swebench/README.md` keep SWE-bench as an
   optional external dependency; the adapter's unit tests (covered by `run_ci.sh`)
   avoid Docker.

   Recommended default: keep official SWE-bench evaluation optional; required CI
   should stay `ty` plus `unittest` until the owner accepts the setup cost.

   Likely artifact if confirmed: update `docs/agent-native/development.md`, CI, and
   `runs/README.md` only if the policy changes.

2. What is the release owner workflow after the first open-source release prep?

   Evidence: recent history includes `Prepare for first open-source release
   (0.1.0)`, and CI is documented, but reviewer, tagging, and publishing rules
   are not described.

   Recommended default: future agents should not tag, publish, or change package
   metadata for a release without explicit owner instruction.

   Likely artifact if confirmed: add a short release section under
   `docs/agent-native/development.md` or a focused release task spec.
