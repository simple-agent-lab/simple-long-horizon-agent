# Owner Questions

Read when:

- Code and existing docs cannot answer a repo-boundary, release, external
  dependency, provider, or owner-preference question.
- You are preparing a future interview or moving confirmed answers into
  canonical docs.

Do not read for:

- Routine code changes that are already covered by `README.md`, `docs/agent-native/`,
  tests, and run scripts.

## Current Questions

1. What is the release owner workflow after the first open-source release prep?

   Evidence: recent history includes `Prepare for first open-source release
   (0.1.0)`, and CI is documented, but reviewer, tagging, and publishing rules
   are not described.

   Recommended default: future agents should not tag, publish, or change package
   metadata for a release without explicit owner instruction.

   Likely artifact if confirmed: add a short release section under
   `docs/agent-native/development.md` or a focused release task spec.
