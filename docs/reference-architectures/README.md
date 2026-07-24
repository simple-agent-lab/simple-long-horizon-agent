# Reference Architectures (local notes)

This directory is a local workspace for reference-architecture research
notes — sketches of how external agent systems are built, captured
before you borrow a pattern in `src/`.

The directory's contents are **gitignored by design** (see the project
`.gitignore`), except for this README and `template.md`. Drop your own
notes here as `<system-name>.md`; they stay on your local disk and out
of the public repository.

Each note should describe:

- What the architecture is optimized for.
- The core loop or control flow.
- How it handles tools.
- How it handles memory or state.
- What is worth borrowing.
- What should be avoided for Simple Agent Lab.

Use [template.md](template.md) for new entries.

When a reference note drives implementation, update the narrowest relevant
topic doc and add a test or other executable validation for the resulting
boundary.
