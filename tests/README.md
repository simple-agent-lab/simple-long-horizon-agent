# Tests

This directory contains focused behavioral tests for the promoted runtime.

Testing and feedback are first-priority design concerns. Now that the balanced
runtime has been promoted into `src`, tests should stay focused on behavior
that helps the project remain simple and understandable.

Useful test targets:

- Agent loop control flow.
- Message and state shape.
- Context visibility.
- Model adapter boundaries.
- Trace or run record behavior.
- Example workflows.

Run the current suite from the repo root:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```
