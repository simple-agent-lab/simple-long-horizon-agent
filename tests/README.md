# Tests

This directory separates fast unit tests from opt-in end-to-end tests.

Testing and feedback are first-priority design concerns. Now that the balanced
runtime has been promoted into `src`, tests should stay focused on behavior
that helps the project remain simple and understandable.

Useful test targets:

- Agent loop control flow.
- Message and state shape.
- Context visibility.
- Memory behavior.
- Model adapter boundaries.
- Trace or run record behavior.
- Example workflows.
- Bash tool execution and the deterministic bash-use demo.

Run the unit suite from the repo root:

```bash
uv run python -m unittest discover -s tests/unit
```

## Live End-To-End Test

`e2e/test_live_bash_e2e.py` is intentionally opt-in because it calls a real model
provider and executes a local bash command through the agent loop. It is skipped
unless all required environment variables are set:

```bash
OPENAI_MODEL=<model> \
OPENAI_AUTH_TOKEN=<api-key> \
uv run python -m unittest tests.e2e.test_live_bash_e2e
```

Set `OPENAI_BASE_URL` too when using an OpenAI-compatible endpoint.

The test also loads a repo-root `.env` file before checking those variables.
Values already set in the shell win over `.env` values.
Use `.env.example` as the local template.

For the OpenAI Responses API live tool-use check, run:

```bash
OPENAI_MODEL=<model> \
OPENAI_AUTH_TOKEN=<api-key> \
uv run python -m unittest tests.e2e.test_live_openai_responses_e2e
```

This test drives a bash agent with `Provider(api="openai-responses")` and checks
that a tool call, tool result, second tool call, and final answer round-trip
through the Responses adapter.

Set `E2E_TRACE_PATH=evals/out/live_openai_responses_tool_trace.json` to write a
provider-neutral trajectory record for inspection. Tool definitions live on each
model turn / model-request event because the available tools can change by step.
