# Example Experiment: Agent-As-Tool Delegation Trace

This experiment follows the runnable 02 demo. A coordinator agent can call a
`run_agent` tool; the tool runs a focused child agent and returns that result as
a normal `tool_result` message.

## Goal

Verify that the promoted balanced runtime can express an AgentTool-style
delegation path while preserving a readable trace for self-evolution evals.

## Agents

```text
coordinator   decides whether to call a child agent tool.
tweet_writer  writes one concise tweet when invoked by `run_agent`.
```

The coordinator uses a reactive `next_agent` function that keeps scheduling the
coordinator until it emits a final answer or reaches a small turn cap.

## Variants

| Variant | Tool description | Child prompt policy |
| --- | --- | --- |
| A (baseline) | "Run a focused child agent and return its final result." | Pass through the task. |
| B (specific tool) | Emphasize that the child must write one tweet. | Pass through the task. |
| C (prompt-shaped) | Same as baseline. | Add a style note such as "Make it casual." |

## Metrics

Read directly from `state.events`:

- Number of `model_request` / `model_response` pairs.
- Whether the coordinator emitted a `ToolCallBlock` for `run_agent`.
- Whether the tool produced `tool_execution_start` and `tool_execution_end`.
- Whether a `tool_result` message returned to the coordinator.
- Whether the final coordinator answer reused the child result.
- The visible-message outline and `llm_payload` on request events.

## Why This Version Suffices

The experiment needs:

- A scheduler that can keep one coordinator running until done.
- Structured assistant tool calls and tool-result messages.
- A local execution boundary for `AgentTool`.
- Request/response trace events around both coordinator and child-agent model
  turns.

The promoted balanced runtime provides those without a graph runtime or a
separate trace store.

## When to Move On

If the next experiment requires streaming partial model output, replaying graph
nodes and edges, or a richer provider lifecycle, switch to the 03 event runtime
reference and fold back only the ideas proven useful.
