# Reference Architecture: pi-mono Agent Runtime

## Source

- Project: [badlogic/pi-mono](https://github.com/badlogic/pi-mono)
- Package reviewed: `packages/agent/` (`@mariozechner/pi-agent-core`)
- Files: `src/types.ts` (~365 lines), `src/agent-loop.ts` (~683 lines),
  `src/agent.ts` (~543 lines), `src/proxy.ts` (~367 lines), `README.md`
- Date reviewed: 2026-04-29
- Reviewer note: Read locally from a shallow clone. The project was not run.

## Summary

pi-mono is a TypeScript "AI agent toolkit" monorepo. The interesting package
for our purposes is `pi-agent-core`: a single-agent stateful runtime that adds
tool calling and event streaming on top of a unified multi-provider LLM
client (`pi-ai`).

The architecture is small enough to read in one sitting and is shaped around
**a clear LLM-call boundary** with two transformation hooks, **a generator
that yields lifecycle events**, and **two independent injection queues** for
mid-run steering and post-stop follow-ups.

The core loop is roughly:

```text
prompt() -> agent_start
         -> turn_start
            -> drain steering queue (inject as messages)
            -> transformContext(messages)        // AgentMessage -> AgentMessage
            -> convertToLlm(messages)            // AgentMessage -> LLM Message
            -> streamFn(model, llmMessages)      // streaming assistant message
            -> emit message_update events
            -> message_end
            -> if tool calls:
                  beforeToolCall hook (can block)
                  execute tools (parallel | sequential)
                  afterToolCall hook (can override result, can terminate)
                  toolResult messages
            -> turn_end
         -> repeat while tool calls or steering pending
         -> if follow-up queue has items: inject and run another turn
         -> agent_end
```

## Core Ideas

- The runtime works in **AgentMessage**, an open union of LLM messages plus
  app-defined custom message types (via TypeScript declaration merging). The
  conversion to a strict LLM Message list happens **only at the LLM call
  boundary**.
- The boundary is split into two well-named hooks. `transformContext` is for
  app-level operations (pruning, summarization, context injection).
  `convertToLlm` is for protocol-level operations (filter UI-only messages,
  map custom kinds onto user/assistant/toolResult).
- The runtime is a **generator that yields events**, not a function that
  returns a final state. Consumers `for await (event of agentLoop(...))`.
- Two distinct injection queues encode two different intents:
  - **steer(message)** injects after the current turn's tools finish and
    before the next LLM call. Used for mid-run nudges or corrections.
  - **followUp(message)** injects only after the agent would otherwise stop.
    Used for "ask another question once it is done".
  - Each queue has a mode: `"all"` (drain everything in one shot) or
    `"one-at-a-time"` (drain one message per turn).
- **`beforeToolCall` / `afterToolCall` hooks have explicit semantics**:
  - `before` can return `{ block: true, reason }` to skip execution and emit
    an error toolResult.
  - `after` can return field-by-field overrides for `content`, `details`,
    `isError`, `terminate`. There is no deep merge.
  - `terminate: true` only stops the loop when **every** finalized tool
    result in the batch agrees. Mixed batches continue normally.
- The runtime exposes **two API layers** with the same semantics:
  - Low-level functional: `agentLoop(prompts, context, config)` -> EventStream.
  - High-level stateful: `Agent` class that owns transcript, queues,
    listeners, and `AbortController`.
- **`prompt()` and `continue()` are first-class entry points**. `continue()`
  resumes from existing state and is the supported way to retry after error
  or to drain a queued steering message that arrived after the agent stopped.
- Tool execution mode is a per-batch decision. Default is `"parallel"`.
  Per-tool `executionMode: "sequential"` forces the whole batch sequential,
  which guarantees mutually exclusive tool semantics when at least one tool
  needs ordering.
- An **AbortSignal is threaded everywhere** (streamFn, beforeToolCall,
  afterToolCall, tool.execute, listeners). The `Agent` class owns one
  AbortController per active run.
- Subscribers (`Agent.subscribe(fn)`) are awaited. `agent_end` does not mark
  the run as idle until awaited subscriber promises settle. This makes
  `agent_end` a real barrier that downstream code can rely on.

## Agent Loop

The control flow is split into outer and inner loops:

```text
outer:
  inner:
    while hasMoreToolCalls or steeringQueue not empty:
      turn_start
      if steering pending: inject as messages, drain
      streamAssistantResponse:
        apply transformContext -> apply convertToLlm
        call streamFn(model, llmMessages) and emit message_update
        finalize assistant message with stopReason
      if assistant has tool calls:
        executeToolCalls (parallel or sequential)
        emit tool_execution_start / update / end
        emit toolResult message_start / message_end
        terminate-if-batch-all-terminate
      turn_end
      drain steeringQueue at top of next iteration
    poll followUpQueue
    if followUp had items: set as pending and continue outer
    else break
agent_end
```

The same shared inner loop powers both `agentLoop` (new prompt path) and
`agentLoopContinue` (resume path). The only difference at the entry is
whether to push the prompt messages into the transcript before the first
LLM call.

## Tool Model

Tools are registered as `AgentTool` objects:

```ts
interface AgentTool {
  name: string;
  label: string;                // for UI
  description: string;
  parameters: TSchema;          // typebox schema -> validated
  prepareArguments?: (raw) => parsed;   // optional pre-validation shim
  execute: (id, params, signal, onUpdate?) => Promise<AgentToolResult>;
  executionMode?: "parallel" | "sequential";
}
```

Key conventions:

- The schema is validated by `validateToolArguments` before `execute` runs.
  Validation failures become an immediate error tool result, no execution.
- `execute` **throws** on failure. Returning error strings inside content is
  explicitly discouraged. The runtime turns thrown errors into `isError:
  true` toolResult messages.
- `execute` may call `onUpdate({ content, details })` to stream progress.
  These become `tool_execution_update` events.
- `terminate: true` on a tool result is a hint, not a stop. Honored only
  when every result in the same batch agrees.
- Tool results carry both `content` (model-facing text/images) and `details`
  (free-form payload for logs and UI). The split keeps the model context
  clean while still preserving structured data.

Execution mode rules:

- Global `toolExecution` config sets the default for the run.
- Per-tool `executionMode` overrides the default.
- If any tool in the batch is sequential, the **entire** batch runs
  sequentially. This is a deliberate safety bias: parallel never silently
  reorders calls that might depend on each other.
- In parallel mode, `tool_execution_end` fires in completion order, but the
  toolResult message events are emitted in **assistant source order**, so the
  transcript stays deterministic.

## State and Memory

State lives on the `Agent` class as a typed `AgentState`:

```ts
interface AgentState {
  systemPrompt: string;
  model: Model;
  thinkingLevel: ThinkingLevel;
  tools: AgentTool[];          // copy-on-assign accessor
  messages: AgentMessage[];    // copy-on-assign accessor
  readonly isStreaming: boolean;
  readonly streamingMessage?: AgentMessage;
  readonly pendingToolCalls: ReadonlySet<string>;
  readonly errorMessage?: string;
}
```

Notable choices:

- Both `tools` and `messages` use accessor properties whose setter copies the
  top-level array. This prevents alias bugs where a caller keeps a reference
  to the array and mutates it during a run.
- `streamingMessage` reflects the partial assistant message during
  streaming. UIs render directly from agent state without subscribing.
- `pendingToolCalls` is a Set of in-flight tool call IDs, also derivable
  from events. Storing it on state means UIs can render a spinner without
  replaying the event log.
- There is **no separate trace/event-log structure**. The transcript
  (`messages`) is the durable record. Events are ephemeral lifecycle
  signals.
- Memory beyond the transcript is left to the consumer. `transformContext`
  is the documented hook for pruning or compaction.

## Provider Boundary

Provider details are isolated behind `streamFn` and `streamProxy`:

- `streamFn: typeof streamSimple` is a pluggable function that produces an
  `AssistantMessageEvent` stream from a model + context. Default uses
  `pi-ai`'s `streamSimple`, which already abstracts OpenAI / Anthropic /
  Google / others.
- `streamProxy` wraps an HTTP backend so a browser app can route through a
  server that holds API keys. The agent still sees the same `streamFn`
  shape.
- `getApiKey(provider)` resolves an API key per call. Documented use: short
  lived OAuth tokens (e.g. GitHub Copilot) that may expire mid-run.
- Provider-native event kinds (`text_delta`, `thinking_delta`,
  `toolcall_delta`, etc.) are passed through `message_update.assistantMessageEvent`
  rather than collapsed. UIs that want raw deltas get them; UIs that just
  want the partial message read `message_update.message`.

## What We Might Borrow

For Simple Agent Lab's balanced runtime (`02_balanced_runtime`):

- **Clear model-call boundary**: keep `context_view(...)` and one optional
  `transform(messages)` hook visible, then record the `to_model_messages(...)`
  payload in `model_request`. Do not expose a second conversion hook until a
  real experiment needs it.
- **Generator-based run()**: yield `Event` instead of returning final state.
  Consumers can subscribe inline. A 1-line `run_to_completion` wrapper covers
  the "I just want the final state" case.
- **Explicit continuation entry points**: keep input policy outside the core
  unless an interactive shell proves queue semantics are needed.
- **Two API layers**: keep `core.run(...)` as a pure generator and provide
  an `AgentRuntime` class for stateful use. Users pick the level they want.
- **`prompt()` vs `resume()`** as separate entry points. Cleaner than
  overloading one function with optional flags.
- **Replace static schedule with `next_agent: State -> str | None`**: the
  same generalization makes mailbox / sequence / round-robin /
  reactive routing all expressible in the same runtime, with the
  caller supplying the rule.

For the rich runtime (`03_event_runtime`):

- **First-class tool calling** with the same content/details split.
- **Parallel and sequential tool execution** with the per-tool override
  rule and the "any sequential implies whole batch sequential" safety bias.
- **AbortSignal threading** through every callback so cancellation is
  cooperative and complete.
- **Pluggable `stream_fn`** with a default that abstracts multiple
  providers. The proxy pattern is also worth keeping in mind for any
  browser-facing future.
- **`terminate` as a batch-consensus hint**, not an immediate stop.

## What We Should Avoid

- **TypeScript declaration merging for custom messages** does not transfer
  to Python. The Python equivalent is a free `kind: str` plus explicit
  projection helpers; do not try to recreate the type-level openness.
- **Two parallel runtimes** (`Agent` class plus `agentLoop` function) only
  pay for themselves because both are exported and documented. Avoid the
  pattern in cases where there is no real consumer of the low-level path.
- **Streaming partial messages as runtime state** (`state.streamingMessage`)
  is convenient for a UI but adds invariants ("never push partial into
  messages list, always replace the tail") that are easy to break in a
  small project. Shared Simple Agent Lab `Message` should stay frozen; if
  streaming is added, use update events or replace unrecorded temporary values.
- **Hidden state on the `Agent` class** (`activeRun`, `pendingToolCalls` as
  a Set, `errorMessage`, `isStreaming`) requires careful lifecycle
  bookkeeping. The runtime spends real complexity ensuring these settle in
  the right order. Worth replicating only when the consumer (a UI) actually
  reads it.
- **`transport: "sse"` defaults and proxy retry logic** are out of scope
  for our balanced runtime. Provider details belong in the rich version
  behind `model_client` style abstractions, not as runtime knobs.

## Notes

- For Simple Agent Lab, borrow pi-mono's boundary semantics rather than its
  names. Our `Message` should play the role of pi-mono's `AgentMessage`
  (runtime transcript plus extensible app-level messages), and our
  `ModelMessage` should play the role of pi-mono / pi-ai's LLM `Message`
  (the provider-neutral model-call payload).
- The `Agent` class deliberately treats `agent_end` as a barrier: it does
  not become idle until awaited listeners settle. Consider whether our
  Python `AgentRuntime.subscribe` should follow the same rule. For pure
  synchronous listeners it is automatic; for async listeners we would need
  to either await them or document the relaxed contract.
- pi-mono's tool model only supports text and image content blocks. If we
  later care about audio/video, the `content` shape needs revisiting.
- The TypeScript `ImageContent`/`TextContent` distinction maps cleanly to
  Python with discriminated content blocks. Because Simple Agent Lab expects
  multimodal providers, `ModelMessage` should use provider-neutral content
  blocks instead of a string-only chat dict. Start with text, image, thinking,
  and tool call blocks; defer file/audio/video until concrete use cases exist.
  For images, follow pi-mono's normalized `data` plus MIME type shape rather
  than provider-specific URL or file-id fields. In Python, `ImageBlock.data`
  should be a plain base64 string without a data-URL prefix.
- Preserve thinking as structured model output when providers expose it. It is
  important for trace, replay, and provider continuity, but display policy
  should remain separate from storage. In Simple Agent Lab, assistant
  `Message` values should keep thinking in an explicit field. Use
  `text`, `signature`, and `redacted` fields; provider identity belongs
  outside each thinking block.
- The promoted `src` runtime does not copy pi-mono's steering queues. If an
  interactive shell later needs them, decide the message-injection order in an
  ADR before adding it to the shared core.
