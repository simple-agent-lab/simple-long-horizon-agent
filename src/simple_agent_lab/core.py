"""Balanced message runtime for Simple Agent Lab.

The core model is small::

    Agent + Message + State + build_context_view() + run()

`run()` drives one agent as a generator: each turn, it builds a context view
of the visible messages, calls `agent.generate(...)`, records request/response
trace events, and dispatches any returned tool calls. The loop stops on the
first turn whose output is the agent's `final` message, or when `max_turns`
is exhausted (truncated runs surface as `agent_end(reason="max_turns")` so
traces can distinguish "agent decided to stop" from "ran out of budget").

`Agent.run(task)` is a convenience wrapper that initializes the state with a
task message and calls `run(self, state, ...)`, returning `(state, events)` so
the caller can stream events and still inspect the populated state.

Multi-agent flows are expressed as a parent agent that delegates through
`tools.task_tool([b, c, d])`: the parent picks one sub-agent via the
`subagent_type` enum and the chosen sub-agent's final message comes back as
the tool result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterator

from .compression import maybe_compress_context
from .context_view import (
    ContextPolicy,
    build_context_view,
)
from .hooks import HookContext, HookMap, HookPoint, fire_hooks
from .llm import llm_message, messages_to_llm_messages
from .messages import (
    AssistantMessage,
    ContentInput,
    Message,
    ToolCallBlock,
    ToolResultBlock,
    message_tool_calls,
    tool_results_message,
)
from .protocols import (
    AgentEndEvent,
    AgentEndReason,
    AgentStartEvent,
    Event,
    ModelRequestEvent,
    ModelResponseEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from .state import State
from .tools import AbortFlag, AgentTool, ToolResult, ToolUpdateFn, text_result

if TYPE_CHECKING:
    from .llm.provider import Provider


# An Agent's `generate` produces the next message from the visible context.
# It takes no other arguments: name/role/tools/etc. are closed over at the
# point where the function is built (see `llm_agent.py` and the test fakes).
GenerateFn = Callable[[list[Message]], Message]

# A state initializer builds the initial `State` for `Agent.run` from a task.
# The default (see `Agent._default_init_state`) records a single task message; a
# higher layer can supply one that records extra context up front — e.g. the
# skills layer records a skills menu and any mentioned skill bodies before the
# task. Kept a plain callable taking the agent explicitly so `core` never
# imports a higher layer: the dependency always points inward (skills imports
# core, not the reverse).
StateInitFn = Callable[["Agent", ContentInput], State]
TraceStateFn = Callable[["Agent", State], State | None]


def records_model_events(agent: "Agent") -> bool:
    """Return whether this agent should produce model request/response events.

    True for any provider-backed agent — including the ``"fake"`` test adapter,
    whose calls are recorded and tagged (``ModelRequestEvent.api``) so a
    consumer can filter fake data downstream. False only for programmatic
    facades (``llm_provider is None``): their deterministic ``generate`` is not
    a model call at all, so it must not masquerade as one in the trace.
    """

    return agent.llm_provider is not None


@dataclass
class Agent:
    name: str
    generate: GenerateFn
    role: str = ""
    # `tools` is a tuple to make explicit that tools are bound at construction:
    # `run()` snapshots them once per call, so mutating this attribute mid-run
    # has no effect on the in-flight loop.
    tools: tuple[AgentTool, ...] = ()
    context_policy: ContextPolicy | None = None
    # Lifecycle hooks consulted during `run()` — observe, block a PRE_TOOL_USE
    # call, or emit messages (append-only; never edit). A sibling of
    # `context_policy`: a pluggable
    # policy bound at construction, read by the loop, defaulting to a no-op
    # empty map. A bare `{HookPoint: [hook, ...]}` dict (like `tools` is a bare
    # tuple). Sub-agents carry their own, so `task_tool` delegation inherits
    # hooks for free. To vary hooks for one invocation without a new agent,
    # `dataclasses.replace(agent, hooks=...)` (same as context_policy).
    hooks: HookMap = field(default_factory=dict)
    # The system prompt `generate` actually sends to the model. Closed over
    # inside `generate` (see `llm_agent.py`), so the loop can't see it on the
    # wire; mirrored here purely so `run()` can record it in the request trace
    # alongside the messages. Empty means "no system prompt was sent".
    system_prompt: str = ""
    # Provider metadata for agents built through `make_llm_agent`. Programmatic
    # facades leave this unset, so their deterministic `generate` calls do not
    # masquerade as model calls in traces.
    llm_provider: Provider | None = None
    # How `run` builds the initial `State` from a task. `None` means the default
    # single-task-message initializer (`_default_init_state`); a `StateInitFn`
    # (e.g. installed by the skills layer) can record extra context messages
    # before the task so they are present at the first sample. The loop (`run`)
    # is unaffected — it always drives whatever `State` the initializer
    # produced.
    init_state: StateInitFn | None = None
    # Optional final trace projection. Workflow facades can run many sub-agents
    # behind one outer `generate`; this lets the agent provide a richer final
    # trace view without changing the message loop or asking callers to know the
    # workflow's internals. None means "trace the run state as-is".
    compose_trace_state: TraceStateFn | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tools, tuple):
            self.tools = tuple(self.tools)

    def run(
        self,
        task: ContentInput,
        *,
        max_turns: int = 10,
        abort: AbortFlag = lambda: False,
    ) -> tuple[State, Iterator[Event]]:
        """Drive this agent on `task` until it emits a final message.

        `task` is `str` or a sequence of content blocks (`ContentInput`), so a
        multimodal task (text + `ImageBlock`) is seeded the same way as plain
        text — the message layer normalizes both to content blocks.

        The initial `State` comes from `self.init_state` when set, else
        `_default_init_state`. A custom initializer is how a layer like skills
        makes a *bare* agent skills-aware: `agent.run(task)` advertises the menu
        and injects bodies because the initializer recorded them — no separate
        run path.

        Returns `(state, events)`. Caller iterates `events` to advance the
        loop and inspects `state` for the message/event history. Callers
        that need to prepend extra messages (e.g. a sub-agent context
        prelude) can `state.record(...)` them after this call and before
        starting to iterate `events`.
        """
        state = (
            self.init_state(self, task)
            if self.init_state is not None
            else self._default_init_state(task)
        )
        events = run(
            self,
            state,
            max_turns=max_turns,
            abort=abort,
        )
        return state, events

    def trace_state(self, state: State) -> State:
        """Return the state that should be used for final trace export."""

        if self.compose_trace_state is None:
            return state
        composed = self.compose_trace_state(self, state)
        return composed if composed is not None else state

    def resume(
        self,
        state: State,
        followup: ContentInput,
        *,
        max_turns: int = 10,
        abort: AbortFlag = lambda: False,
    ) -> tuple[State, Iterator[Event]]:
        """Continue an existing `state` with a follow-up user message.

        Unlike `run`, which seeds a fresh `State`, this appends `followup`
        to the conversation already on `state` and drives the loop over it.
        Both the session (the prior messages stay visible) and the trace
        (events keep accumulating on the same `state.events`) carry over — so
        a `run_trace_from_state(state)` after this spans both turns.

        This is the seam for swapping the agent between turns while keeping
        continuity — e.g. rebuild with a different `reasoning` effort and
        `resume` the same `state`. Build the new agent with the SAME `name`,
        since context visibility/routing is keyed on the agent name; a
        mismatched name would change which prior messages are visible.

        Returns `(state, events)` like `run`, so callers can treat the two
        interchangeably (the returned `state` is the one passed in).
        """
        state.send("task", "user", self.name, followup)
        events = run(
            self,
            state,
            max_turns=max_turns,
            abort=abort,
        )
        return state, events

    def _default_init_state(self, task: ContentInput) -> State:
        """Initialize a fresh `State` with just the task message."""

        state = State(task=task)
        state.send("task", "user", self.name, task)
        return state


def run(
    agent: Agent,
    state: State,
    *,
    max_turns: int = 10,
    abort: AbortFlag = lambda: False,
) -> Iterator[Event]:
    """Run one agent as a generator until it emits `final` or hits `max_turns`.

    Each yielded `Event` is recorded in `state`. Multi-agent flows are
    expressed by giving `agent` a `task_tool` whose sub-agents each call
    their own `run()` inside the tool execute function.

    The agent's `hooks` (an `Agent` field, like `context_policy`) let callers
    observe points in the loop, block a `PRE_TOOL_USE` call, or emit messages —
    append-only, never editing. The default empty map is a no-op, so an agent
    built without hooks runs exactly as before.
    """
    name = agent.name
    tool_by_name = {tool.name: tool for tool in agent.tools}
    hooks = agent.hooks

    def session_hook(point: HookPoint) -> Iterator[Event]:
        _, hook_events = fire_hooks(
            hooks, HookContext(point=point, agent=name, state=state), state
        )
        yield from hook_events

    yield state.record_event(AgentStartEvent())
    yield from session_hook(HookPoint.SESSION_START)
    final_emitted = False
    # Default outcome; overridden when the loop breaks on `final` or terminate.
    end_reason: AgentEndReason = "max_turns"
    for _ in range(max_turns):
        if abort():
            end_reason = "abort"
            break
        yield state.record_event(TurnStartEvent(agent=name))

        policy = agent.context_policy or ContextPolicy()
        for compression_event in maybe_compress_context(agent, state, policy):
            yield compression_event

        context = build_context_view(
            name,
            state.active_context_messages(),
            policy=policy,
        )
        visible = list(context.messages)
        # Match make_llm_agent / provider wire shape (no routing headers).
        # Prepend the agent's system prompt as a leading system message so the
        # recorded request mirrors what actually crosses the wire (adapters
        # send it as the first system entry); without this the trace would
        # silently drop the system prompt that `generate` passes via
        # `LLMRequest.system_prompt`. `visible_count` stays the conversation
        # count; `llm_message_count` reflects the full payload including it.
        llm_payload = messages_to_llm_messages(visible, with_header=False)
        if agent.system_prompt:
            llm_payload = [llm_message("system", agent.system_prompt), *llm_payload]

        if records_model_events(agent):
            api = agent.llm_provider.api if agent.llm_provider is not None else ""
            yield state.record_event(
                ModelRequestEvent(
                    agent=name,
                    api=api,
                    visible_count=len(visible),
                    llm_message_count=len(llm_payload),
                    context_view=context.as_dict(),
                    tools=[
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                        for tool in tool_by_name.values()
                    ],
                    llm_payload=llm_payload,
                )
            )

        output = agent.generate(visible)
        output_tool_calls = message_tool_calls(output)
        if records_model_events(agent):
            api = agent.llm_provider.api if agent.llm_provider is not None else ""
            yield state.record_event(
                ModelResponseEvent(
                    agent=name,
                    api=api,
                    output_kind=output.kind,
                    target=output.target,
                    tool_call_count=len(output_tool_calls),
                    usage=output.usage
                    if isinstance(output, AssistantMessage)
                    else None,
                    model=output.model if isinstance(output, AssistantMessage) else "",
                )
            )

        yield state.record(output)

        if output.sender == name and output.kind == "final":
            final_emitted = True

        if tool_by_name and output_tool_calls:
            tool_terminated = False
            for event in dispatch_tool_calls(
                output, tool_by_name, state, abort=abort, hooks=hooks
            ):
                yield event
                if isinstance(event, ToolExecutionEndEvent) and event.terminate:
                    tool_terminated = True
            if tool_terminated:
                yield state.record_event(TurnEndEvent(agent=name, terminated=True))
                end_reason = "tool_terminate"
                break

        yield state.record_event(TurnEndEvent(agent=name))

        if final_emitted:
            end_reason = "done"
            break

    # Single exit: SESSION_END then agent_end, whatever stopped the loop.
    yield from session_hook(HookPoint.SESSION_END)
    yield state.record_event(AgentEndEvent(reason=end_reason))


def dispatch_tool_calls(
    assistant_msg: Message,
    tools: dict[str, AgentTool],
    state: State,
    *,
    abort: AbortFlag = lambda: False,
    max_concurrency: int = 8,
    hooks: HookMap | None = None,
) -> Iterator[Event]:
    """Run assistant tool calls and append deterministic tool-result messages.

    Before any call reaches the thread pool, each is run through the
    `PRE_TOOL_USE` hook gate — synchronously, in this generator, never in a
    worker thread (hooks emit events and `state.record_event` is not
    thread-safe; the pool runs only `tool.execute`). A hook can **block** a
    call: it never executes, and a synthesized error result is added so the
    model can self-correct next turn, exactly like a tool that raised.
    """
    tool_calls = message_tool_calls(assistant_msg)
    if not tool_calls:
        return

    hooks = hooks or {}
    target = assistant_msg.sender or "agent"

    for tool_call in tool_calls:
        yield state.record_event(
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
            )
        )

    # PRE_TOOL_USE gate (sequential, before the pool). A hook can only block:
    # a blocked call skips the pool and seeds `results` with its error directly
    # (the model self-corrects next turn); the rest go to `effective`.
    effective: list[ToolCallBlock] = []
    results: dict[str, ToolResult] = {}
    for tool_call in tool_calls:
        decision, hook_events = fire_hooks(
            hooks,
            HookContext(
                point=HookPoint.PRE_TOOL_USE,
                agent=target,
                state=state,
                tool_call=tool_call,
            ),
            state,
        )
        yield from hook_events
        if decision.block_reason:
            results[tool_call.id] = text_result(decision.block_reason, is_error=True)
            yield state.record_event(
                ToolExecutionEndEvent(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    is_error=True,
                    terminate=False,
                )
            )
            continue
        effective.append(tool_call)

    sequential = any(
        (tool := tools.get(call.name)) is not None
        and tool.execution_mode == "sequential"
        for call in effective
    )
    workers = 1 if sequential else min(max_concurrency, len(effective))

    update_buffers: dict[str, list[ToolResult]] = {call.id: [] for call in effective}

    def make_on_update(call_id: str) -> ToolUpdateFn:
        def on_update(partial: ToolResult) -> None:
            update_buffers[call_id].append(partial)

        return on_update

    if effective:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_tool_call = {
                pool.submit(
                    _execute_one,
                    call,
                    tools,
                    abort,
                    make_on_update(call.id),
                ): call
                for call in effective
            }

            for future in as_completed(future_to_tool_call):
                call = future_to_tool_call[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = text_result(f"{type(exc).__name__}: {exc}", is_error=True)
                results[call.id] = result

                for partial in update_buffers[call.id]:
                    yield state.record_event(
                        ToolExecutionUpdateEvent(
                            tool_call_id=call.id,
                            tool_name=call.name,
                            partial=partial,
                        )
                    )
                yield state.record_event(
                    ToolExecutionEndEvent(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        is_error=result.is_error,
                        terminate=result.terminate,
                    )
                )

    # `results` now holds every call by its original id (blocked ones seeded in
    # the gate, executed ones filled by the pool). The bundle is built over the
    # original `tool_calls` so each result lands under the id and name the model
    # emitted, whether or not a hook blocked or rewrote the call.
    bundle = tool_results_message(
        [
            ToolResultBlock(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=tuple(results[tool_call.id].content),
                is_error=results[tool_call.id].is_error,
            )
            for tool_call in tool_calls
        ],
        target=target,
        sidecar={
            "details": {
                tool_call.id: results[tool_call.id].details for tool_call in tool_calls
            },
        },
    )
    yield state.record(bundle)

    # POST_TOOL_USE fires only after the provider-required tool-result bundle is
    # in the transcript. Hooks may append reminder/context messages here without
    # orphaning the assistant tool_call -> user tool_result pair.
    for tool_call in tool_calls:
        _, hook_events = fire_hooks(
            hooks,
            HookContext(
                point=HookPoint.POST_TOOL_USE,
                agent=target,
                state=state,
                tool_call=tool_call,
            ),
            state,
        )
        yield from hook_events


def _execute_one(
    tool_call: ToolCallBlock,
    tools: dict[str, AgentTool],
    abort: AbortFlag,
    on_update: ToolUpdateFn | None,
) -> ToolResult:
    tool = tools.get(tool_call.name)
    if tool is None:
        return text_result(f"Tool {tool_call.name!r} not found", is_error=True)

    def run_tool() -> ToolResult:
        try:
            return tool.execute(
                tool_call.id, dict(tool_call.arguments), abort, on_update
            )
        except Exception as exc:
            return text_result(f"{type(exc).__name__}: {exc}", is_error=True)

    if tool.timeout_seconds is None:
        return run_tool()

    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(run_tool)
        try:
            return future.result(timeout=tool.timeout_seconds)
        except FuturesTimeoutError:
            return text_result(
                f"Tool {tool_call.name!r} timed out after {tool.timeout_seconds}s",
                is_error=True,
            )
    finally:
        pool.shutdown(wait=False)
