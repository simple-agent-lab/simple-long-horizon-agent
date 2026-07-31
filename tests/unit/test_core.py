from __future__ import annotations

import unittest
from typing import Any

import simple_long_horizon_agent
from simple_long_horizon_agent import (
    Agent,
    AgentEndEvent,
    CompressionDecision,
    ContextCompressionEvent,
    ContextPolicy,
    EventKind,
    Message,
    ModelRequestEvent,
    ModelResponseEvent,
    State,
    TextBlock,
    TokenUsage,
    ToolCallBlock,
    assistant_message,
    build_context_view,
    make_message,
    message_text,
    run,
    spans_from_events,
    tool_result_message,
)
from simple_long_horizon_agent.compression import (
    _active_context_tokens,
    maybe_compress_context,
)
from simple_long_horizon_agent.llm import Provider
from simple_long_horizon_agent.tools import (
    AbortFlag,
    AgentTool,
    ToolResult,
    ToolUpdateFn,
    task_tool,
    text_result,
    tool_result_text,
)

REAL_PROVIDER = Provider(id="test", api="openai-chat", model="test-model")


def _reply(
    text: str = "done",
    *,
    sender: str = "writer",
    kind: str = "final",
):
    def generate(visible: list[Message]) -> Message:
        del visible
        return assistant_message(text, sender=sender, target="user", kind=kind)

    return generate


def _drain(events: Any) -> None:
    for _ in events:
        pass


def _run(agent: Agent, state: State, **kwargs: Any) -> None:
    _drain(run(agent, state, **kwargs))


def _event(state: State, event_type: type, *, agent: str | None = None) -> Any:
    return next(
        event
        for event in state.events
        if isinstance(event, event_type)
        and (agent is None or getattr(event, "agent", None) == agent)
    )


def _latest_message(state: State, kind: str) -> Message:
    return next(message for message in reversed(state.messages) if message.kind == kind)


def _execute(tool: AgentTool, **args: Any) -> ToolResult:
    return tool.execute("call_1", args, lambda: False, None)


def _tool_state(
    task: str,
    result: str,
    *,
    usage: TokenUsage | None = None,
) -> State:
    state = State(task)
    state.send("task", "user", "writer", task)
    state.record(
        assistant_message(
            [TextBlock("call"), ToolCallBlock("c0", "read", {})],
            sender="writer",
            target="user",
            kind="step",
        )
    )
    state.record(
        tool_result_message(
            result,
            tool_call_id="c0",
            tool_name="read",
            target="writer",
        )
    )
    if usage:
        state.record(
            assistant_message(
                "answer",
                sender="writer",
                target="user",
                kind="step",
                usage=usage,
            )
        )
    return state


def _rewrite_strategy(
    replacement: Message,
    indices: tuple[int, ...] = (2,),
):
    def strategy(
        active: list[tuple[int, Message]], agent_name: str
    ) -> CompressionDecision:
        del active, agent_name
        return CompressionDecision(
            compress_indices=indices,
            replacement=replacement,
            rewrite=True,
        )

    return strategy


def _idle_writer(visible: list[Message]) -> Message:
    """A step fn that compression tests never actually invoke."""
    del visible
    return assistant_message("idle", sender="writer", target="user", kind="final")


class CoreTest(unittest.TestCase):
    def test_records_request_and_response_events(self) -> None:
        state = State("write one sentence")
        state.send("task", "user", "writer", state.task)
        _run(
            Agent(
                "writer",
                _reply(),
                role="Write one sentence.",
                llm_provider=REAL_PROVIDER,
            ),
            state,
        )

        self.assertIs(state.events[0].kind, EventKind.MESSAGE)
        self.assertIs(state.events[-1].kind, EventKind.AGENT_END)
        kinds = [event.kind for event in state.events]
        self.assertEqual(message_text(_latest_message(state, "final")), "done")
        self.assertIn("model_request", kinds)
        self.assertIn("model_response", kinds)
        self.assertEqual(_event(state, ModelRequestEvent).api, "openai-chat")
        self.assertEqual(_event(state, ModelResponseEvent).api, "openai-chat")

    def test_model_events_use_provider_api_tag(self) -> None:
        cases = [
            ("programmatic", None, ""),
            ("fake", Provider(id="fake", api="fake", model="fake-model"), "fake"),
        ]
        for label, provider, expected_api in cases:
            with self.subTest(label):
                state = State("task")
                state.send("task", "user", "writer", state.task)
                _run(Agent("writer", _reply(), llm_provider=provider), state)
                self.assertEqual(_event(state, ModelRequestEvent).api, expected_api)
                self.assertEqual(_event(state, ModelResponseEvent).api, expected_api)

    def test_run_accepts_a_multimodal_content_list_task(self) -> None:
        """`Agent.run` takes `str` or content blocks; a text+image task is
        recorded as one user message carrying both blocks."""
        from simple_long_horizon_agent.messages import ImageBlock

        task = [
            TextBlock("Describe this screenshot:"),
            ImageBlock(data="QUJD", mime_type="image/png"),
        ]
        agent = Agent("writer", _reply(), role="Describe.")
        state, events = agent.run(task, max_turns=2)
        _drain(events)

        blocks = state.messages[0].content
        self.assertEqual(len(blocks), 2)
        self.assertIsInstance(blocks[0], TextBlock)
        self.assertIsInstance(blocks[1], ImageBlock)

    def test_dispatches_agent_tool_result_back_to_agent(self) -> None:
        def coordinator(visible: list[Message]) -> Message:
            if any(message.kind == "tool_result" for message in visible):
                result = next(
                    message
                    for message in reversed(visible)
                    if message.kind == "tool_result"
                )
                return assistant_message(
                    f"final: {message_text(result)}",
                    sender="coordinator",
                    target="user",
                    kind="final",
                )
            return assistant_message(
                [
                    TextBlock("asking tool"),
                    ToolCallBlock("call_1", "echo", {"text": "child ok"}),
                ],
                sender="coordinator",
                target="coordinator",
                kind="step",
            )

        def echo_tool(
            call_id: str,
            args: dict[str, Any],
            abort: AbortFlag,
            on_update: ToolUpdateFn | None,
        ) -> ToolResult:
            del call_id, abort, on_update
            return text_result(str(args["text"]))

        state = State("delegate")
        state.send("task", "user", "coordinator", state.task)
        echo = AgentTool(
            name="echo",
            description="Echo text.",
            parameters={"type": "object"},
            execute=echo_tool,
            execution_mode="sequential",
        )
        _run(
            Agent(
                "coordinator",
                coordinator,
                role="Coordinate tool use.",
                tools=(echo,),
            ),
            state,
            max_turns=2,
        )

        self.assertEqual(
            message_text(_latest_message(state, "tool_result")), "child ok"
        )
        self.assertEqual(
            message_text(_latest_message(state, "final")),
            "final: child ok",
        )
        self.assertTrue(
            any(event.kind == "tool_execution_start" for event in state.events)
        )
        self.assertTrue(
            any(event.kind == "tool_execution_end" for event in state.events)
        )

    def test_agent_run_drives_loop_and_exposes_state(self) -> None:
        agent = Agent("writer", _reply())
        state, events = agent.run("write something")
        _drain(events)

        self.assertEqual(message_text(_latest_message(state, "final")), "done")

    def test_resume_continues_session_and_trace_with_a_rebuilt_agent(self) -> None:
        from simple_long_horizon_agent import AgentStartEvent

        seen: dict[str, Any] = {}

        def second(visible: list[Message]) -> Message:
            seen["texts"] = [message_text(m) for m in visible]
            return assistant_message("two", sender="w", target="user", kind="final")

        agent_a = Agent("w", _reply("one", sender="w"))
        state, events = agent_a.run("first task")
        _drain(events)

        agent_b = Agent("w", second)
        returned_state, events2 = agent_b.resume(state, "follow up")
        _drain(events2)

        self.assertIs(returned_state, state)
        self.assertIn("one", seen["texts"])
        self.assertIn("follow up", seen["texts"])
        starts = [e for e in state.events if isinstance(e, AgentStartEvent)]
        self.assertEqual(len(starts), 2)
        finals = [m for m in state.messages if m.kind == "final"]
        self.assertEqual([message_text(m) for m in finals], ["one", "two"])

    def test_init_state_hook_builds_the_initial_state(self) -> None:
        def init_state(agent: Agent, task: Any) -> State:
            state = State(task=task)
            state.send("context", "primer", agent.name, "PRELUDE")
            state.send("task", "user", agent.name, task)
            return state

        agent = Agent("stateful", _reply(sender="stateful"), init_state=init_state)
        state, events = agent.run("real task")
        _drain(events)

        self.assertTrue(any(m.sender == "primer" for m in state.messages))
        self.assertEqual(
            [message_text(m) for m in state.messages if m.kind != "final"][:2],
            ["PRELUDE", "real task"],
        )

    def test_default_init_state_records_only_the_task(self) -> None:
        agent = Agent("plain", _reply("ok", sender="plain"))
        state, events = agent.run("just the task")
        _drain(events)

        non_final = [m for m in state.messages if m.kind != "final"]
        self.assertEqual(len(non_final), 1)
        self.assertEqual(message_text(non_final[0]), "just the task")

    def test_max_turns_exhausted_reports_truncation_in_agent_end(self) -> None:
        state = State("ramble")
        state.send("task", "user", "chatty", state.task)
        _run(
            Agent("chatty", _reply("still thinking", sender="chatty", kind="step")),
            state,
            max_turns=2,
        )

        end = next(
            event
            for event in reversed(state.events)
            if isinstance(event, AgentEndEvent)
        )
        self.assertEqual(end.reason, "max_turns")

    def test_task_tool_dispatches_to_named_subagent(self) -> None:
        def make_prefix_generate(name: str, prefix: str):
            def generate(visible: list[Message]) -> Message:
                task_msg = next(
                    message for message in visible if message.kind == "task"
                )
                return assistant_message(
                    f"{prefix}:{message_text(task_msg)}",
                    sender=name,
                    target="user",
                    kind="final",
                )

            return generate

        echoer = Agent(
            "echoer", make_prefix_generate("echoer", "echo"), role="Echo back the task."
        )
        shouter = Agent(
            "shouter", make_prefix_generate("shouter", "SHOUT"), role="Shout the task."
        )
        tool = task_tool([echoer, shouter])

        self.assertEqual(tool.name, "task")
        self.assertEqual(tool.parameters["required"], ["subagent_type", "task"])
        self.assertIn("context", tool.parameters["properties"])
        self.assertEqual(
            tool.parameters["properties"]["subagent_type"]["enum"],
            ["echoer", "shouter"],
        )
        self.assertIn("echoer: Echo back the task.", tool.description)
        self.assertIn("shouter: Shout the task.", tool.description)

        result = _execute(tool, subagent_type="shouter", task="hi")
        self.assertFalse(result.is_error)
        self.assertEqual(
            "\n".join(b.text for b in result.content if hasattr(b, "text")),
            "SHOUT:hi",
        )

    def test_task_tool_passes_default_and_call_context_to_subagent(self) -> None:
        def reader(visible: list[Message]) -> Message:
            context = [
                message_text(message)
                for message in visible
                if message.kind == "context"
            ]
            task_msg = next(message for message in visible if message.kind == "task")
            return assistant_message(
                "\n".join([*context, message_text(task_msg)]),
                sender="reader",
                target="user",
                kind="final",
            )

        tool = task_tool(
            [Agent("reader", reader, role="Read delegated task context.")],
            default_context="Repository: Simple Long Horizon Agent.",
        )

        result = _execute(
            tool,
            subagent_type="reader",
            task="Write the summary.",
            context="Caller: include the feedback signal.",
        )

        self.assertFalse(result.is_error)
        self.assertEqual(
            tool_result_text(result),
            "\n".join(
                [
                    "Repository: Simple Long Horizon Agent.",
                    "Caller: include the feedback signal.",
                    "Write the summary.",
                ]
            ),
        )

    def test_task_tool_reports_unknown_subagent_as_tool_error(self) -> None:
        tool = task_tool([Agent("only", _reply("ok", sender="only"))])
        result = _execute(tool, subagent_type="missing", task="hi")
        self.assertTrue(result.is_error)
        text = "\n".join(b.text for b in result.content if hasattr(b, "text"))
        self.assertIn("Unknown subagent_type", text)
        self.assertIn("'only'", text)

    def test_public_api_surface_is_a_known_set(self) -> None:
        expected = {
            "AbortFlag",
            "Agent",
            "AgentEndEvent",
            "AgentName",
            "AgentStartEvent",
            "AgentTool",
            "AssistantMessage",
            "CompactControl",
            "CompressionDecision",
            "CompressionStrategy",
            "ContentBlock",
            "ContextCompressionEvent",
            "ContextPolicy",
            "ContextView",
            "ContextWindowBook",
            "CostBreakdown",
            "Event",
            "EventKind",
            "HookContext",
            "HookDecision",
            "HookFiredEvent",
            "HookPoint",
            "ImageBlock",
            "Message",
            "MessageContent",
            "MessageEvent",
            "MessageKind",
            "MessageSidecar",
            "ModelCost",
            "ModelPrice",
            "ModelRequestEvent",
            "ModelResponseEvent",
            "ModelTurn",
            "PriceBook",
            "Role",
            "RunCost",
            "RunTrace",
            "SkillMetadata",
            "SkillRoot",
            "Span",
            "State",
            "SummarizeStrategy",
            "RuntimeMessage",
            "TextBlock",
            "ThinkingBlock",
            "TieredStrategy",
            "TokenUsage",
            "Tool",
            "ToolCallBlock",
            "ToolExecutionEndEvent",
            "ToolExecutionMode",
            "ToolExecutionStartEvent",
            "ToolExecutionUpdateEvent",
            "ToolResult",
            "ToolResultBlock",
            "ToolUpdateFn",
            "ToolCompactStrategy",
            "TurnEndEvent",
            "TurnStartEvent",
            "UserMessage",
            "append_openai_training_record",
            "assistant_message",
            "build_context_view",
            "default_context_window_book",
            "default_price_book",
            "discover_skills",
            "effective_token_budget",
            "estimate_context_tokens",
            "estimate_message_chars",
            "estimate_message_tokens",
            "event_record",
            "is_tool_result_message",
            "make_compact_control",
            "make_edit_tool",
            "make_llm_agent",
            "make_message",
            "make_read_tool",
            "make_recall_tool",
            "message_text",
            "model_turns_from_events",
            "openai_training_record",
            "print_trace",
            "render_skills_instructions",
            "run",
            "run_trace_from_state",
            "run_with_skills",
            "spans_from_events",
            "runtime_message",
            "task_tool",
            "text_of",
            "text_result",
            "tool_result_message",
            "tool_result_text",
            "tool_results_message",
            "tool_results_of",
            "usage_cost",
            "user_message",
        }
        self.assertEqual(set(simple_long_horizon_agent.__all__), expected)
        self.assertEqual(len(simple_long_horizon_agent.__all__), len(expected))

    def test_context_view_uses_single_agent_transcript(self) -> None:
        state = State("route test")
        state.send("task", "user", "writer", "visible task")
        state.send("message", "planner", "planner", "planner note")
        state.send("message", "planner", "all", "broadcast")
        state.send("summary", "runtime", "writer", "old summary")
        state.send("message", "planner", "writer", "debug note")

        policy = ContextPolicy(model_invisible_kinds=("summary",))
        view = build_context_view(
            "writer", state.active_context_messages(), policy=policy
        )

        self.assertEqual(
            [message_text(message) for message in view.messages],
            ["visible task", "planner note", "broadcast", "debug note"],
        )
        self.assertEqual(view.stats.total_messages, 5)
        self.assertEqual(view.stats.visible_messages, 4)

    def test_run_records_context_view_summary(self) -> None:
        state = State("project context view")
        state.send("task", "user", "writer", state.task)
        state.send("message", "user", "writer", "first note")
        state.send("message", "user", "writer", "second note")
        _run(
            Agent("writer", _reply(), role="Write.", llm_provider=REAL_PROVIDER),
            state,
        )

        context = _event(state, ModelRequestEvent).context_view
        self.assertEqual(context["agent"], "writer")
        self.assertGreater(context["estimated_tokens"], 0)
        self.assertEqual(context["visible_messages"], context["total_messages"])

    def test_run_compresses_context_before_model_request(self) -> None:
        captured: dict[str, Any] = {}

        def writer(visible: list[Message]) -> Message:
            captured["writer_visible_texts"] = [
                message_text(message) for message in visible
            ]
            return assistant_message(
                "done",
                sender="writer",
                target="user",
                kind="final",
            )

        def compressor(visible: list[Message]) -> Message:
            captured["compressor_prompt"] = message_text(visible[0])
            return assistant_message(
                "compressed old context",
                sender="compressor",
                target="runtime",
                kind="final",
                usage=TokenUsage(input_tokens=123, output_tokens=7),
                model="compressor-model",
                sidecar={"raw": {"request": {"model": "compressor-model"}}},
            )

        state = State("compress context")
        state.send("task", "user", "writer", state.task)
        state.send("message", "user", "writer", "old " + ("x" * 120))
        state.send("message", "user", "writer", "recent note")
        compression_policy = ContextPolicy(
            strategy=simple_long_horizon_agent.SummarizeStrategy(
                compressor=Agent(
                    "compressor",
                    compressor,
                    llm_provider=REAL_PROVIDER,
                ),
                threshold_tokens=1,
                keep_recent=1,
            ),
        )

        _run(
            Agent(
                "writer",
                writer,
                role="Write.",
                context_policy=compression_policy,
                llm_provider=REAL_PROVIDER,
            ),
            state,
        )

        summaries = [message for message in state.messages if message.kind == "summary"]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].role, "user")
        summary_text = simple_long_horizon_agent.text_of(summaries[0].content)
        self.assertTrue(summary_text.startswith("[This session continues"))
        self.assertIn("compressed old context", summary_text)
        self.assertEqual(
            summaries[0].sidecar["compression"]["model"],
            "compressor-model",
        )
        self.assertEqual(
            summaries[0].sidecar["compression"]["usage"]["input_tokens"],
            123,
        )
        self.assertEqual(
            summaries[0].sidecar["raw"]["request"]["model"],
            "compressor-model",
        )
        compressor_request = _event(state, ModelRequestEvent, agent="compressor")
        self.assertEqual(
            compressor_request.context_view["agent"],
            "compressor",
        )
        compressor_response = _event(state, ModelResponseEvent, agent="compressor")
        self.assertEqual(compressor_response.model, "compressor-model")
        self.assertEqual(compressor_response.usage.input_tokens, 123)
        self.assertGreater(
            compressor_response.elapsed,
            compressor_request.elapsed,
        )
        writer_request = _event(state, ModelRequestEvent, agent="writer")
        summary_payloads = [
            payload
            for payload in writer_request.llm_payload
            if "compressed old context"
            in simple_long_horizon_agent.text_of(payload.content)
        ]
        self.assertEqual(len(summary_payloads), 1)
        self.assertEqual(summary_payloads[0].role, "user")
        compression = _event(state, ContextCompressionEvent)
        self.assertLess(compression.start_elapsed, compression.elapsed)
        compression_span = next(
            span
            for span in spans_from_events("trace", state.events)
            if span.kind == "compression"
        )
        self.assertLess(compression_span.start, compression_span.end)
        self.assertEqual(compression.compressed_message_indices, [1])
        self.assertEqual(
            state.snapshot.active_context_indices,
            compression.active_context_indices + [len(state.messages) - 1],
        )
        self.assertIn("old", captured["compressor_prompt"])
        writer_texts = captured["writer_visible_texts"]
        self.assertEqual(len(writer_texts), 3)
        self.assertEqual(writer_texts[0], "compress context")
        self.assertTrue(writer_texts[1].startswith("[This session continues"))
        self.assertEqual(writer_texts[2], "recent note")
        next_view = build_context_view("writer", state.active_context_messages())
        self.assertNotIn(
            "old " + ("x" * 116),
            [message_text(message) for message in next_view.messages],
        )
        rebuilt = state.rebuild_snapshot()
        self.assertEqual(
            rebuilt.active_context_indices,
            state.snapshot.active_context_indices,
        )

    def test_compression_span_falls_back_to_prior_compressor_call(self) -> None:
        events = [
            ModelRequestEvent(
                index=0,
                elapsed=1.0,
                agent="context_compressor",
                visible_count=1,
                llm_message_count=1,
                context_view={},
                tools=[],
                llm_payload=[],
            ),
            ModelResponseEvent(
                index=1,
                elapsed=4.0,
                agent="context_compressor",
                output_kind="final",
                target="runtime",
                tool_call_count=0,
                model="compressor-model",
            ),
            ContextCompressionEvent(
                index=2,
                elapsed=4.1,
                agent="writer",
                summary_message_index=3,
                compressed_message_indices=[1, 2],
                active_context_indices=[0, 3],
                before_tokens=4000,
                after_tokens=2000,
                strategy="summarize",
            ),
        ]

        compression_span = next(
            span
            for span in spans_from_events("trace", events)
            if span.kind == "compression"
        )
        self.assertEqual(compression_span.start, 1.0)
        self.assertEqual(compression_span.end, 4.1)

    def test_tiered_strategy_returns_first_applicable_decision(self) -> None:
        calls: list[str] = []

        def make_stage(name: str, decision: CompressionDecision | None):
            def stage(active: list[tuple[int, Message]], agent_name: str):
                del active, agent_name
                calls.append(name)
                return decision

            return stage

        decision = CompressionDecision(
            compress_indices=(0,),
            replacement=make_message("user", "x", kind="summary"),
        )

        tiered = simple_long_horizon_agent.TieredStrategy(
            (make_stage("a", None), make_stage("b", decision))
        )
        self.assertIs(tiered([], "w"), decision)
        self.assertEqual(calls, ["a", "b"])

        calls.clear()
        first_wins = simple_long_horizon_agent.TieredStrategy(
            (make_stage("a", decision), make_stage("b", None))
        )
        self.assertIs(first_wins([], "w"), decision)
        self.assertEqual(calls, ["a"])

        self.assertIsNone(
            simple_long_horizon_agent.TieredStrategy((make_stage("a", None),))([], "w")
        )

    def test_tool_compact_folds_old_tool_exchanges(self) -> None:
        state = State("tool compact")
        state.send("task", "user", "writer", state.task)
        for index, payload in enumerate(
            ("alpha result", "beta result", "gamma result")
        ):
            state.record(
                assistant_message(
                    [
                        TextBlock(f"call-{index}"),
                        ToolCallBlock(f"c{index}", "echo", {"i": index}),
                    ],
                    sender="writer",
                    target="user",
                    kind="step",
                )
            )
            state.record(
                tool_result_message(
                    payload,
                    tool_call_id=f"c{index}",
                    tool_name="echo",
                    target="writer",
                )
            )

        policy = ContextPolicy(
            strategy=simple_long_horizon_agent.ToolCompactStrategy(
                threshold_tokens=1,
                keep_recent_exchanges=1,
            ),
        )
        _run(
            Agent("writer", _reply(), context_policy=policy),
            state,
            max_turns=1,
        )

        compression = _event(state, ContextCompressionEvent)
        self.assertEqual(len(compression.compressed_message_indices), 4)
        replacement = state.messages[compression.summary_message_index]
        self.assertEqual(replacement.kind, "summary")
        self.assertEqual(replacement.role, "user")
        replacement_text = message_text(replacement)
        self.assertIn("Compacted 2 older tool exchange(s)", replacement_text)
        self.assertIn("alpha result", replacement_text)
        self.assertIn("beta result", replacement_text)
        self.assertNotIn("gamma result", replacement_text)
        post_summary_indices = compression.active_context_indices[
            compression.active_context_indices.index(compression.summary_message_index)
            + 1 :
        ]
        recent_texts = [
            message_text(state.messages[index]) for index in post_summary_indices
        ]
        self.assertTrue(any("gamma result" in text for text in recent_texts))

    def test_after_tokens_ignores_stale_pre_compression_usage_baseline(self) -> None:
        def compressor(visible: list[Message]) -> Message:
            del visible
            return assistant_message(
                "short summary", sender="compressor", target="runtime", kind="final"
            )

        state = State("task")
        state.send("task", "user", "writer", state.task)
        state.send("message", "user", "writer", "old context " + ("x" * 400))
        state.record(
            assistant_message(
                "recent answer",
                sender="writer",
                target="user",
                kind="step",
                usage=TokenUsage(input_tokens=6000, output_tokens=120),
            )
        )
        state.send("message", "user", "writer", "newest follow-up")

        policy = ContextPolicy(
            strategy=simple_long_horizon_agent.SummarizeStrategy(
                compressor=Agent("compressor", compressor),
                threshold_tokens=50,
                keep_recent=2,
            ),
        )
        _run(Agent("writer", _reply(), context_policy=policy), state, max_turns=1)

        compression = _event(state, ContextCompressionEvent)
        self.assertLess(compression.after_tokens, 1000)
        self.assertLess(compression.after_tokens, compression.before_tokens)

    def test_rewrite_substitutes_message_in_place(self) -> None:
        state = _tool_state("rewrite", "HUGE " + "x" * 500)
        events = maybe_compress_context(
            Agent("writer", _idle_writer),
            state,
            ContextPolicy(
                strategy=_rewrite_strategy(
                    tool_result_message(
                        "shrunk",
                        tool_call_id="c0",
                        tool_name="read",
                        target="writer",
                    )
                )
            ),
        )

        rewrite = next(e for e in events if isinstance(e, ContextCompressionEvent))
        self.assertEqual(rewrite.compressed_message_indices, [2])
        self.assertEqual(rewrite.summary_message_index, 3)
        self.assertLess(rewrite.after_tokens, rewrite.before_tokens)
        self.assertEqual(state.snapshot.active_context_indices, [0, 1, 3])
        self.assertEqual(
            [message_text(m) for m in state.active_context_messages()],
            ["rewrite", "call", "shrunk"],
        )
        self.assertIn("HUGE", message_text(state.messages[2]))
        self.assertEqual(state.messages[3].sidecar, {})
        self.assertEqual(message_text(state.messages[3]), "shrunk")

    def test_rewrite_invalidates_stale_usage_baseline(self) -> None:
        state = _tool_state(
            "stale",
            "HUGE " + "x" * 800,
            usage=TokenUsage(input_tokens=6000, output_tokens=120),
        )
        maybe_compress_context(
            Agent("writer", _idle_writer),
            state,
            ContextPolicy(
                strategy=_rewrite_strategy(
                    tool_result_message(
                        "short",
                        tool_call_id="c0",
                        tool_name="read",
                        target="writer",
                    )
                )
            ),
        )
        self.assertLess(_active_context_tokens(state.active_context_items()), 1000)

    def test_rewrite_rejects_structure_changing_replacement(self) -> None:
        replacements = {
            "tool call id": tool_result_message(
                "x",
                tool_call_id="OTHER",
                tool_name="read",
                target="writer",
            ),
            "role": assistant_message(
                "x",
                sender="writer",
                target="user",
                kind="step",
            ),
        }
        for label, replacement in replacements.items():
            with self.subTest(label), self.assertRaises(ValueError):
                maybe_compress_context(
                    Agent("writer", _idle_writer),
                    _tool_state("reject", "big"),
                    ContextPolicy(strategy=_rewrite_strategy(replacement)),
                )

    def test_rewrite_requires_a_single_target(self) -> None:
        with self.assertRaises(ValueError):
            maybe_compress_context(
                Agent("writer", _idle_writer),
                _tool_state("multi", "big"),
                ContextPolicy(
                    strategy=_rewrite_strategy(
                        tool_result_message(
                            "x",
                            tool_call_id="c0",
                            tool_name="read",
                            target="writer",
                        ),
                        (1, 2),
                    )
                ),
            )

    def test_framework_auto_fixes_split_tool_pair(self) -> None:
        state = State("tool pair fixup")
        state.send("task", "user", "writer", state.task)
        call_msg = assistant_message(
            [
                TextBlock("calling"),
                ToolCallBlock("call_x", "echo", {"text": "hi"}),
            ],
            sender="writer",
            target="user",
            kind="step",
        )
        state.record(call_msg)
        state.record(
            tool_result_message(
                "ok",
                tool_call_id="call_x",
                tool_name="echo",
                target="writer",
            )
        )

        call_index = state.messages.index(call_msg)
        captured: dict[str, Any] = {}

        class CompressOnlyCall:
            def __call__(self, active, agent_name):
                captured["saw_active_count"] = len(active)
                return CompressionDecision(
                    compress_indices=(call_index,),
                    replacement=make_message(
                        "system",
                        "[call gone]",
                        sender="runtime",
                        target=agent_name,
                        kind="summary",
                    ),
                )

        def writer(visible: list[Message]) -> Message:
            return assistant_message(
                f"saw {len(visible)} msgs",
                sender="writer",
                target="user",
                kind="final",
            )

        policy = ContextPolicy(strategy=CompressOnlyCall())
        for _ in run(Agent("writer", writer, context_policy=policy), state):
            pass

        compressions = [
            event
            for event in state.events
            if isinstance(event, ContextCompressionEvent)
        ]
        self.assertEqual(compressions, [])
