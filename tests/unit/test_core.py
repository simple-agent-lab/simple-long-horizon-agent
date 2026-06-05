from __future__ import annotations

import unittest
from typing import Any

import simple_agent_lab
from simple_agent_lab import (
    Agent,
    AgentEndEvent,
    CompressionDecision,
    ContextCompressionEvent,
    ContextPolicy,
    EventKind,
    Message,
    ModelRequestEvent,
    State,
    TextBlock,
    TokenUsage,
    ToolCallBlock,
    assistant_message,
    build_context_view,
    make_message,
    message_text,
    run,
    tool_result_message,
)
from simple_agent_lab.compression import (
    _active_context_tokens,
    maybe_compress_context,
)
from simple_agent_lab.tools import (
    AbortFlag,
    AgentTool,
    ToolResult,
    ToolUpdateFn,
    task_tool,
    text_result,
    tool_result_text,
)


def _idle_writer(visible: list[Message]) -> Message:
    """A step fn that compression tests never actually invoke."""
    del visible
    return assistant_message("idle", sender="writer", target="user", kind="final")


class CoreTest(unittest.TestCase):
    def test_records_request_and_response_events(self) -> None:
        def writer(visible: list[Message]) -> Message:
            del visible
            return assistant_message(
                "done",
                sender="writer",
                target="user",
                kind="final",
            )

        state = State("write one sentence")
        state.send("task", "user", "writer", state.task)
        for _ in run(
            Agent("writer", writer, role="Write one sentence."),
            state,
        ):
            pass

        self.assertIs(state.events[0].kind, EventKind.MESSAGE)
        self.assertIs(state.events[-1].kind, EventKind.AGENT_END)
        final = next(
            message for message in reversed(state.messages) if message.kind == "final"
        )
        self.assertEqual(message_text(final), "done")
        self.assertIn("model_request", [event.kind for event in state.events])
        self.assertIn("model_response", [event.kind for event in state.events])
        self.assertEqual(state.events[-1].kind, "agent_end")

    def test_run_accepts_a_multimodal_content_list_task(self) -> None:
        """`Agent.run` takes `str` or content blocks; a text+image task is seeded
        as one user message carrying both blocks (the multimodal task path)."""
        from simple_agent_lab.messages import ImageBlock

        def writer(visible: list[Message]) -> Message:
            del visible
            return assistant_message(
                "done", sender="writer", target="user", kind="final"
            )

        task = [
            TextBlock("Describe this screenshot:"),
            ImageBlock(data="QUJD", mime_type="image/png"),
        ]
        agent = Agent("writer", writer, role="Describe.")
        state, events = agent.run(task, max_turns=2)
        for _ in events:
            pass

        # The seeded task message preserves both content blocks (text + image).
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
        for _ in run(
            Agent(
                "coordinator",
                coordinator,
                role="Coordinate tool use.",
                tools=(echo,),
            ),
            state,
            max_turns=2,
        ):
            pass

        self.assertEqual(
            message_text(
                next(
                    message
                    for message in reversed(state.messages)
                    if message.kind == "tool_result"
                )
            ),
            "child ok",
        )
        self.assertEqual(
            message_text(
                next(
                    message
                    for message in reversed(state.messages)
                    if message.kind == "final"
                )
            ),
            "final: child ok",
        )
        self.assertTrue(
            any(event.kind == "tool_execution_start" for event in state.events)
        )
        self.assertTrue(
            any(event.kind == "tool_execution_end" for event in state.events)
        )

    def test_agent_run_drives_loop_and_exposes_state(self) -> None:
        def writer(visible: list[Message]) -> Message:
            del visible
            return assistant_message(
                "done", sender="writer", target="user", kind="final"
            )

        agent = Agent("writer", writer)
        state, events = agent.run("write something")
        for _ in events:
            pass

        final = next(message for message in state.messages if message.kind == "final")
        self.assertEqual(message_text(final), "done")

    def test_seed_hook_builds_the_initial_state(self) -> None:
        # A `seed` callable lets a higher layer (e.g. skills) record extra
        # context messages before the task without touching `run`. Default
        # `run` seeds only the task message; a seed can prepend its own.
        def writer(visible: list[Message]) -> Message:
            del visible
            return assistant_message(
                "done", sender="seeded", target="user", kind="final"
            )

        def seed(agent: Agent, task: Any) -> State:
            state = State(task=task)
            state.send("context", "primer", agent.name, "PRELUDE")
            state.send("task", "user", agent.name, task)
            return state

        agent = Agent("seeded", writer, seed=seed)
        state, events = agent.run("real task")
        for _ in events:
            pass

        self.assertTrue(any(m.sender == "primer" for m in state.messages))
        self.assertEqual(
            [message_text(m) for m in state.messages if m.kind != "final"][:2],
            ["PRELUDE", "real task"],
        )

    def test_default_seed_records_only_the_task(self) -> None:
        # With no seed, `run` keeps its original behavior: one task message.
        def writer(visible: list[Message]) -> Message:
            del visible
            return assistant_message("ok", sender="plain", target="user", kind="final")

        agent = Agent("plain", writer)
        state, events = agent.run("just the task")
        for _ in events:
            pass

        non_final = [m for m in state.messages if m.kind != "final"]
        self.assertEqual(len(non_final), 1)
        self.assertEqual(message_text(non_final[0]), "just the task")

    def test_max_turns_exhausted_reports_truncation_in_agent_end(self) -> None:
        # If the agent never emits `final`, the run was truncated by the
        # turn budget. The trace must say so — otherwise downstream analysis
        # cannot tell "agent decided to stop" from "ran out of turns".
        def chatty(visible: list[Message]) -> Message:
            del visible
            return assistant_message(
                "still thinking",
                sender="chatty",
                target="user",
                kind="step",
            )

        state = State("ramble")
        state.send("task", "user", "chatty", state.task)
        for _ in run(Agent("chatty", chatty), state, max_turns=2):
            pass

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

        result = tool.execute(
            "call_1",
            {"subagent_type": "shouter", "task": "hi"},
            lambda: False,
            None,
        )
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
            default_context="Repository: Simple Agent Lab.",
        )

        result = tool.execute(
            "call_1",
            {
                "subagent_type": "reader",
                "task": "Write the summary.",
                "context": "Caller: include the feedback signal.",
            },
            lambda: False,
            None,
        )

        self.assertFalse(result.is_error)
        self.assertEqual(
            tool_result_text(result),
            "\n".join(
                [
                    "Repository: Simple Agent Lab.",
                    "Caller: include the feedback signal.",
                    "Write the summary.",
                ]
            ),
        )

    def test_task_tool_reports_unknown_subagent_as_tool_error(self) -> None:
        def noop(visible: list[Message]) -> Message:
            del visible
            return assistant_message("ok", sender="only", target="user", kind="final")

        tool = task_tool([Agent("only", noop)])
        result = tool.execute(
            "call_1",
            {"subagent_type": "missing", "task": "hi"},
            lambda: False,
            None,
        )
        self.assertTrue(result.is_error)
        text = "\n".join(b.text for b in result.content if hasattr(b, "text"))
        self.assertIn("Unknown subagent_type", text)
        self.assertIn("'only'", text)

    def test_public_api_surface_is_a_known_set(self) -> None:
        # Pin the public `__all__` to a known set so adding or dropping a
        # symbol is an explicit, reviewable diff rather than silent drift.
        # Catches both regressions (re-exposing removed APIs like StepFn /
        # NextFn / until_final / make_llm_step) and accidental new exports.
        expected = {
            "AbortFlag",
            "Agent",
            "AgentEndEvent",
            "AgentName",
            "AgentStartEvent",
            "AgentTool",
            "AssistantMessage",
            "CompressionDecision",
            "CompressionStrategy",
            "ContentBlock",
            "ContextCompressionEvent",
            "ContextPolicy",
            "ContextView",
            "Event",
            "EventKind",
            "ImageBlock",
            "Message",
            "MessageContent",
            "MessageEvent",
            "MessageKind",
            "MessageSidecar",
            "ModelRequestEvent",
            "ModelResponseEvent",
            "ModelTurn",
            "Role",
            "RunTrace",
            "SkillMetadata",
            "SkillRoot",
            "Span",
            "State",
            "SummarizeStrategy",
            "SystemMessage",
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
            "discover_skills",
            "effective_token_budget",
            "estimate_context_tokens",
            "estimate_message_chars",
            "estimate_message_tokens",
            "event_record",
            "is_tool_result_message",
            "make_edit_tool",
            "make_llm_agent",
            "make_message",
            "make_read_tool",
            "message_text",
            "model_turns_from_events",
            "openai_training_record",
            "print_trace",
            "render_skills_instructions",
            "run",
            "run_trace_from_state",
            "run_with_skills",
            "spans_from_events",
            "system_message",
            "task_tool",
            "text_of",
            "text_result",
            "tool_result_message",
            "tool_result_text",
            "tool_results_message",
            "tool_results_of",
            "user_message",
        }
        self.assertEqual(set(simple_agent_lab.__all__), expected)
        # `__all__` should also have no duplicates.
        self.assertEqual(
            len(simple_agent_lab.__all__), len(set(simple_agent_lab.__all__))
        )

    def test_context_view_uses_single_agent_transcript(self) -> None:
        state = State("route test")
        state.send("task", "user", "writer", "visible task")
        state.send("message", "planner", "planner", "planner note")
        state.send("message", "planner", "all", "broadcast")
        state.send("summary", "runtime", "writer", "old summary")
        state.send("message", "planner", "writer", "debug note")

        # Routing fields (sender/target) never filter the view; only
        # `model_invisible_kinds` does. Skipping "summary" hides exactly
        # that one message.
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
        def writer(visible: list[Message]) -> Message:
            del visible
            return assistant_message(
                "done",
                sender="writer",
                target="user",
                kind="final",
            )

        state = State("project context view")
        state.send("task", "user", "writer", state.task)
        state.send("message", "user", "writer", "first note")
        state.send("message", "user", "writer", "second note")
        for _ in run(
            Agent("writer", writer, role="Write."),
            state,
        ):
            pass

        request = next(
            event
            for event in reversed(state.events)
            if isinstance(event, ModelRequestEvent)
        )
        context = request.context_view
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
            )

        state = State("compress context")
        state.send("task", "user", "writer", state.task)
        state.send("message", "user", "writer", "old " + ("x" * 120))
        state.send("message", "user", "writer", "recent note")
        compression_policy = ContextPolicy(
            strategy=simple_agent_lab.SummarizeStrategy(
                compressor=Agent("compressor", compressor),
                threshold_tokens=1,
                keep_recent=1,
            ),
        )

        for _ in run(
            Agent(
                "writer",
                writer,
                role="Write.",
                context_policy=compression_policy,
            ),
            state,
        ):
            pass

        summaries = [message for message in state.messages if message.kind == "summary"]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(message_text(summaries[0]), "compressed old context")
        compression = next(
            event
            for event in state.events
            if isinstance(event, ContextCompressionEvent)
        )
        self.assertEqual(compression.compressed_message_indices, [1])
        self.assertEqual(
            state.snapshot.active_context_indices,
            compression.active_context_indices + [len(state.messages) - 1],
        )
        self.assertIn("old", captured["compressor_prompt"])
        self.assertEqual(
            captured["writer_visible_texts"],
            ["compress context", "compressed old context", "recent note"],
        )
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

    def test_tiered_strategy_returns_first_applicable_decision(self) -> None:
        # TieredStrategy restores tiering as a single strategy: stages are
        # tried in order, the first to return a decision wins, and later stages
        # are not consulted. Nones fall through; all-None yields None.
        calls: list[str] = []

        def make_stage(name: str, decision: CompressionDecision | None):
            def stage(active: list[tuple[int, Message]], agent_name: str):
                del active, agent_name
                calls.append(name)
                return decision

            return stage

        decision = CompressionDecision(
            compress_indices=(0,),
            replacement=make_message("system", "x", kind="summary"),
        )

        # First declines, second fires -> second's decision; both consulted.
        tiered = simple_agent_lab.TieredStrategy(
            (make_stage("a", None), make_stage("b", decision))
        )
        self.assertIs(tiered([], "w"), decision)
        self.assertEqual(calls, ["a", "b"])

        # First fires -> later stage never consulted.
        calls.clear()
        first_wins = simple_agent_lab.TieredStrategy(
            (make_stage("a", decision), make_stage("b", None))
        )
        self.assertIs(first_wins([], "w"), decision)
        self.assertEqual(calls, ["a"])

        # Every stage declines -> None.
        self.assertIsNone(
            simple_agent_lab.TieredStrategy((make_stage("a", None),))([], "w")
        )

    def test_tool_compact_folds_old_tool_exchanges(self) -> None:
        # ToolCompactStrategy is the rule-based, no-LLM option: when the context
        # exceeds `threshold_tokens`, every tool exchange except the most
        # recent `keep_recent_exchanges` is replaced by one short marker
        # listing the tool name and a result preview.
        def writer(visible: list[Message]) -> Message:
            del visible
            return assistant_message(
                "done", sender="writer", target="user", kind="final"
            )

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
            strategy=simple_agent_lab.ToolCompactStrategy(
                threshold_tokens=1,
                keep_recent_exchanges=1,
            ),
        )
        for _ in run(
            Agent("writer", writer, context_policy=policy),
            state,
            max_turns=1,
        ):
            pass

        compression = next(
            event
            for event in state.events
            if isinstance(event, ContextCompressionEvent)
        )
        # Compacted the first two exchanges (assistant + tool_result each = 4 msgs).
        self.assertEqual(len(compression.compressed_message_indices), 4)
        replacement = state.messages[compression.summary_message_index]
        self.assertEqual(replacement.kind, "summary")
        replacement_text = message_text(replacement)
        self.assertIn("Compacted 2 older tool exchange(s)", replacement_text)
        self.assertIn("alpha result", replacement_text)
        self.assertIn("beta result", replacement_text)
        # The most recent exchange stays verbatim — its messages survive
        # in active and the summary sits between the old block and the
        # kept tail.
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
        # A kept recent assistant carries `usage.context_tokens` from when the
        # window was full. After compression drops the older messages, that
        # baseline references content that is gone, so `after_tokens` must NOT
        # trust it — otherwise the event reports compression as a no-op.
        def writer(visible: list[Message]) -> Message:
            del visible
            return assistant_message(
                "done", sender="writer", target="user", kind="final"
            )

        def compressor(visible: list[Message]) -> Message:
            del visible
            return assistant_message(
                "short summary", sender="compressor", target="runtime", kind="final"
            )

        state = State("task")
        state.send("task", "user", "writer", state.task)
        state.send("message", "user", "writer", "old context " + ("x" * 400))
        # Recent assistant kept by keep_recent; its context_tokens reflects the
        # full pre-compression window (6120), dwarfing the real text size.
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
            strategy=simple_agent_lab.SummarizeStrategy(
                compressor=Agent("compressor", compressor),
                threshold_tokens=50,
                keep_recent=2,
            ),
        )
        for _ in run(
            Agent("writer", writer, context_policy=policy), state, max_turns=1
        ):
            pass

        compression = next(
            event
            for event in state.events
            if isinstance(event, ContextCompressionEvent)
        )
        # The stale baseline would have reported ~6120+; the true post-compression
        # context (summary + short kept messages, the kept assistant at its exact
        # 120 output_tokens) is small. Guard well below the stale value.
        self.assertLess(compression.after_tokens, 1000)
        self.assertLess(compression.after_tokens, compression.before_tokens)

    def test_rewrite_substitutes_message_in_place(self) -> None:
        # A `rewrite=True` decision swaps one message for a shorter,
        # structure-preserving replacement: the tool_result shrinks but stays a
        # first-class turn, its tool_call partner stays paired, and the original
        # remains in the append-only transcript (just no longer active).
        state = State("rewrite")
        state.send("task", "user", "writer", state.task)  # 0
        state.record(
            assistant_message(
                [TextBlock("call"), ToolCallBlock("c0", "read", {})],
                sender="writer",
                target="user",
                kind="step",
            )
        )  # 1
        state.record(
            tool_result_message(
                "HUGE " + "x" * 500,
                tool_call_id="c0",
                tool_name="read",
                target="writer",
            )
        )  # 2

        def shrink(
            active: list[tuple[int, Message]], agent_name: str
        ) -> CompressionDecision | None:
            for index, message in active:
                if message.kind == "tool_result":
                    return CompressionDecision(
                        compress_indices=(index,),
                        replacement=tool_result_message(
                            "shrunk",
                            tool_call_id="c0",
                            tool_name="read",
                            target=agent_name,
                        ),
                        rewrite=True,
                    )
            return None

        events = maybe_compress_context(
            Agent("writer", _idle_writer), state, ContextPolicy(strategy=shrink)
        )

        # A rewrite records the same ContextCompressionEvent as a fold: the
        # single rewritten index folds into its in-place replacement.
        rewrite = next(e for e in events if isinstance(e, ContextCompressionEvent))
        self.assertEqual(rewrite.compressed_message_indices, [2])
        self.assertEqual(rewrite.summary_message_index, 3)
        self.assertLess(rewrite.after_tokens, rewrite.before_tokens)
        # Active view: task, assistant(tool_call), shrunk result — pair intact.
        self.assertEqual(state.snapshot.active_context_indices, [0, 1, 3])
        self.assertEqual(
            [message_text(m) for m in state.active_context_messages()],
            ["rewrite", "call", "shrunk"],
        )
        # The original long result is still in the transcript, just not active.
        self.assertIn("HUGE", message_text(state.messages[2]))
        # The replacement is a plain tool_result — no runtime marker on it; the
        # compaction boundary is inferred from its (out-of-order) index instead.
        self.assertEqual(state.messages[3].sidecar, {})
        self.assertEqual(message_text(state.messages[3]), "shrunk")

    def test_rewrite_invalidates_stale_usage_baseline(self) -> None:
        # After a rewrite shrinks a tool_result, a kept assistant still carries
        # a `context_tokens` that counted the old, larger content. The marker on
        # the replacement must invalidate that baseline so sizing reflects the
        # shrink immediately — otherwise a rewrite meant to dodge compression
        # would look like a no-op.
        state = State("stale")
        state.send("task", "user", "writer", state.task)  # 0
        state.record(
            assistant_message(
                [TextBlock("call"), ToolCallBlock("c0", "read", {})],
                sender="writer",
                target="user",
                kind="step",
            )
        )  # 1
        state.record(
            tool_result_message(
                "HUGE " + "x" * 800,
                tool_call_id="c0",
                tool_name="read",
                target="writer",
            )
        )  # 2
        # Kept assistant whose context_tokens (6120) counted the HUGE result.
        state.record(
            assistant_message(
                "answer",
                sender="writer",
                target="user",
                kind="step",
                usage=TokenUsage(input_tokens=6000, output_tokens=120),
            )
        )  # 3

        def shrink(
            active: list[tuple[int, Message]], agent_name: str
        ) -> CompressionDecision | None:
            for index, message in active:
                if message.kind == "tool_result":
                    return CompressionDecision(
                        compress_indices=(index,),
                        replacement=tool_result_message(
                            "short",
                            tool_call_id="c0",
                            tool_name="read",
                            target=agent_name,
                        ),
                        rewrite=True,
                    )
            return None

        maybe_compress_context(
            Agent("writer", _idle_writer), state, ContextPolicy(strategy=shrink)
        )

        # The stale baseline would report ~6120; the true post-rewrite size is
        # tiny (the kept assistant at its exact 120 output_tokens + short texts).
        size = _active_context_tokens(state.active_context_items())
        self.assertLess(size, 1000)

    def test_rewrite_rejects_structure_changing_replacement(self) -> None:
        # A rewrite may shrink content but must preserve the message's shape:
        # same role and the same tool_call_id linkage. Otherwise the swap would
        # orphan a tool_call or its result, which providers reject.
        def make_state() -> State:
            state = State("reject")
            state.send("task", "user", "writer", state.task)  # 0
            state.record(
                assistant_message(
                    [TextBlock("call"), ToolCallBlock("c0", "read", {})],
                    sender="writer",
                    target="user",
                    kind="step",
                )
            )  # 1
            state.record(
                tool_result_message(
                    "big", tool_call_id="c0", tool_name="read", target="writer"
                )
            )  # 2
            return state

        def decide(replacement: Message) -> object:
            def strategy(
                active: list[tuple[int, Message]], agent_name: str
            ) -> CompressionDecision:
                del active, agent_name
                return CompressionDecision(
                    compress_indices=(2,), replacement=replacement, rewrite=True
                )

            return strategy

        # Different tool_call_id -> would orphan the c0 pair.
        with self.assertRaises(ValueError):
            maybe_compress_context(
                Agent("writer", _idle_writer),
                make_state(),
                ContextPolicy(
                    strategy=decide(
                        tool_result_message(
                            "x",
                            tool_call_id="OTHER",
                            tool_name="read",
                            target="writer",
                        )
                    ),
                ),
            )

        # Different role -> tool_result would become an assistant turn.
        with self.assertRaises(ValueError):
            maybe_compress_context(
                Agent("writer", _idle_writer),
                make_state(),
                ContextPolicy(
                    strategy=decide(
                        assistant_message(
                            "x", sender="writer", target="user", kind="step"
                        )
                    ),
                ),
            )

    def test_rewrite_requires_a_single_target(self) -> None:
        state = State("multi")
        state.send("task", "user", "writer", state.task)  # 0
        state.record(
            assistant_message(
                [TextBlock("call"), ToolCallBlock("c0", "read", {})],
                sender="writer",
                target="user",
                kind="step",
            )
        )  # 1
        state.record(
            tool_result_message(
                "big", tool_call_id="c0", tool_name="read", target="writer"
            )
        )  # 2

        def strategy(
            active: list[tuple[int, Message]], agent_name: str
        ) -> CompressionDecision:
            del active
            return CompressionDecision(
                compress_indices=(1, 2),
                replacement=tool_result_message(
                    "x", tool_call_id="c0", tool_name="read", target=agent_name
                ),
                rewrite=True,
            )

        with self.assertRaises(ValueError):
            maybe_compress_context(
                Agent("writer", _idle_writer),
                state,
                ContextPolicy(strategy=strategy),
            )

    def test_framework_auto_fixes_split_tool_pair(self) -> None:
        # If a strategy puts only one side of a tool_call/tool_result pair
        # in `compress_indices`, the framework un-compresses that side so
        # the model never sees an orphan. This frees strategy authors
        # from re-implementing tool-pair bookkeeping in every strategy.
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
        result_msg = tool_result_message(
            "ok",
            tool_call_id="call_x",
            tool_name="echo",
            target="writer",
        )
        state.record(result_msg)

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

        # No compression event recorded: the only compress index was
        # orphan-fixed away, leaving compress_set empty.
        compressions = [
            event
            for event in state.events
            if isinstance(event, ContextCompressionEvent)
        ]
        self.assertEqual(compressions, [])


if __name__ == "__main__":
    unittest.main()
