from __future__ import annotations

import unittest
from types import ModuleType
from unittest.mock import patch

from simple_agent_lab import Agent, State, message_text
from simple_agent_lab.compression import SummarizeStrategy
from simple_agent_lab.evals.chain import (
    CHAIN_DATA_KEY,
    CHAIN_HANDOFF_CONTEXT_PREFACE,
    INVALID_PROMPT_TOOL_REMINDER,
    _apply_context_window_handoff,
    _context_policy,
    _generate_handoff_doc,
    _recover_invalid_prompt,
    append_chain_task,
    demote_prior_chain_tasks,
    state_from_chain_payload,
    state_to_chain_payload,
)
from simple_agent_lab.evals.protocols import AgentSpec
from simple_agent_lab.llm.env import FAKE_PROVIDER
from simple_agent_lab.messages import (
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolCallBlock,
    ToolResultBlock,
    assistant_message,
    message_tool_calls,
    runtime_message,
    text_of,
    tool_results_message,
)
from simple_agent_lab.protocols import ContextCompressionEvent


class ChainStateCodecTest(unittest.TestCase):
    def test_round_trip_preserves_active_messages_and_metadata(self) -> None:
        state = State((TextBlock("task"), ImageBlock("image", "image/png")))
        state.data[CHAIN_DATA_KEY] = {"chain_id": "repo#1", "window_index": 2}
        state.record(runtime_message("policy", sidecar={"details": {"version": 1}}))
        state.record(
            assistant_message(
                (
                    ThinkingBlock(
                        "reasoning",
                        signature="sig",
                        redacted=True,
                        source_field="reasoning_content",
                    ),
                    ToolCallBlock("call-1", "bash", {"cmd": "pwd"}),
                ),
                sender="solver",
                target="user",
                kind="step",
                usage=TokenUsage(
                    input_tokens=10,
                    output_tokens=2,
                    cache_read_tokens=3,
                    cache_write_tokens=4,
                ),
                model="test-model",
                sidecar={"raw": {"id": "response-1"}},
            )
        )
        state.record(
            tool_results_message(
                [
                    ToolResultBlock(
                        tool_call_id="call-1",
                        tool_name="bash",
                        content=(
                            TextBlock("ok"),
                            ImageBlock("result", "image/jpeg"),
                        ),
                    )
                ],
                target="solver",
            )
        )
        state.record_event(
            ContextCompressionEvent(
                agent="solver",
                summary_message_index=1,
                compressed_message_indices=[0],
                active_context_indices=[1, 2],
                before_tokens=100,
                after_tokens=50,
                strategy="test",
            )
        )

        restored = state_from_chain_payload(state_to_chain_payload(state))

        self.assertEqual(restored.task, state.task)
        self.assertEqual(restored.data, state.data)
        self.assertEqual(
            restored.active_context_messages(), state.active_context_messages()
        )

    def test_rejects_unknown_message_and_block_kinds(self) -> None:
        base = {
            "schema": "simple-agent-lab.eval-chain-state.v1",
            "task": "",
            "data": {},
        }
        cases = [
            [{**base, "messages": [{"role": "developer", "content": []}]}],
            [
                {
                    **base,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"kind": "audio", "data": "..."}],
                        }
                    ],
                }
            ],
        ]

        for (payload,) in cases:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                state_from_chain_payload(payload)


class ChainContextEditTest(unittest.TestCase):
    def test_demoting_prior_task_keeps_it_visible_but_compressible(self) -> None:
        state = State("chain")
        append_chain_task(
            state,
            agent_name="solver",
            item_id="one",
            task="first task",
        )
        state.record(
            assistant_message(
                "first answer", sender="solver", target="user", kind="final"
            )
        )

        demote_prior_chain_tasks(state, agent_name="solver")

        active = state.active_context_messages()
        self.assertEqual(
            [message_text(message) for message in active],
            [
                "first task",
                "first answer",
            ],
        )
        self.assertEqual(active[0].kind, "message")
        event = state.events[-1]
        self.assertIsInstance(event, ContextCompressionEvent)
        assert isinstance(event, ContextCompressionEvent)
        self.assertEqual(event.strategy, "chain-task-demote")

    def test_invalid_prompt_recovery_retries_tool_output_then_ends_unknown(
        self,
    ) -> None:
        state = State("chain")
        append_chain_task(
            state,
            agent_name="solver",
            item_id="one",
            task="solve it",
        )
        state.record(
            assistant_message(
                (ToolCallBlock("call-1", "bash", {"cmd": "bad"}),),
                sender="solver",
                target="user",
                kind="step",
            )
        )
        state.record(
            tool_results_message(
                [
                    ToolResultBlock(
                        tool_call_id="call-1",
                        tool_name="bash",
                        content=(TextBlock("provider-triggering output"),),
                    )
                ],
                target="solver",
            )
        )

        retry = _recover_invalid_prompt(
            state,
            agent_name="solver",
            item_id="one",
            exc=RuntimeError("invalid_prompt"),
            retries=0,
        )

        self.assertIsNotNone(retry)
        assert retry is not None
        self.assertTrue(retry.retry)
        self.assertEqual(retry.retries, 1)
        self.assertFalse(
            any(
                message_tool_calls(message)
                for message in state.active_context_messages()
            )
        )
        self.assertIn(
            INVALID_PROMPT_TOOL_REMINDER,
            [message_text(message) for message in state.active_context_messages()],
        )

        ended = _recover_invalid_prompt(
            state,
            agent_name="solver",
            item_id="one",
            exc=RuntimeError("invalid_prompt"),
            retries=retry.retries,
        )

        self.assertIsNotNone(ended)
        assert ended is not None
        self.assertFalse(ended.retry)
        self.assertEqual(ended.skip_reason, "invalid_prompt_tool_exchange_not_found")
        self.assertEqual(state.active_context_messages(), [])

    def test_invalid_chain_task_is_skipped_without_retry(self) -> None:
        state = State("chain")
        append_chain_task(
            state,
            agent_name="solver",
            item_id="one",
            task="provider-triggering task",
        )

        recovery = _recover_invalid_prompt(
            state,
            agent_name="solver",
            item_id="one",
            exc=RuntimeError("code=-4321"),
            retries=0,
        )

        self.assertIsNotNone(recovery)
        assert recovery is not None
        self.assertFalse(recovery.retry)
        self.assertEqual(recovery.skip_reason, "invalid_prompt_chain_task")
        self.assertEqual(state.active_context_messages(), [])


class ChainHandoffTest(unittest.TestCase):
    def test_mid_instance_handoff_keeps_task_and_new_document(self) -> None:
        state = State("chain")
        append_chain_task(
            state,
            agent_name="solver",
            item_id="one",
            task="current task",
        )
        task_index = len(state.messages) - 1
        state.record(
            assistant_message(
                "work so far", sender="solver", target="user", kind="step"
            )
        )

        with patch(
            "simple_agent_lab.evals.chain._generate_handoff_doc",
            return_value="durable notes",
        ):
            did_reset, generated_turns, before_tokens = _apply_context_window_handoff(
                FAKE_PROVIDER,
                state,
                AgentSpec(name="solver"),
                {},
                window_index=2,
                task_message_index=task_index,
                item_id="one",
            )

        self.assertTrue(did_reset)
        self.assertEqual(generated_turns, 0)
        self.assertGreater(before_tokens, 0)
        self.assertEqual(
            [text_of(message.content) for message in state.active_context_messages()],
            ["current task", CHAIN_HANDOFF_CONTEXT_PREFACE + "durable notes"],
        )
        self.assertEqual(state.data[CHAIN_DATA_KEY]["window_index"], 2)

    def test_failed_handoff_generation_is_transactional(self) -> None:
        state = State("chain")
        state.send("context", "user", "solver", "durable context")
        before_events = list(state.events)
        before_messages = state.messages

        def fail(_: object) -> Message:
            raise RuntimeError("provider unavailable")

        with (
            patch(
                "simple_agent_lab.evals.chain.make_llm_agent",
                return_value=Agent("solver", fail),
            ),
            self.assertRaisesRegex(RuntimeError, "provider unavailable"),
        ):
            _generate_handoff_doc(
                FAKE_PROVIDER,
                state,
                AgentSpec(name="solver"),
                {},
            )

        self.assertEqual(state.events, before_events)
        self.assertEqual(state.messages, before_messages)


class ChainRuntimeConfigTest(unittest.TestCase):
    def test_default_context_policy_reuses_parsed_runtime_values(self) -> None:
        runtime = {
            "threshold_tokens": 123,
            "keep_recent": 2,
            "preserve_kinds": ["task", "context"],
        }

        policy = _context_policy(
            ModuleType("chain_test_container"),
            provider=FAKE_PROVIDER,
            request_extra={},
            config={"config": runtime},
            runtime=runtime,
            compression_strategy="summarize",
        )

        self.assertIsInstance(policy.strategy, SummarizeStrategy)
        assert isinstance(policy.strategy, SummarizeStrategy)
        self.assertEqual(policy.strategy.threshold_tokens, 123)
        self.assertEqual(policy.strategy.keep_recent, 2)
        self.assertEqual(policy.strategy.preserve_kinds, ("task", "context"))
