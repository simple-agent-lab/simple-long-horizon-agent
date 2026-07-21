"""Generic support for chained eval runs.

The normal in-container runner handles one isolated instance. This runner keeps
the same suite container-half contract, but it also restores a continuation
state from ``input/chain_state.json`` and writes the next state to
``out/chain_state.json`` so a host script can run an ordered chain of instances.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Literal, cast

from simple_agent_lab.agents.flavors import build_flavor_agent
from simple_agent_lab.compression import SummarizeStrategy, summarize_compression
from simple_agent_lab.context_view import ContextPolicy, estimate_context_tokens
from simple_agent_lab.core import Agent, run as run_agent
from simple_agent_lab.evals.in_container import provider_from_env
from simple_agent_lab.evals.protocols import (
    INSTANCE_KEY,
    RESULT_KEY,
    TRACE_KEY,
    TRACE_RAW_KEY,
    AgentSpec,
    ArtifactStore,
    ContainerTask,
)
from simple_agent_lab.evals.stores import container_store_from_env
from simple_agent_lab.llm import Provider
from simple_agent_lab.llm.env import API_KIND_CHOICES, request_extra_from_env
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.messages import (
    ContentBlock,
    ContentInput,
    ImageBlock,
    Message,
    MessageKind,
    MessageSidecar,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolCallBlock,
    ToolResultBlock,
    assistant_message,
    is_tool_result_message,
    message_tool_calls,
    runtime_message,
    tool_results_of,
    user_message,
)
from simple_agent_lab.protocols import (
    AgentEndEvent,
    AgentEndReason,
    ContextCompressionEvent,
    Event,
    MessageEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from simple_agent_lab.state import State
from simple_agent_lab.tools import AbortFlag
from simple_agent_lab.trace import event_stream, run_trace_from_state
from simple_agent_lab.trace.jsonl import json_safe
from simple_agent_lab.workflow import final_output, never_abort

CHAIN_DATA_KEY = "eval_chain"
CHAIN_STATE_INPUT_KEY = "input/chain_state.json"
CHAIN_STATE_OUTPUT_KEY = "out/chain_state.json"
CHAIN_CONFIG_KEY = "input/chain_config.json"
CHAIN_STATE_SCHEMA = "simple-agent-lab.eval-chain-state.v1"
ENCRYPTED_REASONING_INCLUDE = "reasoning.encrypted_content"
CHAIN_TASK_DEMOTE_REASON = "chain-task-demote"
INVALID_PROMPT_TOOL_REMINDER = (
    "Removed invalid_prompt-triggering tool call/output. Use another command."
)
INVALID_PROMPT_ITEM_END_MESSAGE = (
    "Chain item {item_id} ends here; it was skipped because tool output kept "
    "triggering invalid_prompt. Continue to the next item."
)
INVALID_PROMPT_TOOL_RETRY_LIMIT = 20
InvalidPromptSource = Literal["chain_task", "tool_output", "unknown"]


@dataclass(frozen=True)
class _InvalidPromptRecovery:
    retry: bool
    retries: int
    skip_reason: str = ""
    error: str = ""


CONTEXT_WINDOW_HANDOFF_REASON = "context_window_handoff"

# Safety net for the mid-instance handoff loop. Each window always runs at least
# one solver turn before it can reset (see ``_context_window_abort``), so the
# per-instance turn budget already bounds the number of resets; this cap only
# guards against a pathological config where the window is smaller than a single
# turn's growth or the handoff document itself.
MAX_CONTEXT_WINDOW_HANDOFFS = 100

CHAIN_HANDOFF_ROLE = (
    "You are writing a handoff document for the next engineer, who will keep "
    "working on this same repository in a fresh session with NO memory of the "
    "transcript above. The transcript is your only source; write down what they "
    "need so they do not have to rediscover it."
)

CHAIN_HANDOFF_PROMPT = (
    "You are close to the context-window limit for this long repo chain. The "
    "next sub-problems will run in a new window that can only see the handoff "
    "you write now, not this transcript. Do not start a new task. Write one "
    "thorough handoff document capturing everything reusable across the "
    "remaining sub-problems, including:\n"
    "- Repository architecture and layout you learned (key directories, "
    "modules, and how they fit together).\n"
    "- Important files and their roles, with paths.\n"
    "- Build, test, run, and lint commands, plus conventions and setup/env "
    "gotchas.\n"
    "- Decisions made and why, and any patterns to follow or avoid.\n"
    "- Current state: what is done, what is in progress, and known issues or "
    "unresolved questions.\n"
    "Write it as durable notes for a teammate; be concrete and specific."
)

CHAIN_HANDOFF_CONTEXT_PREFACE = (
    "HANDOFF FROM EARLIER IN THIS REPO CHAIN (previous context window).\n"
    "You are continuing a long chain of sub-problems on this repository. The "
    "earlier transcript is gone; the notes below are what your previous self "
    "left for you. Treat them as trusted background, reuse them instead of "
    "rediscovering, and solve only the current sub-problem.\n\n"
)


def start_chain_state(task: ContentInput) -> State:
    """Create the persistent transcript seed for one eval chain."""

    return State(task)


def append_chain_task(
    state: State,
    *,
    agent_name: str,
    item_id: str,
    task: str,
    details: Mapping[str, Any] | None = None,
    demote_prior_tasks: bool = True,
) -> None:
    """Append one benchmark item prompt to an existing chain state."""

    if demote_prior_tasks:
        demote_prior_chain_tasks(state, agent_name=agent_name)
    sidecar_details = {
        "chain": {"item_id": item_id},
        **dict(details or {}),
    }
    state.send(
        "task",
        "user",
        agent_name,
        task,
        sidecar={"details": sidecar_details},
    )


def demote_prior_chain_tasks(state: State, *, agent_name: str) -> None:
    """Make already-started chain item prompts compressible.

    Only the current benchmark item should stay pinned as ``kind="task"``.
    When the next item starts, prior item prompts become ordinary messages so a
    later summarize pass can fold them together with their solution transcript.
    The edit is represented as append-only replacement messages plus an active
    context re-point, preserving the original trace entries for audit.
    """

    active_items = state.active_context_items()
    target = next(
        (
            (index, message)
            for index, message in active_items
            if message.kind == "task" and message_chain_item_id(message)
        ),
        None,
    )
    if target is None:
        return

    target_index, message = target
    state.record(replace(message, kind="message"))
    replacement_index = len(state.messages) - 1
    state.record_event(
        ContextCompressionEvent(
            agent=agent_name,
            summary_message_index=replacement_index,
            compressed_message_indices=[target_index],
            active_context_indices=[
                replacement_index if index == target_index else index
                for index, _ in active_items
            ],
            before_tokens=0,
            after_tokens=0,
            strategy=CHAIN_TASK_DEMOTE_REASON,
        )
    )


def state_to_chain_payload(state: State) -> dict[str, Any]:
    """Return a JSON-safe continuation payload for the state's active context."""

    active_items = state.active_context_items()
    return {
        "schema": CHAIN_STATE_SCHEMA,
        "task": _content_input_to_record(state.task),
        "messages": [_message_to_record(message) for _, message in active_items],
        "data": json_safe(state.data),
    }


def state_from_chain_payload(payload: Mapping[str, Any]) -> State:
    """Rebuild a ``State`` from ``state_to_chain_payload`` data."""

    schema = str(payload.get("schema") or "")
    if schema and schema != CHAIN_STATE_SCHEMA:
        raise ValueError(f"Unsupported chain state schema: {schema!r}")

    state = State(task=_content_input_from_record(payload.get("task", "")))
    data = payload.get("data")
    if isinstance(data, Mapping):
        state.data.update(dict(data))
    _chain_data(state)

    messages = payload.get("messages", [])
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("chain payload 'messages' must be a list")
    for record in messages:
        if not isinstance(record, Mapping):
            raise ValueError("chain message records must be objects")
        state.record(_message_from_record(record))
    return state


def is_invalid_prompt_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    code = getattr(exc, "code", None)
    status_code = getattr(exc, "status_code", None)
    return (
        "invalid_prompt" in text
        or "invalid prompt" in text
        or "-4321" in text
        or code == -4321
        or status_code == -4321
    )


def invalid_prompt_source(
    state: Any,
    *,
    item_id: str,
) -> InvalidPromptSource:
    """Classify which latest user-visible message caused invalid_prompt."""

    for _, message in reversed(state.active_context_items()):
        if getattr(message, "role", "") != "user":
            continue
        if is_tool_result_message(message):
            return "tool_output"
        if message_chain_item_id(message) == item_id:
            return "chain_task"
        return "unknown"
    return "unknown"


def replace_latest_tool_exchange_for_invalid_prompt(
    state: Any, *, agent_name: str
) -> bool:
    """Replace the latest active tool call/result exchange with a safe note."""

    active_items = state.active_context_items()
    tool_result_index: int | None = None
    tool_call_ids: set[str] = set()
    for index, message in reversed(active_items):
        if is_tool_result_message(message):
            tool_result_index = index
            tool_call_ids = {
                block.tool_call_id for block in tool_results_of(message.content)
            }
            break
    if tool_result_index is None:
        return False

    dropped = tool_exchange_indices(active_items, tool_call_ids)
    if not dropped:
        return False

    replacement = user_message(
        INVALID_PROMPT_TOOL_REMINDER,
        sender="user",
        target=agent_name,
        kind="message",
    )
    state.record(replacement)
    replacement_index = len(state.messages) - 1
    active_context_indices: list[int] = []
    inserted = False
    for index, _ in active_items:
        if index in dropped:
            if not inserted:
                active_context_indices.append(replacement_index)
                inserted = True
            continue
        active_context_indices.append(index)
    state.record_event(
        ContextCompressionEvent(
            agent=agent_name,
            summary_message_index=replacement_index,
            compressed_message_indices=sorted(dropped),
            active_context_indices=active_context_indices,
            before_tokens=0,
            after_tokens=0,
            strategy="invalid-prompt-tool-exchange-replace",
        )
    )
    return True


def tool_exchange_indices(
    active_items: list[tuple[int, Any]], tool_call_ids: set[str]
) -> set[int]:
    return {
        index
        for index, message in active_items
        if any(call.id in tool_call_ids for call in message_tool_calls(message))
        or any(
            result.tool_call_id in tool_call_ids
            for result in tool_results_of(message.content)
        )
    }


def drop_chain_task_for_invalid_prompt_skip(
    state: Any, *, agent_name: str, item_id: str
) -> None:
    """Drop a skipped chain item prompt from active context."""

    active_items = state.active_context_items()
    target_index: int | None = None
    for index, message in reversed(active_items):
        if (
            getattr(message, "role", "") == "user"
            and message_chain_item_id(message) == item_id
        ):
            target_index = index
            break
    if target_index is None:
        return

    state.record_event(
        ContextCompressionEvent(
            agent=agent_name,
            summary_message_index=target_index,
            compressed_message_indices=[target_index],
            active_context_indices=[
                index for index, _ in active_items if index != target_index
            ],
            before_tokens=0,
            after_tokens=0,
            strategy="invalid-prompt-chain-task-drop",
        )
    )


def end_chain_item_after_invalid_prompt_tool_retry_limit(
    state: Any, *, agent_name: str, item_id: str
) -> None:
    """Clear active context after persistent invalid_prompt for one chain item."""

    active_items = state.active_context_items()
    if not active_items:
        return

    end_message = user_message(
        INVALID_PROMPT_ITEM_END_MESSAGE.format(item_id=item_id),
        sender="user",
        target=agent_name,
        kind="message",
    )
    state.record(end_message)
    end_message_index = len(state.messages) - 1
    state.record_event(
        ContextCompressionEvent(
            agent=agent_name,
            summary_message_index=end_message_index,
            compressed_message_indices=[index for index, _ in active_items],
            active_context_indices=[],
            before_tokens=0,
            after_tokens=0,
            strategy="invalid-prompt-clear-context",
        )
    )


def _recover_invalid_prompt(
    state: State,
    *,
    agent_name: str,
    item_id: str,
    exc: BaseException,
    retries: int,
) -> _InvalidPromptRecovery | None:
    """Apply the shared chain invalid-prompt policy, or return None to re-raise."""

    if not is_invalid_prompt_error(exc):
        return None
    prompt_source = invalid_prompt_source(state, item_id=item_id)
    provider_error = f"{type(exc).__name__}: {exc}"
    if prompt_source == "chain_task":
        drop_chain_task_for_invalid_prompt_skip(
            state, agent_name=agent_name, item_id=item_id
        )
        return _InvalidPromptRecovery(
            retry=False,
            retries=retries,
            skip_reason="invalid_prompt_chain_task",
            error=provider_error,
        )
    if prompt_source != "tool_output" and not retries:
        return None
    if retries >= INVALID_PROMPT_TOOL_RETRY_LIMIT:
        end_chain_item_after_invalid_prompt_tool_retry_limit(
            state, agent_name=agent_name, item_id=item_id
        )
        return _InvalidPromptRecovery(
            retry=False,
            retries=retries,
            skip_reason="invalid_prompt_tool_output_retry_limit",
            error=provider_error,
        )
    if not replace_latest_tool_exchange_for_invalid_prompt(
        state, agent_name=agent_name
    ):
        end_chain_item_after_invalid_prompt_tool_retry_limit(
            state, agent_name=agent_name, item_id=item_id
        )
        return _InvalidPromptRecovery(
            retry=False,
            retries=retries,
            skip_reason="invalid_prompt_tool_exchange_not_found",
            error=provider_error,
        )
    return _InvalidPromptRecovery(retry=True, retries=retries + 1)


def message_chain_item_id(message: Any) -> str:
    details = getattr(message, "sidecar", {}).get("details", {})
    if not isinstance(details, Mapping):
        return ""
    chain = details.get("chain", {})
    if isinstance(chain, Mapping) and chain.get("item_id"):
        return str(chain.get("item_id") or "")
    return ""


def run_chain_in_container(
    *,
    instance: Mapping[str, Any],
    container_module: str,
    provider: Provider,
    workdir: Path,
    max_turns: int,
    wall_time_seconds: float | None = None,
    store: ArtifactStore,
    trace_id: str,
    producer: str,
    request_extra: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], State]:
    """Run one eval instance while preserving chain state."""

    module = importlib.import_module(container_module)
    tasks = cast(ContainerTask, module)
    workdir = Path(workdir)
    config = _load_chain_config(store)
    spec = _chain_agent_spec(module, config)
    state = _load_state(store, module=module, config=config)

    context: dict[str, Any] = {}
    prepare = getattr(module, "prepare", None)
    if callable(prepare):
        context = dict(prepare(workdir, instance) or {})

    task = tasks.build_task(instance, workdir=str(workdir))
    item_id = str(instance.get("instance_id") or "?")
    demote_prior_chain_tasks(state, agent_name=spec.name)
    event_start = len(state.events)
    append_chain_task(
        state,
        agent_name=spec.name,
        item_id=item_id,
        task=str(task),
        demote_prior_tasks=False,
    )
    # The current instance's task message is kept active across mid-instance
    # handoffs so the reset window still shows what problem to solve.
    task_message_index = len(state.messages) - 1

    status = "ok"
    error = ""
    skip_reason = ""
    invalid_prompt_retries = 0
    chain_window_index = int(_chain_data(state).get("window_index", 1) or 1)
    handoff_active = _handoff_active(config)
    window_limit = _context_window_tokens(config)
    position = int(config.get("position", 0) or 0)
    instances_in_chain = int(config.get("instances_in_chain", 0) or 0)
    is_last_instance = instances_in_chain > 0 and position >= instances_in_chain
    handoff_written = False
    boundary_handoff_written = False
    handoff_context_tokens = 0
    context_window_handoffs = 0
    result_product: dict[str, Any] = {"model_patch": ""}
    deadline = (
        time.monotonic() + wall_time_seconds if wall_time_seconds is not None else None
    )

    def deadline_abort() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    try:
        agent = _build_agent(
            provider=provider,
            workdir=workdir,
            request_extra=request_extra,
            config=config,
            container_module=container_module,
        )
        # Turns spent generating handoff docs are overhead, not solver work, so
        # they are excluded from the instance's turn budget below.
        handoff_gen_turns = 0
        while status == "ok":
            turn_budget = max(
                0,
                max_turns
                - (_count_turns(state.events[event_start:]) - handoff_gen_turns),
            )
            if turn_budget <= 0:
                prompt_source = invalid_prompt_source(state, item_id=item_id)
                invalid_retry = prompt_source == "tool_output" or bool(
                    invalid_prompt_retries
                )
                if invalid_retry:
                    if prompt_source == "tool_output":
                        end_chain_item_after_invalid_prompt_tool_retry_limit(
                            state, agent_name=spec.name, item_id=item_id
                        )
                    elif prompt_source == "chain_task":
                        drop_chain_task_for_invalid_prompt_skip(
                            state, agent_name=spec.name, item_id=item_id
                        )
                    status = "skipped"
                    skip_reason = "invalid_prompt_turn_budget_exhausted"
                    error = (
                        "invalid_prompt retry exhausted this chain item's turn budget"
                    )
                break
            since_event_index = len(state.events)
            window_abort = (
                _context_window_abort(
                    state, window_limit, since_event_index=since_event_index
                )
                if handoff_active
                else never_abort
            )
            run_abort = lambda: deadline_abort() or window_abort()  # noqa: E731
            try:
                end_reason: AgentEndReason = "max_turns"
                for event in run_agent(
                    agent, state, max_turns=turn_budget, abort=run_abort
                ):
                    if isinstance(event, ContextCompressionEvent):
                        print(
                            f"[chain] {item_id}: context edit "
                            f"{event.strategy} {event.before_tokens}->{event.after_tokens}",
                            flush=True,
                        )
                    if isinstance(event, AgentEndEvent):
                        end_reason = event.reason
                    if _task_tool_enabled(config):
                        if _message_has_invalid_prompt_task_error(event):
                            raise RuntimeError("invalid_prompt surfaced by task tool")
                # Mid-instance context-window handoff: the run stopped at a turn
                # boundary because the active context hit the window. Reset the
                # context in place (keeping the full trace) and keep solving the
                # SAME instance, as long as turn budget remains.
                remaining = max_turns - (
                    _count_turns(state.events[event_start:]) - handoff_gen_turns
                )
                if (
                    handoff_active
                    and end_reason == "abort"
                    and not deadline_abort()
                    and remaining > 0
                    and _over_window(state, window_limit)
                    and context_window_handoffs < MAX_CONTEXT_WINDOW_HANDOFFS
                ):
                    did_reset, gen_turns, before_tokens = _apply_context_window_handoff(
                        provider,
                        state,
                        spec,
                        request_extra,
                        window_index=chain_window_index + 1,
                        task_message_index=task_message_index,
                        item_id=item_id,
                    )
                    handoff_gen_turns += gen_turns
                    if did_reset:
                        chain_window_index += 1
                        handoff_written = True
                        handoff_context_tokens = before_tokens
                        context_window_handoffs += 1
                        continue
                    # Generation is transactional. If it fails, spend the next
                    # solver turn on real work with the intact full context
                    # instead of ending this otherwise healthy chain item.
                    continue
                break
            except Exception as exc:
                recovery = _recover_invalid_prompt(
                    state,
                    agent_name=spec.name,
                    item_id=item_id,
                    exc=exc,
                    retries=invalid_prompt_retries,
                )
                if recovery is None:
                    raise
                invalid_prompt_retries = recovery.retries
                if not recovery.retry:
                    status = "skipped"
                    skip_reason = recovery.skip_reason
                    error = recovery.error
                    break
                continue

        if status == "ok":
            extract = tasks.extract_result
            result_product = dict(
                extract(
                    workdir,
                    instance,
                    **_context_kwargs(extract, context, state=state),
                )
            )
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    # Boundary handoff: the instance finished cleanly but its context is still
    # at/over the window (it completed just as the window filled). Have the model
    # write a handoff doc and start the NEXT instance in a fresh window seeded
    # with only that doc. The finished instance's own trace/metrics still use the
    # full `state`; only the outgoing chain_state is reset. Mid-instance handoffs
    # (above) instead keep the same instance going, so this fires only for the
    # "just completed at the boundary -> next problem" case.
    outgoing_state = state
    if (
        status == "ok"
        and handoff_active
        and not is_last_instance
        and not deadline_abort()
    ):
        boundary_tokens = estimate_context_tokens(state.active_context_messages())
        if boundary_tokens >= _context_window_tokens(config):
            try:
                doc = _generate_handoff_doc(provider, state, spec, request_extra)
            except Exception as exc:  # never let handoff failure kill the chain
                print(
                    f"[chain] {item_id}: handoff doc generation failed, "
                    f"carrying full context forward: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            else:
                if doc.strip():
                    prior_active = [index for index, _ in state.active_context_items()]
                    before_reset_tokens = estimate_context_tokens(
                        state.active_context_messages()
                    )
                    state.send(
                        "context",
                        "user",
                        spec.name,
                        CHAIN_HANDOFF_CONTEXT_PREFACE + doc,
                        sidecar={"details": {"chain": {"handoff": True}}},
                    )
                    doc_index = len(state.messages) - 1
                    after_tokens = estimate_context_tokens([state.messages[doc_index]])
                    state.record_event(
                        ContextCompressionEvent(
                            agent=spec.name,
                            summary_message_index=doc_index,
                            compressed_message_indices=prior_active,
                            active_context_indices=[doc_index],
                            before_tokens=before_reset_tokens,
                            after_tokens=after_tokens,
                            strategy=CONTEXT_WINDOW_HANDOFF_REASON,
                        )
                    )
                    chain_window_index += 1
                    outgoing_state = _handoff_reset_state(
                        module,
                        config=config,
                        spec=spec,
                        doc=doc,
                        window_index=chain_window_index,
                    )
                    handoff_written = True
                    boundary_handoff_written = True
                    handoff_context_tokens = before_reset_tokens
                    print(
                        f"[chain] {item_id}: context-window handoff "
                        f"{handoff_context_tokens}->{after_tokens} "
                        f"(window {chain_window_index})",
                        flush=True,
                    )

    metrics = summarize_compression(state.events[event_start:])
    chain_id = _chain_id(config, instance)
    result = {
        **result_product,
        "instance_id": item_id,
        "chain_id": chain_id,
        "provider_auth_env": str(config.get("provider_auth_env") or ""),
        "agent_flavor": _agent_flavor(config, default=spec.flavor),
        "task_tool": _task_tool_enabled(config),
        "compression_strategy": _compression_strategy(config),
        "status": status,
        "error": error,
        "skip_reason": skip_reason,
        "invalid_prompt_retries": invalid_prompt_retries,
        "handoff": handoff_active,
        "handoff_written": handoff_written,
        "boundary_handoff_written": boundary_handoff_written,
        "handoff_context_tokens": handoff_context_tokens,
        "context_window_handoffs": context_window_handoffs,
        "chain_window_index": chain_window_index,
        "compression_metrics": metrics.as_dict(),
        "chain_event_start": event_start,
        "chain_event_end": len(state.events),
    }
    store.put(
        RESULT_KEY, (json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8")
    )
    store.put(
        CHAIN_STATE_OUTPUT_KEY,
        (
            json.dumps(state_to_chain_payload(outgoing_state), ensure_ascii=False)
            + "\n"
        ).encode("utf-8"),
    )
    if bool(config.get("write_trajectories", True)):
        trace_bytes, raw_bytes = _trace_artifacts(
            state,
            trace_id=trace_id,
            producer=producer,
            meta={
                "chain_id": chain_id,
                "instance_id": item_id,
                "status": status,
                "provider_auth_env": result["provider_auth_env"],
                "agent_flavor": result["agent_flavor"],
                "compression_strategy": result["compression_strategy"],
            },
        )
        store.put(TRACE_KEY, trace_bytes)
        if raw_bytes is not None:
            store.put(TRACE_RAW_KEY, raw_bytes)
    return result, state


def _build_agent(
    *,
    provider: Provider,
    workdir: Path,
    request_extra: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    container_module: str,
) -> Agent:
    module = importlib.import_module(container_module)
    spec = _chain_agent_spec(module, config)
    return build_flavor_agent(
        flavor=_agent_flavor(config, default=spec.flavor),
        provider=provider,
        cwd=workdir,
        name=spec.name,
        role=spec.role,
        system_prompt=spec.system_prompt,
        request_extra=request_extra,
        context_policy=_context_policy(
            module,
            provider=provider,
            request_extra=request_extra,
            config=config,
        ),
        enable_default_compression=False,
        solver_read=False,
        solver_task=_task_tool_enabled(config),
    )


def _context_policy(
    module: ModuleType,
    *,
    provider: Provider,
    request_extra: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> ContextPolicy:
    hook = getattr(module, "chain_context_policy", None)
    if callable(hook):
        return hook(provider=provider, request_extra=request_extra, config=config)
    if _compression_strategy(config) != "summarize":
        return ContextPolicy()
    runtime = _runtime_config(config)
    compressor = make_llm_agent(
        name="chain_compressor",
        provider=provider,
        role=(
            "Summarize older eval-chain context. Preserve durable facts, "
            "decisions, tool results, constraints, file paths, test signals, "
            "and unresolved questions. Omit low-value wording."
        ),
        request_extra=request_extra,
    )
    return ContextPolicy(
        strategy=SummarizeStrategy(
            compressor=compressor,
            threshold_tokens=int(runtime.get("threshold_tokens", 217600) or 217600),
            keep_recent=int(runtime.get("keep_recent", 4) or 4),
            preserve_kinds=tuple(
                runtime.get("preserve_kinds") or ("task", "system", "context")
            ),
        )
    )


def _load_state(
    store: ArtifactStore,
    *,
    module: ModuleType,
    config: Mapping[str, Any],
) -> State:
    try:
        payload = json.loads(store.get(CHAIN_STATE_INPUT_KEY).decode("utf-8"))
    except OSError:
        return _start_state(module, config=config)
    return state_from_chain_payload(payload)


def _start_state(module: ModuleType, *, config: Mapping[str, Any]) -> State:
    hook = getattr(module, "chain_start_state", None)
    if callable(hook):
        return hook(config=config)
    return start_chain_state(_chain_display_name(config))


def _generate_handoff_doc(
    provider: Provider,
    state: State,
    spec: AgentSpec,
    request_extra: Mapping[str, Any] | None,
) -> str:
    """Have the model write a handoff doc from the current context.

    Builds a tool-less agent with the SAME name as the solver so it inherits the
    solver's visible context. Generation runs on a scratch copy and is committed
    only when it produces a non-empty document, so a provider failure cannot
    leave a dangling prompt or partial handoff exchange in the solver state.
    The returned text seeds the next window, whether that window continues the
    same instance (mid-instance handoff) or starts the next one.
    """

    handoff_agent = make_llm_agent(
        name=spec.name,
        provider=provider,
        role=CHAIN_HANDOFF_ROLE,
        request_extra=request_extra,
    )
    scratch = State(
        task=state.task,
        events=list(state.events),
        data=dict(state.data),
        _monotonic_origin=state._monotonic_origin,
    )
    before = len(scratch.messages)
    _, events = handoff_agent.resume(scratch, CHAIN_HANDOFF_PROMPT, max_turns=2)
    for _ in events:
        pass
    doc = final_output(scratch, spec.name, after_message_index=before)
    if not doc.strip():
        return ""
    for event in scratch.events[len(state.events) :]:
        state.record_event_at(event, elapsed=event.elapsed)
    return doc


def _handoff_reset_state(
    module: ModuleType,
    *,
    config: Mapping[str, Any],
    spec: AgentSpec,
    doc: str,
    window_index: int,
) -> State:
    """Build the next window's state: a fresh chain state seeded with the doc.

    The full transcript is intentionally dropped; only the handoff notes cross
    into the new window, which is the whole point of the handoff mechanism.
    """

    state = _start_state(module, config=config)
    _chain_data(state)["window_index"] = window_index
    state.send(
        "context",
        "user",
        spec.name,
        CHAIN_HANDOFF_CONTEXT_PREFACE + doc,
        sidecar={"details": {"chain": {"handoff": True}}},
    )
    return state


def _count_turns(events: Sequence[Event]) -> int:
    """Number of solver turns started in ``events`` (one per TurnStartEvent)."""

    return sum(1 for event in events if isinstance(event, TurnStartEvent))


def _over_window(state: State, limit: int) -> bool:
    """True when the active context has reached the context-window limit."""

    if limit <= 0:
        return False
    return estimate_context_tokens(state.active_context_messages()) >= limit


def _count_completed_turns(events: Sequence[Event]) -> int:
    """Number of solver turns that fully finished (one per TurnEndEvent)."""

    return sum(1 for event in events if isinstance(event, TurnEndEvent))


def _context_window_abort(
    state: State, limit: int, *, since_event_index: int
) -> AbortFlag:
    """Abort flag that trips once the active context reaches ``limit``.

    The core loop checks this at the top of every turn and also passes it into
    tool execution. It only trips after at least one solver turn has *completed*
    in the current window (``since_event_index`` marks the window start), which
    matters twice: every window makes real progress before a reset (so a tiny
    window can never get stuck resetting without doing any work), and the abort
    can only fire at a clean turn boundary — never mid-turn, which would kill the
    in-flight tool call ("aborted before start") and lose that turn's work.
    """

    def abort() -> bool:
        if _count_completed_turns(state.events[since_event_index:]) < 1:
            return False
        return _over_window(state, limit)

    return abort


def _apply_context_window_handoff(
    provider: Provider,
    state: State,
    spec: AgentSpec,
    request_extra: Mapping[str, Any] | None,
    *,
    window_index: int,
    task_message_index: int,
    item_id: str,
) -> tuple[bool, int, int]:
    """Write a handoff doc mid-instance and reset the ACTIVE context in place.

    Unlike the between-instance ``_handoff_reset_state`` (which builds a fresh
    state for the next instance), this keeps the SAME state so the current
    instance keeps working after the reset. The full transcript stays in
    ``state.messages``/``state.events`` for the trace; only the active context
    the model sees is repointed to the current task plus the fresh handoff doc.

    Returns ``(did_reset, handoff_gen_turns, before_tokens)``. ``did_reset`` is
    False (and the caller carries the full context forward) when generation
    fails or returns an empty document.
    """

    before_tokens = estimate_context_tokens(state.active_context_messages())
    events_before = len(state.events)
    try:
        doc = _generate_handoff_doc(provider, state, spec, request_extra)
    except Exception as exc:  # never let handoff failure kill the chain
        print(
            f"[chain] {item_id}: mid-instance handoff generation failed, "
            f"carrying full context forward: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return False, 0, 0
    gen_turns = _count_turns(state.events[events_before:])
    if not doc.strip():
        return False, gen_turns, 0

    prior_active = [index for index, _ in state.active_context_items()]
    state.send(
        "context",
        "user",
        spec.name,
        CHAIN_HANDOFF_CONTEXT_PREFACE + doc,
        sidecar={"details": {"chain": {"handoff": True}}},
    )
    doc_index = len(state.messages) - 1
    keep = [task_message_index, doc_index]
    dropped = [index for index in prior_active if index not in set(keep)]
    _chain_data(state)["window_index"] = window_index
    after_tokens = estimate_context_tokens([state.messages[index] for index in keep])
    state.record_event(
        ContextCompressionEvent(
            agent=spec.name,
            summary_message_index=doc_index,
            compressed_message_indices=sorted(dropped),
            active_context_indices=keep,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            strategy=CONTEXT_WINDOW_HANDOFF_REASON,
        )
    )
    print(
        f"[chain] {item_id}: mid-instance context-window handoff "
        f"{before_tokens}->{after_tokens} (window {window_index})",
        flush=True,
    )
    return True, gen_turns, before_tokens


def _load_chain_config(store: ArtifactStore) -> dict[str, Any]:
    try:
        raw = store.get(CHAIN_CONFIG_KEY)
    except OSError:
        return {}
    return json.loads(raw.decode("utf-8") or "{}")


def _chain_agent_spec(module: ModuleType, config: Mapping[str, Any]) -> AgentSpec:
    factory = getattr(module, "agent_spec", None)
    spec = factory() if callable(factory) else AgentSpec()
    return replace(spec, flavor=_agent_flavor(config, default=spec.flavor))


def _chain_data(state: State) -> dict[str, Any]:
    data = state.data.setdefault(CHAIN_DATA_KEY, {})
    if isinstance(data, dict):
        return data
    state.data[CHAIN_DATA_KEY] = {}
    return cast(dict[str, Any], state.data[CHAIN_DATA_KEY])


def _content_input_to_record(content: ContentInput) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    return cast(list[dict[str, Any]], json_safe(content))


def _content_input_from_record(value: Any) -> ContentInput:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(_block_from_record(block) for block in value)
    return str(value or "")


def _message_to_record(message: Message) -> dict[str, Any]:
    record = cast(dict[str, Any], json_safe(message))
    if record.get("usage") is None:
        record.pop("usage", None)
    return record


def _message_from_record(record: Mapping[str, Any]) -> Message:
    role = str(record.get("role") or "")
    content = tuple(_block_from_record(block) for block in record.get("content", []))
    sender = str(record.get("sender") or role or "agent")
    target = str(record.get("target") or "all")
    kind = cast(MessageKind, str(record.get("kind") or "message"))
    sidecar = cast(MessageSidecar, _mapping(record.get("sidecar")))
    if role == "user":
        return user_message(
            content, sender=sender, target=target, kind=kind, sidecar=sidecar
        )
    if role == "system":
        return runtime_message(
            content, sender=sender, target=target, kind=kind, sidecar=sidecar
        )
    if role == "assistant":
        usage = record.get("usage")
        return assistant_message(
            content,
            sender=sender,
            target=target,
            kind=kind,
            sidecar=sidecar,
            usage=_usage_from_record(usage) if isinstance(usage, Mapping) else None,
            model=str(record.get("model") or ""),
        )
    raise ValueError(f"Unsupported chain message role: {role!r}")


def _block_from_record(record: Any) -> ContentBlock:
    if not isinstance(record, Mapping):
        raise ValueError("content block records must be objects")
    kind = str(record.get("kind") or "")
    if kind == "text":
        return TextBlock(str(record.get("text") or ""))
    if kind == "image":
        return ImageBlock(
            data=str(record.get("data") or ""),
            mime_type=str(record.get("mime_type") or "image/png"),
        )
    if kind == "thinking":
        return ThinkingBlock(
            text=str(record.get("text") or ""),
            signature=(
                str(record["signature"])
                if record.get("signature") is not None
                else None
            ),
            redacted=bool(record.get("redacted", False)),
            source_field=(
                str(record["source_field"])
                if record.get("source_field") is not None
                else None
            ),
        )
    if kind == "tool_call":
        return ToolCallBlock(
            id=str(record.get("id") or ""),
            name=str(record.get("name") or ""),
            arguments=_mapping(record.get("arguments")),
        )
    if kind == "tool_result":
        return ToolResultBlock(
            tool_call_id=str(record.get("tool_call_id") or ""),
            tool_name=str(record.get("tool_name") or ""),
            content=cast(
                tuple[TextBlock | ImageBlock, ...],
                tuple(_block_from_record(block) for block in record.get("content", [])),
            ),
            is_error=bool(record.get("is_error", False)),
        )
    raise ValueError(f"Unsupported content block kind: {kind!r}")


def _usage_from_record(record: Mapping[str, Any]) -> TokenUsage:
    return TokenUsage(
        input_tokens=int(record.get("input_tokens", 0) or 0),
        output_tokens=int(record.get("output_tokens", 0) or 0),
        cache_read_tokens=int(record.get("cache_read_tokens", 0) or 0),
        cache_write_tokens=int(record.get("cache_write_tokens", 0) or 0),
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _runtime_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("config")
    return value if isinstance(value, Mapping) else {}


def _agent_flavor(config: Mapping[str, Any], *, default: str = "bash") -> str:
    return str(_runtime_config(config).get("agent_flavor") or default)


def _task_tool_enabled(config: Mapping[str, Any]) -> bool:
    value = config.get("task_tool")
    if value is not None:
        return bool(value)
    return bool(_runtime_config(config).get("task_tool", False))


def _compression_strategy(config: Mapping[str, Any]) -> str:
    return str(_runtime_config(config).get("compression_strategy") or "summarize")


def _handoff_enabled(config: Mapping[str, Any]) -> bool:
    return bool(_runtime_config(config).get("handoff", True))


def _context_window_tokens(config: Mapping[str, Any]) -> int:
    return int(_runtime_config(config).get("context_window_tokens", 0) or 0)


def _handoff_active(config: Mapping[str, Any]) -> bool:
    """True when the chain should reset windows via model-authored handoffs.

    Handoff is the default no-compression mechanism for keeping long chains
    under the context window. It is mutually exclusive with ``summarize``
    compression, and needs a positive ``context_window_tokens`` to trigger on.
    """

    return (
        _handoff_enabled(config)
        and _compression_strategy(config) != "summarize"
        and _context_window_tokens(config) > 0
    )


def _chain_id(config: Mapping[str, Any], instance: Mapping[str, Any]) -> str:
    return str(
        config.get("chain_id")
        or config.get("repo")
        or instance.get("instance_id")
        or ""
    )


def _chain_display_name(config: Mapping[str, Any]) -> str:
    return str(config.get("chain_display_name") or config.get("chain_id") or "chain")


def _context_kwargs(
    fn: Callable[..., Any],
    context: Mapping[str, Any],
    *,
    state: State | None = None,
) -> dict[str, Any]:
    parameters = inspect.signature(fn).parameters
    kwargs: dict[str, Any] = {}
    if "context" in parameters:
        kwargs["context"] = context
    if state is not None and "state" in parameters:
        kwargs["state"] = state
    return kwargs


def _task_tool_error_texts(event: Any) -> list[str]:
    from simple_agent_lab.messages import text_of

    if not isinstance(event, MessageEvent):
        return []
    texts: list[str] = []
    for block in tool_results_of(event.message.content):
        if not block.is_error or block.tool_name != "task":
            continue
        texts.append(text_of(block.content))
    return texts


def _message_has_invalid_prompt_task_error(event: Any) -> bool:
    return any(
        is_invalid_prompt_error(RuntimeError(text))
        for text in _task_tool_error_texts(event)
    )


def _trace_artifacts(
    state: State, *, trace_id: str, producer: str, meta: Mapping[str, Any]
) -> tuple[bytes, bytes | None]:
    trace = run_trace_from_state(
        state=state,
        trace_id=trace_id,
        producer=producer,
        meta=dict(meta),
    )
    header, lines, raw_pool = event_stream(trace)
    trace_bytes = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in (header, *lines)
    ).encode("utf-8")
    raw_bytes = (
        "".join(
            json.dumps(blob, ensure_ascii=False) + "\n" for blob in raw_pool
        ).encode("utf-8")
        if raw_pool
        else None
    )
    return trace_bytes, raw_bytes


def _request_extra_for_api_kind(api_kind: str) -> dict[str, Any]:
    extra = request_extra_from_env()
    if api_kind != "openai-responses":
        return extra
    include = list(extra.get("include") or [])
    if ENCRYPTED_REASONING_INCLUDE not in include:
        include.append(ENCRYPTED_REASONING_INCLUDE)
    extra["include"] = include
    return extra


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generic eval-chain in-container runner."
    )
    parser.add_argument("--container-module", required=True)
    parser.add_argument("--suite-name", default="suite")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--workdir", default="/app")
    parser.add_argument("--max-turns", type=int, default=250)
    parser.add_argument("--provider", choices=["fake", "openai"], default="openai")
    parser.add_argument("--api-kind", choices=API_KIND_CHOICES, default="openai-chat")
    parser.add_argument("--wall-time-seconds", type=float, default=None)
    args = parser.parse_args(argv)
    store = container_store_from_env()
    instance = json.loads(store.get(INSTANCE_KEY).decode("utf-8"))
    provider = provider_from_env(kind=args.provider, api_kind=args.api_kind)
    run_chain_in_container(
        instance=instance,
        container_module=args.container_module,
        provider=provider,
        workdir=Path(args.workdir),
        max_turns=args.max_turns,
        wall_time_seconds=args.wall_time_seconds,
        store=store,
        trace_id=f"{args.suite_name}.{args.instance_id}",
        producer=f"suite:{args.suite_name}",
        request_extra=_request_extra_for_api_kind(args.api_kind),
    )
    print(f"wrote chain result for {args.instance_id} via artifact store")


if __name__ == "__main__":
    main()
