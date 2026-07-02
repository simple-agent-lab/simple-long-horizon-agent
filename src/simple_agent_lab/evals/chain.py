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
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Literal, cast

from simple_agent_lab.agents.flavors import build_flavor_agent, run_goal_flavor
from simple_agent_lab.compression import SummarizeStrategy, summarize_compression
from simple_agent_lab.context_view import ContextPolicy, estimate_context_tokens
from simple_agent_lab.core import Agent, run as run_agent
from simple_agent_lab.evals.in_container import provider_from_env
from simple_agent_lab.evals.protocols import (
    INSTANCE_KEY,
    RESULT_KEY,
    TRACE_KEY,
    AgentSpec,
    ArtifactStore,
    ContainerTask,
)
from simple_agent_lab.evals.stores import container_store_from_env
from simple_agent_lab.llm import Provider
from simple_agent_lab.llm.env import API_KIND_CHOICES, request_extra_from_env
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.messages import (
    AssistantMessage,
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
    ContextCompressionEvent,
    Event,
    MessageEvent,
    TurnStartEvent,
)
from simple_agent_lab.state import State
from simple_agent_lab.trace import event_stream, run_trace_from_state
from simple_agent_lab.trace.jsonl import json_safe
from simple_agent_lab.workflow import final_output

CHAIN_DATA_KEY = "eval_chain"
CHAIN_STATE_INPUT_KEY = "input/chain_state.json"
CHAIN_STATE_OUTPUT_KEY = "out/chain_state.json"
CHAIN_CONFIG_KEY = "input/chain_config.json"
CHAIN_STATE_SCHEMA = "simple-agent-lab.eval-chain-state.v1"
ENCRYPTED_REASONING_INCLUDE = "reasoning.encrypted_content"
INVALID_PROMPT_TOOL_REMINDER = (
    "Removed invalid_prompt-triggering tool call/output. Use another command."
)
INVALID_PROMPT_ITEM_END_MESSAGE = (
    "Chain item {item_id} ends here; it was skipped because tool output kept "
    "triggering invalid_prompt. Continue to the next item."
)
INVALID_PROMPT_TOOL_RETRY_LIMIT = 20
InvalidPromptSource = Literal["chain_task", "tool_output", "unknown"]

CHAIN_GOAL_PREFACE = (
    "You are working through one long chained problem made of many smaller "
    "sub-problems, delivered to you in order. The transcript above is your own "
    "accumulated work on the earlier sub-problems in this chain; reuse that "
    "context and what you already learned instead of starting over. Focus only "
    "on the current sub-problem now, and make its solution address just that "
    "sub-problem."
)

CONTEXT_WINDOW_HANDOFF_REASON = "context_window_handoff"

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


def start_chain_state(
    task: ContentInput,
    *,
    agent_name: str,
    metadata: Mapping[str, Any] | None = None,
) -> State:
    """Create the persistent transcript seed for one eval chain."""

    state = State(task)
    state.data[CHAIN_DATA_KEY] = {
        "agent_name": agent_name,
        **dict(metadata or {}),
    }
    return state


def append_chain_task(
    state: State,
    *,
    agent_name: str,
    item_id: str,
    task: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    """Append one benchmark item prompt to an existing chain state."""

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


def state_to_chain_payload(state: State) -> dict[str, Any]:
    """Return a JSON-safe continuation payload for the state's active context."""

    active_items = state.active_context_items()
    return {
        "schema": CHAIN_STATE_SCHEMA,
        "task": _content_input_to_record(state.task),
        "messages": [_message_to_record(message) for _, message in active_items],
        "active_context_indices": list(range(len(active_items))),
        "data": json_safe(state.data),
        "meta": {
            "source_message_count": len(state.messages),
            "source_event_count": len(state.events),
        },
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
    _ensure_chain_data(state)

    messages = payload.get("messages", [])
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("chain payload 'messages' must be a list")
    for record in messages:
        if not isinstance(record, Mapping):
            raise ValueError("chain message records must be objects")
        state.record(_message_from_record(record))

    indices = payload.get("active_context_indices")
    if indices is None:
        indices = list(range(len(state.messages)))
    if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
        raise ValueError("chain payload 'active_context_indices' must be a list")
    active = [int(index) for index in indices]
    chain_data = _ensure_chain_data(state)
    state.record_event(
        ContextCompressionEvent(
            agent=str(chain_data.get("agent_name") or ""),
            summary_message_index=active[-1] if active else -1,
            compressed_message_indices=[],
            active_context_indices=active,
            before_tokens=0,
            after_tokens=0,
            strategy="chain-state-restore",
        )
    )
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


def invalid_prompt_source(state: Any, *, item_id: str) -> InvalidPromptSource:
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
        kind="context",
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
    """Return the connected tool-call/tool-result component for call ids."""

    wanted = set(tool_call_ids)
    dropped: set[int] = set()
    changed = True
    while changed:
        changed = False
        for index, message in active_items:
            calls = message_tool_calls(message)
            result_ids = {
                block.tool_call_id for block in tool_results_of(message.content)
            }
            if calls and any(call.id in wanted for call in calls):
                before = len(wanted)
                wanted.update(call.id for call in calls)
                dropped.add(index)
                changed = changed or len(wanted) != before
            if result_ids and result_ids & wanted:
                before = len(wanted)
                wanted.update(result_ids)
                dropped.add(index)
                changed = changed or len(wanted) != before
    return dropped


def repair_active_tool_pairs(state: Any, *, agent_name: str) -> bool:
    """Drop active tool-call/result orphans before the next provider request."""

    active_items = state.active_context_items()
    kept = tool_pair_safe_indices(active_items)
    if len(kept) == len(active_items):
        return False

    dropped = [index for index, _ in active_items if index not in set(kept)]
    note = user_message(
        "Removed an incomplete tool call/tool result exchange from context.",
        sender="user",
        target=agent_name,
        kind="context",
    )
    state.record(note)
    note_index = len(state.messages) - 1
    state.record_event(
        ContextCompressionEvent(
            agent=agent_name,
            summary_message_index=note_index,
            compressed_message_indices=dropped,
            active_context_indices=[*kept, note_index],
            before_tokens=0,
            after_tokens=0,
            strategy="tool-pair-orphan-repair",
        )
    )
    return True


def tool_pair_safe_indices(active_items: list[tuple[int, Any]]) -> list[int]:
    remaining = {index for index, _ in active_items}
    messages = dict(active_items)
    changed = True
    while changed:
        changed = False
        call_ids = {
            call.id
            for index in remaining
            for call in message_tool_calls(messages[index])
        }
        result_ids = {
            block.tool_call_id
            for index in remaining
            for block in tool_results_of(messages[index].content)
        }
        drop: set[int] = set()
        for index in remaining:
            calls = message_tool_calls(messages[index])
            if calls and any(call.id not in result_ids for call in calls):
                drop.add(index)
            results = tool_results_of(messages[index].content)
            if results and any(block.tool_call_id not in call_ids for block in results):
                drop.add(index)
        if drop:
            remaining -= drop
            changed = True
    return [index for index, _ in active_items if index in remaining]


def drop_chain_task_for_invalid_prompt_skip(
    state: Any, *, agent_name: str, item_id: str
) -> bool:
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
        return False

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
    return True


def end_chain_item_after_invalid_prompt_tool_retry_limit(
    state: Any, *, agent_name: str, item_id: str
) -> bool:
    """Clear active context after persistent invalid_prompt for one chain item."""

    active_items = state.active_context_items()
    if not active_items:
        return False

    end_message = user_message(
        INVALID_PROMPT_ITEM_END_MESSAGE.format(item_id=item_id),
        sender="user",
        target=agent_name,
        kind="context",
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
    return True


def message_chain_item_id(message: Any) -> str:
    details = getattr(message, "sidecar", {}).get("details", {})
    if not isinstance(details, Mapping):
        return ""
    chain = details.get("chain", {})
    if isinstance(chain, Mapping) and chain.get("item_id"):
        return str(chain.get("item_id") or "")
    swebench = details.get("swebench", {})
    if isinstance(swebench, Mapping):
        return str(swebench.get("instance_id") or "")
    return ""


def run_chain_in_container(
    *,
    instance: Mapping[str, Any],
    container_module: str,
    provider: Provider,
    workdir: Path,
    max_turns: int,
    store: ArtifactStore,
    trace_id: str,
    producer: str,
    suite_name: str,
    request_extra: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], State]:
    """Run one eval instance while preserving chain state."""

    del suite_name
    module = importlib.import_module(container_module)
    tasks = cast(ContainerTask, module)
    workdir = Path(workdir)
    config = _load_chain_config(store)
    spec = _chain_agent_spec(module, config)
    state = _load_state(store, module=module, config=config, agent_name=spec.name)
    _update_state_metadata(module, state, instance=instance, config=config, spec=spec)

    context: dict[str, Any] = {}
    prepare = getattr(module, "prepare", None)
    if callable(prepare):
        context = dict(prepare(workdir, instance) or {})

    task = tasks.build_task(instance, workdir=str(workdir))
    item_id = str(instance.get("instance_id") or "?")
    event_start = len(state.events)
    append_chain_task(
        state,
        agent_name=spec.name,
        item_id=item_id,
        task=str(task),
        details=_chain_task_details(module, instance=instance, config=config),
    )

    status = "ok"
    error = ""
    skip_reason = ""
    invalid_prompt_retries = 0
    chain_window_index = int(
        _chain_data(state).get("window_index", 1)
        if isinstance(_chain_data(state), Mapping)
        else 1
    )
    handoff_active = _handoff_active(config)
    position = int(config.get("position", 0) or 0)
    instances_in_chain = int(config.get("instances_in_chain", 0) or 0)
    is_last_instance = instances_in_chain > 0 and position >= instances_in_chain
    handoff_written = False
    handoff_context_tokens = 0
    result_product: dict[str, Any] = {"model_patch": ""}

    try:
        flavor = _agent_flavor(config, default=spec.flavor)
        if flavor == "goal":
            # Approach A: run the ORIGINAL Codex-style thread-goal loop, but seed
            # it with the shared chain state so the goal solver inherits every
            # earlier instance's context (the whole point in a repo chain). The
            # goal mechanism is unchanged (same solver, steering, budgets, and
            # update_goal completion); only the starting state and a chain preface
            # differ. Simple flavors keep the run_agent loop below.
            run_goal_flavor(
                provider,
                workdir,
                request_extra,
                name=spec.name,
                role=spec.role,
                system_prompt=spec.system_prompt,
                context_policy=_context_policy(
                    module,
                    provider=provider,
                    request_extra=request_extra,
                    config=config,
                ),
                solver_read=_solver_read(config),
                solver_task=_task_tool_enabled(config),
                objective=str(task),
                state=state,
                steering_preface=CHAIN_GOAL_PREFACE,
            )
        agent = (
            None
            if flavor == "goal"
            else _build_agent(
                provider=provider,
                workdir=workdir,
                request_extra=request_extra,
                config=config,
                container_module=container_module,
            )
        )
        while status == "ok" and agent is not None:
            repair_active_tool_pairs(state, agent_name=spec.name)
            turn_budget = _remaining_turn_budget(state.events[event_start:], max_turns)
            if turn_budget <= 0:
                prompt_source = invalid_prompt_source(state, item_id=item_id)
                if prompt_source == "tool_output" or invalid_prompt_retries:
                    end_chain_item_after_invalid_prompt_tool_retry_limit(
                        state, agent_name=spec.name, item_id=item_id
                    )
                elif prompt_source == "chain_task":
                    drop_chain_task_for_invalid_prompt_skip(
                        state, agent_name=spec.name, item_id=item_id
                    )
                status = "skipped"
                skip_reason = "invalid_prompt_turn_budget_exhausted"
                error = "invalid_prompt retry exhausted this chain item's turn budget"
                break
            try:
                for event in run_agent(agent, state, max_turns=turn_budget):
                    if isinstance(event, ContextCompressionEvent):
                        print(
                            f"[chain] {item_id}: context edit "
                            f"{event.strategy} {event.before_tokens}->{event.after_tokens}",
                            flush=True,
                        )
                    if _task_tool_enabled(config):
                        if _message_has_invalid_prompt_task_error(event):
                            raise RuntimeError("invalid_prompt surfaced by task tool")
                break
            except Exception as exc:
                if not is_invalid_prompt_error(exc):
                    raise
                prompt_source = invalid_prompt_source(state, item_id=item_id)
                provider_error = f"{type(exc).__name__}: {exc}"
                if prompt_source == "chain_task":
                    drop_chain_task_for_invalid_prompt_skip(
                        state, agent_name=spec.name, item_id=item_id
                    )
                    status = "skipped"
                    skip_reason = "invalid_prompt_chain_task"
                    error = provider_error
                    break
                if prompt_source == "tool_output" or invalid_prompt_retries:
                    if invalid_prompt_retries >= INVALID_PROMPT_TOOL_RETRY_LIMIT:
                        end_chain_item_after_invalid_prompt_tool_retry_limit(
                            state, agent_name=spec.name, item_id=item_id
                        )
                        status = "skipped"
                        skip_reason = "invalid_prompt_tool_output_retry_limit"
                        error = provider_error
                        break
                    if not replace_latest_tool_exchange_for_invalid_prompt(
                        state, agent_name=spec.name
                    ):
                        end_chain_item_after_invalid_prompt_tool_retry_limit(
                            state, agent_name=spec.name, item_id=item_id
                        )
                        status = "skipped"
                        skip_reason = "invalid_prompt_tool_exchange_not_found"
                        error = provider_error
                        break
                    invalid_prompt_retries += 1
                    continue
                raise

        if status == "ok":
            extract = tasks.extract_result
            result_product = dict(
                extract(workdir, instance, **_context_kwargs(extract, context))
            )
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    # Context-window handoff: when this instance finished cleanly and the
    # accumulated context is at/over the configured window, have the model write
    # a handoff doc and start the NEXT instance in a fresh window seeded with
    # only that doc. The finished instance's own trace/metrics still use the
    # full `state`; only the outgoing chain_state is reset.
    outgoing_state = state
    if status == "ok" and handoff_active and not is_last_instance:
        handoff_context_tokens = estimate_context_tokens(
            state.active_context_messages()
        )
        if handoff_context_tokens >= _context_window_tokens(config):
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
                    chain_window_index += 1
                    outgoing_state = _handoff_reset_state(
                        module,
                        config=config,
                        spec=spec,
                        doc=doc,
                        window_index=chain_window_index,
                    )
                    handoff_written = True
                    after_tokens = estimate_context_tokens(
                        outgoing_state.active_context_messages()
                    )
                    state.record_event(
                        ContextCompressionEvent(
                            agent=spec.name,
                            summary_message_index=-1,
                            compressed_message_indices=[],
                            active_context_indices=[
                                index for index, _ in state.active_context_items()
                            ],
                            before_tokens=handoff_context_tokens,
                            after_tokens=after_tokens,
                            strategy=CONTEXT_WINDOW_HANDOFF_REASON,
                        )
                    )
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
        "chain_part_index": int(config.get("part_index", 1) or 1),
        "chain_part_count": int(config.get("part_count", 1) or 1),
        "provider_auth_env": str(config.get("provider_auth_env") or ""),
        "agent_flavor": _agent_flavor(config, default=spec.flavor),
        "solver_read": _solver_read(config),
        "task_tool": _task_tool_enabled(config),
        "compression_strategy": _compression_strategy(config),
        "status": status,
        "error": error,
        "skip_reason": skip_reason,
        "invalid_prompt_retries": invalid_prompt_retries,
        "handoff": handoff_active,
        "handoff_written": handoff_written,
        "handoff_context_tokens": handoff_context_tokens,
        "chain_window_index": chain_window_index,
        "compression_metrics": metrics.as_dict(),
        "chain_event_start": event_start,
        "chain_event_end": len(state.events),
        **_chain_result_metadata(
            module, instance=instance, config=config, context=context
        ),
    }
    state.data["result"] = result
    _chain_data(state)["last_item_id"] = item_id
    if outgoing_state is not state:
        _chain_data(outgoing_state)["last_item_id"] = item_id

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
        store.put(
            TRACE_KEY,
            _trace_bytes(
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
                    **_chain_trace_metadata(
                        module, instance=instance, config=config, result=result
                    ),
                },
            ),
        )
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
        solver_read=_solver_read(config),
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
    agent_name: str,
) -> State:
    try:
        payload = json.loads(store.get(CHAIN_STATE_INPUT_KEY).decode("utf-8"))
    except (FileNotFoundError, OSError):
        return _start_state(module, config=config, agent_name=agent_name)
    return state_from_chain_payload(payload)


def _start_state(
    module: ModuleType, *, config: Mapping[str, Any], agent_name: str
) -> State:
    hook = getattr(module, "chain_start_state", None)
    if callable(hook):
        return hook(config=config, agent_name=agent_name)
    return start_chain_state(
        _chain_display_name(config),
        agent_name=agent_name,
        metadata={
            "chain_id": _chain_id(config, {}),
            "part_index": int(config.get("part_index", 1) or 1),
            "part_count": int(config.get("part_count", 1) or 1),
        },
    )


def _generate_handoff_doc(
    provider: Provider,
    state: State,
    spec: AgentSpec,
    request_extra: Mapping[str, Any] | None,
) -> str:
    """Have the model write a handoff doc from the finished instance's context.

    Builds a tool-less agent with the SAME name as the solver so it inherits the
    solver's visible context, then resumes one turn on the shared ``state``. The
    returned text is what seeds the next instance's fresh window.
    """

    handoff_agent = make_llm_agent(
        name=spec.name,
        provider=provider,
        role=CHAIN_HANDOFF_ROLE,
        request_extra=request_extra,
    )
    before = len(state.messages)
    _, events = handoff_agent.resume(state, CHAIN_HANDOFF_PROMPT, max_turns=2)
    for _ in events:
        pass
    return final_output(state, spec.name, after_message_index=before)


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

    state = _start_state(module, config=config, agent_name=spec.name)
    _chain_data(state)["window_index"] = window_index
    state.send(
        "context",
        "user",
        spec.name,
        CHAIN_HANDOFF_CONTEXT_PREFACE + doc,
        sidecar={"details": {"chain": {"handoff": True}}},
    )
    return state


def _load_chain_config(store: ArtifactStore) -> dict[str, Any]:
    try:
        raw = store.get(CHAIN_CONFIG_KEY)
    except (FileNotFoundError, OSError):
        return {}
    return json.loads(raw.decode("utf-8") or "{}")


def _update_state_metadata(
    module: ModuleType,
    state: State,
    *,
    instance: Mapping[str, Any],
    config: Mapping[str, Any],
    spec: AgentSpec,
) -> None:
    data = _chain_data(state)
    data.update(
        {
            "chain_id": _chain_id(config, instance),
            "part_index": int(config.get("part_index", 1) or 1),
            "part_count": int(config.get("part_count", 1) or 1),
            "agent_name": spec.name,
        }
    )
    hook = getattr(module, "chain_state_metadata", None)
    if callable(hook):
        data.update(dict(hook(instance=instance, config=config) or {}))


def _chain_agent_spec(module: ModuleType, config: Mapping[str, Any]) -> AgentSpec:
    hook = getattr(module, "chain_agent_spec", None)
    if callable(hook):
        return hook(config=config)
    factory = getattr(module, "agent_spec", None)
    spec = factory() if callable(factory) else AgentSpec()
    return replace(spec, flavor=_agent_flavor(config, default=spec.flavor))


def _chain_task_details(
    module: ModuleType, *, instance: Mapping[str, Any], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    hook = getattr(module, "chain_task_details", None)
    if callable(hook):
        return dict(hook(instance=instance, config=config) or {})
    return {}


def _chain_result_metadata(
    module: ModuleType,
    *,
    instance: Mapping[str, Any],
    config: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    hook = getattr(module, "chain_result_metadata", None)
    if callable(hook):
        return dict(hook(instance=instance, config=config, context=context) or {})
    return {}


def _chain_trace_metadata(
    module: ModuleType,
    *,
    instance: Mapping[str, Any],
    config: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    hook = getattr(module, "chain_trace_metadata", None)
    if callable(hook):
        return dict(hook(instance=instance, config=config, result=result) or {})
    return {}


def _chain_data(state: State) -> dict[str, Any]:
    data = state.data.setdefault(CHAIN_DATA_KEY, {})
    if isinstance(data, dict):
        return data
    state.data[CHAIN_DATA_KEY] = {}
    return cast(dict[str, Any], state.data[CHAIN_DATA_KEY])


def _ensure_chain_data(state: State) -> dict[str, Any]:
    existing = state.data.get(CHAIN_DATA_KEY)
    if isinstance(existing, dict):
        return existing
    state.data[CHAIN_DATA_KEY] = {}
    return cast(dict[str, Any], state.data[CHAIN_DATA_KEY])


def _content_input_to_record(content: ContentInput) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    return [_block_to_record(block) for block in content]


def _content_input_from_record(value: Any) -> ContentInput:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(_block_from_record(block) for block in value)
    return str(value or "")


def _message_to_record(message: Message) -> dict[str, Any]:
    record = {
        "role": message.role,
        "sender": message.sender,
        "target": message.target,
        "kind": message.kind,
        "content": [_block_to_record(block) for block in message.content],
        "sidecar": json_safe(message.sidecar),
    }
    if isinstance(message, AssistantMessage):
        record["model"] = message.model
        if message.usage is not None:
            record["usage"] = {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "cache_read_tokens": message.usage.cache_read_tokens,
                "cache_write_tokens": message.usage.cache_write_tokens,
            }
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


def _block_to_record(block: ContentBlock) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"kind": "text", "text": block.text}
    if isinstance(block, ImageBlock):
        return {"kind": "image", "data": block.data, "mime_type": block.mime_type}
    if isinstance(block, ThinkingBlock):
        return {
            "kind": "thinking",
            "text": block.text,
            "signature": block.signature,
            "redacted": block.redacted,
            "source_field": block.source_field,
        }
    if isinstance(block, ToolCallBlock):
        return {
            "kind": "tool_call",
            "id": block.id,
            "name": block.name,
            "arguments": json_safe(dict(block.arguments)),
        }
    if isinstance(block, ToolResultBlock):
        return {
            "kind": "tool_result",
            "tool_call_id": block.tool_call_id,
            "tool_name": block.tool_name,
            "content": [_block_to_record(item) for item in block.content],
            "is_error": block.is_error,
        }
    raise TypeError(f"Unsupported content block: {type(block)!r}")


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


def _solver_read(config: Mapping[str, Any]) -> bool:
    return bool(_runtime_config(config).get("solver_read", True))


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
    display = str(config.get("chain_display_name") or "")
    if display:
        return display
    chain_id = str(config.get("chain_id") or "chain")
    part_index = int(config.get("part_index", 1) or 1)
    part_count = int(config.get("part_count", 1) or 1)
    if part_count <= 1:
        return chain_id
    return f"{chain_id} part {part_index}/{part_count}"


def _remaining_turn_budget(events: list[Event], max_turns: int) -> int:
    used = sum(1 for event in events if isinstance(event, TurnStartEvent))
    return max(0, max_turns - used)


def _context_kwargs(
    fn: Callable[..., Any], context: Mapping[str, Any]
) -> dict[str, Any]:
    return {"context": context} if "context" in inspect.signature(fn).parameters else {}


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


def _trace_bytes(
    state: State, *, trace_id: str, producer: str, meta: Mapping[str, Any]
) -> bytes:
    trace = run_trace_from_state(
        state=state,
        trace_id=trace_id,
        producer=producer,
        meta=dict(meta),
    )
    header, lines, _raw_pool = event_stream(trace)
    return "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in (header, *lines)
    ).encode("utf-8")


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
    del args.wall_time_seconds

    store = container_store_from_env()
    instance = json.loads(store.get(INSTANCE_KEY).decode("utf-8"))
    provider = provider_from_env(kind=args.provider, api_kind=args.api_kind)
    run_chain_in_container(
        instance=instance,
        container_module=args.container_module,
        provider=provider,
        workdir=Path(args.workdir),
        max_turns=args.max_turns,
        store=store,
        trace_id=f"{args.suite_name}.{args.instance_id}",
        producer=f"suite:{args.suite_name}",
        suite_name=args.suite_name,
        request_extra=_request_extra_for_api_kind(args.api_kind),
    )
    print(f"wrote chain result for {args.instance_id} via artifact store")


if __name__ == "__main__":
    main()
