"""In-container runner for SWE-bench Pro repo-session experiments.

Unlike the generic runner, this runner resumes a repository-level conversation
from ``input/session_state.json``, appends the current instance task, runs the
agent inside the instance container, and writes the next continuation payload to
``out/session_state.json``.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Literal, cast

from simple_agent_lab.agents.flavors import build_flavor_agent
from simple_agent_lab.compression import SummarizeStrategy, summarize_compression
from simple_agent_lab.context_view import ContextPolicy
from simple_agent_lab.core import Agent, run as run_agent
from simple_agent_lab.evals.in_container import provider_from_env
from simple_agent_lab.evals.protocols import (
    INSTANCE_KEY,
    RESULT_KEY,
    TRACE_KEY,
    ContainerTask,
    ArtifactStore,
)
from simple_agent_lab.evals.stores import container_store_from_env
from simple_agent_lab.llm import Provider
from simple_agent_lab.llm.env import API_KIND_CHOICES, request_extra_from_env
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.messages import (
    is_tool_result_message,
    message_tool_calls,
    tool_results_of,
    user_message,
)
from simple_agent_lab.protocols import (
    ContextCompressionEvent,
    Event,
    MessageEvent,
    ModelRequestEvent,
    TurnStartEvent,
)
from simple_agent_lab.state import State
from simple_agent_lab.trace import event_stream, run_trace_from_state

from .container import AGENT_NAME, AGENT_ROLE, AGENT_SYSTEM_PROMPT
from .repo_session_state import (
    SESSION_CONFIG_KEY,
    SESSION_STATE_INPUT_KEY,
    SESSION_STATE_OUTPUT_KEY,
    append_repo_session_task,
    start_repo_session_state,
    state_from_session_payload,
    state_to_session_payload,
)

ENCRYPTED_REASONING_INCLUDE = "reasoning.encrypted_content"
INVALID_PROMPT_TOOL_REMINDER = (
    "刚刚的工具调用及其输出会触发 invalid_prompt，已从上下文移除。请使用其他命令继续。"
)
INVALID_PROMPT_INSTANCE_END_MESSAGE = (
    "上一道题 {instance_id} 在这里结束；因为工具输出持续触发 invalid_prompt，"
    "已跳过该实例。继续下一道题。"
)
INVALID_PROMPT_TOOL_RETRY_LIMIT = 20
CONTEXT_WINDOW_RESET_REASON = "context_window_reset"
InvalidPromptSource = Literal["instance_task", "tool_output", "unknown"]


def run_repo_session_in_container(
    *,
    instance: Mapping[str, Any],
    provider: Provider,
    workdir: Path,
    max_turns: int,
    store: ArtifactStore,
    trace_id: str,
    producer: str,
    suite_name: str,
    request_extra: Mapping[str, Any] | None = None,
    container_module: str = "simple_agent_lab.evals.suites.swebench.container",
) -> tuple[dict[str, Any], State]:
    """Run one SWE-bench Pro instance while preserving repo-session state."""

    del suite_name
    module = importlib.import_module(container_module)
    tasks = cast(ContainerTask, module)
    workdir = Path(workdir)
    config = _load_session_config(store)
    state = _load_state(store, config)
    _update_state_metadata(state, instance=instance, config=config)

    context: dict[str, Any] = {}
    prepare = getattr(module, "prepare", None)
    if callable(prepare):
        context = dict(prepare(workdir, instance) or {})

    task = tasks.build_task(instance, workdir=str(workdir))
    instance_id = str(instance.get("instance_id") or "?")
    event_start = len(state.events)
    append_repo_session_task(
        state,
        agent_name=AGENT_NAME,
        instance_id=instance_id,
        task=str(task),
    )

    status = "ok"
    error = ""
    skip_reason = ""
    invalid_prompt_retries = 0
    context_window_restarts = 0
    chain_window_index = int(
        state.data.get("repo_chain", {}).get("window_index", 1)
        if isinstance(state.data.get("repo_chain"), Mapping)
        else 1
    )
    max_context_restarts = int(config.get("max_context_restarts_per_instance", 0) or 0)
    result_product: dict[str, Any] = {"model_patch": ""}

    try:
        agent = _build_agent(
            provider=provider,
            workdir=workdir,
            request_extra=request_extra,
            config=config,
        )
        while status == "ok":
            _repair_active_tool_pairs(state, agent_name=AGENT_NAME)
            turn_budget = _remaining_turn_budget(state.events[event_start:], max_turns)
            if turn_budget <= 0:
                prompt_source = _invalid_prompt_source(state, instance_id=instance_id)
                if prompt_source == "tool_output" or invalid_prompt_retries:
                    _end_instance_after_invalid_prompt_tool_retry_limit(
                        state, agent_name=AGENT_NAME, instance_id=instance_id
                    )
                elif prompt_source == "instance_task":
                    _drop_instance_task_for_invalid_prompt_skip(
                        state, agent_name=AGENT_NAME, instance_id=instance_id
                    )
                status = "skipped"
                skip_reason = "invalid_prompt_turn_budget_exhausted"
                error = "invalid_prompt retry exhausted this instance's turn budget"
                break
            try:
                for event in run_agent(agent, state, max_turns=turn_budget):
                    if isinstance(event, ContextCompressionEvent):
                        print(
                            f"[repo_session] {instance_id}: context edit "
                            f"{event.strategy} {event.before_tokens}->{event.after_tokens}",
                            flush=True,
                        )
                    if _mode(config) == "chain_task":
                        _raise_chain_context_errors(event, config=config)
                        if _message_has_invalid_prompt_task_error(event):
                            raise RuntimeError("invalid_prompt surfaced by task tool")
                break
            except Exception as exc:
                if _mode(config) == "chain_task" and _is_context_window_error(exc):
                    if context_window_restarts >= max_context_restarts:
                        raise
                    context_window_restarts += 1
                    chain_window_index += 1
                    state = start_repo_session_state(
                        _session_display_name(config), agent_name=AGENT_NAME
                    )
                    state.data["repo_chain"] = {
                        "window_index": chain_window_index,
                        "agent_flavor": "bash_task",
                        "compression_strategy": "none",
                    }
                    _update_state_metadata(state, instance=instance, config=config)
                    event_start = len(state.events)
                    append_repo_session_task(
                        state,
                        agent_name=AGENT_NAME,
                        instance_id=instance_id,
                        task=str(task),
                    )
                    continue
                if not _is_invalid_prompt_error(exc):
                    raise
                prompt_source = _invalid_prompt_source(state, instance_id=instance_id)
                provider_error = f"{type(exc).__name__}: {exc}"
                if prompt_source == "instance_task":
                    _drop_instance_task_for_invalid_prompt_skip(
                        state, agent_name=AGENT_NAME, instance_id=instance_id
                    )
                    status = "skipped"
                    skip_reason = "invalid_prompt_instance_task"
                    error = provider_error
                    break
                if prompt_source == "tool_output" or invalid_prompt_retries:
                    if invalid_prompt_retries >= INVALID_PROMPT_TOOL_RETRY_LIMIT:
                        _end_instance_after_invalid_prompt_tool_retry_limit(
                            state, agent_name=AGENT_NAME, instance_id=instance_id
                        )
                        status = "skipped"
                        skip_reason = "invalid_prompt_tool_output_retry_limit"
                        error = provider_error
                        break
                    if not _replace_latest_tool_exchange_for_invalid_prompt(
                        state, agent_name=AGENT_NAME
                    ):
                        _end_instance_after_invalid_prompt_tool_retry_limit(
                            state, agent_name=AGENT_NAME, instance_id=instance_id
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

    metrics = summarize_compression(state.events[event_start:])
    result = {
        **result_product,
        "instance_id": instance_id,
        "repo": str(config.get("repo") or instance.get("repo") or ""),
        "session_id": str(config.get("session_id") or config.get("repo") or ""),
        "session_part_index": int(config.get("part_index", 1) or 1),
        "session_part_count": int(config.get("part_count", 1) or 1),
        "provider_auth_env": str(config.get("provider_auth_env") or ""),
        "agent_flavor": _agent_flavor(config),
        "compression_strategy": _compression_strategy(config),
        "status": status,
        "error": error,
        "skip_reason": skip_reason,
        "invalid_prompt_retries": invalid_prompt_retries,
        "context_window_restarts": context_window_restarts,
        "chain_window_index": chain_window_index,
        "baseline_commit": str(context.get("baseline_commit") or ""),
        "compression_metrics": metrics.as_dict(),
        "session_event_start": event_start,
        "session_event_end": len(state.events),
    }
    state.data["result"] = result
    state.data.setdefault("repo_session", {})["last_instance_id"] = instance_id

    store.put(
        RESULT_KEY, (json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8")
    )
    store.put(
        SESSION_STATE_OUTPUT_KEY,
        (json.dumps(state_to_session_payload(state), ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )
    if bool(config.get("write_trajectories", False)):
        store.put(
            TRACE_KEY,
            _trace_bytes(
                state,
                trace_id=trace_id,
                producer=producer,
                meta={
                    "repo": result["repo"],
                    "session_id": result["session_id"],
                    "instance_id": instance_id,
                    "status": status,
                    "provider_auth_env": result["provider_auth_env"],
                    "agent_flavor": result["agent_flavor"],
                    "compression_strategy": result["compression_strategy"],
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
) -> Agent:
    return build_flavor_agent(
        flavor=_agent_flavor(config),
        provider=provider,
        cwd=workdir,
        name=AGENT_NAME,
        role=AGENT_ROLE,
        system_prompt=AGENT_SYSTEM_PROMPT,
        request_extra=request_extra,
        context_policy=_context_policy(
            provider=provider,
            request_extra=request_extra,
            config=config,
        ),
        enable_default_compression=False,
    )


def _context_policy(
    *,
    provider: Provider,
    request_extra: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> ContextPolicy:
    if _mode(config) == "chain_task" or _compression_strategy(config) != "summarize":
        return ContextPolicy()
    runtime = _runtime_config(config)
    compressor = make_llm_agent(
        name="swebench_compressor",
        provider=provider,
        role=(
            "Summarize older SWE-bench repo-session context. Preserve durable "
            "facts, decisions, tool results, constraints, file paths, test "
            "signals, and unresolved questions. Omit low-value wording."
        ),
        request_extra=request_extra,
    )
    return ContextPolicy(
        strategy=SummarizeStrategy(
            compressor=compressor,
            threshold_tokens=int(runtime.get("threshold_tokens", 217600) or 217600),
            keep_recent=int(runtime.get("keep_recent", 12) or 12),
            preserve_kinds=tuple(runtime.get("preserve_kinds") or ()),
        )
    )


def _load_state(store: ArtifactStore, config: Mapping[str, Any]) -> State:
    try:
        payload = json.loads(store.get(SESSION_STATE_INPUT_KEY).decode("utf-8"))
    except (FileNotFoundError, OSError):
        return start_repo_session_state(
            _session_display_name(config), agent_name=AGENT_NAME
        )
    return state_from_session_payload(payload)


def _load_session_config(store: ArtifactStore) -> dict[str, Any]:
    try:
        raw = store.get(SESSION_CONFIG_KEY)
    except (FileNotFoundError, OSError):
        return {}
    return json.loads(raw.decode("utf-8") or "{}")


def _update_state_metadata(
    state: State, *, instance: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    repo_session = state.data.setdefault("repo_session", {})
    if isinstance(repo_session, dict):
        repo_session.update(
            {
                "repo": str(config.get("repo") or instance.get("repo") or ""),
                "session_id": str(config.get("session_id") or ""),
                "part_index": int(config.get("part_index", 1) or 1),
                "part_count": int(config.get("part_count", 1) or 1),
                "agent_name": AGENT_NAME,
            }
        )


def _runtime_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("config")
    return value if isinstance(value, Mapping) else {}


def _mode(config: Mapping[str, Any]) -> str:
    return str(config.get("mode") or "compression")


def _agent_flavor(config: Mapping[str, Any]) -> str:
    if _mode(config) == "chain_task":
        return "bash_task"
    return str(_runtime_config(config).get("agent_flavor") or "bash")


def _compression_strategy(config: Mapping[str, Any]) -> str:
    if _mode(config) == "chain_task":
        return "none"
    return str(_runtime_config(config).get("compression_strategy") or "summarize")


def _session_display_name(config: Mapping[str, Any]) -> str:
    display = str(config.get("session_display_name") or "")
    if display:
        return display
    repo = str(config.get("repo") or "unknown")
    part_index = int(config.get("part_index", 1) or 1)
    part_count = int(config.get("part_count", 1) or 1)
    if part_count <= 1:
        return repo
    return f"{repo} part {part_index}/{part_count}"


def _remaining_turn_budget(events: list[Event], max_turns: int) -> int:
    used = sum(1 for event in events if isinstance(event, TurnStartEvent))
    return max(0, max_turns - used)


def _context_kwargs(
    fn: Callable[..., Any], context: Mapping[str, Any]
) -> dict[str, Any]:
    return {"context": context} if "context" in inspect.signature(fn).parameters else {}


def _is_invalid_prompt_error(exc: BaseException) -> bool:
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


def _invalid_prompt_source(state: State, *, instance_id: str) -> InvalidPromptSource:
    for _, message in reversed(state.active_context_items()):
        if getattr(message, "role", "") != "user":
            continue
        if is_tool_result_message(message):
            return "tool_output"
        if _message_swebench_instance_id(message) == instance_id:
            return "instance_task"
        return "unknown"
    return "unknown"


def _replace_latest_tool_exchange_for_invalid_prompt(
    state: State, *, agent_name: str
) -> bool:
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
    dropped = _tool_exchange_indices(active_items, tool_call_ids)
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


def _tool_exchange_indices(
    active_items: list[tuple[int, Any]], tool_call_ids: set[str]
) -> set[int]:
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


def _repair_active_tool_pairs(state: State, *, agent_name: str) -> bool:
    active_items = state.active_context_items()
    kept = _tool_pair_safe_indices(active_items)
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


def _tool_pair_safe_indices(active_items: list[tuple[int, Any]]) -> list[int]:
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


def _drop_instance_task_for_invalid_prompt_skip(
    state: State, *, agent_name: str, instance_id: str
) -> bool:
    active_items = state.active_context_items()
    target_index: int | None = None
    for index, message in reversed(active_items):
        if (
            getattr(message, "role", "") == "user"
            and _message_swebench_instance_id(message) == instance_id
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
            strategy="invalid-prompt-instance-task-drop",
        )
    )
    return True


def _end_instance_after_invalid_prompt_tool_retry_limit(
    state: State, *, agent_name: str, instance_id: str
) -> bool:
    active_items = state.active_context_items()
    if not active_items:
        return False
    end_message = user_message(
        INVALID_PROMPT_INSTANCE_END_MESSAGE.format(instance_id=instance_id),
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


def _message_swebench_instance_id(message: Any) -> str:
    details = getattr(message, "sidecar", {}).get("details", {})
    if not isinstance(details, Mapping):
        return ""
    swebench = details.get("swebench", {})
    if not isinstance(swebench, Mapping):
        return ""
    return str(swebench.get("instance_id") or "")


def _is_context_window_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    code = getattr(exc, "code", None)
    return (
        "context_length_exceeded" in text
        or "context window" in text
        or "maximum context" in text
        or "context length" in text
        or "too many tokens" in text
        or "input is too long" in text
        or code == "context_length_exceeded"
    )


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
        _is_invalid_prompt_error(RuntimeError(text))
        for text in _task_tool_error_texts(event)
    )


def _raise_chain_context_errors(event: Any, *, config: Mapping[str, Any]) -> None:
    if isinstance(event, ModelRequestEvent):
        context_window_tokens = int(
            _runtime_config(config).get("context_window_tokens", 0) or 0
        )
        if context_window_tokens > 0:
            estimated_tokens = int(event.context_view.get("estimated_tokens", 0) or 0)
            if estimated_tokens >= context_window_tokens:
                raise RuntimeError(
                    "context_length_exceeded: model request estimated context "
                    f"{estimated_tokens} reached configured limit "
                    f"{context_window_tokens}"
                )
    for text in _task_tool_error_texts(event):
        if _is_context_window_error(RuntimeError(text)):
            raise RuntimeError("context_length_exceeded surfaced by task tool")


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
        description="SWE-bench Pro repo-session in-container runner."
    )
    parser.add_argument("--container-module", required=True)
    parser.add_argument("--suite-name", default="swebench_pro")
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
    run_repo_session_in_container(
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
    print(f"wrote repo-session result for {args.instance_id} via artifact store")


if __name__ == "__main__":
    main()
