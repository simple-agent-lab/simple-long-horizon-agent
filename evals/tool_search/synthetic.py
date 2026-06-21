"""Synthetic end-to-end tool-search execution bench.

This is intentionally local and deterministic: it runs the real Simple Agent
Lab loop plus real `AgentTool.execute` calls, but uses scripted agents so the
bench is reproducible without model credentials. That makes it a fast harness
for comparing tool exposure strategies before plugging in a live LLM.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from simple_agent_lab import (
    Agent,
    AgentTool,
    Message,
    TextBlock,
    ToolCallBlock,
    assistant_message,
    message_text,
    tool_results_of,
)
from simple_agent_lab.context_view import estimate_message_tokens
from simple_agent_lab.llm import Provider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.messages import make_message
from simple_agent_lab.messages import text_of
from simple_agent_lab.protocols import ModelRequestEvent
from simple_agent_lab.state import State
from simple_agent_lab.tool_search import (
    BM25ToolRetriever,
    ToolRecord,
    ToolRegistry,
    make_invoke_tool,
    make_search_tools_tool,
)
from simple_agent_lab.tools import AbortFlag, ToolResult, ToolUpdateFn, text_result

BenchMode = Literal["proxy", "dynamic_topk", "static_budgeted"]
BenchRunner = Literal["scripted", "llm"]
AGENT_NAME = "tool_search_agent"


@dataclass(frozen=True)
class SyntheticTask:
    instance_id: str
    prompt: str
    required_tool: str
    arguments: Mapping[str, Any]
    expected: Any


@dataclass(frozen=True)
class BenchResult:
    instance_id: str
    mode: str
    runner: str
    success: bool
    correct_tool: bool
    selected_tool: str
    required_tool: str
    expected: Any
    observed: Any
    gold_rank: int
    gold_in_candidates: bool
    tool_schema_tokens: int
    peak_context_tokens: int
    tool_calls: int
    errors: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "mode": self.mode,
            "runner": self.runner,
            "success": self.success,
            "correct_tool": self.correct_tool,
            "selected_tool": self.selected_tool,
            "required_tool": self.required_tool,
            "expected": self.expected,
            "observed": self.observed,
            "gold_rank": self.gold_rank,
            "gold_in_candidates": self.gold_in_candidates,
            "tool_schema_tokens": self.tool_schema_tokens,
            "peak_context_tokens": self.peak_context_tokens,
            "tool_calls": self.tool_calls,
            "errors": self.errors,
        }


@dataclass(frozen=True)
class BenchReport:
    mode: str
    runner: str
    results: tuple[BenchResult, ...]

    def summary(self) -> dict[str, Any]:
        total = len(self.results)
        if total == 0:
            return {"mode": self.mode, "runner": self.runner, "total": 0}
        return {
            "mode": self.mode,
            "runner": self.runner,
            "total": total,
            "success_rate": round(
                sum(result.success for result in self.results) / total, 3
            ),
            "correct_tool_rate": round(
                sum(result.correct_tool for result in self.results) / total, 3
            ),
            "gold_in_candidates_rate": round(
                sum(result.gold_in_candidates for result in self.results) / total, 3
            ),
            "mean_gold_rank": _mean_rank(self.results),
            "mean_schema_tokens": round(
                sum(result.tool_schema_tokens for result in self.results) / total, 1
            ),
            "mean_peak_context_tokens": round(
                sum(result.peak_context_tokens for result in self.results) / total, 1
            ),
            "tool_calls": sum(result.tool_calls for result in self.results),
            "errors": sum(result.errors for result in self.results),
        }


def default_tasks() -> tuple[SyntheticTask, ...]:
    return (
        SyntheticTask(
            "add-1",
            'Choose the right tool to add two integers. Arguments: {"a": 17, "b": 25}.',
            "add_numbers",
            {"a": 17, "b": 25},
            42,
        ),
        SyntheticTask(
            "mul-1",
            'Choose the right tool to multiply two integers. Arguments: {"a": 6, "b": 7}.',
            "multiply_numbers",
            {"a": 6, "b": 7},
            42,
        ),
        SyntheticTask(
            "reverse-1",
            'Choose the right tool to reverse text. Arguments: {"text": "drawer"}.',
            "reverse_text",
            {"text": "drawer"},
            "reward",
        ),
        SyntheticTask(
            "upper-1",
            'Choose the right tool to uppercase text. Arguments: {"text": "quiet lab"}.',
            "uppercase_text",
            {"text": "quiet lab"},
            "QUIET LAB",
        ),
        SyntheticTask(
            "words-1",
            'Choose the right tool to count words in text. Arguments: {"text": "small agents use tools"}.',
            "count_words",
            {"text": "small agents use tools"},
            4,
        ),
    )


def build_synthetic_registry(*, distractors: int = 200, seed: int = 7) -> ToolRegistry:
    records = list(_core_tool_records())
    records.extend(_distractor_records(distractors))
    rng = random.Random(seed)
    rng.shuffle(records)
    return ToolRegistry(records)


def run_bench(
    *,
    registry: ToolRegistry,
    tasks: Sequence[SyntheticTask] = (),
    mode: BenchMode,
    runner: BenchRunner = "scripted",
    top_k: int = 8,
    static_tool_limit: int = 64,
    provider: Provider | None = None,
    request_extra: Mapping[str, Any] | None = None,
    max_turns: int = 6,
) -> BenchReport:
    selected_tasks = tuple(tasks) if tasks else default_tasks()
    results = tuple(
        run_task(
            task,
            registry=registry,
            mode=mode,
            runner=runner,
            top_k=top_k,
            static_tool_limit=static_tool_limit,
            provider=provider,
            request_extra=request_extra,
            max_turns=max_turns,
        )
        for task in selected_tasks
    )
    return BenchReport(mode=mode, runner=runner, results=results)


def run_task(
    task: SyntheticTask,
    *,
    registry: ToolRegistry,
    mode: BenchMode,
    runner: BenchRunner,
    top_k: int,
    static_tool_limit: int,
    provider: Provider | None = None,
    request_extra: Mapping[str, Any] | None = None,
    max_turns: int = 6,
) -> BenchResult:
    if runner == "llm" and provider is None:
        raise ValueError("runner='llm' requires a Provider")
    if mode == "proxy":
        diagnostics = _retrieval_diagnostics(registry, task, limit=top_k)
        return _run_proxy_task(
            task,
            registry=registry,
            runner=runner,
            top_k=top_k,
            gold_rank=diagnostics[0],
            gold_in_candidates=diagnostics[1],
            provider=provider,
            request_extra=request_extra,
            max_turns=max_turns,
        )
    if mode == "dynamic_topk":
        retrieved = BM25ToolRetriever(registry).search(task.prompt, k=top_k)
        visible = tuple(result.record for result in retrieved)
        diagnostics = _rank_in_results(retrieved, task.required_tool)
        return _run_direct_task(
            task,
            mode=mode,
            runner=runner,
            visible_records=visible,
            gold_rank=diagnostics[0],
            gold_in_candidates=diagnostics[1],
            provider=provider,
            request_extra=request_extra,
            max_turns=max_turns,
        )
    if mode == "static_budgeted":
        visible_records = registry.records[: max(1, static_tool_limit)]
        visible_registry = ToolRegistry(visible_records)
        selected = tuple(BM25ToolRetriever(visible_registry).search(task.prompt, k=1))
        selected_record = selected[0].record if selected else None
        diagnostics = _retrieval_diagnostics(
            visible_registry,
            task,
            limit=len(visible_registry.records),
        )
        return _run_direct_task(
            task,
            mode=mode,
            runner=runner,
            visible_records=visible_records,
            selected_record=selected_record,
            gold_rank=diagnostics[0],
            gold_in_candidates=diagnostics[1],
            provider=provider,
            request_extra=request_extra,
            max_turns=max_turns,
        )
    raise ValueError(f"unknown bench mode: {mode}")


def write_report(path: Path, reports: Sequence[BenchReport]) -> None:
    payload = {
        "summaries": [report.summary() for report in reports],
        "results": [
            result.as_dict() for report in reports for result in report.results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run_proxy_task(
    task: SyntheticTask,
    *,
    registry: ToolRegistry,
    runner: BenchRunner,
    top_k: int,
    gold_rank: int,
    gold_in_candidates: bool,
    provider: Provider | None,
    request_extra: Mapping[str, Any] | None,
    max_turns: int,
) -> BenchResult:
    retriever = BM25ToolRetriever(registry)
    search_tool = make_search_tools_tool(registry, retriever, default_k=top_k)
    invoke_tool = make_invoke_tool(registry)
    tools = (search_tool, invoke_tool)

    if runner == "scripted":

        def brain(visible: list[Message]) -> Message:
            results = _tool_result_blocks(visible)
            if not results:
                return assistant_message(
                    [
                        TextBlock("Searching for the right executable tool."),
                        ToolCallBlock(
                            "search_1",
                            "search_tools",
                            {"query": task.prompt, "k": top_k},
                        ),
                    ],
                    sender=AGENT_NAME,
                    target="user",
                    kind="step",
                )
            if not any(block.tool_name == "invoke_tool" for block in results):
                candidate = _first_candidate_name(results)
                if not candidate:
                    return assistant_message(
                        "No candidate tool was found.",
                        sender=AGENT_NAME,
                        target="user",
                        kind="final",
                    )
                return assistant_message(
                    [
                        TextBlock(f"Invoking {candidate}."),
                        ToolCallBlock(
                            "invoke_1",
                            "invoke_tool",
                            {
                                "tool_name": candidate,
                                "arguments": dict(task.arguments),
                            },
                        ),
                    ],
                    sender=AGENT_NAME,
                    target="user",
                    kind="step",
                )
            return assistant_message(
                "Done.",
                sender=AGENT_NAME,
                target="user",
                kind="final",
            )

        agent = Agent(AGENT_NAME, brain, tools=tools)
    else:
        assert provider is not None
        agent = make_llm_agent(
            name=AGENT_NAME,
            provider=provider,
            tools=tools,
            system_prompt=_proxy_system_prompt(top_k),
            request_extra=request_extra,
        )

    state, events = agent.run(
        task.prompt,
        max_turns=max_turns,
    )
    list(events)
    return _result_from_state(
        task,
        mode="proxy",
        runner=runner,
        gold_rank=gold_rank,
        gold_in_candidates=gold_in_candidates,
        state=state,
        exposed_tools=tools,
    )


def _run_direct_task(
    task: SyntheticTask,
    *,
    mode: str,
    runner: BenchRunner,
    visible_records: Sequence[ToolRecord],
    selected_record: ToolRecord | None = None,
    gold_rank: int = 0,
    gold_in_candidates: bool = False,
    provider: Provider | None = None,
    request_extra: Mapping[str, Any] | None = None,
    max_turns: int = 6,
) -> BenchResult:
    tools = tuple(record.tool for record in visible_records)
    selected = (
        selected_record.tool
        if selected_record is not None
        else (tools[0] if tools else None)
    )

    if runner == "scripted":

        def brain(visible: list[Message]) -> Message:
            if selected is None:
                return assistant_message(
                    "No visible tool matched the task.",
                    sender=AGENT_NAME,
                    target="user",
                    kind="final",
                )
            if not _tool_result_blocks(visible):
                return assistant_message(
                    [
                        TextBlock(f"Calling {selected.name}."),
                        ToolCallBlock("direct_1", selected.name, dict(task.arguments)),
                    ],
                    sender=AGENT_NAME,
                    target="user",
                    kind="step",
                )
            return assistant_message(
                "Done.",
                sender=AGENT_NAME,
                target="user",
                kind="final",
            )

        agent = Agent(AGENT_NAME, brain, tools=tools)
    else:
        assert provider is not None
        agent = make_llm_agent(
            name=AGENT_NAME,
            provider=provider,
            tools=tools,
            system_prompt=_direct_system_prompt(),
            request_extra=request_extra,
        )

    state, events = agent.run(
        task.prompt,
        max_turns=max_turns,
    )
    list(events)
    return _result_from_state(
        task,
        mode=mode,
        runner=runner,
        gold_rank=gold_rank,
        gold_in_candidates=gold_in_candidates,
        state=state,
        exposed_tools=tools,
    )


def _result_from_state(
    task: SyntheticTask,
    *,
    mode: str,
    runner: str,
    gold_rank: int,
    gold_in_candidates: bool,
    state: State,
    exposed_tools: Sequence[AgentTool],
) -> BenchResult:
    selected = ""
    observed: Any = None
    tool_calls = 0
    errors = 0
    for message in state.messages:
        for block in tool_results_of(message.content):
            tool_calls += 1
            if block.is_error:
                errors += 1
            if block.tool_name == "search_tools":
                continue
            if block.tool_name == "invoke_tool":
                details = message.sidecar.get("details", {})
                if isinstance(details, Mapping):
                    invoke_details = details.get(block.tool_call_id)
                    if isinstance(invoke_details, Mapping):
                        selected = str(invoke_details.get("tool_name") or "")
                        result_details = invoke_details.get("result_details")
                        observed = _observed_value(result_details)
                if observed is None:
                    observed = _parse_result_text(message_text(message))
                continue
            selected = block.tool_name
            details = message.sidecar.get("details", {})
            if isinstance(details, Mapping):
                observed = _observed_value(details.get(block.tool_call_id))
            if observed is None:
                observed = _parse_result_text(message_text(message))

    success = observed == task.expected
    return BenchResult(
        instance_id=task.instance_id,
        mode=mode,
        runner=runner,
        success=success,
        correct_tool=selected == task.required_tool,
        selected_tool=selected,
        required_tool=task.required_tool,
        expected=task.expected,
        observed=observed,
        gold_rank=gold_rank,
        gold_in_candidates=gold_in_candidates,
        tool_schema_tokens=sum(_tool_schema_tokens(tool) for tool in exposed_tools),
        peak_context_tokens=_peak_context_tokens(state),
        tool_calls=tool_calls,
        errors=errors,
    )


def _core_tool_records() -> tuple[ToolRecord, ...]:
    return (
        _record(
            "add_numbers",
            "Add two integers and return their sum.",
            {"a": "integer", "b": "integer"},
            lambda args: int(args["a"]) + int(args["b"]),
            tags=("math", "addition", "integer"),
            examples=("add 17 and 25",),
        ),
        _record(
            "multiply_numbers",
            "Multiply two integers and return their product.",
            {"a": "integer", "b": "integer"},
            lambda args: int(args["a"]) * int(args["b"]),
            tags=("math", "multiplication", "integer"),
            examples=("multiply 6 by 7",),
        ),
        _record(
            "reverse_text",
            "Reverse the characters in a text string.",
            {"text": "string"},
            lambda args: str(args["text"])[::-1],
            tags=("text", "reverse", "string"),
            examples=("reverse drawer",),
        ),
        _record(
            "uppercase_text",
            "Convert text to uppercase letters.",
            {"text": "string"},
            lambda args: str(args["text"]).upper(),
            tags=("text", "uppercase", "case"),
            examples=("uppercase quiet lab",),
        ),
        _record(
            "count_words",
            "Count whitespace-separated words in text.",
            {"text": "string"},
            lambda args: len(str(args["text"]).split()),
            tags=("text", "word-count", "string"),
            examples=("count words in a sentence",),
        ),
    )


def _distractor_records(count: int) -> tuple[ToolRecord, ...]:
    topics = (
        (
            "add_shipping_fee",
            "Add a shipping fee to an invoice amount.",
            {"a": "integer", "b": "integer"},
            lambda args: int(args["a"]) + 5,
        ),
        (
            "multiply_pixels",
            "Multiply image width and height to estimate pixels.",
            {"a": "integer", "b": "integer"},
            lambda args: int(args["a"]) * int(args["b"]) + 1,
        ),
        (
            "reverse_dns_lookup",
            "Reverse lookup a DNS-style hostname string.",
            {"text": "string"},
            lambda args: "not-found",
        ),
        (
            "uppercase_title_words",
            "Uppercase only title words in a heading.",
            {"text": "string"},
            lambda args: str(args["text"]).title(),
        ),
        (
            "count_unique_terms",
            "Count unique terms in text.",
            {"text": "string"},
            lambda args: len(set(str(args["text"]).split())),
        ),
        (
            "add_calendar_days",
            "Add calendar days to a date-like value.",
            {"a": "integer", "b": "integer"},
            lambda args: int(args["a"]) + int(args["b"]) + 7,
        ),
    )
    records: list[ToolRecord] = []
    for index in range(max(0, count)):
        base, description, schema, fn = topics[index % len(topics)]
        name = f"{base}_{index:04d}"
        records.append(
            _record(
                name,
                description + f" Synthetic distractor #{index}.",
                schema,
                fn,
                tags=("synthetic", "distractor", base.replace("_", "-")),
            )
        )
    return tuple(records)


def _record(
    name: str,
    description: str,
    schema: Mapping[str, str],
    fn: Callable[[Mapping[str, Any]], Any],
    *,
    tags: Sequence[str] = (),
    examples: Sequence[str] = (),
) -> ToolRecord:
    parameters = {
        "type": "object",
        "properties": {
            arg: {"type": typ, "description": f"{arg} input."}
            for arg, typ in schema.items()
        },
        "required": list(schema),
        "additionalProperties": False,
    }

    def execute(
        call_id: str,
        args: dict[str, Any],
        abort: AbortFlag,
        on_update: ToolUpdateFn | None,
    ) -> ToolResult:
        del call_id, abort, on_update
        value = fn(args)
        return text_result(
            json.dumps({"value": value}, sort_keys=True), details={"value": value}
        )

    return ToolRecord(
        AgentTool(
            name=name,
            description=description,
            parameters=parameters,
            execute=execute,
        ),
        namespace="synthetic",
        tags=tuple(tags),
        examples=tuple(examples),
        when_to_use=description,
    )


def _tool_result_blocks(messages: Sequence[Message]):
    return [block for message in messages for block in tool_results_of(message.content)]


def _first_candidate_name(blocks: Sequence[Any]) -> str:
    for block in blocks:
        if block.tool_name != "search_tools":
            continue
        try:
            payload = json.loads(text_of(block.content))
        except (TypeError, ValueError):
            payload = {}
        tools = payload.get("tools")
        if isinstance(tools, list) and tools:
            first = tools[0]
            if isinstance(first, Mapping):
                return str(first.get("name") or first.get("tool_id") or "")
    return ""


def _observed_value(details: Any) -> Any:
    if isinstance(details, Mapping) and "value" in details:
        return details["value"]
    return None


def _parse_result_text(text: str) -> Any:
    matches = re.findall(r"\{[^{}]*\"value\"[^{}]*\}", text)
    for match in reversed(matches):
        try:
            payload = json.loads(match)
        except ValueError:
            continue
        if isinstance(payload, Mapping) and "value" in payload:
            return payload["value"]
    return None


def _tool_schema_tokens(tool: AgentTool) -> int:
    msg = make_message(
        "system",
        json.dumps(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
            sort_keys=True,
        ),
        sender="runtime",
        target="agent",
        kind="context",
    )
    return estimate_message_tokens(msg)


def _peak_context_tokens(state: State) -> int:
    return max(
        (
            int(event.context_view.get("estimated_tokens", 0))
            for event in state.events
            if isinstance(event, ModelRequestEvent)
        ),
        default=0,
    )


def _retrieval_diagnostics(
    registry: ToolRegistry,
    task: SyntheticTask,
    *,
    limit: int,
) -> tuple[int, bool]:
    results = BM25ToolRetriever(registry).search(
        task.prompt,
        k=max(0, limit),
    )
    return _rank_in_results(results, task.required_tool)


def _rank_in_results(
    results: Sequence[Any],
    required_tool: str,
) -> tuple[int, bool]:
    for result in results:
        if result.record.tool.name == required_tool:
            return int(result.rank), True
    return 0, False


def _mean_rank(results: Sequence[BenchResult]) -> float:
    ranks = [result.gold_rank for result in results if result.gold_rank > 0]
    if not ranks:
        return 0.0
    return round(sum(ranks) / len(ranks), 1)


def _proxy_system_prompt(top_k: int) -> str:
    return (
        "You are running a tool-search execution benchmark. The task text gives "
        "the exact arguments object to use. First call search_tools with a short "
        f"query and k={top_k}. Then choose the best returned concrete tool and "
        "call invoke_tool with its name and the exact arguments from the task. "
        "After the tool result, give a short final answer with the observed value."
    )


def _direct_system_prompt() -> str:
    return (
        "You are running a tool execution benchmark. The task text gives the "
        "exact arguments object to use. Choose exactly one visible concrete tool "
        "whose schema and description match the task, call it with those exact "
        "arguments, then give a short final answer with the observed value."
    )
