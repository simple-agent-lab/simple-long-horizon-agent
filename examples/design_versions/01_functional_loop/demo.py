"""Functional agent-loop sketch.

Run:

    python3 examples/design_versions/01_functional_loop/demo.py
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Optional

from simple_agent_lab.llm import (
    LLMRequest,
    llm_response_to_assistant_message,
    messages_to_llm_messages,
    tool_to_llm_tool,
)
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.llm import complete as llm_complete
from simple_agent_lab.messages import (
    Message,
    MessageContent,
    message_tool_calls,
    system_message,
    tool_result_message,
    user_message,
)
from simple_agent_lab.tools import (
    AgentTool as Tool,
    ToolResult,
    text_result,
    tool_result_text,
)


ModelFn = Callable[[list["Message"], list["Tool"]], "Message"]


TASK = (
    "I bought 3 shirts at $19.99 each and 2 pairs of pants at $34.50 each. "
    "With 8.5% tax, what's the total?"
)
INSTRUCTION = (
    "You are a helpful shopping assistant. Use `calculate` for any arithmetic."
)


def dispatch_tool_calls(
    assistant_msg: Message,
    tools: dict[str, Tool],
) -> tuple[list[Message], bool]:
    """Run the explicit tool calls carried by an assistant message.

    Returns (tool_result messages in original tool-call order, terminated).
    Exceptions from `execute` are caught and converted into
    `is_error=True` results — they never bubble out. The model sees the
    error text and can self-correct.
    """
    out: list[Message] = []
    terminated = False
    for tc in message_tool_calls(assistant_msg):
        name = tc.name
        call_id = tc.id
        args = dict(tc.arguments)

        tool = tools.get(name)
        if tool is None:
            result = text_result(f"Tool {name!r} not found", is_error=True)
        elif tool.execute is None:
            result = text_result(
                f"Tool {name!r} has no execute function",
                is_error=True,
            )
        else:
            try:
                result = tool.execute(args)
            except Exception as exc:
                result = text_result(f"{type(exc).__name__}: {exc}", is_error=True)

        out.append(tool_result_message(
            tool_result_text(result),
            tool_call_id=call_id,
            tool_name=name,
            sender=name,
            target=assistant_msg.sender or "assistant",
            is_error=result.is_error,
            data={
                "details": result.details,
                "content_blocks": [
                    {"kind": c.kind, "text": c.text, "image_url": c.image_url}
                    for c in result.content
                ],
            },
        ))
        if result.terminate:
            terminated = True
    return out, terminated


def context_view(
    messages: list[Message],
    instruction: str,
    last: int | None = None,
) -> list[Message]:
    visible = messages[-last:] if last is not None else messages
    return [
        system_message(instruction, sender="system", kind="instruction"),
        *visible,
    ]


def make_model(
    provider: LLMProvider,
    system_prompt: str = "",
    request_extra: Optional[dict[str, Any]] = None,
) -> ModelFn:
    """Build a `ModelFn` that talks to `provider` via `simple_agent_lab.llm`.

    Pass a real `Provider(api="anthropic-messages", ...)` once that adapter
    lands; for now, `Provider(api="fake", ...)` works out of the box.
    """
    request_options = dict(request_extra or {})

    def model(messages: list[Message], tools: list[Tool] = ()) -> Message:
        req = LLMRequest(
            provider=provider,
            messages=messages_to_llm_messages(messages),
            tools=[tool_to_llm_tool(t) for t in tools],
            system_prompt=system_prompt or None,
            extra=request_options,
        )
        resp = llm_complete(req)
        return llm_response_to_assistant_message(
            resp,
            sender=provider.model,
            target="user" if resp.stop_reason == "end_turn" else "assistant",
            kind="final" if resp.stop_reason == "end_turn" else "thought",
        )

    return model


def as_agent_message(agent_name: str, output: Message) -> Message:
    return replace(
        output,
        sender=agent_name,
        target="user" if output.kind == "final" else agent_name,
        data={**dict(output.data), "model_sender": output.sender},
    )


def run_loop(
    task: str,
    instruction: str,
    model: ModelFn,
    tools: list[Tool] = (),
    max_steps: int = 3,
) -> list[Message]:
    tool_registry: dict[str, Tool] = {t.name: t for t in tools}
    messages = [
        user_message(task, sender="user", target="assistant", kind="task")
    ]
    for _ in range(max_steps):
        output = model(context_view(messages, instruction), list(tools))
        message = as_agent_message("assistant", output)
        messages.append(message)

        # Dispatch any tool_calls embedded in the assistant message.
        # Tool calls do not stop the loop — the model gets results next turn.
        if tool_registry and message_tool_calls(message):
            results, terminated = dispatch_tool_calls(message, tool_registry)
            messages.extend(results)
            if terminated:
                break
            continue

        if message.kind == "final":
            break
    return messages


def print_trace(messages: list[Message]) -> None:
    for index, message in enumerate(messages):
        print(f"{index:02d} {message.kind:<11} {message.sender:>10} {message.content}")


# --------------------------------------------------------------------------
# Demo: a shopping-helper agent that uses a `calculate` tool to do arithmetic.
#
# Realistic shape of how a user writes an agent with this loop:
#   1. Define one or more `Tool`s.
#   2. Configure a model (here, the LLM layer's `fake` adapter — swap for
#      `Provider(api="anthropic-messages", ...)` once that adapter lands).
#   3. Call `run_loop` with the task, instruction, model, and tools.


def calculate(args: dict) -> ToolResult:
    """Evaluate a simple arithmetic expression."""
    expression = args.get("expression", "")
    try:
        # `eval` with empty builtins is OK for arithmetic from a known
        # caller. Real tools should use a safe expression parser.
        value = eval(expression, {"__builtins__": {}}, {})
        return text_result(
            f"{expression} = {value}",
            details={"value": value},
        )
    except Exception as exc:
        return text_result(
            f"Could not evaluate {expression!r}: {exc}",
            is_error=True,
        )


calculator = Tool(
    name="calculate",
    description="Evaluate a math expression and return the numeric result.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A Python-style arithmetic expression.",
            },
        },
        "required": ["expression"],
    },
    execute=calculate,
)


def shopping_helper_model() -> ModelFn:
    """A model that calls `calculate` first, then narrates the answer.

    The fake adapter is deterministic, but it still reads the same
    messages, tools, and tool results that a real provider adapter would see.
    """
    provider = LLMProvider(id="fake", api="fake", model="fake-model")
    instruction = "You are a helpful shopping assistant. Use `calculate` for any arithmetic."
    return make_model(provider, system_prompt=instruction)


def run_demo(model: ModelFn | None = None) -> list[Message]:
    return run_loop(
        task=TASK,
        instruction=INSTRUCTION,
        model=model or shopping_helper_model(),
        tools=[calculator],
        max_steps=4,
    )


def main() -> None:
    messages = run_demo()
    print_trace(messages)


if __name__ == "__main__":
    main()
