"""Run an agent that calls a multimodal MCP tool.

This demo starts the local MCP server in `scripts/mcp_demo_server.py` as a
subprocess (stdio transport), discovers its tools, wraps them as runtime
`AgentTool`s, and runs an agent that calls the multimodal `render_swatch`
tool. The PNG the server returns flows back as an `ImageBlock` inside the
tool result — the same boundary the bash tool's image attachments use — so
the model would see a real image on its next turn.

Examples:

    uv run --extra mcp python -m scripts.run_mcp_agent_demo
    uv run --extra mcp python -m scripts.run_mcp_agent_demo --list
    uv run --extra mcp python -m scripts.run_mcp_agent_demo --color gold --save-image /tmp/swatch.png

The default run is deterministic: a small scripted `generate` issues the
tool call (no API key needed) while still going through the real runtime
path — model request, MCP tool call, tool result, final answer. The block
of code building the tool list is exactly what you would hand to
`make_llm_agent(..., tools=tools)` to let a live model drive these tools.
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

from simple_agent_lab import (
    Agent,
    ImageBlock,
    State,
    ToolCallBlock,
    assistant_message,
    message_text,
    print_trace,
    run,
    text_of,
    tool_results_of,
)
from simple_agent_lab.mcp import MCPServerConfig, connect_mcp


ROOT = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = ROOT / "scripts" / "mcp_demo_server.py"


def make_scripted_generate(tool_name: str, color: str):
    """A deterministic `generate`: call the MCP tool, then finish.

    Swap this for `make_llm_agent(..., tools=tools).generate` to let a real
    model decide when and how to call the MCP tools.
    """

    def generate(visible: list) -> object:
        if any(message.kind == "tool_result" for message in visible):
            return assistant_message(
                "Rendered the swatch via the MCP server.",
                sender="mcp_demo",
                target="user",
                kind="final",
            )
        return assistant_message(
            (
                ToolCallBlock(
                    id="call_1",
                    name=tool_name,
                    arguments={"color": color, "size": 48},
                ),
            ),
            sender="mcp_demo",
            target="user",
            kind="step",
        )

    return generate


def print_tool_result_blocks(state: State, *, save_image: Path | None) -> None:
    saved = False  # write only the first image so multiple results don't collide
    for message in state.messages:
        if message.kind != "tool_result":
            continue
        for block in tool_results_of(message.content):
            for inner in block.content:
                if isinstance(inner, ImageBlock):
                    raw = base64.b64decode(inner.data)
                    print(
                        f"  [image] {inner.mime_type}, {len(raw)} bytes "
                        f"(from {block.tool_name})"
                    )
                    if save_image is not None and not saved:
                        save_image.write_bytes(raw)
                        print(f"  [image] saved to {save_image}")
                        saved = True
                else:
                    print(f"  [text ] {text_of((inner,))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal MCP agent demo")
    parser.add_argument(
        "--color",
        default="crimson",
        help="Palette color to render (crimson, teal, gold, indigo).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Just connect and list the server's tools, then exit.",
    )
    parser.add_argument(
        "--save-image",
        default=None,
        metavar="PATH",
        help="Write the returned PNG swatch to PATH so you can open it.",
    )
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()

    # This is the whole integration surface: describe the server, connect,
    # and turn its tools into runtime AgentTools.
    config = MCPServerConfig.stdio("swatch", sys.executable, str(SERVER_SCRIPT))
    print(f"=== connecting to MCP server: {config.name} (stdio) ===")
    with connect_mcp(config) as conn:
        tools = conn.agent_tools()
        print(f"discovered {len(tools)} tool(s): {', '.join(t.name for t in tools)}")
        if args.list:
            for tool in tools:
                print(f"\n- {tool.name}: {tool.description}")
            return

        render_tool = next(t for t in tools if t.name.endswith("render_swatch"))
        agent = Agent(
            name="mcp_demo",
            generate=make_scripted_generate(render_tool.name, args.color),
            tools=tuple(tools),
        )
        state = State(task=f"render a {args.color} swatch")
        state.send("task", "user", agent.name, state.task)
        for _ in run(agent, state, max_turns=4):
            pass

    save_image = Path(args.save_image) if args.save_image else None
    print("\n=== MCP tool result blocks ===")
    print_tool_result_blocks(state, save_image=save_image)

    final = next(
        message for message in reversed(state.messages) if message.kind == "final"
    )
    print("\n=== final ===")
    print(message_text(final))

    if not args.no_trace:
        print("\n=== full trace ===")
        print_trace(state)


if __name__ == "__main__":
    main()
