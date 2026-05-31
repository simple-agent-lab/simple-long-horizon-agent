"""Tests for the MCP integration.

The content-mapping tests are pure and always run when the `mcp` SDK is
installed. The connection/tool tests drive a real `MCPConnection` over the
SDK's in-memory transport against a `FastMCP` server, so the async->sync
bridge, tool discovery, multimodal result mapping, and the full runtime
loop are all exercised without a subprocess or a socket.

The whole module is skipped when the optional `mcp` extra is absent, so
the core unit gate (which installs only the base + dev dependencies) stays
green.
"""

from __future__ import annotations

import base64
import logging
import unittest
from contextlib import asynccontextmanager
from typing import AsyncIterator

from simple_agent_lab.core import Agent, run
from simple_agent_lab.messages import (
    ImageBlock,
    TextBlock,
    ToolCallBlock,
    assistant_message,
    tool_results_of,
)
from simple_agent_lab.state import State

try:
    from mcp import ClientSession, types as mcp_types
    from mcp.server.fastmcp import FastMCP
    from mcp.shared.memory import (
        create_connected_server_and_client_session as connect_session,
    )

    from simple_agent_lab.mcp import (
        MCPConnection,
        MCPServerConfig,
        mcp_content_to_blocks,
    )

    HAS_MCP = True
except ImportError:  # pragma: no cover - exercised only without the extra
    HAS_MCP = False

_SKIP_REASON = "mcp extra not installed (install with: uv sync --extra mcp)"

# The in-memory MCP server logs every request at INFO; quiet it so the test
# runner's output stays readable.
logging.getLogger("mcp").setLevel(logging.WARNING)


# A 1x1 PNG, base64-encoded — enough to assert image bytes survive the trip.
_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


# --------------------------------------------------------------------------
# Pure content mapping (the multimodal heart)
# --------------------------------------------------------------------------


@unittest.skipUnless(HAS_MCP, _SKIP_REASON)
class McpContentMappingTest(unittest.TestCase):
    def test_text_content_becomes_text_block(self) -> None:
        blocks = mcp_content_to_blocks(
            [mcp_types.TextContent(type="text", text="hello")]
        )
        self.assertEqual(blocks, (TextBlock("hello"),))

    def test_image_content_becomes_image_block(self) -> None:
        blocks = mcp_content_to_blocks(
            [mcp_types.ImageContent(type="image", data=_PNG_B64, mimeType="image/png")]
        )
        self.assertEqual(blocks, (ImageBlock(data=_PNG_B64, mime_type="image/png"),))

    def test_mixed_content_preserves_order(self) -> None:
        blocks = mcp_content_to_blocks(
            [
                mcp_types.TextContent(type="text", text="see image:"),
                mcp_types.ImageContent(
                    type="image", data=_PNG_B64, mimeType="image/png"
                ),
            ]
        )
        self.assertEqual(len(blocks), 2)
        self.assertIsInstance(blocks[0], TextBlock)
        self.assertIsInstance(blocks[1], ImageBlock)

    def test_audio_content_becomes_placeholder_text(self) -> None:
        blocks = mcp_content_to_blocks(
            [mcp_types.AudioContent(type="audio", data=_PNG_B64, mimeType="audio/wav")]
        )
        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        assert isinstance(block, TextBlock)
        self.assertIn("audio", block.text)
        self.assertIn("audio/wav", block.text)

    def test_embedded_text_resource_becomes_text_block(self) -> None:
        resource = mcp_types.TextResourceContents(
            uri="file:///notes.txt", mimeType="text/plain", text="resource body"
        )
        blocks = mcp_content_to_blocks(
            [mcp_types.EmbeddedResource(type="resource", resource=resource)]
        )
        self.assertEqual(blocks, (TextBlock("resource body"),))

    def test_embedded_image_blob_resource_becomes_image_block(self) -> None:
        resource = mcp_types.BlobResourceContents(
            uri="file:///pixel.png", mimeType="image/png", blob=_PNG_B64
        )
        blocks = mcp_content_to_blocks(
            [mcp_types.EmbeddedResource(type="resource", resource=resource)]
        )
        self.assertEqual(blocks, (ImageBlock(data=_PNG_B64, mime_type="image/png"),))

    def test_embedded_binary_resource_becomes_placeholder(self) -> None:
        resource = mcp_types.BlobResourceContents(
            uri="file:///data.bin",
            mimeType="application/octet-stream",
            blob=_PNG_B64,
        )
        blocks = mcp_content_to_blocks(
            [mcp_types.EmbeddedResource(type="resource", resource=resource)]
        )
        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        assert isinstance(block, TextBlock)
        self.assertIn("file:///data.bin", block.text)
        self.assertIn("application/octet-stream", block.text)

    def test_resource_link_becomes_placeholder(self) -> None:
        link = mcp_types.ResourceLink(
            type="resource_link",
            name="report",
            uri="https://example.com/report.pdf",
            mimeType="application/pdf",
        )
        blocks = mcp_content_to_blocks([link])
        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        assert isinstance(block, TextBlock)
        self.assertIn("report", block.text)
        self.assertIn("https://example.com/report.pdf", block.text)


# --------------------------------------------------------------------------
# Connection + tool wrapping over the in-memory transport
# --------------------------------------------------------------------------


def _demo_server() -> FastMCP:
    server = FastMCP("demo")

    @server.tool(description="Echo text back as plain text.")
    def echo(text: str) -> str:
        return f"echo: {text}"

    @server.tool(description="Return a tiny PNG plus a caption (multimodal).")
    def render(label: str) -> list[mcp_types.ContentBlock]:
        return [
            mcp_types.TextContent(type="text", text=f"badge: {label}"),
            mcp_types.ImageContent(type="image", data=_PNG_B64, mimeType="image/png"),
        ]

    @server.tool(description="Always fails, to exercise error mapping.")
    def boom() -> str:
        raise ValueError("intentional failure")

    return server


def _open_connection(server: FastMCP, *, name: str = "demo") -> MCPConnection:
    @asynccontextmanager
    async def factory() -> AsyncIterator[ClientSession]:
        async with connect_session(server._mcp_server) as session:
            yield session

    return MCPConnection(factory, name=name).open()


@unittest.skipUnless(HAS_MCP, _SKIP_REASON)
class McpConnectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _open_connection(_demo_server())
        self.addCleanup(self.conn.close)

    def test_discovers_tools(self) -> None:
        names = {tool.name for tool in self.conn.tools}
        self.assertEqual(names, {"echo", "render", "boom"})

    def test_agent_tools_are_prefixed_with_server_name(self) -> None:
        tools = self.conn.agent_tools()
        names = {tool.name for tool in tools}
        self.assertEqual(names, {"demo_echo", "demo_render", "demo_boom"})

    def test_agent_tool_carries_input_schema(self) -> None:
        echo = next(t for t in self.conn.agent_tools() if t.name == "demo_echo")
        self.assertEqual(echo.parameters.get("type"), "object")
        self.assertIn("text", echo.parameters.get("properties", {}))

    def test_text_tool_call_returns_text_block(self) -> None:
        echo = next(t for t in self.conn.agent_tools() if t.name == "demo_echo")
        result = echo.execute("c1", {"text": "hi"}, lambda: False, None)
        self.assertFalse(result.is_error)
        self.assertEqual(result.content, (TextBlock("echo: hi"),))

    def test_multimodal_tool_call_returns_text_and_image(self) -> None:
        render = next(t for t in self.conn.agent_tools() if t.name == "demo_render")
        result = render.execute("c2", {"label": "release"}, lambda: False, None)
        self.assertFalse(result.is_error)
        self.assertEqual(len(result.content), 2)
        self.assertIsInstance(result.content[0], TextBlock)
        image = result.content[1]
        assert isinstance(image, ImageBlock)
        self.assertEqual(image.mime_type, "image/png")
        # The image bytes survive the round trip unchanged.
        self.assertEqual(base64.b64decode(image.data)[:8], b"\x89PNG\r\n\x1a\n")

    def test_failing_tool_maps_to_error_result(self) -> None:
        boom = next(t for t in self.conn.agent_tools() if t.name == "demo_boom")
        result = boom.execute("c3", {}, lambda: False, None)
        self.assertTrue(result.is_error)
        self.assertTrue(result.content)  # carries the server's error text

    def test_abort_before_call_short_circuits(self) -> None:
        echo = next(t for t in self.conn.agent_tools() if t.name == "demo_echo")
        result = echo.execute("c4", {"text": "hi"}, lambda: True, None)
        self.assertTrue(result.is_error)


# --------------------------------------------------------------------------
# Full runtime loop: a model calling an MCP multimodal tool
# --------------------------------------------------------------------------


@unittest.skipUnless(HAS_MCP, _SKIP_REASON)
class McpRuntimeLoopTest(unittest.TestCase):
    def test_mcp_image_result_lands_in_transcript(self) -> None:
        conn = _open_connection(_demo_server())
        self.addCleanup(conn.close)
        tools = conn.agent_tools()
        render_name = "demo_render"

        def generate(visible: list) -> object:
            # Finish once the tool result is in context; otherwise call render.
            if any(m.kind == "tool_result" for m in visible):
                return assistant_message(
                    "Rendered the badge via MCP.",
                    sender="mcp_demo",
                    target="user",
                    kind="final",
                )
            return assistant_message(
                (
                    ToolCallBlock(
                        id="call_1", name=render_name, arguments={"label": "v1"}
                    ),
                ),
                sender="mcp_demo",
                target="user",
                kind="step",
            )

        agent = Agent(name="mcp_demo", generate=generate, tools=tuple(tools))
        state = State(task="render a badge")
        state.send("task", "user", agent.name, "render a badge")
        for _ in run(agent, state, max_turns=4):
            pass

        tool_result_msgs = [m for m in state.messages if m.kind == "tool_result"]
        self.assertEqual(len(tool_result_msgs), 1)
        result_blocks = tool_results_of(tool_result_msgs[0].content)
        self.assertEqual(len(result_blocks), 1)
        inner = result_blocks[0].content
        self.assertTrue(any(isinstance(b, ImageBlock) for b in inner))
        self.assertEqual(state.messages[-1].kind, "final")


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------


@unittest.skipUnless(HAS_MCP, _SKIP_REASON)
class McpConfigTest(unittest.TestCase):
    def test_stdio_requires_command(self) -> None:
        with self.assertRaises(ValueError):
            MCPServerConfig(name="x", transport="stdio")

    def test_http_requires_url(self) -> None:
        with self.assertRaises(ValueError):
            MCPServerConfig(name="x", transport="http")

    def test_stdio_factory_helper(self) -> None:
        config = MCPServerConfig.stdio("fs", "npx", "-y", "server")
        self.assertEqual(config.command, "npx")
        self.assertEqual(config.args, ("-y", "server"))

    def test_http_factory_helper(self) -> None:
        config = MCPServerConfig.http("remote", "https://example.com/mcp")
        self.assertEqual(config.transport, "http")
        self.assertEqual(config.url, "https://example.com/mcp")


if __name__ == "__main__":
    unittest.main()
