"""A tiny multimodal MCP server for the demo, spoken over stdio.

Run indirectly: `scripts/run_mcp_agent_demo.py` launches this file as a
subprocess and connects to it as an MCP client. You can also smoke it by
hand with any MCP client that speaks stdio.

It exposes two tools:

  * `describe_palette(name)` — plain-text tool, returns a short blurb.
  * `render_swatch(color, size)` — multimodal tool, returns a caption plus
    a PNG image of a solid color square. This is the path the demo cares
    about: an image content block flowing back to the model.

The PNG is generated with the standard library only (`zlib` + `struct`),
so the demo has no image dependency.
"""

from __future__ import annotations

import base64
import struct
import zlib

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent


# Quiet the per-request INFO logs so the demo's own output reads cleanly;
# the server's stderr is inherited by the client subprocess.
server = FastMCP("swatch-demo", log_level="WARNING")

# A small, fixed palette so the demo is deterministic.
_PALETTE: dict[str, tuple[int, int, int]] = {
    "crimson": (220, 20, 60),
    "teal": (0, 128, 128),
    "gold": (255, 215, 0),
    "indigo": (75, 0, 130),
}


@server.tool(description="Describe one of the demo palette colors in a sentence.")
def describe_palette(name: str) -> str:
    rgb = _PALETTE.get(name.lower())
    if rgb is None:
        return f"Unknown color {name!r}. Try one of: {', '.join(sorted(_PALETTE))}."
    return f"{name.title()} is RGB{rgb} — a good accent for buttons and badges."


@server.tool(
    description=(
        "Render a solid-color square as a PNG and return it as an image, "
        "with a caption. `color` is a palette name; `size` is the side length "
        "in pixels (8-128)."
    )
)
def render_swatch(color: str, size: int = 48) -> list[TextContent | ImageContent]:
    rgb = _PALETTE.get(color.lower(), (128, 128, 128))
    side = max(8, min(128, int(size)))
    png_b64 = _solid_png_base64(side, rgb)
    return [
        TextContent(type="text", text=f"{color} swatch, {side}x{side}px, RGB{rgb}"),
        ImageContent(type="image", data=png_b64, mimeType="image/png"),
    ]


def _solid_png_base64(side: int, rgb: tuple[int, int, int]) -> str:
    """Encode a `side`x`side` solid-color RGB image as base64 PNG (stdlib only)."""

    r, g, b = rgb
    row = b"\x00" + bytes((r, g, b)) * side  # filter byte 0 + RGB pixels
    raw = row * side
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, 9))
        + _png_chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


if __name__ == "__main__":
    server.run()
