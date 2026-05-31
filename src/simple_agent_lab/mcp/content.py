"""Map MCP tool-result content into the runtime's visible blocks.

This is the multimodal heart of the MCP integration. An MCP server can
return five content shapes (text, image, audio, embedded resource,
resource link); the runtime tool boundary only carries two visible block
types (`TextBlock` and `ImageBlock`, see ``simple_agent_lab.messages``).
This module is the single, pure translation between the two.

The mapping is deliberately lossless where the runtime has a matching
block and explicit where it does not:

  * text                      -> ``TextBlock``
  * image                     -> ``ImageBlock`` (the multimodal path)
  * embedded image resource   -> ``ImageBlock``
  * embedded text resource    -> ``TextBlock``
  * audio / binary resource / -> ``TextBlock`` placeholder noting the
    resource link                kind, mime type, and size, so the model
                                 still learns the artifact exists even
                                 though the runtime can't render it yet.

Keeping this a pure ``content -> blocks`` function (no session, no I/O)
makes the whole multimodal surface unit-testable without a live server.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mcp import types as mcp_types

from simple_agent_lab.messages import ImageBlock, TextBlock


VisibleBlock = TextBlock | ImageBlock

_IMAGE_MIME_PREFIX = "image/"


def mcp_content_to_blocks(content: Iterable[Any]) -> tuple[VisibleBlock, ...]:
    """Translate an MCP tool result's content list into runtime blocks.

    Accepts the parsed ``CallToolResult.content`` items (pydantic models
    from ``mcp.types``). Unknown or unrenderable items become a short text
    placeholder rather than being dropped, so nothing the server returned
    disappears silently from the transcript.
    """

    blocks: list[VisibleBlock] = []
    for item in content:
        blocks.extend(_block_for(item))
    return tuple(blocks)


def _block_for(item: Any) -> list[VisibleBlock]:
    if isinstance(item, mcp_types.TextContent):
        return [TextBlock(item.text)]
    if isinstance(item, mcp_types.ImageContent):
        return [ImageBlock(data=item.data, mime_type=item.mimeType)]
    if isinstance(item, mcp_types.AudioContent):
        # No audio block exists at the runtime boundary yet; surface a
        # placeholder so the model still knows audio was produced.
        return [TextBlock(_placeholder("audio", item.mimeType, item.data))]
    if isinstance(item, mcp_types.EmbeddedResource):
        return [_block_for_resource(item.resource)]
    if isinstance(item, mcp_types.ResourceLink):
        return [TextBlock(_resource_link_text(item))]
    return [TextBlock(f"[unsupported MCP content: {type(item).__name__}]")]


def _block_for_resource(resource: Any) -> VisibleBlock:
    """Map an embedded resource's contents to a single block.

    Text resources become text. Binary (``blob``) resources become an
    image when their mime type is an image, otherwise a placeholder that
    records the uri, mime, and size.
    """

    text = getattr(resource, "text", None)
    if text is not None:
        return TextBlock(text)
    blob = getattr(resource, "blob", None)
    if blob is not None:
        mime = getattr(resource, "mimeType", None) or ""
        if mime.startswith(_IMAGE_MIME_PREFIX):
            return ImageBlock(data=blob, mime_type=mime)
        uri = getattr(resource, "uri", "")
        return TextBlock(_placeholder("resource", mime, blob, uri=str(uri)))
    return TextBlock(f"[empty MCP resource: {type(resource).__name__}]")


def _resource_link_text(link: mcp_types.ResourceLink) -> str:
    parts = [f"[resource link: {link.name}"]
    if link.uri:
        parts.append(f" <{link.uri}>")
    if link.mimeType:
        parts.append(f" ({link.mimeType})")
    parts.append("]")
    return "".join(parts)


def _placeholder(kind: str, mime: str, data: str, *, uri: str = "") -> str:
    """Build a compact `[kind: mime, N bytes]` note for unrenderable data."""

    detail = mime or "unknown type"
    location = f" {uri}" if uri else ""
    return f"[{kind} content:{location} {detail}, ~{_approx_bytes(data)} bytes (not rendered)]"


def _approx_bytes(b64: str) -> int:
    """Approximate the decoded byte count of a base64 string without decoding."""

    if not b64:
        return 0
    padding = b64.count("=")
    return max(0, (len(b64) * 3) // 4 - padding)
