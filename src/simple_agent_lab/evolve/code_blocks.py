"""EVOLVE-BLOCK utilities for ShinkaEvolve-style constrained code mutation.

A code payload can mark which regions a proposer may rewrite:

    # EVOLVE-BLOCK-START
    ...mutable code...
    # EVOLVE-BLOCK-END

Everything outside the markers (entry points, the scoring interface) is
immutable. These helpers extract the mutable regions, splice replacements
back, and — the part that keeps LLM edits honest — verify after a mutation
that the immutable scaffold did not change.

This is one payload convention, not a harness requirement: a code-evolution
experiment stores the full source under one payload key and applies these in
its proposer/evaluator.
"""

from __future__ import annotations

EVOLVE_START = "# EVOLVE-BLOCK-START"
EVOLVE_END = "# EVOLVE-BLOCK-END"


def _split(source: str) -> tuple[list[str], list[str]]:
    """Split `source` into (immutable segments, mutable blocks).

    Immutable segments include the marker lines themselves, so rejoining as
    `seg[0] + block[0] + seg[1] + block[1] + ... + seg[n]` reproduces the
    source exactly. Raises `ValueError` on unbalanced or nested markers.
    """

    segments: list[str] = []
    blocks: list[str] = []
    seg_begin = 0  # where the current immutable segment started
    search = 0  # where to look for the next marker pair
    while True:
        start = source.find(EVOLVE_START, search)
        end = source.find(EVOLVE_END, search)
        if start == -1 and end == -1:
            segments.append(source[seg_begin:])
            return segments, blocks
        if start == -1 or (end != -1 and end < start):
            raise ValueError(
                f"unbalanced markers: {EVOLVE_END!r} without a preceding "
                f"{EVOLVE_START!r}"
            )
        if end == -1:
            raise ValueError(
                f"unbalanced markers: {EVOLVE_START!r} without a closing {EVOLVE_END!r}"
            )
        # The start marker (and its line break) stays on the immutable side.
        newline = source.find("\n", start)
        block_begin = len(source) if newline == -1 else newline + 1
        if block_begin > end:
            raise ValueError("EVOLVE-BLOCK markers must be on separate lines")
        block = source[block_begin:end]
        if EVOLVE_START in block:
            raise ValueError("nested EVOLVE-BLOCK markers are not supported")
        segments.append(source[seg_begin:block_begin])
        blocks.append(block)
        # The end marker opens the next immutable segment; keep searching
        # past it so it is not re-found as a stray closer.
        seg_begin = end
        search = end + len(EVOLVE_END)


def evolve_blocks(source: str) -> list[str]:
    """The mutable region contents, in order of appearance."""

    _, blocks = _split(source)
    return blocks


def replace_evolve_blocks(source: str, replacements: list[str]) -> str:
    """Splice new contents into the marked regions, scaffold untouched.

    `replacements` must match the block count; raises `ValueError` otherwise.
    """

    segments, blocks = _split(source)
    if len(replacements) != len(blocks):
        raise ValueError(
            f"expected {len(blocks)} replacement blocks, got {len(replacements)}"
        )
    parts: list[str] = []
    for segment, replacement in zip(segments, replacements):
        parts.append(segment)
        parts.append(replacement)
    parts.append(segments[-1])
    return "".join(parts)


def check_immutable_regions(original: str, mutated: str) -> None:
    """Raise `ValueError` if `mutated` touched anything outside the blocks.

    The guard for LLM whole-file rewrites: run it in the proposer (or
    evaluator) and a candidate that edited the scaffold is recorded as a
    failure instead of scoring with an illegal solution.
    """

    original_segments, _ = _split(original)
    mutated_segments, _ = _split(mutated)
    if len(original_segments) != len(mutated_segments):
        raise ValueError(
            "mutation changed the number of EVOLVE-BLOCK regions: "
            f"{len(original_segments) - 1} -> {len(mutated_segments) - 1}"
        )
    if original_segments != mutated_segments:
        raise ValueError("mutation changed code outside the EVOLVE-BLOCK regions")
