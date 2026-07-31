"""Regenerate the registry-backed config table in the configuration reference.

The env knobs that have moved into the central registry
(`simple_long_horizon_agent.config.REGISTRY`) are the source of truth for their name /
default / docs. This script renders them into the marked block of
`docs/configuration.md`, grouped by the `domain.subsystem`
hierarchy, so the catalog cannot drift from the code.

Usage:
    uv run python -m scripts.build_config_reference           # rewrite the block
    uv run python -m scripts.build_config_reference --check   # fail if stale

Knobs not yet migrated (provider, model aliases, pricing, memory, ...) stay in
the hand-written sections of the same file until they move into the registry.
"""

from __future__ import annotations

import sys
from pathlib import Path

from simple_long_horizon_agent.config import REGISTRY, EnvVar

DOC_PATH = Path(__file__).resolve().parents[1] / "docs/configuration.md"
BEGIN = "<!-- BEGIN GENERATED: config-registry (scripts/build_config_reference.py) -->"
END = "<!-- END GENERATED: config-registry -->"


def _default_cell(var: EnvVar) -> str:
    if var.default is None:
        return "unset"
    return f"`{var.default}`"


def _render() -> str:
    groups: dict[str, list[EnvVar]] = {}
    for var in REGISTRY:
        groups.setdefault(var.group, []).append(var)

    lines = [
        BEGIN,
        "<!-- Generated from simple_long_horizon_agent.config.REGISTRY — do not edit by "
        "hand; run scripts/build_config_reference.py. -->",
        "",
    ]
    for group in sorted(groups):
        lines.append(f"### `{group}`")
        lines.append("")
        lines.append("| Variable | Default | Purpose |")
        lines.append("| --- | --- | --- |")
        for var in groups[group]:
            lines.append(f"| `{var.name}` | {_default_cell(var)} | {var.doc} |")
        lines.append("")
    lines.append(END)
    return "\n".join(lines)


def _splice(text: str, block: str) -> str:
    if BEGIN not in text or END not in text:
        raise SystemExit(
            f"{DOC_PATH.name}: missing generated markers; add the BEGIN/END "
            "comments where the registry table should go."
        )
    head = text[: text.index(BEGIN)]
    tail = text[text.index(END) + len(END) :]
    return head + block + tail


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    check = "--check" in args
    current = DOC_PATH.read_text(encoding="utf-8")
    updated = _splice(current, _render())
    if check:
        if current != updated:
            print(
                f"{DOC_PATH.name} config-registry block is stale.\n"
                "Run: uv run python -m scripts.build_config_reference",
                file=sys.stderr,
            )
            return 1
        print(f"{DOC_PATH.name} config-registry block is up to date")
        return 0
    DOC_PATH.write_text(updated, encoding="utf-8")
    print(f"Regenerated config-registry block in {DOC_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
