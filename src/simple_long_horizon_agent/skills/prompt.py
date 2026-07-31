"""Render the <skills_instructions> menu and wrap it as a runtime message.

The menu is the only "matching engine": a list of name + description + path,
plus the verbatim "how to use skills" prose that tells the model how to load
and use a skill (it reads SKILL.md, resolves scripts/ relative to the skill
dir, loads only the references it needs). The prose is adapted from Codex
(``.idea/codex/codex-rs/core-skills/src/render.rs``); the deferred 2%-budget
and path-aliasing machinery is intentionally not implemented.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from simple_long_horizon_agent.messages import RuntimeMessage, runtime_message

from .discovery import (
    SkillMetadata,
    SkillRoot,
    default_skill_roots,
    discover_skills,
)


# Adapted verbatim from Codex (core-skills/src/render.rs:
# SKILLS_INTRO_WITH_ABSOLUTE_PATHS + SKILLS_HOW_TO_USE_WITH_ABSOLUTE_PATHS).
# The only change is "open its SKILL.md" -> "open it with the `read` tool",
# since this repo loads content through a dedicated reader rather than an
# implicit file-open. Keeping the standard wording is deliberate: it is
# well-tuned prompt engineering that drives reliable skill selection.
SKILLS_INTRO = (
    "A skill is a set of local instructions to follow that is stored in a "
    "`SKILL.md` file. Below is the list of skills that can be used. Each entry "
    "includes a name, description, and file path so you can open the source "
    "for full instructions when using a specific skill."
)

SKILLS_HOW_TO_USE = """\
- Discovery: The list above is the skills available in this session (name + description + file path). Skill bodies live on disk at the listed paths.
- Trigger rules: If the user names a skill (with `/skill-name` or plain text) OR the task clearly matches a skill's description shown above, you must use that skill for that turn. Multiple mentions mean use them all. Do not carry skills across turns unless re-mentioned.
- Missing/blocked: If a named skill isn't in the list or the path can't be read, say so briefly and continue with the best fallback.
- How to use a skill (progressive disclosure):
  1) After deciding to use a skill, open its `SKILL.md` with the `read` tool. Read only enough to follow the workflow.
  2) When `SKILL.md` references relative paths (e.g., `scripts/foo.py`), resolve them relative to the skill directory listed above first, and only consider other paths if needed.
  3) If `SKILL.md` points to extra folders such as `references/`, load only the specific files needed for the request; don't bulk-load everything. (`read` a directory path to list a skill's files if you're unsure what it bundles.)
  4) If `scripts/` exist, prefer running or patching them with the `bash` tool instead of retyping large code blocks.
  5) If `assets/` or templates exist, reuse them instead of recreating from scratch. Loading content uses `read`; executing uses `bash`.
- Coordination and sequencing:
  - If multiple skills apply, choose the minimal set that covers the request and state the order you'll use them.
  - Announce which skill(s) you're using and why (one short line). If you skip an obvious skill, say why.
- Context hygiene:
  - Keep context small: summarize long sections instead of pasting them; only load extra files when needed.
  - Avoid deep reference-chasing: prefer opening only files directly linked from `SKILL.md` unless you're blocked.
  - When variants exist (frameworks, providers, domains), pick only the relevant reference file(s) and note that choice.
- Safety and fallback: If a skill can't be applied cleanly (missing files, unclear instructions), state the issue, pick the next-best approach, and continue."""


def render_skills_instructions(skills: Sequence[SkillMetadata]) -> str:
    """Render the literal ``<skills_instructions>`` block, or ``""`` if empty."""

    if not skills:
        return ""
    lines = [
        f"- {skill.name}: {skill.description} (file: {skill.path_to_skill_md})"
        for skill in skills
    ]
    return (
        "<skills_instructions>\n"
        f"{SKILLS_INTRO}\n\n"
        "### Available skills\n"
        + "\n".join(lines)
        + "\n\n### How to use skills\n"
        + SKILLS_HOW_TO_USE
        + "\n</skills_instructions>"
    )


def skills_menu_message(
    skills: Sequence[SkillMetadata], *, target: str = "all"
) -> RuntimeMessage | None:
    """Wrap the menu as a runtime message, or ``None`` when there are no skills.

    A system-kind message is folded into the provider's system area by both
    adapters (OpenAI Chat emits a ``role="system"`` entry; Anthropic appends it
    to the top-level ``system`` string), so the menu lands where Codex puts it
    while staying a visible, toggleable entry in ``state.events``."""

    block = render_skills_instructions(skills)
    if not block:
        return None
    return runtime_message(block, sender="skills", target=target, kind="system")


def system_prompt_with_skills(
    base_prompt: str,
    *,
    cwd: str | Path,
    home: str | None = None,
    roots: Sequence[SkillRoot] | None = None,
) -> str:
    """Fold the discovered skills menu into a base system prompt.

    For agent flavors that run through the generic ``agent.run`` loop (e.g. the
    benchmark path), there is no per-turn message seam to record a menu into, so
    the ``<skills_instructions>`` menu rides the system prompt instead — the same
    content ``run_with_skills`` records as a runtime message on the interactive
    edge. When no skill is discovered the prompt is returned unchanged, so an
    empty bundled library leaves the baseline untouched.

    Pass explicit ``roots`` to control discovery; otherwise the default scopes
    (bundled library, project under ``cwd``, user under ``home``) are scanned.
    """

    skill_roots = (
        list(roots) if roots is not None else default_skill_roots(str(cwd), home)
    )
    menu = render_skills_instructions(discover_skills(skill_roots))
    if not menu:
        return base_prompt
    return f"{base_prompt}\n\n{menu}" if base_prompt else menu
