"""Parse user-facing skill directives out of a task string.

This is the entire "/" command surface: a thin user->harness layer parsed
before the model, deliberately not generalized into a slash-command framework
(matching how Codex/Claude Code keep `/` as a control layer separate from
model-invoked tools). Exactly two slash directives are understood:

- ``/no-skills`` disables skills for this run (the token is stripped so the
  model never sees it). ``no-skills`` is reserved and never treated as a skill.
- ``/name`` explicitly invokes a known skill (left in the task text so the
  model sees the mention too). A ``/name`` token only counts when ``name``
  matches a discovered skill, so ordinary paths like ``/usr/bin`` are ignored.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


# Whole-word `/no-skills` directive, not glued to surrounding non-space text.
NO_SKILLS_RE = re.compile(r"(?<!\S)/no-skills(?!\S)")
# A `/name` skill mention: a slash at a token boundary (start-of-text or after
# whitespace) so a mid-path slash (`/usr/bin`) does not match the inner
# segments. Resolution against the known-skills set is what ultimately gates it.
MENTION_RE = re.compile(r"(?<!\S)/([A-Za-z0-9][A-Za-z0-9_-]*)")
# Slash tokens that are control directives, never skill mentions.
RESERVED_DIRECTIVES = {"no-skills"}


@dataclass(frozen=True)
class SkillDirectives:
    skills_enabled: bool
    mentions: tuple[str, ...]
    cleaned_task: str


def parse_skill_directives(task: str, skill_names: Iterable[str]) -> SkillDirectives:
    """Parse ``/no-skills`` and ``/name`` skill mentions from ``task``."""

    known = set(skill_names)

    disabled = NO_SKILLS_RE.search(task) is not None
    cleaned = NO_SKILLS_RE.sub("", task) if disabled else task
    # Collapse the whitespace left by a stripped directive.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    mentions: list[str] = []
    seen: set[str] = set()
    if not disabled:
        for match in MENTION_RE.finditer(task):
            token = match.group(1)
            if token in seen or token in RESERVED_DIRECTIVES:
                continue
            if token not in known:
                continue
            seen.add(token)
            mentions.append(token)

    return SkillDirectives(
        skills_enabled=not disabled,
        mentions=tuple(mentions),
        cleaned_task=cleaned,
    )
