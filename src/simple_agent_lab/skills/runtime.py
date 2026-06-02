"""The visible edge that wires skills into a run.

``run_with_skills`` is the skills analogue of ``Agent.run``: it builds the
``State``, records the menu and any ``$mention`` skill bodies *before* the
task (so they are present at the first sample), seeds the task, and drives the
existing ``core.run`` loop. Recording up front — rather than mutating context
mid-flight — keeps every message the model sees visible in ``state.events``,
exactly like ``task_tool`` records its context messages. The core loop is
untouched: loading a skill (the ``read`` tool) and running its scripts (the
``bash`` tool) are ordinary tool calls.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence

from simple_agent_lab.core import Agent, run
from simple_agent_lab.messages import Message, user_message
from simple_agent_lab.protocols import Event
from simple_agent_lab.state import State
from simple_agent_lab.tools import AbortFlag

from .directives import parse_skill_directives
from .discovery import SkillMetadata, SkillRoot, default_skill_roots, discover_skills
from .prompt import skills_menu_message


SKILL_MANIFEST_MAX_ENTRIES = 50
SKILL_MANIFEST_MAX_DEPTH = 2


def skill_body_messages(
    skills: Sequence[SkillMetadata], *, target: str = "all"
) -> list[Message]:
    """Wrap each skill's body in a ``<skill>`` context message, with a short
    ``<files>`` manifest of the skill directory so the model knows which
    references/scripts/schemas it can load next (it enumerates; it does not
    eagerly read them). Bodies that can't be read are skipped (the menu still
    advertises them)."""

    messages: list[Message] = []
    for skill in skills:
        try:
            body = skill.read_body()
        except OSError:
            continue
        manifest = _skill_file_manifest(skill.base_dir)
        files_block = f"<files>\n{manifest}\n</files>\n" if manifest else ""
        content = (
            f"<skill>\n<name>{skill.name}</name>\n"
            f"<path>{skill.path_to_skill_md}</path>\n{body}\n{files_block}</skill>"
        )
        messages.append(
            user_message(content, sender="skills", target=target, kind="context")
        )
    return messages


def _skill_file_manifest(base_dir: str) -> str:
    """A shallow, sorted list of a skill's files (excluding SKILL.md and
    dotfiles), one relative path per line, capped for context hygiene."""

    entries: list[str] = []
    for root, dirnames, filenames in os.walk(base_dir):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        depth = os.path.relpath(root, base_dir).count(os.sep) if root != base_dir else 0
        if depth >= SKILL_MANIFEST_MAX_DEPTH:
            dirnames[:] = []
        for filename in sorted(filenames):
            if filename.startswith(".") or (
                root == base_dir and filename == "SKILL.md"
            ):
                continue
            rel = os.path.relpath(os.path.join(root, filename), base_dir)
            entries.append(rel.replace(os.sep, "/"))
            if len(entries) >= SKILL_MANIFEST_MAX_ENTRIES:
                return "\n".join(entries)
    return "\n".join(entries)


def run_with_skills(
    agent: Agent,
    task: str,
    *,
    skills: Sequence[SkillMetadata] | None = None,
    roots: Sequence[SkillRoot] | None = None,
    preload: Sequence[str] = (),
    cwd: str = ".",
    max_turns: int = 10,
    abort: AbortFlag = lambda: False,
) -> tuple[State, Iterator[Event]]:
    """Run ``agent`` on ``task`` with skills advertised and injected.

    Skills are on by default. ``skills`` (pre-discovered) takes precedence over
    ``roots``; if neither is given, discovery uses ``default_skill_roots(cwd)``.
    Pass a filtered ``skills=`` list (and/or ``preload`` of names) to scope
    skills per agent — there are no per-agent directories. ``preload`` names
    have their full bodies injected up front (like Claude Code's subagent
    ``skills:`` field), in addition to any ``$mention`` in the task. A
    ``/no-skills`` directive disables the whole layer for this run. ``agent`` is
    expected to already carry a ``read`` tool (to load skill content) and a
    ``bash`` tool (to run skill scripts).

    Returns ``(state, events)`` like ``Agent.run``: ``events`` is a lazy
    generator the caller iterates to advance the loop, and ``state`` is
    populated as it runs.
    """

    discovered: Sequence[SkillMetadata]
    if skills is not None:
        discovered = skills
    else:
        discovered = discover_skills(
            list(roots) if roots is not None else default_skill_roots(cwd)
        )

    directives = parse_skill_directives(task, [s.name for s in discovered])

    state = State(task=directives.cleaned_task)
    if directives.skills_enabled and discovered:
        menu = skills_menu_message(discovered, target=agent.name)
        if menu is not None:
            state.record(menu)
        inject_names = set(directives.mentions) | set(preload)
        inject = [s for s in discovered if s.name in inject_names]
        for message in skill_body_messages(inject, target=agent.name):
            state.record(message)

    state.send("task", "user", agent.name, directives.cleaned_task)
    events = run(agent, state, max_turns=max_turns, abort=abort)
    return state, events
