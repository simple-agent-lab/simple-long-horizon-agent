"""Agent skills: discover SKILL.md packages, advertise them, inject bodies.

A skill is a directory with a ``SKILL.md`` (YAML frontmatter + body) plus
optional ``scripts/`` and ``references/``. The framework only does three
deterministic things: discover skills into metadata (path, not body), render
a ``<skills_instructions>`` menu, and inject a named skill's body on a
``$mention``. The model loads a skill by *reading* its ``SKILL.md`` (the
``read`` tool) and runs its scripts via ``bash`` — there is no skill-execution
engine. See ``docs/decisions/0021-add-agent-skills.md``.
"""

from .discovery import (
    BUNDLED_LIBRARY_DIR,
    SkillMetadata,
    SkillRoot,
    default_skill_roots,
    discover_skills,
    load_skill_from_file,
    parse_frontmatter,
)

__all__ = [
    "BUNDLED_LIBRARY_DIR",
    "SkillMetadata",
    "SkillRoot",
    "default_skill_roots",
    "discover_skills",
    "load_skill_from_file",
    "parse_frontmatter",
]
