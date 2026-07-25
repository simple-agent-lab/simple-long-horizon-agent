"""Discover SKILL.md files on disk into a deduplicated list of metadata.

The cornerstone decision (from Codex): store the *path* to ``SKILL.md``,
never the body. The body is read lazily, only when a skill is actually used,
so the always-present menu stays cheap even with many skills.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkillMetadata:
    """One discovered skill. ``read_body`` is the only place the body loads."""

    name: str
    description: str
    path_to_skill_md: str  # absolute path to SKILL.md -- NOT the body
    base_dir: str  # directory holding SKILL.md; resolves scripts/ and references/
    scope: str = "user"  # "repo" | "user" | "bundled"

    def read_body(self) -> str:
        with open(self.path_to_skill_md, "r", encoding="utf-8") as handle:
            return handle.read()


@dataclass(frozen=True)
class SkillRoot:
    path: str
    scope: str


# The skill library shipped inside the package (and the wheel).
BUNDLED_LIBRARY_DIR = str(Path(__file__).resolve().parent / "library")

# Lower rank wins on a name collision. A project ("repo") skill overrides a
# user-global one, which overrides a bundled default of the same name.
SCOPE_RANK = {"repo": 0, "user": 1, "bundled": 2}

DEFAULT_PROJECT_SUBDIRS = (".agents/skills", ".simple_agent_lab/skills")

MAX_DESCRIPTION_LEN = 1024


def default_skill_roots(
    cwd: str,
    home: str | None = None,
    *,
    project_subdirs: tuple[str, ...] = DEFAULT_PROJECT_SUBDIRS,
) -> list[SkillRoot]:
    """Bundled library first, then project scope (walk up to the git root),
    then user scope. Order does not set priority — ``SCOPE_RANK`` does."""

    resolved_home = home if home is not None else os.path.expanduser("~")
    roots: list[SkillRoot] = [SkillRoot(BUNDLED_LIBRARY_DIR, "bundled")]

    current = os.path.abspath(cwd)
    while True:
        for sub in project_subdirs:
            roots.append(SkillRoot(os.path.join(current, sub), "repo"))
        parent = os.path.dirname(current)
        if parent == current or os.path.isdir(os.path.join(current, ".git")):
            break
        current = parent

    for sub in project_subdirs:
        roots.append(SkillRoot(os.path.join(resolved_home, sub), "user"))

    return roots


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Return ``(frontmatter, body)``. Minimal YAML-ish parser: reads only
    ``key: value`` lines from the leading ``---`` block. Good enough for the
    ``name``/``description`` we need; the body is opaque."""

    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    _, raw_fm, body = parts
    frontmatter: dict[str, str] = {}
    for line in raw_fm.splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition(":")
            frontmatter[key.strip()] = value.strip().strip('"').strip("'")
    return frontmatter, body.lstrip("\n")


def load_skill_from_file(path: str, scope: str) -> SkillMetadata | None:
    """Build metadata from one SKILL.md. A missing description is the one
    hard skip (the description is the entire reason a skill gets picked)."""

    with open(path, "r", encoding="utf-8") as handle:
        frontmatter, _body = parse_frontmatter(handle.read())

    base_dir = os.path.dirname(os.path.abspath(path))
    name = (frontmatter.get("name") or os.path.basename(base_dir)).strip()
    description = (frontmatter.get("description") or "").strip()

    if not description:
        return None
    if len(description) > MAX_DESCRIPTION_LEN:
        description = description[:MAX_DESCRIPTION_LEN]

    return SkillMetadata(
        name=name,
        description=description,
        path_to_skill_md=os.path.abspath(path),
        base_dir=base_dir,
        scope=scope,
    )


def discover_skills_under_root(
    root: SkillRoot, *, max_depth: int = 6
) -> list[SkillMetadata]:
    """Find skills under one root. A directory that contains a SKILL.md *is* a
    skill; do not recurse into it (so references/ etc. are not re-scanned)."""

    skills: list[SkillMetadata] = []
    if not os.path.isdir(root.path):
        return skills
    base_depth = root.path.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root.path):
        if dirpath.count(os.sep) - base_depth >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in dirnames if not d.startswith(".") and d != "node_modules"
        ]
        if "SKILL.md" in filenames:
            skill = load_skill_from_file(os.path.join(dirpath, "SKILL.md"), root.scope)
            if skill is not None:
                skills.append(skill)
            dirnames[:] = []  # a skill dir is a leaf
    return skills


def discover_skills(roots: list[SkillRoot]) -> list[SkillMetadata]:
    """Discover, dedup by canonical path, first-wins by scope rank on name
    collision. Returns skills sorted by name for a stable menu."""

    seen_paths: set[str] = set()
    by_name: dict[str, SkillMetadata] = {}
    for root in roots:
        for skill in discover_skills_under_root(root):
            canonical = os.path.realpath(skill.path_to_skill_md)
            if canonical in seen_paths:
                continue
            seen_paths.add(canonical)
            existing = by_name.get(skill.name)
            if existing is None or _scope_rank(skill.scope) < _scope_rank(
                existing.scope
            ):
                by_name[skill.name] = skill
    return sorted(by_name.values(), key=lambda s: s.name)


def _scope_rank(scope: str) -> int:
    return SCOPE_RANK.get(scope, len(SCOPE_RANK))
