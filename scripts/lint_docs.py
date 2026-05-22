"""Lint local documentation links and path references.

Usage:
    python scripts/lint_docs.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
CODE_PATH_REFERENCE = re.compile(
    r"`("
    r"(?:\.{1,2}/)?(?:AGENTS|CONTEXT|CONTRIBUTING|README|LICENSE)\.md|"
    r"(?:\.{1,2}/)?(?:\.agents|\.github|docs|runs|tests|evals|src|scripts)/"
    r"[A-Za-z0-9_./-]+"
    r")`"
)


@dataclass(frozen=True)
class Issue:
    path: Path
    line: int
    message: str

    def format(self) -> str:
        return f"{self.path.relative_to(ROOT)}:{self.line}: {self.message}"


def tracked_markdown_files() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.md",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return sorted(
            path
            for line in result.stdout.splitlines()
            if line
            for path in [ROOT / line]
            if path.exists()
        )

    ignored_parts = {".git", ".venv", "dist", "__pycache__"}
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not ignored_parts.intersection(path.relative_to(ROOT).parts)
    )


def line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if not target:
        return target
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    return target.split()[0]


def is_external_link(target: str) -> bool:
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target))


def resolve_local_link(source: Path, target: str) -> Path | None:
    if is_external_link(target):
        return None
    target = unquote(target.split("#", 1)[0])
    if not target:
        return source
    return (source.parent / target).resolve()


def resolve_code_path(source: Path, target: str) -> Path:
    if target.startswith(("./", "../")):
        return (source.parent / target).resolve()
    return (ROOT / target).resolve()


def local_reference_exists(path: Path) -> bool:
    if path.exists():
        return True
    if path.is_dir():
        return True
    return (path / "README.md").exists()


def lint_local_links(path: Path, text: str) -> list[Issue]:
    issues: list[Issue] = []
    for match in MARKDOWN_LINK.finditer(text):
        target = link_target(match.group(1))
        resolved = resolve_local_link(path, target)
        if resolved is None:
            continue
        if not local_reference_exists(resolved):
            issues.append(
                Issue(
                    path,
                    line_number(text, match.start(1)),
                    f"Broken local Markdown link: {target}",
                )
            )
    return issues


def lint_code_path_references(path: Path, text: str) -> list[Issue]:
    issues: list[Issue] = []
    for match in CODE_PATH_REFERENCE.finditer(text):
        target = match.group(1)
        resolved = resolve_code_path(path, target)
        if local_reference_exists(resolved):
            continue
        issues.append(
            Issue(
                path,
                line_number(text, match.start(1)),
                f"Broken local code-path reference: {target}",
            )
        )
    return issues


def lint_file(path: Path) -> list[Issue]:
    text = path.read_text(encoding="utf-8")
    return [
        *lint_local_links(path, text),
        *lint_code_path_references(path, text),
    ]


def main() -> int:
    issues: list[Issue] = []
    for path in tracked_markdown_files():
        issues.extend(lint_file(path))

    if issues:
        print("Docs lint failed:")
        for issue in issues:
            print(f"  {issue.format()}")
        return 1

    print("Docs lint passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
