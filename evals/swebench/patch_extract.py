"""Patch extraction helpers for SWE-bench-style agent workspaces.

The filtering rules mirror SWALM's SWE task patch collection rules, but this
adapter writes them to `.git/info/exclude` so the generated ignore block itself
does not become part of the model patch.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


IGNORE_BLOCK_START = "# === SIMPLE AGENT LAB SWE-BENCH AUTO-GENERATED START ==="
IGNORE_BLOCK_END = "# === SIMPLE AGENT LAB SWE-BENCH AUTO-GENERATED END ==="

DEFAULT_GITIGNORE_RULES = [
    "*.jpg",
    "*.png",
    "*.jpeg",
    "*.o",
    "*.out",
    "*.obj",
    "*.so",
    "build",
    "Build",
]

LANGUAGE_GITIGNORE_RULES = {
    "c": ["bin/", "lib/", "*.dylib"],
    "cpp": ["bin/", "lib/", "*.dylib"],
    "java": ["target/", "out/", "*.class", "*.jar", ".gradle/"],
    "js": [
        "node_modules/",
        "dist/",
        ".next/",
        "coverage/",
        ".env",
        "npm-debug.log*",
        "yarn-debug.log*",
        "yarn-error.log*",
    ],
    "ts": [
        "node_modules/",
        "build/",
        "dist/",
        ".next/",
        "coverage/",
        ".env",
        "npm-debug.log*",
        "yarn-debug.log*",
        "yarn-error.log*",
        "*.js",
        "*.js.map",
        "*.d.ts",
        ".tsbuildinfo",
    ],
    "go": ["pkg/", "vendor/", "bin/", "*.test"],
    "rust": ["target/", "Cargo.lock", "*.rs.bk"],
    "python": [],
    "csharp": [
        "bin/",
        "obj/",
        "*.suo",
        "*.user",
        "*.userosscache",
        "*.sln.docstates",
        "*.vs/",
        "*.cache/",
        "*.pdb",
        "*.dll",
        "*.exe",
    ],
    "kotlin": [
        "build/",
        "out/",
        "*.class",
        "*.jar",
        ".gradle/",
        "buildSrc/build/",
        "*.kt.bak",
    ],
    "php": [
        "vendor/",
        "composer.lock",
        "composer.phar",
        "*.log",
        "*.cache",
        "*.tmp",
        "*.swp",
        ".env",
        "phpunit.xml",
    ],
    "ruby": [
        "*.gem",
        "Gemfile.lock",
        "vendor/",
        "log/",
        "tmp/",
        "*.bundle",
        "*.so",
        "*.o",
        "*.a",
        "mkmf.log",
    ],
    "scala": [
        "target/",
        "project/target/",
        "project/project/",
        "*.class",
        "*.jar",
        ".sbt/",
        ".scala/",
        "*.log",
    ],
    "swift": [
        ".build/",
        "Packages/",
        "*.xcworkspace/",
        "*.xcuserstate",
        "*.xcprofdata",
        "DerivedData/",
        "*.swp",
        "*.swo",
        "*.log",
        "Pods/",
        "Podfile.lock",
    ],
}

LANGUAGE_ALIASES = {
    "java": ["java"],
    "cpp": ["cpp", "c++"],
    "c": ["c"],
    "js": ["js", "javascript"],
    "ts": ["ts", "typescript"],
    "go": ["go", "golang"],
    "rust": ["rust"],
    "python": ["python"],
    "csharp": ["csharp", "c#", "cs"],
    "kotlin": ["kotlin"],
    "php": ["php"],
    "ruby": ["ruby"],
    "scala": ["scala"],
    "swift": ["swift"],
}


def instance_language(instance: dict[str, Any]) -> str:
    """Return the normalized SWE-bench language for an instance record."""

    raw = instance.get("language") or instance.get("programming_language") or "python"
    return normalize_language(str(raw))


def instance_base_commit(instance: dict[str, Any]) -> str | None:
    """Return the base commit field used by SWE-bench and multilingual records."""

    base_commit = instance.get("base_commit")
    if base_commit:
        return str(base_commit)
    base = instance.get("base")
    if isinstance(base, dict) and base.get("sha"):
        return str(base["sha"])
    return None


def normalize_language(language: str) -> str:
    """Normalize SWALM-style language aliases."""

    language = language.strip().casefold()
    for normalized, aliases in LANGUAGE_ALIASES.items():
        if language in aliases:
            return normalized
    return language or "python"


def gitignore_rules(language: str = "python") -> list[str]:
    """Return SWALM-style default plus language-specific generated-file rules."""

    normalized = normalize_language(language)
    return [
        *DEFAULT_GITIGNORE_RULES,
        *LANGUAGE_GITIGNORE_RULES.get(normalized, []),
    ]


def git_diff(
    repo_dir: Path,
    *,
    language: str = "python",
    commit: str | None = None,
) -> str:
    """Return a staged SWE-bench prediction diff after filtering generated files."""

    if not (repo_dir / ".git").exists():
        return ""
    update_info_exclude(repo_dir, language=language)
    subprocess.run(
        ["git", "add", "-A"],
        cwd=repo_dir,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    command = ["git", "diff", "--cached", "--src-prefix=a/", "--dst-prefix=b/"]
    if commit:
        command.append(commit)
    completed = subprocess.run(
        command,
        cwd=repo_dir,
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    stripped = completed.stdout.strip()
    return stripped + ("\n" if stripped else "")


def prepare_baseline_commit(repo_dir: Path, *, language: str = "python") -> str | None:
    """Commit the pre-agent workspace state and return the baseline commit id."""

    if not (repo_dir / ".git").exists():
        return None
    update_info_exclude(repo_dir, language=language)
    subprocess.run(
        ["git", "add", "-A"],
        cwd=repo_dir,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=repo_dir,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if status.returncode != 0:
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Simple Agent Lab",
                "-c",
                "user.email=simple-agent-lab@example.invalid",
                "commit",
                "--no-verify",
                "-m",
                "simple-agent-lab pre-agent baseline",
            ],
            cwd=repo_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def update_info_exclude(repo_dir: Path, *, language: str = "python") -> None:
    """Install the generated ignore block in this checkout's private git exclude."""

    exclude = repo_dir / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    filtered = _without_generated_block(existing).rstrip()
    block = "\n".join(
        [
            IGNORE_BLOCK_START,
            *gitignore_rules(language),
            IGNORE_BLOCK_END,
        ]
    )
    content = f"{filtered}\n\n{block}\n" if filtered else f"{block}\n"
    exclude.write_text(content, encoding="utf-8")


def _without_generated_block(text: str) -> str:
    lines: list[str] = []
    skipping = False
    for line in text.splitlines():
        if line == IGNORE_BLOCK_START:
            skipping = True
            continue
        if line == IGNORE_BLOCK_END:
            skipping = False
            continue
        if not skipping:
            lines.append(line)
    return "\n".join(lines)
