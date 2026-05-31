"""SWE-bench patch helpers — re-export shim.

The canonical implementation now ships in the wheel at
``simple_agent_lab.evals.suites.swebench.patch`` so the in-container runner can
use it without copying files in (ADR 0017). This module keeps the historical
import path (`evals.swebench.patch_extract`) working for the legacy launcher and
the existing tests.
"""

from __future__ import annotations

from simple_agent_lab.evals.suites.swebench.patch import (
    DEFAULT_GITIGNORE_RULES,
    IGNORE_BLOCK_END,
    IGNORE_BLOCK_START,
    LANGUAGE_ALIASES,
    LANGUAGE_GITIGNORE_RULES,
    git_diff,
    gitignore_rules,
    instance_base_commit,
    instance_language,
    normalize_language,
    prepare_baseline_commit,
    update_info_exclude,
)

__all__ = [
    "DEFAULT_GITIGNORE_RULES",
    "IGNORE_BLOCK_END",
    "IGNORE_BLOCK_START",
    "LANGUAGE_ALIASES",
    "LANGUAGE_GITIGNORE_RULES",
    "git_diff",
    "gitignore_rules",
    "instance_base_commit",
    "instance_language",
    "normalize_language",
    "prepare_baseline_commit",
    "update_info_exclude",
]
