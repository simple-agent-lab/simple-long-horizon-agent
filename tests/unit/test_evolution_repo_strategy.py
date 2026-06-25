from __future__ import annotations

import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path

from simple_agent_lab.evolution.components.repo_strategy import (
    proposal_from_candidate_tree,
    source_tree_agent_strategy,
)
from simple_agent_lab.evolution.source_tree import SOURCE_ROOT
from simple_agent_lab.evolution.types import Context


class FakeVersion:
    hash = "abc123def456"


class RepoStrategyTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.base = self.root / "base"
        self.changed = self.root / "changed"
        package = self.base / SOURCE_ROOT
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("VALUE = 1\n")
        (package / "core.py").write_text("def run() -> str:\n    return 'old'\n")
        (self.base / "README.md").write_text("old notes\n")
        self._copy_tree(self.base, self.changed)

    def test_proposal_contains_only_source_changes_and_records_outside_changes(
        self,
    ) -> None:
        (self.changed / SOURCE_ROOT / "core.py").write_text(
            "def run() -> str:\n    return 'new'\n"
        )
        (self.changed / "README.md").write_text("new notes\n")

        proposal = proposal_from_candidate_tree(
            self.base,
            self.changed,
            base_hash="basehash",
            note="agent edit",
            evidence=("agent-ran",),
        )

        self.assertEqual(proposal.base, "basehash")
        self.assertEqual(proposal.kind, "source")
        self.assertEqual(proposal.note, "agent edit")
        self.assertEqual(
            proposal.edits,
            {SOURCE_ROOT + "/core.py": "def run() -> str:\n    return 'new'\n"},
        )
        self.assertIn("agent-ran", proposal.evidence)
        self.assertTrue(
            any("discarded-outside-source:README.md" == e for e in proposal.evidence)
        )

    def test_added_python_file_under_source_root_is_captured(self) -> None:
        (self.changed / SOURCE_ROOT / "helpers.py").write_text("ANSWER = 42\n")

        proposal = proposal_from_candidate_tree(
            self.base, self.changed, base_hash="basehash", note="add helper"
        )

        self.assertEqual(proposal.edits, {SOURCE_ROOT + "/helpers.py": "ANSWER = 42\n"})

    def test_non_python_source_change_is_ignored_with_evidence(self) -> None:
        (self.changed / SOURCE_ROOT / "notes.md").write_text("# changed\n")

        proposal = proposal_from_candidate_tree(
            self.base, self.changed, base_hash="basehash", note="notes"
        )

        self.assertEqual(proposal.edits, {})
        self.assertTrue(
            any(
                "discarded-non-python-source:" + SOURCE_ROOT + "/notes.md" == e
                for e in proposal.evidence
            )
        )

    def test_deleted_source_file_is_ignored_with_evidence(self) -> None:
        (self.changed / SOURCE_ROOT / "core.py").unlink()

        proposal = proposal_from_candidate_tree(
            self.base, self.changed, base_hash="basehash", note="delete"
        )

        self.assertEqual(proposal.edits, {})
        self.assertTrue(
            any(
                "discarded-deleted-source:" + SOURCE_ROOT + "/core.py" == e
                for e in proposal.evidence
            )
        )

    def test_source_tree_agent_strategy_runs_fake_agent_and_validates_edits(
        self,
    ) -> None:
        validations: list[tuple[Path, dict[str, str]]] = []
        fake_agent = FakeAgent()

        def agent_builder(**kwargs):
            fake_agent.cwd = Path(kwargs["cwd"])
            return fake_agent

        def validation(repo_root: Path, files: dict[str, str]) -> None:
            validations.append((repo_root, dict(files)))

        ctx = Context(
            runs=(),
            current=FakeVersion(),
            workspace=self.root / "workspace",
            decisions=(),
        )
        strategy = source_tree_agent_strategy(
            provider=object(),
            repo_root=self.base,
            agent_builder=agent_builder,
            validation=validation,
            max_turns=3,
        )

        proposal = strategy(ctx)

        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertTrue(fake_agent.events_consumed)
        self.assertEqual(proposal.base, FakeVersion.hash)
        self.assertEqual(
            proposal.edits,
            {SOURCE_ROOT + "/core.py": "def run() -> str:\n    return 'agent'\n"},
        )
        self.assertEqual(validations, [(self.base, dict(proposal.edits))])

    @staticmethod
    def _copy_tree(src: Path, dst: Path) -> None:
        for path in src.rglob("*"):
            target = dst / path.relative_to(src)
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())


class FakeAgent:
    def __init__(self) -> None:
        self.cwd: Path | None = None
        self.events_consumed = False

    def run(self, task: str, *, max_turns: int) -> tuple[object, Iterator[object]]:
        self.task = task
        self.max_turns = max_turns

        def events() -> Iterator[object]:
            assert self.cwd is not None
            (self.cwd / SOURCE_ROOT / "core.py").write_text(
                "def run() -> str:\n    return 'agent'\n"
            )
            self.events_consumed = True
            yield object()

        return object(), events()


if __name__ == "__main__":
    unittest.main()
