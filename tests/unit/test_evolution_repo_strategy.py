from __future__ import annotations

import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path

from simple_agent_lab.evolution.components.repo_strategy import (
    proposal_from_candidate_tree,
    source_tree_agent_strategy,
)
from simple_agent_lab.evolution import Experiment
from simple_agent_lab.evolution import registry
from simple_agent_lab.evolution.source_tree import SOURCE_ROOT
from simple_agent_lab.evolution.types import Context


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

    def test_non_utf8_python_change_is_ignored_with_evidence(self) -> None:
        (self.changed / SOURCE_ROOT / "core.py").write_bytes(b"\xff\xfe")

        proposal = proposal_from_candidate_tree(
            self.base, self.changed, base_hash="basehash", note="bad encoding"
        )

        self.assertEqual(proposal.edits, {})
        self.assertTrue(
            any(
                "discarded-non-utf8-source:" + SOURCE_ROOT + "/core.py" == e
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

        exp = Experiment(
            self.root / "strategy-workspace",
            rollout=lambda _version, _slice: [],
            seed={SOURCE_ROOT + "/core.py": "def run() -> str:\n    return 'old'\n"},
        )
        current = exp.current()
        ctx = Context(
            runs=(),
            current=current,
            workspace=self.root / "strategy-workspace",
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
        self.assertEqual(proposal.base, current.hash)
        self.assertEqual(
            proposal.edits,
            {SOURCE_ROOT + "/core.py": "def run() -> str:\n    return 'agent'\n"},
        )
        self.assertEqual(validations, [(fake_agent.base_tree, dict(proposal.edits))])

    def test_source_tree_agent_strategy_overlays_current_version_before_agent(
        self,
    ) -> None:
        validations: list[tuple[Path, dict[str, str]]] = []
        repo_core = self.base / SOURCE_ROOT / "core.py"
        repo_core.write_text("def run() -> str:\n    return 'repo'\n")
        exp = Experiment(
            self.root / "evolution",
            rollout=lambda _version, _slice: [],
            seed={
                SOURCE_ROOT + "/core.py": "def run() -> str:\n    return 'version'\n"
            },
        )
        fake_agent = FakeAgent(expected_before="version")

        def agent_builder(**kwargs):
            fake_agent.cwd = Path(kwargs["cwd"])
            return fake_agent

        def validation(repo_root: Path, files: dict[str, str]) -> None:
            validations.append((repo_root, dict(files)))

        current = exp.current()
        ctx = Context(runs=(), current=current, workspace=self.root / "evolution")
        strategy = source_tree_agent_strategy(
            provider=object(),
            repo_root=self.base,
            agent_builder=agent_builder,
            validation=validation,
        )

        proposal = strategy(ctx)

        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal.base, current.hash)
        self.assertEqual(
            proposal.edits,
            {SOURCE_ROOT + "/core.py": "def run() -> str:\n    return 'agent'\n"},
        )
        self.assertEqual(validations, [(fake_agent.base_tree, dict(proposal.edits))])

    def test_source_tree_agent_registry_accepts_config_compatibility_kwargs(
        self,
    ) -> None:
        strategy = registry.build(
            "strategy",
            registry.Use(
                "source_tree_agent",
                provider=object(),
                repo_root=self.base,
                surface=object(),
                editable_components=("everything",),
            ),
        )

        self.assertTrue(callable(strategy))

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
    def __init__(self, expected_before: str | None = None) -> None:
        self.cwd: Path | None = None
        self.base_tree: Path | None = None
        self.events_consumed = False
        self.expected_before = expected_before

    def run(self, task: str, *, max_turns: int) -> tuple[object, Iterator[object]]:
        self.task = task
        self.max_turns = max_turns

        def events() -> Iterator[object]:
            assert self.cwd is not None
            self.base_tree = self.cwd.parent / "base"
            if self.expected_before is not None:
                before = (self.cwd / SOURCE_ROOT / "core.py").read_text()
                self.assert_expected_before(before)
            (self.cwd / SOURCE_ROOT / "core.py").write_text(
                "def run() -> str:\n    return 'agent'\n"
            )
            self.events_consumed = True
            yield object()

        return object(), events()

    def assert_expected_before(self, content: str) -> None:
        if self.expected_before is None:
            return
        if self.expected_before not in content:
            raise AssertionError(
                f"expected candidate to contain {self.expected_before!r}: {content!r}"
            )


if __name__ == "__main__":
    unittest.main()
