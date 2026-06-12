from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evolution.kernel import store
from simple_agent_lab.evolution.types import Manifest


class StoreTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ws = Path(tmp.name)

    def test_stage_is_content_addressed(self) -> None:
        a = store.stage(self.ws, base=None, edits={"prompt.md": "hi"})
        b = store.stage(self.ws, base=None, edits={"prompt.md": "hi"})
        self.assertEqual(a.hash, b.hash)  # identical content -> same hash
        c = store.stage(self.ws, base=None, edits={"prompt.md": "bye"})
        self.assertNotEqual(a.hash, c.hash)

    def test_stage_applies_edits_over_base_and_tombstones(self) -> None:
        base = store.stage(self.ws, base=None, edits={"prompt.md": "p", "old.md": "x"})
        child = store.stage(
            self.ws, base=base, edits={"prompt.md": "p2", "old.md": None}
        )
        self.assertEqual(child.read("prompt.md"), "p2")
        self.assertEqual(child.read("old.md"), "")  # tombstoned
        self.assertEqual(child.parent, base.hash)

    def test_promote_current_rollback(self) -> None:
        a = store.stage(self.ws, base=None, edits={"prompt.md": "a"})
        store.promote(self.ws, a)
        self.assertEqual(store.current(self.ws).hash, a.hash)
        b = store.stage(self.ws, base=a, edits={"prompt.md": "b"})
        store.promote(self.ws, b)
        self.assertEqual(store.current(self.ws).hash, b.hash)
        store.promote(self.ws, store.version(self.ws, b.parent))  # rollback
        self.assertEqual(store.current(self.ws).hash, a.hash)

    def test_restage_preserves_original_manifest(self) -> None:
        first = store.stage(self.ws, base=None, edits={"p": "x"}, manifest=Manifest(note="first"))
        again = store.stage(self.ws, base=None, edits={"p": "x"}, manifest=Manifest(note="second"))
        self.assertEqual(first.hash, again.hash)
        self.assertEqual(again.manifest.note, "first")  # first provenance wins

    def test_current_missing_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            store.current(self.ws)


if __name__ == "__main__":
    unittest.main()
