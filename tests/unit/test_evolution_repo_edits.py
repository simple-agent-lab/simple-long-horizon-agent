from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evolution.repo_edits import (
    directory_edits,
    proposal_from_changed_tree,
    touched_paths,
)


class RepoEditsTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_directory_edits_snapshots_files_with_excludes(self) -> None:
        root = self.tmp / "repo"
        (root / "agent").mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "agent" / "task_agent.py").write_text("print('hi')\n")
        (root / ".git" / "HEAD").write_text("ref: main\n")

        edits = directory_edits(root)

        self.assertEqual(edits["agent/task_agent.py"], "print('hi')\n")
        self.assertNotIn(".git/HEAD", edits)

    def test_proposal_from_changed_tree_captures_add_modify_delete(self) -> None:
        base = self.tmp / "base"
        changed = self.tmp / "changed"
        (base / "agent").mkdir(parents=True)
        (changed / "agent").mkdir(parents=True)
        (base / "agent" / "task_agent.py").write_text("old\n")
        (base / "agent" / "old.py").write_text("remove\n")
        (changed / "agent" / "task_agent.py").write_text("new\n")
        (changed / "agent" / "meta_agent.py").write_text("add\n")

        proposal = proposal_from_changed_tree(
            base,
            changed,
            note="meta-agent patch",
            evidence=("model_patch.diff",),
            kind="code",
        )

        self.assertEqual(proposal.edits["agent/task_agent.py"], "new\n")
        self.assertEqual(proposal.edits["agent/meta_agent.py"], "add\n")
        self.assertIsNone(proposal.edits["agent/old.py"])
        self.assertEqual(proposal.note, "meta-agent patch")
        self.assertEqual(proposal.evidence, ("model_patch.diff",))
        self.assertEqual(proposal.kind, "code")

    def test_touched_paths_reads_unified_diff_headers(self) -> None:
        diff = """diff --git a/task_agent.py b/task_agent.py
--- a/task_agent.py
+++ b/task_agent.py
@@ -1 +1 @@
-old
+new
diff --git a/old.py b/old.py
deleted file mode 100644
--- a/old.py
+++ /dev/null
"""

        self.assertEqual(touched_paths(diff), ("old.py", "task_agent.py"))


if __name__ == "__main__":
    unittest.main()
