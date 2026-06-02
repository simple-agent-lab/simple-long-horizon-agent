"""In-process test for the replay-to-rebuild container entry (no Docker).

Drives `run_in_container` to produce a prior trace whose tool calls mutate a
real workspace, wipes the workspace to mimic a fresh container, then
`resume_in_container` rebuilds the workspace from the recorded calls and
resumes the model loop — all in-process via an ad-hoc container module.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Mapping

from simple_agent_lab import (
    Agent,
    Message,
    ToolCallBlock,
    assistant_message,
    recorded_tool_calls,
)
from simple_agent_lab.evals.in_container import resume_in_container, run_in_container
from simple_agent_lab.evals.protocols import TRACE_KEY
from simple_agent_lab.evals.stores import LocalDirStore
from simple_agent_lab.llm import Provider
from simple_agent_lab.tools import AgentTool, text_result

_PATHS = ("a.txt", "b.txt")


def _make_container_module(name: str) -> str:
    """Register a container module whose agent writes real files via a tool."""

    def build_agent(*, provider: Provider, cwd: Path, request_extra=None) -> Agent:
        del provider, request_extra

        def touch(call_id, args, abort, on_update):  # noqa: ANN001 - stub
            del call_id, abort, on_update
            (cwd / args["path"]).write_text("x", encoding="utf-8")
            return text_result(f"created {args['path']}")

        tool = AgentTool(
            name="touch",
            description="create a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            execute=touch,
        )

        def gen(visible: list[Message]) -> Message:
            # Decide from context (like a real agent), not an internal counter,
            # so a resumed loop that already sees both calls goes straight to
            # final instead of re-issuing them.
            done = len(recorded_tool_calls(visible))
            if done < len(_PATHS):
                path = _PATHS[done]
                return assistant_message(
                    [
                        ToolCallBlock(
                            id=f"c{done}", name="touch", arguments={"path": path}
                        )
                    ],
                    sender="w",
                    target="user",
                    kind="step",
                )
            return assistant_message("done", sender="w", target="user", kind="final")

        return Agent("w", gen, tools=(tool,))

    mod = types.ModuleType(name)
    mod.build_task = lambda instance, *, workdir: "make the files"  # type: ignore[attr-defined]
    mod.build_agent = build_agent  # type: ignore[attr-defined]
    mod.extract_result = (  # type: ignore[attr-defined]
        lambda workspace, instance: {
            "files": sorted(p.name for p in Path(workspace).glob("*.txt"))
        }
    )
    sys.modules[name] = mod
    return name


class ResumeInContainerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._mod_name = _make_container_module("sal_test_resume_container")
        self.addCleanup(lambda: sys.modules.pop(self._mod_name, None))
        self._provider = Provider(id="fake", api="fake", model="fake-model")

    def _store(self, root: Path, run: str) -> Any:
        return LocalDirStore(root).bind(root / run)

    def test_resume_rebuilds_workspace_and_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workdir = root / "work"
            workdir.mkdir()
            instance: Mapping[str, Any] = {"instance_id": "demo-1"}

            # 1) Original run: writes a.txt + b.txt, produces the trace.
            result, state = run_in_container(
                instance=instance,
                container_module=self._mod_name,
                provider=self._provider,
                workdir=workdir,
                max_turns=5,
                store=self._store(root, "run0"),
                trace_id="t",
                producer="p",
                suite_name="demo",
            )
            self.assertEqual(result["files"], ["a.txt", "b.txt"])

            prior_trace = json.loads(
                self._store(root, "run0").get(TRACE_KEY).decode("utf-8")
            )

            # 2) Fresh container: wipe the workspace.
            for path in workdir.glob("*.txt"):
                path.unlink()
            self.assertEqual(list(workdir.glob("*.txt")), [])

            # 3) Resume from the second tool-result message (index 4): the
            # recorded touch calls are replayed to rebuild the workspace, then
            # the model loop resumes and finishes immediately.
            result2, state2 = resume_in_container(
                prior_trace=prior_trace,
                fork_message_index=4,
                instance=instance,
                container_module=self._mod_name,
                provider=self._provider,
                workdir=workdir,
                max_turns=5,
                store=self._store(root, "run1"),
                trace_id="t2",
                producer="p",
                suite_name="demo",
            )

            self.assertEqual(result2["files"], ["a.txt", "b.txt"])  # rebuilt
            final = next(m for m in reversed(state2.messages) if m.kind == "final")
            self.assertEqual(final.kind, "final")
            # Resumed trace records where it forked from.
            resumed_trace = json.loads(
                self._store(root, "run1").get(TRACE_KEY).decode("utf-8")
            )
            self.assertEqual(resumed_trace["meta"]["resumed_from_message"], 4)
            self.assertFalse(resumed_trace["meta"]["in_progress"])

    def test_resume_without_rebuild_leaves_workspace_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workdir = root / "work"
            workdir.mkdir()
            instance: Mapping[str, Any] = {"instance_id": "demo-1"}

            _, _ = run_in_container(
                instance=instance,
                container_module=self._mod_name,
                provider=self._provider,
                workdir=workdir,
                max_turns=5,
                store=self._store(root, "run0"),
                trace_id="t",
                producer="p",
                suite_name="demo",
            )
            prior_trace = json.loads(
                self._store(root, "run0").get(TRACE_KEY).decode("utf-8")
            )
            for path in workdir.glob("*.txt"):
                path.unlink()

            # rebuild_side_effects=False: the model loop resumes but the
            # workspace is NOT reconstructed, so extraction sees nothing.
            result, _ = resume_in_container(
                prior_trace=prior_trace,
                fork_message_index=4,
                instance=instance,
                container_module=self._mod_name,
                provider=self._provider,
                workdir=workdir,
                max_turns=5,
                store=self._store(root, "run1"),
                trace_id="t2",
                producer="p",
                suite_name="demo",
                rebuild_side_effects=False,
            )
            self.assertEqual(result["files"], [])


if __name__ == "__main__":
    unittest.main()
