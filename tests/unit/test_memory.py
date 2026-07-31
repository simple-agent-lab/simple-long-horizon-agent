from __future__ import annotations

import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from simple_long_horizon_agent import (
    Agent,
    HookFiredEvent,
    HookPoint,
    State,
    assistant_message,
    make_llm_agent,
    message_text,
    text_of,
)
from simple_long_horizon_agent.llm import Provider
from simple_long_horizon_agent.memory import (
    FilesystemArtifact,
    FilesystemMemory,
    FilesystemMemoryLimits,
    FilesystemMemoryPayload,
    Memory,
    MemoryContext,
)
from simple_long_horizon_agent.memory.filesystem import (
    DEFAULT_MAX_HANDBOOK_CHARS,
    filesystem_distillation_prompt,
    sanitize_summary,
)
from simple_long_horizon_agent.memory.transcript import extract_memory_text
from simple_long_horizon_agent.messages import runtime_message
from simple_long_horizon_agent.protocols import ModelRequestEvent
from simple_long_horizon_agent.tools import AgentTool, text_result


def _memory_context(
    task: str = "t",
    *,
    final: bool = False,
    **kwargs: Any,
) -> MemoryContext:
    state = State(task)
    state.send("task", "user", "agent", task)
    if final:
        state.record(
            assistant_message("done", sender="agent", target="user", kind="final")
        )
    return MemoryContext(agent="agent", task=task, state=state, **kwargs)


def _finish_memory_process(
    root: str,
    run_id: str,
    attempting: Any,
    entered_distiller: Any,
    release_distiller: Any | None,
    observed_notes: Any,
) -> None:
    """Process target used to exercise the root file lock, not a thread lock."""

    def distill(payload: FilesystemMemoryPayload) -> dict[str, Any]:
        entered_distiller.set()
        if release_distiller is not None:
            release_distiller.wait(timeout=5)
        observed_notes.put((run_id, payload.notes))
        lessons = "- first lesson\n"
        if run_id == "second":
            lessons += "- second lesson\n"
        return {
            "memory_name": "shared",
            "summary_md": f"## Task\n{run_id}\n",
            "memory_md": f"# Memory Handbook\n\n## Lessons\n\n{lessons}",
        }

    attempting.set()
    FilesystemMemory(root=root, distiller=distill).finish(
        _memory_context(run_id, run_id=run_id)
    )


class MemoryBaseTest(unittest.TestCase):
    def test_memory_binding_declares_tools_and_runtime_hooks(
        self,
    ) -> None:
        class FakeMemory(Memory):
            finished = False

            def initial(self, ctx: MemoryContext):
                return (
                    runtime_message(
                        "remembered initial context",
                        sender="memory",
                        target=ctx.agent,
                        kind="context",
                    ),
                )

            def finish(self, ctx: MemoryContext) -> None:
                self.finished = True
                assert ctx.state is not None
                self.final_count = sum(
                    1 for message in ctx.state.messages if message.kind == "final"
                )

        memory = FakeMemory()
        binding = memory.bind(
            MemoryContext(
                agent="agent",
                task="task",
                session_id="s1",
            )
        )

        agent = Agent(
            "agent",
            lambda visible: assistant_message(
                "done", sender="agent", target="user", kind="final"
            ),
            hooks=binding.hooks,
        )
        state, events = agent.run("task")
        list(events)

        self.assertEqual(binding.tools, ())
        self.assertIn(
            "remembered initial context",
            [message_text(message) for message in state.messages],
        )
        fired_points = [
            event.point for event in state.events if isinstance(event, HookFiredEvent)
        ]
        self.assertEqual(
            fired_points,
            [str(HookPoint.SESSION_START), str(HookPoint.SESSION_END)],
        )
        self.assertTrue(memory.finished)
        self.assertEqual(memory.final_count, 1)

    def test_memory_initial_failure_is_recorded_not_silently_swallowed(self) -> None:
        class FailingInitialMemory(Memory):
            def initial(self, ctx: MemoryContext):
                del ctx
                raise RuntimeError("boom")

        memory = FailingInitialMemory()
        binding = memory.bind(MemoryContext(agent="agent", task="task"))
        agent = Agent(
            "agent",
            lambda visible: assistant_message(
                "done", sender="agent", target="user", kind="final"
            ),
            hooks=binding.hooks,
        )
        state, events = agent.run("task")
        list(events)

        memory_notes = [
            message_text(message)
            for message in state.messages
            if message.sender == "memory"
        ]
        self.assertTrue(
            any("skipped after an error" in note for note in memory_notes),
            memory_notes,
        )
        self.assertTrue(any("RuntimeError: boom" in note for note in memory_notes))

    def test_llm_agent_factory_closes_over_bound_memory_tools(self) -> None:
        class ToolMemory(Memory):
            def tools(self, ctx: MemoryContext):
                del ctx
                return (
                    AgentTool(
                        name="memory_probe",
                        description="Probe memory tool binding.",
                        parameters={"type": "object", "additionalProperties": False},
                        execute=lambda call_id, args, abort, on_update: text_result(
                            "ok"
                        ),
                    ),
                )

        memory = ToolMemory()
        binding = memory.bind(
            MemoryContext(
                agent="agent",
                task="",
                memory_name="demo",
            )
        )
        provider = Provider(id="fake", api="fake", model="fake-model")
        agent = make_llm_agent(
            name="agent",
            provider=provider,
            tools=binding.tools,
        )

        state, events = agent.run("answer directly")
        seen_events = list(events)

        self.assertIn("memory_probe", [tool.name for tool in agent.tools])
        # The fake provider's calls are recorded and tagged api="fake" (so a
        # consumer can filter), not hidden.
        model_requests = [
            event for event in seen_events if isinstance(event, ModelRequestEvent)
        ]
        self.assertTrue(model_requests)
        self.assertTrue(all(event.api == "fake" for event in model_requests))
        self.assertEqual(state.messages[-1].kind, "final")

    def test_llm_agent_factory_closes_over_bound_memory_hooks(self) -> None:
        class InitialMemory(Memory):
            finished = False

            def initial(self, ctx: MemoryContext):
                return (
                    runtime_message(
                        "remembered initial context",
                        sender="memory",
                        target=ctx.agent,
                        kind="context",
                    ),
                )

            def finish(self, ctx: MemoryContext) -> None:
                assert ctx.state is not None
                self.finished = True

        memory = InitialMemory()
        binding = memory.bind(
            MemoryContext(
                agent="agent",
                task="",
                memory_name="demo",
            )
        )
        provider = Provider(id="fake", api="fake", model="fake-model")
        agent = make_llm_agent(
            name="agent",
            provider=provider,
            hooks=binding.hooks,
        )

        state, events = agent.run("answer directly")
        list(events)

        self.assertIn(
            "remembered initial context",
            [message_text(message) for message in state.messages],
        )
        self.assertTrue(memory.finished)

    def test_bound_memory_hooks_are_not_executed_unless_passed_to_agent(self) -> None:
        class FakeMemory(Memory):
            def initial(self, ctx: MemoryContext):
                return (
                    runtime_message(
                        "remembered initial context",
                        sender="memory",
                        target=ctx.agent,
                        kind="context",
                    ),
                )

        memory = FakeMemory()
        binding = memory.bind(
            MemoryContext(
                agent="agent",
                task="",
                session_id="s1",
            )
        )
        provider = Provider(id="fake", api="fake", model="fake-model")
        agent = make_llm_agent(
            name="agent",
            provider=provider,
            tools=binding.tools,
        )
        state, events = agent.run("task")
        list(events)

        self.assertNotIn(
            "remembered initial context",
            [message_text(message) for message in state.messages],
        )


class FilesystemMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.root = Path(self._temp_dir.name)

    def test_filesystem_memory_prompt_preserves_quality_gate(self) -> None:
        payload = FilesystemMemoryPayload(
            task="fix login",
            transcript="user asked to fix login\nassistant edited auth.py",
            artifacts=(),
            memory_summary="v1\n\n# Memory Summary\n",
            index="# Memory Index\n",
            notes="# Memory Handbook\n",
            run_path="runs/r1",
            available_memories=("auth",),
            context=MemoryContext(agent="agent", task="fix login"),
        )
        prompt = filesystem_distillation_prompt(payload)

        self.assertIn("No-op is allowed and preferred", prompt)
        self.assertIn("future user time saved", prompt)
        self.assertIn("Do not store secrets", prompt)
        self.assertIn("stable user preferences", prompt)

    def test_filesystem_memory_writes_evidence_and_distilled_outputs(self) -> None:
        payloads = []

        def distill(payload):
            payloads.append(payload)
            return {
                "memory_name": "ignored",
                "summary_md": "# inst-1\n\n## Task\nShort\n\n## Outcome\npass\n",
                "index_row": {
                    "summary": "short summary",
                    "scope": "auth flow",
                    "signals": "auth error",
                    "keywords": "auth, focused checks",
                    "artifacts": "submission.txt",
                },
                "memory_md": "# Memory Handbook\n\n## Lessons\n\n"
                "- Use focused checks for recurring auth errors. [runs/run_42]\n",
            }

        memory = FilesystemMemory(root=self.root, distiller=distill)
        ctx = _memory_context(
            "fix auth",
            final=True,
            session_id="inst/1",
            run_id="run/42",
            memory_name="repo/name",
        )
        assert ctx.state is not None
        ctx.state.data["model_patch"] = "diff --git a/core.py b/core.py\n"

        context = memory.initial(ctx)
        memory.finish(ctx)

        memory_dir = memory.memory_dir(ctx)
        run_dir = memory_dir / "runs" / "run_42"
        summary = (run_dir / "summary.md").read_text()
        index = (memory_dir / "INDEX.md").read_text()
        handbook = (memory_dir / "MEMORY.md").read_text()
        memory_summary = (memory_dir / "memory_summary.md").read_text()
        manifest = (run_dir / "artifacts.md").read_text()

        self.assertEqual(len(context), 1)
        self.assertIn("filesystem memory", message_text(context[0]))
        self.assertTrue((run_dir / "task.md").exists())
        self.assertIn("fix auth", (run_dir / "transcript.md").read_text())
        self.assertIn("diff --git", (run_dir / "artifacts/submission.txt").read_text())
        for expected in ("submission.txt", "Final run submission"):
            self.assertIn(expected, manifest)
        self.assertIn("## Task", summary)
        self.assertNotIn("Outcome", summary)
        self.assertEqual(payloads[0].run_path, "runs/run_42")
        self.assertIn("v1", payloads[0].memory_summary)
        for expected in (
            "short summary",
            "auth flow",
            "auth error",
            "auth, focused checks",
        ):
            self.assertIn(expected, index)
        self.assertEqual(index.count("runs/run_42/summary.md"), 1)
        self.assertEqual(
            handbook,
            "# Memory Handbook\n\n## Lessons\n\n"
            "- Use focused checks for recurring auth errors. [runs/run_42]\n",
        )
        self.assertNotIn("## Run Updates", handbook)
        for expected in ("short summary", "runs/run_42/summary.md"):
            self.assertIn(expected, memory_summary)

    def test_filesystem_memory_distiller_chooses_memory_name_when_omitted(
        self,
    ) -> None:
        existing = self.root / "auth"
        memory = FilesystemMemory(root=self.root)
        memory.ensure_layout(existing)
        (existing / "MEMORY.md").write_text(
            "# Memory Handbook\n\n## Patterns\n\n- Existing auth notes.\n"
        )
        payloads = []

        def distill(payload):
            payloads.append(payload)
            return {
                "memory_name": "auth",
                "summary_md": "## Task\nFix login callback\n",
                "index_row": {"summary": "login callback", "signals": "auth"},
                "memory_md": "# Memory Handbook\n\n## Lessons\n\n"
                "- Existing auth notes.\n"
                "- Login callback fixes should inspect auth logs. [runs/r1]\n",
            }

        memory = FilesystemMemory(root=self.root, distiller=distill)
        ctx = _memory_context(
            "fix login callback", final=True, session_id="s1", run_id="r1"
        )
        initial = memory.initial(ctx)
        memory.finish(ctx)

        self.assertEqual(len(initial), 1)
        self.assertIn("- auth", text_of(initial[0].content))
        self.assertEqual(payloads[0].available_memories, ("auth",))
        for expected in ("auth/MEMORY.md", "Existing auth notes"):
            self.assertIn(expected, payloads[0].notes)
        self.assertTrue((existing / "runs/r1").exists())
        handbook = (existing / "MEMORY.md").read_text()
        for expected in ("Login callback fixes", "Existing auth notes"):
            self.assertIn(expected, handbook)
        self.assertNotIn("## Run Updates", handbook)
        self.assertFalse((self.root / "default").exists())

    def test_filesystem_memory_distiller_failure_still_writes_evidence(self) -> None:
        def distill(payload):
            raise RuntimeError("distiller unavailable")

        memory = FilesystemMemory(root=self.root, distiller=distill)
        ctx = _memory_context(
            "collect evidence",
            final=True,
            session_id="fallback/session",
            run_id="run/99",
        )
        memory.finish(ctx)

        run_dir = memory.memory_dir(ctx) / "runs/run_99"
        for name in ("task.md", "transcript.md"):
            self.assertTrue((run_dir / name).exists())
        self.assertIn("Distillation unavailable", (run_dir / "summary.md").read_text())
        self.assertIn(
            "runs/run_99/summary.md", (run_dir.parent.parent / "INDEX.md").read_text()
        )
        self.assertIn("RuntimeError", (run_dir / "memory_error.md").read_text())
        self.assertIn(
            "collect evidence",
            (run_dir.parent.parent / "memory_summary.md").read_text(),
        )

    def test_distiller_can_drop_a_run_without_creating_a_namespace(self) -> None:
        FilesystemMemory(
            root=self.root,
            distiller=lambda payload: {"retain_run": False},
        ).finish(_memory_context("routine run", run_id="r1"))

        self.assertFalse(
            any(path for path in self.root.iterdir() if not path.name.startswith("."))
        )

    def test_filesystem_memory_initial_lists_available_names_when_name_omitted(
        self,
    ) -> None:
        memory = FilesystemMemory(root=self.root)
        memory.ensure_layout(self.root / "auth")

        context = memory.initial(MemoryContext(agent="agent", task="fix login"))

        self.assertEqual(len(context), 1)
        text = text_of(context[0].content)
        self.assertNotIn("MEMORY_ROOT=", text)
        for expected in (
            "absolute path",
            "persists between separate actions",
            "- auth",
        ):
            self.assertIn(expected, text)

    def test_handbook_rewrite_acceptance_and_guards(self) -> None:
        short = "# Memory Handbook\n\n## Lessons\n\n- Keep me.\n"
        full = (
            "# Memory Handbook\n\n## Lessons\n\n"
            "- Prefer a tiny reproduction script first. [runs/r1]\n"
        )
        oversize = "# Memory Handbook\n\n## Lessons\n\n" + "- runaway bullet\n" * (
            DEFAULT_MAX_HANDBOOK_CHARS // 8
        )
        cases = (
            ("full", None, full, full, ("## Run Updates", "## Durable Lessons"), False),
            ("empty", short, "", short, (), False),
            ("oversize", short, oversize, short, ("runaway bullet",), True),
            (
                "erase",
                "# Memory Handbook\n\n## Lessons\n\n- One.\n- Two.\n- Three.\n",
                "# Memory Handbook\n\n## Lessons\n",
                "# Memory Handbook\n\n## Lessons\n\n- One.\n- Two.\n- Three.\n",
                (),
                True,
            ),
        )
        for name, existing, rewrite, expected, absent, rejected in cases:
            with self.subTest(name=name):
                memory_dir = self.root / name
                FilesystemMemory(root=self.root).ensure_layout(memory_dir)
                if existing is not None:
                    (memory_dir / "MEMORY.md").write_text(existing)
                memory = FilesystemMemory(
                    root=self.root,
                    distiller=lambda payload, text=rewrite: {
                        "summary_md": "## Task\nt\n",
                        "memory_md": text,
                    },
                )
                memory.finish(_memory_context("t", memory_name=name, run_id="r1"))

                handbook = (memory_dir / "MEMORY.md").read_text()
                self.assertEqual(handbook, expected)
                for unexpected in absent:
                    self.assertNotIn(unexpected, handbook)
                self.assertEqual(
                    (memory_dir / "runs/r1/memory_error.md").exists(),
                    rejected,
                )
                if rejected:
                    self.assertIn(
                        "Handbook rewrite rejected",
                        (memory_dir / "runs/r1/memory_error.md").read_text(),
                    )

    def test_finish_stores_state_memory_artifacts_without_duplicate_submission(
        self,
    ) -> None:
        memory = FilesystemMemory(root=self.root)
        ctx = _memory_context("solve instance", memory_name="demo", run_id="r1")
        assert ctx.state is not None
        ctx.state.data["memory_artifacts"] = [
            {
                "name": "model_patch.diff",
                "content": "diff --git a/x b/x\n+new line\n",
                "description": "Final unified git diff (model_patch) produced by the run.",
            }
        ]
        memory.finish(ctx)

        run_dir = self.root / "demo/runs/r1"
        self.assertIn("+new line", (run_dir / "artifacts/model_patch.diff").read_text())
        self.assertIn("model_patch.diff", (run_dir / "artifacts.md").read_text())
        self.assertFalse((run_dir / "artifacts/submission.txt").exists())
        self.assertIn("model_patch.diff", (self.root / "demo/INDEX.md").read_text())

    def test_artifact_builder_products_reach_distiller_payload(self) -> None:
        payloads: list[FilesystemMemoryPayload] = []

        def distill(payload):
            payloads.append(payload)
            return {"memory_md": ""}

        memory = FilesystemMemory(
            root=self.root,
            distiller=distill,
            artifact_builder=lambda ctx: [
                FilesystemArtifact(
                    "model_patch.diff",
                    "diff --git a/x b/x\n+remember me\n",
                    "Final unified git diff (model_patch).",
                )
            ],
        )
        memory.finish(
            _memory_context("solve instance", memory_name="demo", run_id="r1")
        )
        run_dir = self.root / "demo/runs/r1"

        self.assertEqual(len(payloads), 1)
        self.assertEqual([a.name for a in payloads[0].artifacts], ["model_patch.diff"])
        self.assertIn("+remember me", payloads[0].artifacts[0].content)
        self.assertIn(
            "+remember me", (run_dir / "artifacts/model_patch.diff").read_text()
        )
        self.assertIn("model_patch.diff", (run_dir / "artifacts.md").read_text())

    def test_distiller_prompt_forbids_raw_line_number_citations(self) -> None:
        payload = FilesystemMemoryPayload(
            task="fix login",
            transcript="## 0. user (task, user -> agent)\n\nfix login",
            artifacts=(),
            memory_summary="v1\n",
            index="# Memory Index\n",
            notes="# Memory Handbook\n",
            run_path="runs/r1",
            available_memories=(),
            context=MemoryContext(agent="agent", task="fix login"),
        )
        prompt = filesystem_distillation_prompt(payload)

        self.assertIn("greppable anchors", prompt)
        self.assertIn("Never cite raw line numbers", prompt)
        self.assertIn("transcript.md ## <n>", prompt)

    def test_policy_block_inlines_summary_and_bans_line_numbers(self) -> None:
        memory = FilesystemMemory(root=self.root)
        text = text_of(
            memory.initial(MemoryContext(agent="agent", task="t", memory_name="demo"))[
                0
            ].content
        )

        for expected in (
            "<memory_summary.md_excerpt>",
            "searching for the cited anchor",
            "persists between separate actions",
            f"{self.root}/demo/MEMORY.md",
        ):
            self.assertIn(expected, text)
        for forbidden in (
            "targeted line ranges",
            "MEMORY_DIR=",
            "copy this exact assignment",
            "cat ",
            "grep",
        ):
            self.assertNotIn(forbidden, text)

    def test_ensure_layout_writes_skeletons_without_readme(self) -> None:
        memory_dir = self.root / "demo"
        FilesystemMemory(root=self.root).ensure_layout(memory_dir)

        self.assertFalse((memory_dir / "README.md").exists())
        for name in ("INDEX.md", "MEMORY.md", "memory_summary.md"):
            self.assertTrue((memory_dir / name).is_file())
        self.assertTrue((memory_dir / "runs").is_dir())

    def test_policy_block_maps_namespace_directory_structure(self) -> None:
        memory = FilesystemMemory(root=self.root)
        ctx = MemoryContext(agent="agent", task="t", memory_name="demo")
        text = text_of(memory.initial(ctx)[0].content)

        for expected in (
            "runs/<run_id>/",
            "task.md",
            "artifacts.md",
            "artifacts/",
            "keywords",
            "## <n>. <role> (<kind>, <sender> -> <target>)",
        ):
            self.assertIn(expected, text)

    def test_filesystem_memory_repeated_run_is_idempotent(self) -> None:
        memory = FilesystemMemory(root=self.root)
        ctx = _memory_context("repeat run", memory_name="demo", run_id="r1")
        memory.finish(ctx)
        memory.finish(ctx)

        memory_dir = self.root / "demo"
        self.assertTrue((memory_dir / "runs/r1").exists())
        self.assertFalse((memory_dir / "runs/r1_2").exists())
        index = (memory_dir / "INDEX.md").read_text()
        self.assertEqual(index.count("runs/r1/summary.md"), 1)

    def test_finish_serializes_dynamic_routing_distillation_and_commit(self) -> None:
        """A second writer must distill from the first writer's committed snapshot."""

        processes = multiprocessing.get_context("spawn")
        first_attempting, first_inside, release_first = (
            processes.Event() for _ in range(3)
        )
        second_attempting, second_inside = (processes.Event() for _ in range(2))
        observed_notes = processes.Queue()
        first = processes.Process(
            target=_finish_memory_process,
            args=(
                str(self.root),
                "first",
                first_attempting,
                first_inside,
                release_first,
                observed_notes,
            ),
        )
        second = processes.Process(
            target=_finish_memory_process,
            args=(
                str(self.root),
                "second",
                second_attempting,
                second_inside,
                None,
                observed_notes,
            ),
        )
        first.start()
        self.assertTrue(first_attempting.wait(timeout=5))
        self.assertTrue(first_inside.wait(timeout=5))
        second.start()
        self.assertTrue(second_attempting.wait(timeout=5))
        blocked_by_root_lock = not second_inside.wait(timeout=0.1)
        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)

        self.assertTrue(blocked_by_root_lock)
        self.assertEqual((first.exitcode, second.exitcode), (0, 0))
        notes = dict(observed_notes.get(timeout=1) for _ in range(2))
        self.assertIn("first lesson", notes["second"])
        shared = self.root / "shared/runs"
        for run_id in ("first", "second"):
            self.assertTrue((shared / run_id / ".complete").is_file())

    def test_atomic_write_replaces_or_leaves_previous_file_intact(self) -> None:
        from simple_long_horizon_agent.memory import filesystem

        path = self.root / "memory.md"
        path.write_text("before")
        with (
            mock.patch.object(
                filesystem.os,
                "replace",
                side_effect=OSError("replace failed"),
            ),
            self.assertRaisesRegex(OSError, "replace failed"),
        ):
            filesystem._write_text_atomic(path, "broken")
        self.assertEqual(path.read_text(), "before")
        self.assertEqual(list(self.root.iterdir()), [path])

        with mock.patch.object(
            filesystem.os, "replace", side_effect=os.replace
        ) as replace:
            filesystem._write_text_atomic(path, "after")
        replace.assert_called_once()
        self.assertEqual(path.read_text(), "after")

    def test_child_namespace_mount_uses_the_shared_root_lock_directory(self) -> None:
        from simple_long_horizon_agent.evals.backends.docker_local import (
            with_local_mounts,
        )
        from simple_long_horizon_agent.evals.protocols import ContainerBinding

        root = self.root / "memory"
        lock_dir = root / ".memory-lock"
        binding = with_local_mounts(
            ContainerBinding(),
            wheelhouse=None,
            wheelhouse_mount=None,
            uv_binary=None,
            memory_home=root / "shared",
            memory_mount="/agent/memory/shared",
            memory_env_home="/agent/memory",
            memory_lock_dir=lock_dir,
        )

        self.assertEqual(
            binding.mounts[str(lock_dir.resolve())],
            {"bind": "/agent/memory/.memory-lock", "mode": "rw"},
        )
        self.assertTrue((lock_dir / "memory.lock").is_file())

    def test_filesystem_memory_keeps_only_the_newest_bounded_runs(self) -> None:
        memory = FilesystemMemory(
            root=self.root,
            limits=FilesystemMemoryLimits(max_runs_per_memory=2),
        )
        for run_id in ("r1", "r2", "r3"):
            memory.finish(_memory_context(run_id, memory_name="demo", run_id=run_id))

        memory_dir = self.root / "demo"
        self.assertEqual(
            sorted(path.name for path in (memory_dir / "runs").iterdir()),
            ["r2", "r3"],
        )
        self.assertNotIn("runs/r1/summary.md", (memory_dir / "INDEX.md").read_text())

    def test_filesystem_memory_refuses_namespaces_above_the_cap(self) -> None:
        memory = FilesystemMemory(
            root=self.root,
            limits=FilesystemMemoryLimits(max_namespaces_per_root=1),
        )
        self.assertTrue(memory.admit_namespaces(("first",)))
        self.assertFalse(memory.admit_namespaces(("second",)))

    def test_filesystem_memory_sanitizes_duplicate_artifact_names(self) -> None:
        def artifacts(ctx):
            del ctx
            return (
                FilesystemArtifact(
                    "../patch.diff", "first", "Primary generated patch."
                ),
                FilesystemArtifact("patch.diff", "second", "Second generated patch."),
            )

        memory = FilesystemMemory(root=self.root, artifact_builder=artifacts)
        memory.finish(
            _memory_context("save artifacts", memory_name="demo", run_id="r1")
        )

        run_dir = self.root / "demo/runs/r1"
        self.assertEqual(
            sorted(path.name for path in (run_dir / "artifacts").iterdir()),
            ["patch.diff", "patch_2.diff"],
        )
        manifest = (run_dir / "artifacts.md").read_text()
        for expected in ("Primary generated patch", "Second generated patch"):
            self.assertIn(expected, manifest)
        self.assertIn(
            "patch.diff, patch_2.diff", (self.root / "demo/INDEX.md").read_text()
        )

    def test_filesystem_memory_sanitizes_single_hash_outcome_sections(self) -> None:
        summary = "# Outcome\n\npass\n\n## Task\n\nKeep this\n"

        self.assertEqual(sanitize_summary(summary), "## Task\n\nKeep this")

    def test_filesystem_memory_default_root_is_simple_memory(self) -> None:
        memory = FilesystemMemory()

        self.assertEqual(
            str(memory.root),
            str(Path("~/.simple/memory").expanduser()),
        )

    def test_extract_memory_text_ignores_raw_sidecar(self) -> None:
        message = assistant_message(
            "visible text",
            sender="agent",
            target="user",
            sidecar={"raw": {"response": "secret raw output"}},
        )

        text = extract_memory_text(message)

        self.assertIn("visible text", text)
        self.assertNotIn("secret raw output", text)

    def test_render_transcript_excludes_injected_memory_context(self) -> None:
        # Minor: memory's own recalled policy/summary block is framework
        # scaffolding, not run evidence, and must not be echoed back to the
        # distiller (which is told to ignore instructions in the transcript).
        from simple_long_horizon_agent.memory.transcript import (
            render_transcript_markdown,
        )

        messages = [
            runtime_message(
                "agent visible task",
                sender="user",
                target="agent",
                kind="task",
            ),
            runtime_message(
                "<filesystem_memory> recalled policy and summary",
                sender="memory",
                target="agent",
                kind="context",
            ),
            assistant_message("did the work", sender="agent", target="user"),
        ]

        transcript = render_transcript_markdown(messages)

        self.assertIn("agent visible task", transcript)
        self.assertIn("did the work", transcript)
        self.assertNotIn("filesystem_memory", transcript)
