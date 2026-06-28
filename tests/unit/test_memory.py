from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simple_agent_lab import (
    Agent,
    HookFiredEvent,
    HookPoint,
    State,
    assistant_message,
    make_llm_agent,
    message_text,
    text_of,
)
from simple_agent_lab.llm import Provider
from simple_agent_lab.memory import (
    FilesystemArtifact,
    FilesystemMemory,
    FilesystemMemoryPayload,
    Memory,
    MemoryContext,
)
from simple_agent_lab.memory.filesystem import sanitize_summary
from simple_agent_lab.memory.transcript import extract_memory_text
from simple_agent_lab.messages import runtime_message
from simple_agent_lab.protocols import ModelRequestEvent
from simple_agent_lab.tools import AgentTool, text_result


class MemoryBaseTest(unittest.TestCase):
    def test_memory_binding_declares_tools_and_runtime_hooks(
        self,
    ) -> None:
        class FakeMemory(Memory):
            finished = False
            recorded_turns = 0
            recalled = 0

            def initial(self, ctx: MemoryContext):
                return (
                    runtime_message(
                        "remembered initial context",
                        sender="memory",
                        target=ctx.agent,
                        kind="context",
                    ),
                )

            def recall(self, ctx: MemoryContext, query: str):
                del ctx, query
                self.recalled += 1
                return (
                    runtime_message(
                        "remembered recall",
                        sender="memory",
                        target="agent",
                        kind="context",
                    ),
                )

            def record(
                self,
                ctx: MemoryContext,
                messages: tuple,
            ) -> None:
                del ctx
                self.recorded_turns += 1
                self.recorded_message_count = len(messages)

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
        self.assertEqual(memory.recalled, 0)
        self.assertEqual(memory.recorded_turns, 0)
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
    def test_filesystem_memory_prompt_preserves_quality_gate(self) -> None:
        from simple_agent_lab.memory.filesystem import filesystem_distillation_prompt

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
        with tempfile.TemporaryDirectory() as tmp:
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
                    "memory_md": "# Memory Handbook\n\n## Lessons\n\n- Use focused checks for recurring auth errors. [runs/run_42]\n",
                }

            memory = FilesystemMemory(root=tmp, distiller=distill)
            state = State("fix auth")
            state.data["model_patch"] = "diff --git a/core.py b/core.py\n"
            state.send("task", "user", "agent", "fix auth")
            state.record(
                assistant_message("done", sender="agent", target="user", kind="final")
            )
            ctx = MemoryContext(
                agent="agent",
                task="fix auth",
                session_id="inst/1",
                run_id="run/42",
                memory_name="repo/name",
                step_index=2,
                state=state,
            )

            context = memory.initial(ctx)
            memory.finish(ctx)

            memory_dir = memory.memory_dir(ctx)
            run_dir = memory_dir / "runs" / "run_42"
            summary = (run_dir / "summary.md").read_text(encoding="utf-8")
            index = (memory_dir / "INDEX.md").read_text(encoding="utf-8")
            handbook = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
            memory_summary = (memory_dir / "memory_summary.md").read_text(
                encoding="utf-8"
            )

            self.assertEqual(len(context), 1)
            self.assertIn("filesystem memory", message_text(context[0]))
            self.assertEqual(memory_dir, Path(tmp) / "repo_name")
            self.assertTrue((run_dir / "task.md").exists())
            self.assertIn(
                "fix auth", (run_dir / "transcript.md").read_text(encoding="utf-8")
            )
            self.assertIn(
                "diff --git",
                (run_dir / "artifacts" / "submission.txt").read_text(encoding="utf-8"),
            )
            manifest = (run_dir / "artifacts.md").read_text(encoding="utf-8")
            self.assertIn("submission.txt", manifest)
            self.assertIn("Final run submission", manifest)
            self.assertIn("## Task", summary)
            self.assertNotIn("Outcome", summary)
            self.assertEqual(payloads[0].run_path, "runs/run_42")
            self.assertIn("v1", payloads[0].memory_summary)
            self.assertIn("short summary", index)
            self.assertIn("auth flow", index)
            self.assertIn("auth error", index)
            self.assertIn("auth, focused checks", index)
            self.assertEqual(index.count("runs/run_42/summary.md"), 1)
            self.assertEqual(
                handbook,
                "# Memory Handbook\n\n## Lessons\n\n"
                "- Use focused checks for recurring auth errors. [runs/run_42]\n",
            )
            self.assertNotIn("## Run Updates", handbook)
            self.assertIn("short summary", memory_summary)
            self.assertIn("runs/run_42/summary.md", memory_summary)

    def test_filesystem_memory_distiller_chooses_memory_name_when_omitted(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "auth"
            memory = FilesystemMemory(root=tmp)
            memory.ensure_layout(existing)
            (existing / "MEMORY.md").write_text(
                "# Memory Handbook\n\n## Patterns\n\n- Existing auth notes.\n",
                encoding="utf-8",
            )
            payloads = []

            def distill(payload):
                payloads.append(payload)
                return {
                    "memory_name": "auth",
                    "summary_md": "## Task\nFix login callback\n",
                    "index_row": {"summary": "login callback", "signals": "auth"},
                    "memory_md": "# Memory Handbook\n\n## Lessons\n\n- Existing auth notes.\n- Login callback fixes should inspect auth logs. [runs/r1]\n",
                }

            memory = FilesystemMemory(root=tmp, distiller=distill)
            state = State("fix login callback")
            state.send("task", "user", "agent", "fix login callback")
            state.record(
                assistant_message("done", sender="agent", target="user", kind="final")
            )
            ctx = MemoryContext(
                agent="agent",
                task="fix login callback",
                session_id="s1",
                run_id="r1",
                state=state,
            )

            initial = memory.initial(ctx)
            self.assertEqual(len(initial), 1)
            self.assertIn("- auth", text_of(initial[0].content))
            memory.finish(ctx)

            self.assertEqual(payloads[0].available_memories, ("auth",))
            self.assertIn("auth/MEMORY.md", payloads[0].notes)
            self.assertIn("Existing auth notes", payloads[0].notes)
            self.assertTrue((Path(tmp) / "auth" / "runs" / "r1").exists())
            handbook = (Path(tmp) / "auth" / "MEMORY.md").read_text(encoding="utf-8")
            self.assertIn(
                "Login callback fixes should inspect auth logs. [runs/r1]", handbook
            )
            self.assertIn("Existing auth notes", handbook)
            self.assertNotIn("## Run Updates", handbook)
            self.assertFalse((Path(tmp) / "default").exists())

    def test_filesystem_memory_distiller_failure_still_writes_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:

            def distill(payload):
                raise RuntimeError("distiller unavailable")

            memory = FilesystemMemory(root=tmp, distiller=distill)
            state = State("collect evidence")
            state.send("task", "user", "agent", "collect evidence")
            state.record(
                assistant_message("done", sender="agent", target="user", kind="final")
            )
            ctx = MemoryContext(
                agent="agent",
                task="collect evidence",
                session_id="fallback/session",
                run_id="run/99",
                state=state,
            )

            memory.finish(ctx)

            run_dir = Path(tmp) / "fallback_session" / "runs" / "run_99"
            self.assertTrue((run_dir / "task.md").exists())
            self.assertTrue((run_dir / "transcript.md").exists())
            summary = (run_dir / "summary.md").read_text(encoding="utf-8")
            index = (Path(tmp) / "fallback_session" / "INDEX.md").read_text(
                encoding="utf-8"
            )
            error = (run_dir / "memory_error.md").read_text(encoding="utf-8")
            memory_summary = (
                Path(tmp) / "fallback_session" / "memory_summary.md"
            ).read_text(encoding="utf-8")

            self.assertIn("Distillation unavailable", summary)
            self.assertIn("runs/run_99/summary.md", index)
            self.assertIn("RuntimeError", error)
            self.assertIn("collect evidence", memory_summary)

    def test_filesystem_memory_initial_lists_available_names_when_name_omitted(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = FilesystemMemory(root=tmp)
            memory.ensure_layout(Path(tmp) / "auth")

            context = memory.initial(MemoryContext(agent="agent", task="fix login"))

            self.assertEqual(len(context), 1)
            text = text_of(context[0].content)
            # P5: guide the agent to read memory by absolute path, not via an
            # env-var assignment or other state that does not persist between
            # separate actions.
            self.assertNotIn("MEMORY_ROOT=", text)
            self.assertIn("absolute path", text)
            self.assertIn("persists between separate actions", text)
            self.assertIn("- auth", text)

    def test_handbook_rewrite_persists_full_model_output(self) -> None:
        # The distiller owns the whole handbook: its memory_md is written verbatim
        # as the new MEMORY.md, with no deterministic sectioning or promotion.
        with tempfile.TemporaryDirectory() as tmp:
            new_handbook = (
                "# Memory Handbook\n\n## Lessons\n\n"
                "- Prefer a tiny reproduction script first. [runs/r1]\n"
            )

            def distill(payload):
                return {"summary_md": "## Task\nt\n", "memory_md": new_handbook}

            memory = FilesystemMemory(root=tmp, distiller=distill)
            state = State("t")
            state.send("task", "user", "agent", "t")
            memory.finish(
                MemoryContext(
                    agent="agent",
                    task="t",
                    memory_name="demo",
                    run_id="r1",
                    state=state,
                )
            )
            handbook = (Path(tmp) / "demo" / "MEMORY.md").read_text(encoding="utf-8")

        self.assertEqual(handbook, new_handbook)
        self.assertNotIn("## Run Updates", handbook)
        self.assertNotIn("## Durable Lessons", handbook)

    def test_handbook_rewrite_empty_keeps_existing(self) -> None:
        # An empty memory_md means "no durable change"; the prior handbook stays.
        with tempfile.TemporaryDirectory() as tmp:
            memory = FilesystemMemory(root=tmp)
            memory_dir = Path(tmp) / "demo"
            memory.ensure_layout(memory_dir)
            (memory_dir / "MEMORY.md").write_text(
                "# Memory Handbook\n\n## Lessons\n\n- Keep me.\n", encoding="utf-8"
            )

            def distill(payload):
                return {"summary_md": "## Task\nt\n", "memory_md": ""}

            memory = FilesystemMemory(root=tmp, distiller=distill)
            state = State("t")
            state.send("task", "user", "agent", "t")
            memory.finish(
                MemoryContext(
                    agent="agent",
                    task="t",
                    memory_name="demo",
                    run_id="r1",
                    state=state,
                )
            )
            handbook = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")

        self.assertIn("- Keep me.", handbook)

    def test_handbook_rewrite_rejected_when_oversize_keeps_existing(self) -> None:
        # Guard: a rewrite past the size cap is rejected; the prior handbook is kept
        # and a marker records the skip instead of persisting a runaway response.
        from simple_agent_lab.memory.filesystem import DEFAULT_MAX_HANDBOOK_CHARS

        with tempfile.TemporaryDirectory() as tmp:
            memory = FilesystemMemory(root=tmp)
            memory_dir = Path(tmp) / "demo"
            memory.ensure_layout(memory_dir)
            (memory_dir / "MEMORY.md").write_text(
                "# Memory Handbook\n\n## Lessons\n\n- Keep me.\n", encoding="utf-8"
            )
            oversize = "# Memory Handbook\n\n## Lessons\n\n" + "- runaway bullet\n" * (
                DEFAULT_MAX_HANDBOOK_CHARS // 8
            )

            def distill(payload):
                return {"summary_md": "## Task\nt\n", "memory_md": oversize}

            memory = FilesystemMemory(root=tmp, distiller=distill)
            state = State("t")
            state.send("task", "user", "agent", "t")
            memory.finish(
                MemoryContext(
                    agent="agent",
                    task="t",
                    memory_name="demo",
                    run_id="r1",
                    state=state,
                )
            )
            handbook = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
            error = (memory_dir / "runs" / "r1" / "memory_error.md").read_text(
                encoding="utf-8"
            )

        self.assertIn("- Keep me.", handbook)
        self.assertNotIn("runaway bullet", handbook)
        self.assertIn("Handbook rewrite rejected", error)

    def test_handbook_rewrite_rejected_when_it_would_erase_lessons(self) -> None:
        # Guard: dropping every bullet from a non-empty handbook (e.g. a truncated
        # response) is rejected to prevent catastrophic loss.
        with tempfile.TemporaryDirectory() as tmp:
            memory = FilesystemMemory(root=tmp)
            memory_dir = Path(tmp) / "demo"
            memory.ensure_layout(memory_dir)
            (memory_dir / "MEMORY.md").write_text(
                "# Memory Handbook\n\n## Lessons\n\n- One.\n- Two.\n- Three.\n",
                encoding="utf-8",
            )

            def distill(payload):
                return {
                    "summary_md": "## Task\nt\n",
                    "memory_md": "# Memory Handbook\n\n## Lessons\n",
                }

            memory = FilesystemMemory(root=tmp, distiller=distill)
            state = State("t")
            state.send("task", "user", "agent", "t")
            memory.finish(
                MemoryContext(
                    agent="agent",
                    task="t",
                    memory_name="demo",
                    run_id="r1",
                    state=state,
                )
            )
            handbook = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")

        self.assertIn("- One.", handbook)
        self.assertIn("- Three.", handbook)

    def test_finish_stores_state_memory_artifacts_without_duplicate_submission(
        self,
    ) -> None:
        # P2: the eval records its product (e.g. the patch) into
        # state.data["memory_artifacts"] and the post-loop finish stores it. With
        # no state.data["model_patch"] set, there is no duplicate submission.txt.
        with tempfile.TemporaryDirectory() as tmp:
            memory = FilesystemMemory(root=tmp)
            state = State("solve instance")
            state.send("task", "user", "agent", "solve instance")
            state.data["memory_artifacts"] = [
                {
                    "name": "model_patch.diff",
                    "content": "diff --git a/x b/x\n+new line\n",
                    "description": "Final unified git diff (model_patch) produced by the run.",
                }
            ]
            ctx = MemoryContext(
                agent="agent",
                task="solve instance",
                memory_name="demo",
                run_id="r1",
                state=state,
            )

            memory.finish(ctx)

            run_dir = Path(tmp) / "demo" / "runs" / "r1"
            patch_artifact = (run_dir / "artifacts" / "model_patch.diff").read_text(
                encoding="utf-8"
            )
            manifest = (run_dir / "artifacts.md").read_text(encoding="utf-8")

            self.assertIn("+new line", patch_artifact)
            self.assertIn("model_patch.diff", manifest)
            self.assertFalse((run_dir / "artifacts" / "submission.txt").exists())
            index = (Path(tmp) / "demo" / "INDEX.md").read_text(encoding="utf-8")
            self.assertIn("model_patch.diff", index)

    def test_artifact_builder_products_reach_distiller_payload(self) -> None:
        # How each suite registers its run products: an injected artifact_builder
        # (e.g. the SWE-bench patch collector). Its output must enter the
        # distillation payload — before the model call — so distilled lessons can
        # still cite `model_patch.diff` as evidence, and must also persist on disk.
        with tempfile.TemporaryDirectory() as tmp:
            payloads: list[FilesystemMemoryPayload] = []

            def distill(payload):
                payloads.append(payload)
                return {"memory_md": ""}

            def artifact_builder(ctx):
                del ctx  # products come from the workspace, not the run State
                return [
                    FilesystemArtifact(
                        name="model_patch.diff",
                        content="diff --git a/x b/x\n+remember me\n",
                        description="Final unified git diff (model_patch).",
                    )
                ]

            memory = FilesystemMemory(
                root=tmp, distiller=distill, artifact_builder=artifact_builder
            )
            state = State("solve instance")
            state.send("task", "user", "agent", "solve instance")
            ctx = MemoryContext(
                agent="agent",
                task="solve instance",
                memory_name="demo",
                run_id="r1",
                state=state,
            )

            memory.finish(ctx)

            run_dir = Path(tmp) / "demo" / "runs" / "r1"
            patch_artifact = (run_dir / "artifacts" / "model_patch.diff").read_text(
                encoding="utf-8"
            )
            manifest = (run_dir / "artifacts.md").read_text(encoding="utf-8")

        self.assertEqual(len(payloads), 1)
        self.assertEqual([a.name for a in payloads[0].artifacts], ["model_patch.diff"])
        self.assertIn("+remember me", payloads[0].artifacts[0].content)
        self.assertIn("+remember me", patch_artifact)
        self.assertIn("model_patch.diff", manifest)

    def test_distiller_prompt_forbids_raw_line_number_citations(self) -> None:
        # P1: citations must be greppable anchors, never raw line numbers.
        from simple_agent_lab.memory.filesystem import filesystem_distillation_prompt

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
        # P3: the summary excerpt is inlined directly into the policy block.
        # P1: locate transcript evidence by searching for an anchor, not line ranges.
        # P5: read memory by absolute path, not via an env-var assignment or other
        # state that does not persist between separate actions.
        with tempfile.TemporaryDirectory() as tmp:
            memory = FilesystemMemory(root=tmp)
            ctx = MemoryContext(agent="agent", task="t", memory_name="demo")
            messages = memory.initial(ctx)
            text = text_of(messages[0].content)

        # P3: the memory_summary.md excerpt is carried inline in the block.
        self.assertIn("<memory_summary.md_excerpt>", text)
        self.assertIn("searching for the cited anchor", text)
        self.assertNotIn("targeted line ranges", text)
        # P5: no broken MEMORY_DIR= prefix guidance; show an absolute-path read.
        self.assertNotIn("MEMORY_DIR=", text)
        self.assertNotIn("copy this exact assignment", text)
        self.assertIn("persists between separate actions", text)
        self.assertIn(f"{tmp}/demo/MEMORY.md", text)
        # Tool-neutral: no shell-specific tool names baked into the prompt.
        self.assertNotIn("cat ", text)
        self.assertNotIn("grep", text)

    def test_ensure_layout_writes_skeletons_without_readme(self) -> None:
        # README.md was an unreferenced orphan (no injected text pointed at it),
        # so ensure_layout must no longer create it; the policy block now carries
        # the directory map instead.
        with tempfile.TemporaryDirectory() as tmp:
            memory = FilesystemMemory(root=tmp)
            memory_dir = Path(tmp) / "demo"
            memory.ensure_layout(memory_dir)

            self.assertFalse((memory_dir / "README.md").exists())
            self.assertTrue((memory_dir / "INDEX.md").exists())
            self.assertTrue((memory_dir / "MEMORY.md").exists())
            self.assertTrue((memory_dir / "memory_summary.md").exists())
            self.assertTrue((memory_dir / "runs").is_dir())

    def test_policy_block_maps_namespace_directory_structure(self) -> None:
        # A zero-prior agent must be able to discover task.md, artifacts.md, the
        # raw artifacts/ dir, the per-run dir, and the INDEX columns from the
        # injected policy text alone — these files are otherwise undiscoverable.
        with tempfile.TemporaryDirectory() as tmp:
            memory = FilesystemMemory(root=tmp)
            ctx = MemoryContext(agent="agent", task="t", memory_name="demo")
            text = text_of(memory.initial(ctx)[0].content)

        self.assertIn("runs/<run_id>/", text)
        self.assertIn("task.md", text)
        self.assertIn("artifacts.md", text)
        self.assertIn("artifacts/", text)
        self.assertIn("keywords", text)  # one of the INDEX table columns
        # transcript heading format is spelled out so the grep anchor (P1) lands.
        self.assertIn("## <n>. <role> (<kind>, <sender> -> <target>)", text)

    def test_filesystem_memory_uses_unique_run_directory_for_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = FilesystemMemory(root=tmp)
            state = State("repeat run")
            state.send("task", "user", "agent", "repeat run")
            ctx = MemoryContext(
                agent="agent",
                task="repeat run",
                memory_name="demo",
                run_id="r1",
                state=state,
            )

            memory.finish(ctx)
            memory.finish(ctx)

            memory_dir = Path(tmp) / "demo"
            self.assertTrue((memory_dir / "runs" / "r1").exists())
            self.assertTrue((memory_dir / "runs" / "r1_2").exists())
            index = (memory_dir / "INDEX.md").read_text(encoding="utf-8")
            self.assertIn("runs/r1/summary.md", index)
            self.assertIn("runs/r1_2/summary.md", index)

    def test_filesystem_memory_sanitizes_duplicate_artifact_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:

            def artifacts(ctx):
                del ctx
                return (
                    FilesystemArtifact(
                        "../patch.diff",
                        "first",
                        "Primary generated patch.",
                    ),
                    FilesystemArtifact(
                        "patch.diff",
                        "second",
                        "Second generated patch.",
                    ),
                )

            memory = FilesystemMemory(root=tmp, artifact_builder=artifacts)
            state = State("save artifacts")
            state.send("task", "user", "agent", "save artifacts")
            ctx = MemoryContext(
                agent="agent",
                task="save artifacts",
                memory_name="demo",
                run_id="r1",
                state=state,
            )

            memory.finish(ctx)

            artifact_dir = Path(tmp) / "demo" / "runs" / "r1" / "artifacts"
            self.assertEqual(
                sorted(path.name for path in artifact_dir.iterdir()),
                ["patch.diff", "patch_2.diff"],
            )
            manifest = (Path(tmp) / "demo" / "runs" / "r1" / "artifacts.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Primary generated patch", manifest)
            self.assertIn("Second generated patch", manifest)
            index = (Path(tmp) / "demo" / "INDEX.md").read_text(encoding="utf-8")
            self.assertIn("patch.diff, patch_2.diff", index)

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
        from simple_agent_lab.memory.transcript import render_transcript_markdown

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


if __name__ == "__main__":
    unittest.main()
