from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simple_agent_lab import (
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
from simple_agent_lab.messages import system_message
from simple_agent_lab.protocols import ModelRequestEvent
from simple_agent_lab.tools import AgentTool, text_result


class MemoryBaseTest(unittest.TestCase):
    def test_memory_binding_declares_tools_and_future_hooks(
        self,
    ) -> None:
        class FakeMemory(Memory):
            finished = False
            recorded_turns = 0

            def initial(self, ctx: MemoryContext):
                return (
                    system_message(
                        "remembered initial context",
                        sender="memory",
                        target=ctx.agent,
                        kind="context",
                    ),
                )

            def recall(self, ctx: MemoryContext, query: str):
                return (
                    system_message(
                        f"remembered recall for {query}",
                        sender="memory",
                        target=ctx.agent,
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
                task="",
                session_id="s1",
            )
        )
        state = State("task")
        state.send("task", "user", "agent", "task")

        assert binding.hooks.before_run is not None
        assert binding.hooks.before_model_request is not None
        assert binding.hooks.after_turn is not None
        assert binding.hooks.after_run is not None

        initial = tuple(binding.hooks.before_run(state))
        recall = tuple(binding.hooks.before_model_request(state))
        state.record(initial[0])
        state.record(recall[0])
        state.record(
            assistant_message("done", sender="agent", target="user", kind="final")
        )
        binding.hooks.after_turn(state, tuple(state.messages))
        binding.hooks.after_run(state)

        self.assertEqual(binding.tools, ())
        self.assertEqual(message_text(initial[0]), "remembered initial context")
        self.assertEqual(message_text(recall[0]), "remembered recall for task")
        self.assertEqual(memory.recorded_turns, 1)
        self.assertGreaterEqual(memory.recorded_message_count, 4)
        self.assertTrue(memory.finished)
        self.assertEqual(memory.final_count, 1)

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

        request = next(
            event for event in seen_events if isinstance(event, ModelRequestEvent)
        )
        self.assertIn("memory_probe", [tool["name"] for tool in request.tools])
        self.assertEqual(state.messages[-1].kind, "final")

    def test_bound_memory_hooks_are_not_executed_without_hook_runtime(self) -> None:
        class FakeMemory(Memory):
            def initial(self, ctx: MemoryContext):
                return (
                    system_message(
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
                    "memory_updates": "- Use focused checks for recurring auth errors.",
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
            self.assertIn("Use focused checks", handbook)
            stable_sections = handbook.split("## Run Updates", maxsplit=1)[0]
            self.assertIn(
                "Use focused checks for recurring auth errors. [runs/run_42]",
                stable_sections,
            )
            self.assertIn("### runs/run_42", handbook)
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
                    "memory_updates": "- Login callback fixes should inspect auth logs.",
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
            stable_sections = handbook.split("## Run Updates", maxsplit=1)[0]
            self.assertIn("Login callback fixes", handbook)
            self.assertIn(
                "Login callback fixes should inspect auth logs. [runs/r1]",
                stable_sections,
            )
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
            self.assertIn("MEMORY_ROOT=", text)
            self.assertIn("- auth", text)

    def test_filesystem_memory_finish_auto_consolidates_and_prunes_run_updates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            updates_by_run = {
                "r1": "- User prefers small targeted memory changes.",
                "r2": "\n".join(
                    [
                        "- User prefers small targeted memory changes.",
                        "- Avoid broad transcript scans when INDEX.md already points to a run.",
                    ]
                ),
                "r3": "- Use `docs/auth.md` as the stable auth reference.",
                "r4": "- Keep auth memory updates evidence-backed.",
                "r5": "- Keep auth memory updates evidence-backed.",
                "r6": "- Re-run auth smoke before changing shared memory.",
            }

            def distill(payload):
                run_id = payload.context.run_id
                return {
                    "summary_md": f"## Task\n{payload.task}\n",
                    "index_row": {
                        "summary": f"auth summary {run_id}",
                        "scope": "auth",
                        "signals": "auth error",
                        "keywords": "auth, memory",
                    },
                    "memory_updates": updates_by_run[run_id],
                }

            memory = FilesystemMemory(root=tmp, distiller=distill)
            ctx = MemoryContext(agent="agent", task="fix auth", memory_name="demo")
            for run_id in updates_by_run:
                state = State(f"fix auth {run_id}")
                state.send("task", "user", "agent", f"fix auth {run_id}")
                memory.finish(
                    MemoryContext(
                        agent="agent",
                        task=f"fix auth {run_id}",
                        memory_name="demo",
                        run_id=run_id,
                        state=state,
                    )
                )

            memory_dir = memory.memory_dir(ctx)

            handbook = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
            stable_sections = handbook.split("## Run Updates", maxsplit=1)[0]
            memory_summary = (memory_dir / "memory_summary.md").read_text(
                encoding="utf-8"
            )

            self.assertIn("## Durable Lessons", handbook)
            self.assertIn(
                "User prefers small targeted memory changes. [runs/r1]", handbook
            )
            self.assertIn(
                "Avoid broad transcript scans when INDEX.md already points to a run.",
                handbook,
            )
            self.assertIn("Use `docs/auth.md` as the stable auth reference.", handbook)
            self.assertIn(
                "Keep auth memory updates evidence-backed. [runs/r4]", handbook
            )
            self.assertEqual(
                stable_sections.count("User prefers small targeted memory changes."),
                1,
            )
            self.assertEqual(
                stable_sections.count("Keep auth memory updates evidence-backed."),
                1,
            )
            self.assertNotIn("### runs/r1", handbook)
            self.assertIn("### runs/r2", handbook)
            self.assertIn("### runs/r6", handbook)
            self.assertTrue(memory_summary.startswith("v1\n"))
            self.assertIn("auth summary r6", memory_summary)
            self.assertIn("runs/r6/summary.md", memory_summary)

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


if __name__ == "__main__":
    unittest.main()
