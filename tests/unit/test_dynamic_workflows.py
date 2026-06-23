from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab import Agent, Message, assistant_message, text_of
from simple_agent_lab.dynamic_workflows import (
    AgentCallOptions,
    DynamicWorkflowRuntime,
    SimpleAgentCallRunner,
    extract_javascript,
)
from simple_agent_lab.llm import Provider

FAKE_PROVIDER = Provider(id="fake", api="fake", model="fake-model")


def _task_text(visible: list[Message]) -> str:
    task = next(message for message in visible if message.kind == "task")
    return text_of(task.content)


def _echo_builder(options: AgentCallOptions, provider: Provider, cwd: Path) -> Agent:
    del provider, cwd

    def generate(visible: list[Message]) -> Message:
        return assistant_message(
            f"{options.name}:{_task_text(visible)}",
            sender=options.name,
            target="user",
            kind="final",
        )

    return Agent(options.name, generate, role=options.role)


class DynamicWorkflowRuntimeTest(unittest.TestCase):
    def test_executes_js_parallel_agent_calls_and_writes_artifacts(self) -> None:
        script = """
phase("fanout");
const results = await parallel([
  () => agent("alpha", { name: "a", cacheKey: "a" }),
  () => agent("beta", { name: "b", cacheKey: "b" })
], { maxConcurrency: 2 });
log("fanout complete");
return results.map((r) => r.output).join("|");
"""
        with tempfile.TemporaryDirectory() as tmp:
            runner = SimpleAgentCallRunner(
                provider=FAKE_PROVIDER,
                cwd=tmp,
                build_agent=_echo_builder,
            )
            result = DynamicWorkflowRuntime(runner=runner).run(
                script=script,
                task="main task",
                artifacts_dir=Path(tmp) / "artifacts",
            )

            self.assertEqual(result.output, "a:alpha|b:beta")
            self.assertTrue(result.script_path.exists())
            self.assertTrue(result.journal_path.exists())
            self.assertTrue((result.artifacts_dir / "subagents").exists())
            self.assertEqual(len(result.agent_calls), 2)

            journal = result.journal_path.read_text(encoding="utf-8")
            self.assertIn("phase_started", journal)
            self.assertIn("agent_completed", journal)

    def test_rerun_reuses_completed_agent_calls_from_journal(self) -> None:
        calls = {"n": 0}

        def builder(options: AgentCallOptions, provider: Provider, cwd: Path) -> Agent:
            del provider, cwd

            def generate(visible: list[Message]) -> Message:
                calls["n"] += 1
                return assistant_message(
                    f"{options.name}:{_task_text(visible)}",
                    sender=options.name,
                    target="user",
                    kind="final",
                )

            return Agent(options.name, generate)

        script = 'return (await agent("once", { name: "worker", cacheKey: "stable" })).output;'
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp) / "artifacts"
            runner = SimpleAgentCallRunner(
                provider=FAKE_PROVIDER,
                cwd=tmp,
                build_agent=builder,
            )
            runtime = DynamicWorkflowRuntime(runner=runner)
            first = runtime.run(script=script, task="task", artifacts_dir=artifacts)
            second = runtime.run(script=script, task="task", artifacts_dir=artifacts)

            self.assertEqual(first.output, "worker:once")
            self.assertEqual(second.output, "worker:once")
            self.assertEqual(calls["n"], 1)
            journal = (artifacts / "workflow_journal.jsonl").read_text(encoding="utf-8")
            self.assertIn("agent_reused", journal)

    def test_script_has_no_direct_require_process_or_fetch(self) -> None:
        script = """
return {
  requireType: typeof require,
  processType: typeof process,
  fetchType: typeof fetch
};
"""
        with tempfile.TemporaryDirectory() as tmp:
            runner = SimpleAgentCallRunner(
                provider=FAKE_PROVIDER,
                cwd=tmp,
                build_agent=_echo_builder,
            )
            result = DynamicWorkflowRuntime(runner=runner).run(
                script=script,
                task="task",
                artifacts_dir=Path(tmp) / "artifacts",
            )
        self.assertEqual(
            result.raw_result,
            {
                "requireType": "undefined",
                "processType": "undefined",
                "fetchType": "undefined",
            },
        )

    def test_escaped_process_does_not_inherit_parent_environment(self) -> None:
        key = "SIMPLE_AGENT_LAB_SECRET_CANARY"
        previous = os.environ.get(key)
        os.environ[key] = "should-not-leak"
        script = f"""
let envValue = "";
try {{
  const escapedProcess = this.constructor.constructor("return process")();
  envValue = escapedProcess.env.{key} || "";
}} catch (err) {{
  envValue = "escape-blocked";
}}
return {{ envValue }};
"""
        try:
            with tempfile.TemporaryDirectory() as tmp:
                runner = SimpleAgentCallRunner(
                    provider=FAKE_PROVIDER,
                    cwd=tmp,
                    build_agent=_echo_builder,
                )
                result = DynamicWorkflowRuntime(runner=runner).run(
                    script=script,
                    task="task",
                    artifacts_dir=Path(tmp) / "artifacts",
                )
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

        self.assertIn(result.raw_result["envValue"], {"", "escape-blocked"})

    def test_extract_javascript_from_fence(self) -> None:
        text = "Here:\n```js\nphase('x');\nreturn 'ok';\n```"
        self.assertEqual(extract_javascript(text), "phase('x');\nreturn 'ok';")


if __name__ == "__main__":
    unittest.main()
