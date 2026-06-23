from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from simple_agent_lab import Agent, Message, assistant_message, text_of
from simple_agent_lab.dynamic_workflows import (
    AgentCallOptions,
    AgentCallResult,
    DynamicWorkflowRuntime,
    SimpleAgentCallRunner,
    WorkflowJournal,
    WorkflowRuntimeOptions,
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

    def test_script_has_no_direct_or_escaped_host_capabilities(self) -> None:
        script = """
const escaped = this.constructor.constructor("return process")();
let fsReadAllowed = true;
try {
  escaped.getBuiltinModule("fs").readFileSync("/etc/hosts", "utf8");
} catch (err) {
  fsReadAllowed = false;
}
let shellAllowed = true;
try {
  escaped.getBuiltinModule("child_process").execSync("echo hi");
} catch (err) {
  shellAllowed = false;
}
return {
  requireType: typeof require,
  processType: typeof process,
  fetchType: typeof fetch,
  leakedSecret: escaped.env.SAL_DYNAMIC_WORKFLOW_SECRET || "",
  fsReadAllowed,
  shellAllowed
};
"""
        with tempfile.TemporaryDirectory() as tmp:
            runner = SimpleAgentCallRunner(
                provider=FAKE_PROVIDER,
                cwd=tmp,
                build_agent=_echo_builder,
            )
            old_secret = os.environ.get("SAL_DYNAMIC_WORKFLOW_SECRET")
            os.environ["SAL_DYNAMIC_WORKFLOW_SECRET"] = "must-not-leak"
            try:
                result = DynamicWorkflowRuntime(runner=runner).run(
                    script=script,
                    task="task",
                    artifacts_dir=Path(tmp) / "artifacts",
                )
            finally:
                if old_secret is None:
                    os.environ.pop("SAL_DYNAMIC_WORKFLOW_SECRET", None)
                else:
                    os.environ["SAL_DYNAMIC_WORKFLOW_SECRET"] = old_secret
        self.assertEqual(
            result.raw_result,
            {
                "requireType": "undefined",
                "processType": "undefined",
                "fetchType": "undefined",
                "leakedSecret": "",
                "fsReadAllowed": False,
                "shellAllowed": False,
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

    def test_journal_appends_parallel_records_as_valid_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = WorkflowJournal(Path(tmp) / "workflow_journal.jsonl")

            def append(index: int) -> None:
                journal.append(
                    "agent_completed",
                    cache_key=f"call-{index}",
                    result={"name": f"worker-{index}"},
                )

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(append, range(100)))

            lines = journal.path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 100)
            records = [json.loads(line) for line in lines]
            self.assertEqual(
                {record["kind"] for record in records}, {"agent_completed"}
            )

    def test_timeout_does_not_wait_for_blocked_subagent(self) -> None:
        class BlockingRunner:
            def __init__(self) -> None:
                self.release = threading.Event()

            def run_agent(
                self,
                prompt: str,
                *,
                options: AgentCallOptions,
                call_id: str,
                phase: str,
                artifacts_dir: Path,
            ) -> AgentCallResult:
                del prompt, options, phase, artifacts_dir
                self.release.wait(timeout=1.5)
                return AgentCallResult(
                    call_id=call_id,
                    name="blocked",
                    phase="",
                    status="completed",
                    output="too late",
                    trace_path="",
                )

        runner = BlockingRunner()
        with tempfile.TemporaryDirectory() as tmp:
            runtime = DynamicWorkflowRuntime(
                runner=runner,
                options=WorkflowRuntimeOptions(
                    max_concurrency=1,
                    timeout_seconds=0.1,
                ),
            )
            started = time.monotonic()
            try:
                with self.assertRaises(TimeoutError):
                    runtime.run(
                        script='return (await agent("slow")).output;',
                        task="task",
                        artifacts_dir=Path(tmp) / "artifacts",
                    )
            finally:
                runner.release.set()
            self.assertLess(time.monotonic() - started, 0.8)

    def test_extract_javascript_from_fence(self) -> None:
        text = "Here:\n```js\nphase('x');\nreturn 'ok';\n```"
        self.assertEqual(extract_javascript(text), "phase('x');\nreturn 'ok';")


if __name__ == "__main__":
    unittest.main()
