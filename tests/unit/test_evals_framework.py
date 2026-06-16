"""Unit-smoke for the generic eval framework (ADR generic-containerized-eval-framework).

No Docker. Covers the two seams — `ContainerBackend` (`FakeBackend` for
orchestration, `LocalProcessBackend` for a real in-process agent run) and
`ArtifactStore` (`LocalDirStore`, plus `HostHttpStore` over real loopback HTTP)
— and the unified `run_suite_instance` entry point that runs identically in or
out of a container by swapping the backend.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping
from unittest import mock

from simple_agent_lab.evals import (
    DEFAULT_MEMORY_CONTAINER_HOME,
    EVAL_KEY,
    INSTANCE_KEY,
    MEMORY_HOME_ENV,
    RESULT_KEY,
    TRACE_KEY,
    FakeBackend,
    HostHttpStore,
    LaunchSpec,
    LocalDirStore,
    LocalProcessBackend,
    RunOutcome,
    RunSpec,
    Suite,
    run_suite_instance,
)
from simple_agent_lab.evals.protocols import MCP_KEY

# Internal helpers live in their own modules, not the top-level facade.
from simple_agent_lab.evals.runner import build_command, container_name
from simple_agent_lab.evals.stores import HttpArtifactClient

SWEBENCH_CONTAINER = "simple_agent_lab.evals.suites.swebench.container"
try:
    import mcp  # noqa: F401

    HAS_MCP = True
except ImportError:  # pragma: no cover - exercised only without the extra
    HAS_MCP = False

_MCP_SKIP_REASON = "mcp extra not installed (install with: uv sync --extra mcp)"


class _DemoSuite:
    """Minimal host-side surface for a new benchmark."""

    name = "demo"
    container_module = "demo.container"

    def launch_spec(self, instance: Mapping[str, Any]) -> LaunchSpec:
        return LaunchSpec(image="demo:latest", workdir="/work")

    def task_input(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in instance.items() if k != "gold"}

    def eval_inputs(self, instance: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return None


def _simulate(answer: str):
    """A FakeBackend `on_run` that writes outputs through the bound store."""

    def on_run(spec: RunSpec, store) -> None:
        json.loads(store.get(INSTANCE_KEY).decode("utf-8"))  # container reads input
        store.put(TRACE_KEY, b'{"meta": {"suite": "demo"}}\n')
        store.put(RESULT_KEY, (json.dumps({"answer": answer}) + "\n").encode("utf-8"))

    return on_run


class OrchestrationTest(unittest.TestCase):
    def test_demo_suite_satisfies_protocol(self) -> None:
        self.assertIsInstance(_DemoSuite(), Suite)

    def test_run_suite_instance_fake_backend(self) -> None:
        instance = {"instance_id": "demo-1", "problem": "p", "gold": "SECRET"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            backend = FakeBackend(on_run=_simulate("42"), log_text="ok\n")
            artifacts = run_suite_instance(
                suite=_DemoSuite(),
                instance=instance,
                backend=backend,
                store=LocalDirStore(root),
                run_root=root,
                run_id="run-x",
                provider="fake",
            )
            # The backend received a structured spec, not a shell command.
            self.assertEqual(backend.runs[0].suite_name, "demo")
            self.assertEqual(backend.runs[0].instance_id, "demo-1")
            self.assertEqual(artifacts.logs, "ok\n")

            written = json.loads(
                (artifacts.run_dir / "input" / "instance.json").read_text()
            )
            self.assertNotIn("gold", written)  # sanitized through the store

            # A run produces result.json (+ trajectory); nothing else is shaped.
            self.assertFalse((artifacts.run_dir / "out" / "prediction.jsonl").exists())
            result = json.loads((artifacts.run_dir / RESULT_KEY).read_text())
            self.assertEqual(result["answer"], "42")

    def test_run_suite_instance_stages_mcp_config_separately(self) -> None:
        instance = {"instance_id": "demo-1", "problem": "p", "gold": "SECRET"}
        mcp_config = {
            "servers": [
                {
                    "name": "workspace",
                    "transport": "stdio",
                    "command": "python",
                    "args": ["-m", "server"],
                    "cwd": "/testbed",
                }
            ]
        }

        def on_run(spec: RunSpec, store) -> None:
            agent_input = json.loads(store.get(INSTANCE_KEY).decode("utf-8"))
            staged_mcp = json.loads(store.get(MCP_KEY).decode("utf-8"))
            self.assertNotIn("servers", agent_input)
            self.assertEqual(staged_mcp, mcp_config)
            store.put(TRACE_KEY, b'{"meta": {"suite": "demo"}}\n')
            store.put(RESULT_KEY, b'{"answer": "ok"}\n')

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            artifacts = run_suite_instance(
                suite=_DemoSuite(),
                instance=instance,
                backend=FakeBackend(on_run=on_run),
                store=LocalDirStore(root),
                run_root=root,
                run_id="run-x",
                provider="fake",
                mcp_config=mcp_config,
            )

            staged = json.loads((artifacts.run_dir / MCP_KEY).read_text())
            self.assertEqual(staged, mcp_config)

    def test_build_command_targets_the_generic_runner(self) -> None:
        spec = RunSpec(
            suite_name="s",
            container_module="m",
            instance_id="i",
            launch_spec=LaunchSpec(image="img", workdir="/w"),
            max_turns=5,
            provider="fake",
            api_kind="openai-chat",
            wheelhouse_mount="/wh",
            run_name="n",
        )
        cmd = build_command(spec)
        self.assertEqual(cmd[:2], ("bash", "-lc"))
        self.assertIn("simple_agent_lab.evals.in_container", cmd[-1])
        self.assertIn("--find-links /wh", cmd[-1])

    def test_build_command_installs_mcp_extra_when_requested(self) -> None:
        spec = RunSpec(
            suite_name="s",
            container_module="m",
            instance_id="i",
            launch_spec=LaunchSpec(image="img", workdir="/w"),
            max_turns=5,
            provider="fake",
            api_kind="openai-chat",
            wheelhouse_mount="/wh",
            run_name="n",
            package_extras=("mcp",),
        )
        cmd = build_command(spec)

        self.assertIn("simple-agent-lab[mcp]", cmd[-1])


class HostHttpStoreTest(unittest.TestCase):
    def test_http_client_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            with HostHttpStore(base, container_host="127.0.0.1") as store:
                bound = store.bind(base / "run-1" / "inst-1")
                binding = bound.container_binding()
                client = HttpArtifactClient(
                    binding.env["SAL_STORE_URL"], binding.env["SAL_STORE_TOKEN"]
                )
                client.put("out/result.json", b'{"ok": true}')
                self.assertEqual(bound.get("out/result.json"), b'{"ok": true}')
                self.assertEqual(client.get("out/result.json"), b'{"ok": true}')
                self.assertFalse(client.exists("out/missing.json"))

    def test_bad_token_is_rejected(self) -> None:
        import urllib.error

        with tempfile.TemporaryDirectory() as tmp:
            with HostHttpStore(Path(tmp), container_host="127.0.0.1") as store:
                url = store.bind(Path(tmp)).container_binding().env["SAL_STORE_URL"]
                client = HttpArtifactClient(url, "wrong-token")
                with self.assertRaises(urllib.error.HTTPError):
                    client.put("x", b"y")


class _SwebenchLikeSuite:
    """Reuses the real SWE-bench container half; trivial host half (no swebench dep)."""

    name = "swebench"
    container_module = SWEBENCH_CONTAINER

    def launch_spec(self, instance: Mapping[str, Any]) -> LaunchSpec:
        return LaunchSpec(image="(in-process)", workdir="(in-process)")

    def task_input(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        return dict(instance)

    def eval_inputs(self, instance: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return None


class LocalProcessBackendTest(unittest.TestCase):
    """Unified entry point runs a real agent in-process — no Docker, no network."""

    def test_run_suite_instance_in_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "testbed"
            repo.mkdir()

            def git(*args: str) -> None:
                subprocess.run(
                    ["git", *args], cwd=repo, check=True, capture_output=True
                )

            git("init")
            git("config", "user.email", "t@example.invalid")
            git("config", "user.name", "T")
            git("config", "commit.gpgsign", "false")
            (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
            git("add", "-A")
            git("commit", "-m", "base")

            root = Path(tmp) / "runs"
            store = LocalDirStore(root)
            instance = {
                "instance_id": "demo__repo-1",
                "problem_statement": "Make it better.",
                "language": "python",
            }
            artifacts = run_suite_instance(
                suite=_SwebenchLikeSuite(),
                instance=instance,
                backend=LocalProcessBackend(workspace=repo),
                store=store,
                run_root=root,
                run_id="r",
                provider="fake",
                max_turns=3,
            )

            self.assertEqual(artifacts.status_code, 0)
            bound = store.bind(artifacts.run_dir)
            result = json.loads(bound.get(RESULT_KEY).decode("utf-8"))
            self.assertIn("model_patch", result)
            trace = json.loads(bound.get(TRACE_KEY).decode("utf-8"))
            self.assertEqual(trace["meta"]["suite"], "swebench")
            self.assertFalse(trace["meta"]["in_progress"])

    @unittest.skipUnless(HAS_MCP, _MCP_SKIP_REASON)
    def test_run_suite_instance_in_process_with_mcp_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "testbed"
            repo.mkdir()

            def git(*args: str) -> None:
                subprocess.run(
                    ["git", *args], cwd=repo, check=True, capture_output=True
                )

            git("init")
            git("config", "user.email", "t@example.invalid")
            git("config", "user.name", "T")
            git("config", "commit.gpgsign", "false")
            (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
            git("add", "-A")
            git("commit", "-m", "base")

            root = Path(tmp) / "runs"
            store = LocalDirStore(root)
            artifacts = run_suite_instance(
                suite=_SwebenchLikeSuite(),
                instance={
                    "instance_id": "demo__repo-mcp",
                    "problem_statement": "Inspect the workspace.",
                    "language": "python",
                },
                backend=LocalProcessBackend(workspace=repo),
                store=store,
                run_root=root,
                run_id="mcp",
                provider="fake",
                max_turns=1,
                mcp_config={
                    "servers": [
                        {
                            "name": "workspace",
                            "transport": "stdio",
                            "command": sys.executable,
                            "args": [
                                "-m",
                                "simple_agent_lab.mcp.workspace_server",
                            ],
                            "cwd": str(repo),
                        }
                    ]
                },
            )

            self.assertEqual(artifacts.status_code, 0, artifacts.logs)
            trace = json.loads(store.bind(artifacts.run_dir).get(TRACE_KEY))
            tools = trace["model_turns"][0]["tools"]
            self.assertIn("workspace_list_files", {tool["name"] for tool in tools})

    def test_oracle_run_reproduces_gold_patch(self) -> None:
        """Oracle mode applies the gold patch (no model) and extract reproduces it.

        This is the suite self-check the integration doc prescribes: a tiny
        self-contained instance + its gold `patch`, driven in-process with
        `provider="oracle"`, must yield a `model_patch` equal to the gold change.
        No Docker, no network, no LLM — just the real container half end to end.
        """

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "testbed"
            repo.mkdir()

            def git(*args: str) -> str:
                return subprocess.run(
                    ["git", *args], cwd=repo, check=True, capture_output=True, text=True
                ).stdout

            git("init")
            git("config", "user.email", "t@example.invalid")
            git("config", "user.name", "T")
            git("config", "commit.gpgsign", "false")
            (repo / "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
            git("add", "-A")
            git("commit", "-m", "base")

            # Build a real gold patch, then revert so the oracle has to re-apply it.
            (repo / "app.py").write_text("def f():\n    return 2\n", encoding="utf-8")
            gold_patch = git("diff")
            git("checkout", "--", "app.py")
            self.assertIn("return 2", gold_patch)

            root = Path(tmp) / "runs"
            store = LocalDirStore(root)
            instance = {
                "instance_id": "demo__repo-oracle",
                "problem_statement": "Make f return 2.",
                "language": "python",
                "patch": gold_patch,  # gold/private — kept only for oracle mode
            }
            artifacts = run_suite_instance(
                suite=_SwebenchLikeSuite(),
                instance=instance,
                backend=LocalProcessBackend(workspace=repo),
                store=store,
                run_root=root,
                run_id="oracle",
                provider="oracle",  # no model: apply the reference solution
            )

            self.assertEqual(artifacts.status_code, 0)
            bound = store.bind(artifacts.run_dir)
            result = json.loads(bound.get(RESULT_KEY).decode("utf-8"))
            self.assertIn("return 2", result["model_patch"])
            # Oracle mode keeps the gold field in the stored instance (trusted).
            stored = json.loads(
                (artifacts.run_dir / "input" / "instance.json").read_text()
            )
            self.assertIn("patch", stored)
            # The trajectory marks the run as oracle.
            trace = json.loads(bound.get(TRACE_KEY).decode("utf-8"))
            self.assertTrue(trace["meta"]["oracle"])

    def test_oracle_without_apply_hook_fails_clearly(self) -> None:
        """A suite whose container half has no apply_oracle errors, not no-ops."""

        import sys
        import types

        # A real, importable container module that lacks apply_oracle.
        mod_name = "sal_test_nooracle_container"
        mod = types.ModuleType(mod_name)
        mod.build_task = lambda instance, *, workdir: "noop task"  # type: ignore[attr-defined]
        mod.extract_result = (  # type: ignore[attr-defined]
            lambda workspace, instance, *, context=None: {"model_patch": ""}
        )
        sys.modules[mod_name] = mod
        self.addCleanup(lambda: sys.modules.pop(mod_name, None))

        class _NoOracleSuite(_DemoSuite):
            container_module = mod_name

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            instance = {"instance_id": "demo-1", "problem": "p"}
            artifacts = run_suite_instance(
                suite=_NoOracleSuite(),
                instance=instance,
                backend=LocalProcessBackend(),
                store=LocalDirStore(root),
                run_root=root,
                run_id="oracle",
                provider="oracle",
            )
            self.assertEqual(artifacts.status_code, 1)  # surfaced as a failed run
            self.assertIn("apply_oracle", artifacts.logs)

    def test_workspace_factory_isolates_concurrent_runs(self) -> None:
        """A workspace factory gives each run its own dir, so concurrency is safe."""
        from simple_agent_lab.evals import run_dataset

        with tempfile.TemporaryDirectory() as tmp:
            ws_base = Path(tmp) / "ws"
            root = Path(tmp) / "runs"
            instances = [
                {
                    "instance_id": f"i-{n}",
                    "problem_statement": "p",
                    "language": "python",
                }
                for n in range(5)
            ]
            # Each run gets ws_base/<instance_id>; no two runs share a workspace.
            backend = LocalProcessBackend(
                workspace=lambda spec: ws_base / spec.instance_id
            )
            report = run_dataset(
                suite=_SwebenchLikeSuite(),
                instances=instances,
                backend=backend,
                store=LocalDirStore(root),
                run_root=root,
                run_id="batch",
                concurrency=4,
                provider="fake",
                max_turns=2,
            )
            self.assertEqual(report.summary(), {"total": 5, "ok": 5, "failed": 0})
            # Each run materialized its own workspace dir.
            for n in range(5):
                self.assertTrue((ws_base / f"i-{n}").is_dir())


class InEnvScoringTest(unittest.TestCase):
    """In-environment scoring: the container-half `evaluate` hook writes the
    verdict into result.json during the run, gated on staged `eval_inputs`
    (ADR collapse-scorer-seam-into-run-primitive). No separate scoring driver."""

    @staticmethod
    def _reuse_module() -> str:
        """Register a throwaway container module whose `evaluate` grades gold."""

        import sys
        import types

        mod_name = "sal_test_reuse_container"
        mod = types.ModuleType(mod_name)
        mod.build_task = lambda instance, *, workdir: "noop"  # type: ignore[attr-defined]
        mod.extract_result = (  # type: ignore[attr-defined]
            lambda workspace, instance, *, context=None: {"answer": "solved"}
        )

        def _evaluate(workspace, instance, *, context=None):  # type: ignore[no-untyped-def]
            # The host staged gold inputs under EVAL_KEY → context["eval"].
            staged = (context or {}).get("eval") or {}
            return {"resolved": staged.get("expected") == "solved"}

        mod.evaluate = _evaluate  # type: ignore[attr-defined]
        sys.modules[mod_name] = mod
        return mod_name

    def test_evaluate_hook_merges_verdict_into_result(self) -> None:
        import sys

        mod_name = self._reuse_module()
        self.addCleanup(lambda: sys.modules.pop(mod_name, None))

        class _ReuseSuite(_DemoSuite):
            container_module = mod_name

            def eval_inputs(self, instance):  # type: ignore[no-untyped-def]
                return {"expected": "solved"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            artifacts = run_suite_instance(
                suite=_ReuseSuite(),
                instance={"instance_id": "r-0", "gold": "solved"},
                backend=LocalProcessBackend(),
                store=LocalDirStore(root),
                run_root=root,
                run_id="reuse",
                provider="fake",
                max_turns=1,
            )
            # Gold was staged privately (EVAL_KEY), not in the agent's instance.
            staged = json.loads((artifacts.run_dir / EVAL_KEY).read_text())
            self.assertEqual(staged["expected"], "solved")
            agent_view = json.loads((artifacts.run_dir / INSTANCE_KEY).read_text())
            self.assertNotIn("gold", agent_view)
            # The evaluate hook merged its verdict into result.json — the verdict
            # lives next to the product, so there is no second phase to run.
            result = json.loads((artifacts.run_dir / RESULT_KEY).read_text())
            self.assertTrue(result["resolved"])
            self.assertEqual(result["answer"], "solved")

    def test_hook_is_skipped_without_staged_eval_inputs(self) -> None:
        import sys

        mod_name = self._reuse_module()
        self.addCleanup(lambda: sys.modules.pop(mod_name, None))

        class _NoGoldSuite(_DemoSuite):
            container_module = mod_name
            # eval_inputs inherited from _DemoSuite returns None → no staged gold.

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            artifacts = run_suite_instance(
                suite=_NoGoldSuite(),
                instance={"instance_id": "r-1"},
                backend=LocalProcessBackend(),
                store=LocalDirStore(root),
                run_root=root,
                run_id="nogold",
                provider="fake",
                max_turns=1,
            )
            result = json.loads((artifacts.run_dir / RESULT_KEY).read_text())
            # No gold staged → the hook never ran → only the raw product remains.
            self.assertNotIn("resolved", result)
            self.assertEqual(result["answer"], "solved")


class RunDatasetTest(unittest.TestCase):
    """The minimal controller: run many instances over a pool, aggregate outcomes."""

    def test_concurrent_run_with_ordering_and_callback(self) -> None:
        from simple_agent_lab.evals import run_dataset

        instances = [{"instance_id": f"i-{n}", "n": n} for n in range(6)]
        seen: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            backend = FakeBackend(on_run=_simulate("ok"))
            report = run_dataset(
                suite=_DemoSuite(),
                instances=instances,
                backend=backend,
                store=LocalDirStore(root),
                run_root=root,
                run_id="batch",
                concurrency=4,
                provider="fake",
                on_result=lambda r: seen.append(r.instance_id),
            )
        # All ran; results follow input order regardless of completion order.
        self.assertEqual(report.summary(), {"total": 6, "ok": 6, "failed": 0})
        self.assertEqual(
            [r.instance_id for r in report.results], [f"i-{n}" for n in range(6)]
        )
        self.assertEqual(len(seen), 6)
        # Each instance got its own run dir under the batch.
        self.assertEqual(len(backend.runs), 6)

    def test_error_is_captured_and_retried(self) -> None:
        from simple_agent_lab.evals import run_dataset

        attempts: dict[str, int] = {}

        def flaky(spec, store) -> None:
            n = attempts.get(spec.instance_id, 0) + 1
            attempts[spec.instance_id] = n
            if spec.instance_id == "bad":
                raise RuntimeError("boom")
            _simulate("ok")(spec, store)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            backend = FakeBackend(on_run=flaky)
            report = run_dataset(
                suite=_DemoSuite(),
                instances=[{"instance_id": "good"}, {"instance_id": "bad"}],
                backend=backend,
                store=LocalDirStore(root),
                run_root=root,
                run_id="batch",
                concurrency=2,
                max_attempts=3,
                provider="fake",
            )
        self.assertEqual(report.summary(), {"total": 2, "ok": 1, "failed": 1})
        bad = [r for r in report.results if r.instance_id == "bad"][0]
        self.assertFalse(bad.ok)
        self.assertIn("boom", bad.error)
        self.assertEqual(bad.attempts, 3)  # retried up to max_attempts
        self.assertEqual(attempts["bad"], 3)


class SubmitReconcileTest(unittest.TestCase):
    """Host-reentrant batch: submit, drop all memory, reconcile from disk only."""

    def test_submit_then_reconcile_from_fresh_process(self) -> None:
        from simple_agent_lab.evals import reconcile_dataset, submit_dataset

        instances = [{"instance_id": f"i-{n}", "problem": "p"} for n in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()

            # --- "first process": submit, then throw the backend/store away ---
            submit_dataset(
                suite=_DemoSuite(),
                instances=instances,
                backend=FakeBackend(on_run=_simulate("42")),
                store=LocalDirStore(root),
                run_root=root,
                run_id="batch",
                provider="fake",
            )
            # The manifest is on disk; nothing else carried over.
            self.assertTrue((root / "batch" / "batch.json").exists())

            # --- "fresh process": new backend + new store, recover from disk ---
            seen: list[str] = []
            report = reconcile_dataset(
                suite=_DemoSuite(),
                backend=FakeBackend(),  # no on_run, no memory of the submit
                store=LocalDirStore(root),
                run_root=root,
                run_id="batch",
                poll_interval_s=0,
                on_result=lambda r: seen.append(r.instance_id),
            )

            self.assertEqual(report.summary(), {"total": 4, "ok": 4, "failed": 0})
            self.assertEqual(len(seen), 4)
            # result.json is the decoupling artifact: a fresh process can read
            # each run's product back without re-running.
            for r in report.results:
                assert r.artifacts is not None  # reconcile completed every run
                result = json.loads((r.artifacts.run_dir / RESULT_KEY).read_text())
                self.assertEqual(result["answer"], "42")

    def test_reconcile_waits_on_pending_runs(self) -> None:
        from simple_agent_lab.evals import reconcile_dataset, submit_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            submit_dataset(
                suite=_DemoSuite(),
                instances=[{"instance_id": "slow"}],
                backend=FakeBackend(on_run=_simulate("ok")),
                store=LocalDirStore(root),
                run_root=root,
                run_id="b",
                provider="fake",
            )
            # poll() returns None twice before reporting done — exercises the loop.
            report = reconcile_dataset(
                suite=_DemoSuite(),
                backend=FakeBackend(pending_polls=2),
                store=LocalDirStore(root),
                run_root=root,
                run_id="b",
                poll_interval_s=0,
            )
        self.assertEqual(report.summary(), {"total": 1, "ok": 1, "failed": 0})

    def test_submit_requires_detaching_backend(self) -> None:
        from simple_agent_lab.evals import LocalProcessBackend, submit_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with self.assertRaises(TypeError):
                submit_dataset(
                    suite=_DemoSuite(),
                    instances=[{"instance_id": "x"}],
                    backend=LocalProcessBackend(),  # run-only, cannot outlive host
                    store=LocalDirStore(root),
                    run_root=root,
                    run_id="b",
                )

    def test_mid_submit_crash_leaves_recoverable_manifest(self) -> None:
        """A crash partway through submit still records every started container."""
        from simple_agent_lab.evals import reconcile_dataset, submit_dataset
        from simple_agent_lab.evals.batch import BATCH_KEY, _batch_store

        class _CrashAtThird(FakeBackend):
            def __init__(self) -> None:
                super().__init__(on_run=_simulate("ok"))
                self.n = 0

            def submit(self, spec, *, store, binding):  # type: ignore[no-untyped-def]
                self.n += 1
                if self.n == 3:
                    raise RuntimeError("host died mid-submit")
                return super().submit(spec, store=store, binding=binding)

        instances = [{"instance_id": f"i-{n}"} for n in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            store = LocalDirStore(root)
            with self.assertRaises(RuntimeError):
                submit_dataset(
                    suite=_DemoSuite(),
                    instances=instances,
                    backend=_CrashAtThird(),
                    store=store,
                    run_root=root,
                    run_id="b",
                    provider="fake",
                )
            # The 2 containers started before the crash are in the manifest.
            manifest = json.loads(
                _batch_store(store, root, "b").get(BATCH_KEY).decode("utf-8")
            )
            self.assertEqual(len(manifest), 2)

            # A fresh process can reconcile the partial batch — no orphans.
            report = reconcile_dataset(
                suite=_DemoSuite(),
                backend=FakeBackend(),
                store=LocalDirStore(root),
                run_root=root,
                run_id="b",
                poll_interval_s=0,
            )
            self.assertEqual(report.summary(), {"total": 2, "ok": 2, "failed": 0})

    def test_reconcile_uses_result_when_poll_never_reports_done(self) -> None:
        """poll() that never returns done still completes if result.json exists."""
        from simple_agent_lab.evals import reconcile_dataset, submit_dataset

        class _NeverDone(FakeBackend):
            def poll(self, handle):  # type: ignore[no-untyped-def]
                return None  # daemon never reports completion (e.g. already gone)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            submit_dataset(
                suite=_DemoSuite(),
                instances=[{"instance_id": "x"}],
                backend=FakeBackend(on_run=_simulate("ok")),  # writes result.json
                store=LocalDirStore(root),
                run_root=root,
                run_id="b",
                provider="fake",
            )
            report = reconcile_dataset(
                suite=_DemoSuite(),
                backend=_NeverDone(),
                store=LocalDirStore(root),
                run_root=root,
                run_id="b",
                poll_interval_s=0,
            )
        self.assertEqual(report.summary(), {"total": 1, "ok": 1, "failed": 0})

    def test_reconcile_completes_off_result_without_instance_record(self) -> None:
        """Reconcile keys completion on result.json — decoupled from the instance.

        The run/score split (ADR scorer-seam-and-scoring-topology) means reconcile no longer needs the
        instance record; a missing input/instance.json does not fail a run whose
        result.json landed. The instance re-enters only at the score phase.
        """
        from simple_agent_lab.evals import reconcile_dataset, submit_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            submit_dataset(
                suite=_DemoSuite(),
                instances=[{"instance_id": "x"}],
                backend=FakeBackend(on_run=_simulate("ok")),
                store=LocalDirStore(root),
                run_root=root,
                run_id="b",
                provider="fake",
            )
            (root / "b" / "x" / "input" / "instance.json").unlink()
            report = reconcile_dataset(
                suite=_DemoSuite(),
                backend=FakeBackend(),  # poll() -> done immediately
                store=LocalDirStore(root),
                run_root=root,
                run_id="b",
                poll_interval_s=0,
            )
            # The result is present, so the run is done regardless of the instance.
            self.assertEqual(report.summary(), {"total": 1, "ok": 1, "failed": 0})

    def test_reconcile_floors_the_idle_wait(self) -> None:
        """poll_interval_s=0 must not busy-spin: the loop sleeps a real floor."""
        from simple_agent_lab.evals import reconcile_dataset, submit_dataset

        slept: list[float] = []

        # Returns None a few times so the loop has to wait, then completes.
        class _SlowThenDone(FakeBackend):
            def __init__(self) -> None:
                super().__init__()
                self.n = 0

            def poll(self, handle):  # type: ignore[no-untyped-def]
                self.n += 1
                return None if self.n < 3 else RunOutcome(status_code=0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            # No on_run → no result.json, so poll()==None falls through to the
            # idle wait (instead of the result-on-disk short-circuit).
            submit_dataset(
                suite=_DemoSuite(),
                instances=[{"instance_id": "x"}],
                backend=FakeBackend(),
                store=LocalDirStore(root),
                run_root=root,
                run_id="b",
                provider="fake",
            )
            reconcile_dataset(
                suite=_DemoSuite(),
                backend=_SlowThenDone(),
                store=LocalDirStore(root),
                run_root=root,
                run_id="b",
                poll_interval_s=0,  # caller asks for 0 …
                sleep_fn=slept.append,
            )
        # … but the loop floored every wait above 0 (no busy-spin).
        self.assertTrue(slept)
        self.assertTrue(all(s >= 0.5 for s in slept))


class ReviewFixesTest(unittest.TestCase):
    """Docker-path guards (no real Docker): exit-code parsing, atomic put, names."""

    def test_exit_status_guards_null_and_nonint(self) -> None:
        from simple_agent_lab.evals.backends.docker_local import exit_status

        self.assertEqual(exit_status({"ExitCode": 0}), 0)
        self.assertEqual(exit_status({"ExitCode": 137}), 137)
        self.assertEqual(exit_status({"ExitCode": None}), 1)  # null → failure
        self.assertEqual(exit_status({}), 1)  # missing → failure

    def test_host_http_put_is_atomic(self) -> None:
        """put writes via a temp file + replace (no torn read), like LocalDirStore."""
        import os as _os

        from simple_agent_lab.evals import HostHttpStore

        seen_tmp: list[str] = []
        real_replace = _os.replace

        def spy_replace(src, dst):  # type: ignore[no-untyped-def]
            seen_tmp.append(str(src))
            return real_replace(src, dst)

        with tempfile.TemporaryDirectory() as tmp:
            store = HostHttpStore(Path(tmp), container_host="127.0.0.1")
            with mock.patch(
                "simple_agent_lab.evals.stores.host_http.os.replace", spy_replace
            ):
                with store:
                    bound = store.bind(Path(tmp) / "r" / "i")
                    bound.put("out/result.json", b'{"ok": true}')
                    self.assertEqual(bound.get("out/result.json"), b'{"ok": true}')
        self.assertTrue(seen_tmp and seen_tmp[0].endswith(".tmp"))

    def test_container_name_clamped_and_distinct(self) -> None:
        short = container_name("swebench", "sympy__sympy-23824", "run-1")
        self.assertLessEqual(len(short), 200)
        self.assertEqual(short, "swebench.sympy__sympy-23824.run-1")  # unchanged

        long_a = container_name("swebench", "x" * 300, "run-1")
        long_b = container_name("swebench", "x" * 300 + "y", "run-1")
        self.assertLessEqual(len(long_a), 200)
        self.assertNotEqual(long_a, long_b)  # distinct overflowing names stay distinct

    def test_container_name_is_namespaced_by_run_root(self) -> None:
        instance = {"instance_id": "astropy__astropy-13398"}
        run_id = "003af6184cce-9119f5eb01dc"

        with tempfile.TemporaryDirectory() as tmp:
            left_root = Path(tmp) / "self_evolving" / "swebench_runs"
            right_root = Path(tmp) / "hyperagents" / "swebench_runs"
            left_backend = FakeBackend()
            right_backend = FakeBackend()

            run_suite_instance(
                suite=_DemoSuite(),
                instance=instance,
                backend=left_backend,
                store=LocalDirStore(left_root),
                run_root=left_root,
                run_id=run_id,
            )
            run_suite_instance(
                suite=_DemoSuite(),
                instance=instance,
                backend=right_backend,
                store=LocalDirStore(right_root),
                run_root=right_root,
                run_id=run_id,
            )

        left_name = left_backend.runs[0].run_name
        right_name = right_backend.runs[0].run_name
        self.assertNotEqual(left_name, right_name)
        self.assertTrue(left_name.startswith("demo."))
        self.assertTrue(left_name.endswith(f".{run_id}"))

    def test_docker_backends_import_without_docker_and_error_clearly(self) -> None:
        """The optional `docker` dep is import-guarded: the module loads without it,
        and using a docker backend gives an actionable install hint, not ImportError."""
        from simple_agent_lab.evals.backends import docker_local

        if docker_local.docker is not None:
            self.skipTest("docker is installed; can't exercise the missing-dep path")
        with self.assertRaises(RuntimeError) as ctx:
            docker_local._require_docker()
        self.assertIn("swebench", str(ctx.exception))

    def test_wheelhouse_and_uv_are_bind_mounted_read_only(self) -> None:
        """The offline path: a host wheelhouse + uv binary become read-only mounts
        at the in-container find-links path and /tmp/uv, beside the store mount."""
        from simple_agent_lab.evals.backends.docker_local import (
            UV_CONTAINER_PATH,
            with_local_mounts,
        )
        from simple_agent_lab.evals.protocols import ContainerBinding

        with tempfile.TemporaryDirectory() as tmp:
            wheelhouse = Path(tmp) / "wheelhouse"
            wheelhouse.mkdir()
            uv = Path(tmp) / "uv"
            uv.write_bytes(b"#!/bin/sh\n")
            store_mount = {"/host/run": {"bind": "/agent/run", "mode": "rw"}}

            bound = with_local_mounts(
                ContainerBinding(
                    mounts=dict(store_mount), env={"SAL_STORE": "localdir"}
                ),
                wheelhouse=wheelhouse,
                wheelhouse_mount="/agent/wheelhouse",
                uv_binary=uv,
            )

        # The store's mount and env are preserved alongside the new mounts.
        self.assertEqual(bound.mounts["/host/run"], store_mount["/host/run"])
        self.assertEqual(bound.env, {"SAL_STORE": "localdir"})
        self.assertEqual(
            bound.mounts[str(wheelhouse.resolve())],
            {"bind": "/agent/wheelhouse", "mode": "ro"},
        )
        self.assertEqual(
            bound.mounts[str(uv.resolve())],
            {"bind": UV_CONTAINER_PATH, "mode": "ro"},
        )

    def test_memory_home_is_bind_mounted_read_write(self) -> None:
        from simple_agent_lab.evals.backends.docker_local import with_local_mounts
        from simple_agent_lab.evals.protocols import ContainerBinding

        with tempfile.TemporaryDirectory() as tmp:
            memory_home = Path(tmp) / "memory"
            bound = with_local_mounts(
                ContainerBinding(env={"SAL_STORE": "localdir"}),
                wheelhouse=None,
                wheelhouse_mount="/agent/wheelhouse",
                uv_binary=None,
                memory_home=memory_home,
            )

            self.assertTrue(memory_home.exists())
            self.assertEqual(
                bound.mounts[str(memory_home.resolve())],
                {"bind": DEFAULT_MEMORY_CONTAINER_HOME, "mode": "rw"},
            )
            self.assertEqual(bound.env["SAL_STORE"], "localdir")
            self.assertEqual(bound.env[MEMORY_HOME_ENV], DEFAULT_MEMORY_CONTAINER_HOME)

    def test_memory_home_from_env_reads_optional_mount_path(self) -> None:
        from simple_agent_lab.evals.in_container import memory_home_from_env

        self.assertIsNone(memory_home_from_env(env={}))
        self.assertEqual(
            memory_home_from_env(env={MEMORY_HOME_ENV: "/agent/memory"}),
            Path("/agent/memory"),
        )

    def test_wheelhouse_without_mount_path_fails_clearly(self) -> None:
        """A wheelhouse with no in-container find-links path is a misconfiguration."""
        from simple_agent_lab.evals.backends.docker_local import with_local_mounts
        from simple_agent_lab.evals.protocols import ContainerBinding

        with self.assertRaises(ValueError) as ctx:
            with_local_mounts(
                ContainerBinding(),
                wheelhouse="/some/wheelhouse",
                wheelhouse_mount=None,
                uv_binary=None,
            )
        self.assertIn("wheelhouse_mount", str(ctx.exception))

    def test_no_offline_inputs_leaves_binding_untouched(self) -> None:
        from simple_agent_lab.evals.backends.docker_local import with_local_mounts
        from simple_agent_lab.evals.protocols import ContainerBinding

        binding = ContainerBinding(
            mounts={"/host/run": {"bind": "/agent/run", "mode": "rw"}}
        )
        self.assertIs(
            with_local_mounts(
                binding,
                wheelhouse=None,
                wheelhouse_mount="/agent/wheelhouse",
                uv_binary=None,
            ),
            binding,
        )

    def test_local_docker_backend_sets_client_timeout(self) -> None:
        from types import SimpleNamespace

        from simple_agent_lab.evals.backends import docker_local
        from simple_agent_lab.evals.backends.docker_local import LocalDockerBackend

        fake_client = object()
        fake_docker = SimpleNamespace(from_env=mock.Mock(return_value=fake_client))
        with mock.patch.object(docker_local, "docker", fake_docker):
            client = LocalDockerBackend(docker_timeout_s=180.0)._client()

        self.assertIs(client, fake_client)
        fake_docker.from_env.assert_called_once_with(timeout=180.0)

    def test_remote_docker_backend_sets_client_timeout(self) -> None:
        from types import SimpleNamespace

        from simple_agent_lab.evals.backends import remote_docker

        fake_client = SimpleNamespace()
        fake_docker = SimpleNamespace(
            DockerClient=mock.Mock(return_value=fake_client),
            from_env=mock.Mock(return_value=fake_client),
        )

        with mock.patch.object(
            remote_docker, "_require_docker", return_value=fake_docker
        ):
            Remote = remote_docker.RemoteDockerBackend
            Remote(base_url="ssh://worker", docker_timeout_s=240.0)._client()
            Remote(docker_timeout_s=120.0)._client()

        fake_docker.DockerClient.assert_called_once_with(
            base_url="ssh://worker", timeout=240.0
        )
        fake_docker.from_env.assert_called_once_with(timeout=120.0)

    def test_start_container_removes_created_container_on_timeout(self) -> None:
        """A Docker SDK timeout after create() must not leave a named container."""
        from types import SimpleNamespace

        from simple_agent_lab.evals.backends import docker_local
        from simple_agent_lab.evals.backends.docker_local import start_container
        from simple_agent_lab.evals.protocols import ContainerBinding

        class FakeImageNotFound(Exception):
            pass

        class FakeAPIError(Exception):
            pass

        class FakeStartTimeout(Exception):
            pass

        class FakeImages:
            def get(self, image: str) -> object:
                return object()

        class FakeContainer:
            def __init__(self) -> None:
                self.removed = False

            def start(self) -> None:
                raise FakeStartTimeout("start timed out")

            def remove(self, *, force: bool) -> None:
                self.removed = force

        class FakeContainers:
            def __init__(self) -> None:
                self.created = FakeContainer()

            def create(self, **kwargs: Any) -> FakeContainer:
                return self.created

        class FakeClient:
            def __init__(self) -> None:
                self.images = FakeImages()
                self.containers = FakeContainers()

        client = FakeClient()
        spec = RunSpec(
            suite_name="s",
            container_module="m",
            instance_id="i",
            launch_spec=LaunchSpec(image="img", workdir="/w"),
            max_turns=1,
            provider="fake",
            api_kind="openai-chat",
            run_name="run-1",
        )

        fake_docker = SimpleNamespace(
            errors=SimpleNamespace(
                ImageNotFound=FakeImageNotFound,
                APIError=FakeAPIError,
            )
        )
        with mock.patch.object(docker_local, "docker", fake_docker):
            with self.assertRaises(FakeStartTimeout):
                start_container(
                    client,
                    spec,
                    ContainerBinding(),
                    user="root",
                    pull="missing",
                    environment={},
                )
        self.assertTrue(client.containers.created.removed)

    def test_local_docker_run_removes_container_when_wait_times_out(self) -> None:
        """A blocking run timeout after submit() must force-remove the container."""
        from types import SimpleNamespace

        from simple_agent_lab.evals.backends import docker_local
        from simple_agent_lab.evals.backends.docker_local import LocalDockerBackend
        from simple_agent_lab.evals.protocols import ContainerBinding
        from simple_agent_lab.evals.protocols import RunHandle

        class FakeNotFound(Exception):
            pass

        class FakeWaitTimeout(Exception):
            pass

        class FakeContainer:
            def __init__(self) -> None:
                self.removed = False

            def wait(self) -> object:
                raise FakeWaitTimeout("wait timed out")

            def remove(self, *, force: bool) -> None:
                self.removed = force

        class FakeContainers:
            def __init__(self, container: FakeContainer) -> None:
                self.container = container

            def get(self, name: str) -> FakeContainer:
                return self.container

        class FakeClient:
            def __init__(self, container: FakeContainer) -> None:
                self.containers = FakeContainers(container)

        container = FakeContainer()
        backend = LocalDockerBackend()
        backend.submit = mock.Mock(
            return_value=RunHandle(backend_kind="local-docker", ref="run-1", run_dir="")
        )
        backend._client = mock.Mock(return_value=FakeClient(container))
        spec = RunSpec(
            suite_name="s",
            container_module="m",
            instance_id="i",
            launch_spec=LaunchSpec(image="img", workdir="/w"),
            max_turns=1,
            provider="fake",
            api_kind="openai-chat",
            run_name="run-1",
        )

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalDirStore(Path(tmp)).bind(Path(tmp) / "run")
            fake_docker = SimpleNamespace(errors=SimpleNamespace(NotFound=FakeNotFound))
            with mock.patch.object(docker_local, "docker", fake_docker):
                with self.assertRaises(FakeWaitTimeout):
                    backend.run(spec, store=store, binding=ContainerBinding())
        self.assertTrue(container.removed)


class _FakeRemoteContainer:
    """Stand-in for a docker-py container: tar in (put), local FS, tar out (get)."""

    def __init__(self, root: Path) -> None:
        self.root = root  # the container's filesystem rooted here

    def put_archive(self, dest: str, tar_bytes: bytes) -> bool:
        import tarfile
        from io import BytesIO

        with tarfile.open(fileobj=BytesIO(tar_bytes), mode="r") as tar:
            tar.extractall(self.root / dest.lstrip("/"))
        return True

    def get_archive(self, path: str):
        import io
        import tarfile

        src = self.root / path.lstrip("/")
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            # docker names members relative to the requested path's parent, e.g.
            # get_archive(".../out") -> "out/<file>".
            tar.add(src, arcname=src.name)
        return [buffer.getvalue()], {}


class RemoteDockerHostPullTest(unittest.TestCase):
    """Host-pull copy logic without a daemon: push instance in, pull out/ back."""

    def test_push_inputs_then_pull_outputs(self) -> None:
        from simple_agent_lab.evals.backends._archive import (
            pack_file_to_root,
            unpack_members,
        )
        from simple_agent_lab.evals.backends.remote_docker import (
            RUN_MOUNT,
            pull_outputs,
            push_inputs,
        )

        # pure-archive round trip
        tar = pack_file_to_root("/agent/run/input/instance.json", b'{"x":1}')
        self.assertEqual(
            unpack_members(tar)["agent/run/input/instance.json"], b'{"x":1}'
        )

        with tempfile.TemporaryDirectory() as tmp:
            host_root = Path(tmp) / "host"
            container_fs = Path(tmp) / "worker"
            container_fs.mkdir()
            store = LocalDirStore(host_root).bind(host_root / "run" / "inst")
            store.put(INSTANCE_KEY, b'{"instance_id":"inst"}')

            container = _FakeRemoteContainer(container_fs)
            # 1. host pushes the instance into the worker container
            push_inputs(container, store)
            staged = container_fs / RUN_MOUNT.lstrip("/") / INSTANCE_KEY
            self.assertTrue(staged.exists())

            # 2. worker writes its outputs locally (simulated)
            out = container_fs / RUN_MOUNT.lstrip("/") / "out"
            out.mkdir(parents=True)
            (out / "result.json").write_text('{"model_patch":"diff"}')
            (out / "trajectory.jsonl").write_text('{"meta":{"in_progress":false}}\n')

            # 3. host pulls out/ back into its store (host-initiated, no reverse conn)
            pulled = pull_outputs(container, store)
            self.assertEqual(set(pulled), {RESULT_KEY, TRACE_KEY})
            self.assertEqual(json.loads(store.get(RESULT_KEY))["model_patch"], "diff")


class SwebenchSuiteDriverTest(unittest.TestCase):
    def test_pro_instance_plan_is_data(self) -> None:
        from evals.swebench.suite import SwebenchSuite

        suite = SwebenchSuite(dataset_name="SWE-bench_Pro")
        self.assertIsInstance(suite, Suite)
        self.assertEqual(suite.name, "swebench_pro")
        self.assertEqual(suite.container_module, SWEBENCH_CONTAINER)
        instance = {
            "instance_id": "instance_acme__widget-abc123",
            "repo": "acme/widget",
            "dockerhub_tag": "acme.widget-abc123",
        }
        launch_spec = suite.launch_spec(instance)
        self.assertEqual(launch_spec.workdir, "/app")
        self.assertEqual(launch_spec.shell, ("/bin/sh", "-lc"))
        self.assertEqual(launch_spec.entrypoint, "")
        self.assertIn("sweap-images", launch_spec.image)
        # Pro images carry no test-spec caps (Verified ones come from the spec,
        # which needs the swebench harness installed — not asserted here).
        self.assertEqual(launch_spec.cap_add, ())

    def test_multilingual_suite_name_is_distinct(self) -> None:
        from evals.swebench.suite import SwebenchSuite

        suite = SwebenchSuite(dataset_name="SWE-bench/SWE-bench_Multilingual")
        self.assertEqual(suite.name, "swebench_multilingual")


class SafePartTest(unittest.TestCase):
    def test_distinct_ids_never_collide_after_sanitization(self) -> None:
        from simple_agent_lab.evals.runner import _safe_part

        # Plain ids (alnum / _.-) are unchanged — SWE-bench ids stay readable.
        self.assertEqual(_safe_part("sympy__sympy-23824"), "sympy__sympy-23824")
        # Ids differing only in replaced chars must not map to the same dir/name.
        self.assertNotEqual(_safe_part("org/repo#1"), _safe_part("org_repo_1"))
        self.assertNotEqual(_safe_part("a:b"), _safe_part("a_b"))
        # Deterministic: same raw id → same safe part.
        self.assertEqual(_safe_part("a:b"), _safe_part("a:b"))


class CreateKwargsTest(unittest.TestCase):
    def test_shared_create_kwargs_carries_plan_fields(self) -> None:
        from simple_agent_lab.evals.backends.docker_local import _create_kwargs

        spec = RunSpec(
            suite_name="s",
            container_module="m",
            instance_id="i",
            launch_spec=LaunchSpec(
                image="img", workdir="/w", cap_add=("SYS_PTRACE",), entrypoint=""
            ),
            max_turns=3,
            provider="fake",
            api_kind="openai-chat",
            run_name="run-1",
        )
        from simple_agent_lab.evals import ContainerBinding

        binding = ContainerBinding(
            mounts={"/host": {"bind": "/agent/run", "mode": "rw"}},
            env={"SAL_STORE": "localdir"},
        )
        kwargs = _create_kwargs(spec, binding, user="root", environment={"X": "1"})
        self.assertEqual(kwargs["image"], "img")
        self.assertEqual(kwargs["name"], "run-1")
        self.assertEqual(
            kwargs["cap_add"], ["SYS_PTRACE"]
        )  # launch_spec cap_add plumbed
        self.assertEqual(
            kwargs["volumes"], {"/host": {"bind": "/agent/run", "mode": "rw"}}
        )
        self.assertEqual(kwargs["environment"], {"X": "1"})
        self.assertEqual(
            kwargs["entrypoint"], ""
        )  # "" included (clears image ENTRYPOINT)
        self.assertNotIn("platform", kwargs)  # omitted when unset


class RequestExtraFromEnvTest(unittest.TestCase):
    """request_extra now carries only session headers; reasoning moved to the
    provider so adapters map it per-model (no API-kind branching here)."""

    def test_no_headers_yields_empty(self) -> None:
        from simple_agent_lab.evals.in_container import request_extra_from_env

        self.assertEqual(request_extra_from_env(env={}), {})

    def test_session_headers_only(self) -> None:
        from simple_agent_lab.evals.in_container import request_extra_from_env

        extra = request_extra_from_env(
            env={"OPENAI_SESSION_ID": "s", "OPENAI_LOG_ID": "l"}
        )
        self.assertEqual(
            extra,
            {"extra_headers": {"extra": '{"session_id":"s"}', "X-TT-logid": "l"}},
        )

    def test_reasoning_no_longer_in_request_extra(self) -> None:
        from simple_agent_lab.evals.in_container import request_extra_from_env

        # Effort is now a provider field, not a request_extra key.
        self.assertEqual(request_extra_from_env(env={"REASONING_EFFORT": "high"}), {})


class ProviderReasoningFromEnvTest(unittest.TestCase):
    """OPENAI provider picks up a normalized, validated reasoning effort that is
    independent of the API kind — the adapter maps it to the wire shape."""

    _BASE = {"OPENAI_MODEL": "gpt-x", "OPENAI_AUTH_TOKEN": "tok"}

    def _provider(self, **extra: str):
        from simple_agent_lab.evals.in_container import provider_from_env

        return provider_from_env(kind="openai", env={**self._BASE, **extra})

    def test_default_reasoning_from_neutral_env(self) -> None:
        self.assertEqual(
            self._provider(REASONING_EFFORT="high").default_reasoning, "high"
        )

    def test_legacy_openai_env_still_honored(self) -> None:
        self.assertEqual(
            self._provider(OPENAI_REASONING_EFFORT="low").default_reasoning, "low"
        )

    def test_neutral_env_wins_over_legacy(self) -> None:
        prov = self._provider(REASONING_EFFORT="high", OPENAI_REASONING_EFFORT="low")
        self.assertEqual(prov.default_reasoning, "high")

    def test_unset_is_none(self) -> None:
        self.assertIsNone(self._provider().default_reasoning)

    def test_invalid_effort_raises(self) -> None:
        with self.assertRaises(SystemExit):
            self._provider(REASONING_EFFORT="ultra")

    def test_effort_is_api_kind_independent(self) -> None:
        from simple_agent_lab.evals.in_container import provider_from_env

        for api_kind in ("openai-chat", "openai-responses"):
            prov = provider_from_env(
                kind="openai",
                api_kind=api_kind,
                env={**self._BASE, "REASONING_EFFORT": "medium"},
            )
            self.assertEqual(prov.default_reasoning, "medium")


class RunInContainerMemoryWiringTest(unittest.TestCase):
    """memory.finish runs at SESSION_END with the suite's artifact_builder."""

    def test_memory_finish_persists_suite_artifact_via_session_end(self) -> None:
        import types

        from simple_agent_lab.evals.in_container import run_in_container
        from simple_agent_lab.evals.protocols import (
            AgentSpec,
            MEMORY_NAME_ENV,
            MEMORY_RUN_ID_ENV,
        )
        from simple_agent_lab.llm import Provider
        from simple_agent_lab.memory import FilesystemArtifact

        observed: dict[str, Any] = {}
        patch_text = "diff --git a/x b/x\n+stub change\n"

        module = types.ModuleType("sal_test_stub_container_mem")

        def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
            del instance, workdir
            return "do the stub task"

        def agent_spec() -> AgentSpec:
            return AgentSpec(name="stub_agent", flavor="bash")

        def extract_result(
            workspace: Any,
            instance: Mapping[str, Any],
            *,
            context: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            del workspace, instance, context
            return {"model_patch": patch_text}

        def memory_artifacts(
            workspace: Any,
            instance: Mapping[str, Any],
            *,
            context: Mapping[str, Any] | None = None,
        ) -> list[FilesystemArtifact]:
            del context
            observed["artifact_workspace"] = workspace
            observed["artifact_instance_id"] = dict(instance).get("instance_id")
            return [
                FilesystemArtifact(
                    name="model_patch.diff",
                    content=patch_text,
                    description="Final unified git diff (model_patch).",
                )
            ]

        module.build_task = build_task  # type: ignore[attr-defined]
        module.agent_spec = agent_spec  # type: ignore[attr-defined]
        module.extract_result = extract_result  # type: ignore[attr-defined]
        module.memory_artifacts = memory_artifacts  # type: ignore[attr-defined]
        sys.modules[module.__name__] = module
        self.addCleanup(lambda: sys.modules.pop(module.__name__, None))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mem_home = tmp_path / "memory"
            workdir = tmp_path / "work"
            workdir.mkdir()
            store = LocalDirStore(tmp_path / "run")
            env = {
                MEMORY_HOME_ENV: str(mem_home),
                MEMORY_NAME_ENV: "stub-namespace",
                MEMORY_RUN_ID_ENV: "run-1",
            }
            with mock.patch.dict("os.environ", env, clear=False):
                result, _state = run_in_container(
                    instance={"instance_id": "stub-1"},
                    container_module=module.__name__,
                    provider=Provider(id="fake", api="fake", model="fake-model"),
                    workdir=workdir,
                    max_turns=2,
                    store=store,
                    trace_id="stub.stub-1",
                    producer="suite:stub",
                    suite_name="stub",
                )

            mem_run_dir = mem_home / "stub-namespace" / "runs" / "run-1"
            patch_artifact = (mem_run_dir / "artifacts" / "model_patch.diff").read_text(
                encoding="utf-8"
            )
            manifest = (mem_run_dir / "artifacts.md").read_text(encoding="utf-8")
            index_exists = (mem_home / "stub-namespace" / "INDEX.md").exists()

        self.assertEqual(result["model_patch"], patch_text)
        self.assertEqual(observed["artifact_workspace"], workdir)
        self.assertEqual(observed["artifact_instance_id"], "stub-1")
        self.assertIn("+stub change", patch_artifact)
        self.assertIn("model_patch.diff", manifest)
        self.assertTrue(index_exists)

    def test_no_memory_home_leaves_run_unchanged(self) -> None:
        import types

        from simple_agent_lab.evals.in_container import run_in_container
        from simple_agent_lab.evals.protocols import AgentSpec
        from simple_agent_lab.llm import Provider

        observed: dict[str, Any] = {}
        module = types.ModuleType("sal_test_stub_container_nomem")

        def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
            del instance, workdir
            return "do the stub task"

        def agent_spec() -> AgentSpec:
            return AgentSpec(name="stub_agent", flavor="bash")

        def extract_result(
            workspace: Any,
            instance: Mapping[str, Any],
            *,
            context: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            del workspace, instance, context
            return {"model_patch": ""}

        def memory_artifacts(
            workspace: Any,
            instance: Mapping[str, Any],
            *,
            context: Mapping[str, Any] | None = None,
        ) -> list[Any]:
            del workspace, instance, context
            observed["collector_ran"] = True
            return []

        module.build_task = build_task  # type: ignore[attr-defined]
        module.agent_spec = agent_spec  # type: ignore[attr-defined]
        module.extract_result = extract_result  # type: ignore[attr-defined]
        module.memory_artifacts = memory_artifacts  # type: ignore[attr-defined]
        sys.modules[module.__name__] = module
        self.addCleanup(lambda: sys.modules.pop(module.__name__, None))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workdir = tmp_path / "work"
            workdir.mkdir()
            mem_home = tmp_path / "memory"
            store = LocalDirStore(tmp_path / "run")
            with mock.patch.dict("os.environ", {MEMORY_HOME_ENV: ""}, clear=False):
                result, _state = run_in_container(
                    instance={"instance_id": "stub-2"},
                    container_module=module.__name__,
                    provider=Provider(id="fake", api="fake", model="fake-model"),
                    workdir=workdir,
                    max_turns=2,
                    store=store,
                    trace_id="stub.stub-2",
                    producer="suite:stub",
                    suite_name="stub",
                )

            self.assertFalse(mem_home.exists())

        self.assertEqual(result["model_patch"], "")
        self.assertNotIn("collector_ran", observed)


if __name__ == "__main__":
    unittest.main()
