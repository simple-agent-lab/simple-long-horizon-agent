"""Unit smoke for the generic eval framework.

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
import types
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from unittest import mock

from simple_agent_lab.evals import (
    AgentSpec,
    ContainerBinding,
    DEFAULT_MEMORY_CONTAINER_HOME,
    EVAL_KEY,
    INSTANCE_KEY,
    MEMORY_HOME_ENV,
    MEMORY_NAME_ENV,
    MEMORY_RUN_ID_ENV,
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
    reconcile_dataset,
    run_dataset,
    run_suite_instance,
    submit_dataset,
)
from simple_agent_lab.evals.bootstrap import bootstrap_script
from simple_agent_lab.llm import Provider
from simple_agent_lab.memory import FilesystemArtifact

# Internal helpers live in their own modules, not the top-level facade.
from simple_agent_lab.evals.runner import build_command, container_name
from simple_agent_lab.evals.stores import HttpArtifactClient

SWEBENCH_CONTAINER = "simple_agent_lab.evals.suites.swebench.container"


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


def _run_spec(**overrides: Any) -> RunSpec:
    values: dict[str, Any] = {
        "suite_name": "s",
        "container_module": "m",
        "instance_id": "i",
        "launch_spec": LaunchSpec(image="img", workdir="/w"),
        "max_turns": 1,
        "provider": "fake",
        "api_kind": "openai-chat",
        "run_name": "run-1",
    }
    values.update(overrides)
    return RunSpec(**values)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


def _git_repo(root: Path, contents: str = "x = 1\n") -> Path:
    repo = root / "testbed"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "T")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "app.py").write_text(contents, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "base")
    return repo


class _TempDirTest(unittest.TestCase):
    root: Path

    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name).resolve()


def _memory_container_module(
    name: str,
    *,
    patch_text: str,
    observed: dict[str, Any],
    artifact: FilesystemArtifact | None,
) -> types.ModuleType:
    module = types.ModuleType(name)
    module.build_task = lambda instance, *, workdir: "do the stub task"  # type: ignore[attr-defined]
    module.agent_spec = lambda: AgentSpec(name="stub_agent", flavor="bash")  # type: ignore[attr-defined]
    module.extract_result = (  # type: ignore[attr-defined]
        lambda workspace, instance, *, context=None: {"model_patch": patch_text}
    )

    def memory_artifacts(
        workspace: Any,
        instance: Mapping[str, Any],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> list[FilesystemArtifact]:
        del context
        observed.update(
            collector_ran=True,
            artifact_workspace=workspace,
            artifact_instance_id=instance["instance_id"],
        )
        return [artifact] if artifact else []

    module.memory_artifacts = memory_artifacts  # type: ignore[attr-defined]
    return module


class OrchestrationTest(_TempDirTest):
    def test_demo_suite_satisfies_protocol(self) -> None:
        self.assertIsInstance(_DemoSuite(), Suite)

    def test_run_suite_instance_fake_backend(self) -> None:
        instance = {"instance_id": "demo-1", "problem": "p", "gold": "SECRET"}
        backend = FakeBackend(on_run=_simulate("42"), log_text="ok\n")
        artifacts = run_suite_instance(
            suite=_DemoSuite(),
            instance=instance,
            backend=backend,
            store=LocalDirStore(self.root),
            run_root=self.root,
            run_id="run-x",
            provider="fake",
        )
        self.assertEqual(
            (backend.runs[0].suite_name, backend.runs[0].instance_id),
            ("demo", "demo-1"),
        )
        self.assertEqual(artifacts.logs, "ok\n")
        self.assertNotIn("gold", _read_json(artifacts.run_dir / INSTANCE_KEY))
        self.assertFalse((artifacts.run_dir / "out" / "prediction.jsonl").exists())
        self.assertEqual(_read_json(artifacts.run_dir / RESULT_KEY)["answer"], "42")

    def test_build_command_targets_the_generic_runner(self) -> None:
        spec = _run_spec(
            max_turns=5,
            wheelhouse_mount="/wh",
            run_name="n",
        )
        cmd = build_command(spec)
        self.assertEqual(cmd[:2], ("bash", "-lc"))
        self.assertIn("simple_agent_lab.evals.in_container", cmd[-1])
        self.assertIn("--find-links /wh", cmd[-1])

    def test_build_command_installs_mcp_extra_when_requested(self) -> None:
        spec = _run_spec(
            max_turns=5,
            wheelhouse_mount="/wh",
            run_name="n",
            package_extras=("mcp",),
        )
        cmd = build_command(spec)

        self.assertIn("simple-agent-lab[mcp]", cmd[-1])


class BootstrapScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.offline_script = bootstrap_script(
            runner_argv=("-m", "runner"),
            install=False,
            wheelhouse_mount="/agent/wheelhouse",
        )

    def test_wheelhouse_bootstrap_selects_python_for_container_libc(self) -> None:
        script = self.offline_script
        self.assertIn('_PYTHON_LIBC="linux-x86_64-musl"', script)
        self.assertIn('_PYTHON_LIBC="linux-x86_64-gnu"', script)
        self.assertIn(
            '"/agent/wheelhouse/uv-python"/cpython-3.11.*-"$_PYTHON_LIBC"'
            "/bin/python3.11",
            script,
        )
        self.assertIn('"$OFFLINE_PYTHON" -m venv /opt/agent-venv', script)
        self.assertLess(
            script.index('if [ -x "$OFFLINE_PYTHON" ]; then'),
            script.index('elif [ -n "$UV_BIN" ]; then'),
        )

    def test_wheelhouse_bootstrap_requires_cpython_311(self) -> None:
        self.assertIn("wheelhouse installs require CPython 3.11", self.offline_script)
        self.assertIn("sys.version_info[:2] == (3, 11)", self.offline_script)

    def test_online_bootstrap_keeps_python310_fallback(self) -> None:
        script = bootstrap_script(
            runner_argv=("-m", "runner"),
            install=False,
            wheelhouse_mount=None,
        )

        self.assertIn('"$UV_BIN" venv --python 3.11', script)
        self.assertIn('|| "$UV_BIN" venv --python python3', script)
        self.assertIn("sys.version_info >= (3, 10)", script)
        self.assertIn("WHEELHOUSE_PYTHON_REQUIRED=0", script)


class HostHttpStoreTest(_TempDirTest):
    def test_http_client_round_trip(self) -> None:
        with HostHttpStore(self.root, container_host="127.0.0.1") as store:
            bound = store.bind(self.root / "run-1" / "inst-1")
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

        with HostHttpStore(self.root, container_host="127.0.0.1") as store:
            url = store.bind(self.root).container_binding().env["SAL_STORE_URL"]
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


class LocalProcessBackendTest(_TempDirTest):
    """Unified entry point runs a real agent in-process — no Docker, no network."""

    def test_run_suite_instance_in_process(self) -> None:
        repo = _git_repo(self.root)
        runs = self.root / "runs"
        store = LocalDirStore(runs)
        artifacts = run_suite_instance(
            suite=_SwebenchLikeSuite(),
            instance={
                "instance_id": "demo__repo-1",
                "problem_statement": "Make it better.",
                "language": "python",
            },
            backend=LocalProcessBackend(workspace=repo),
            store=store,
            run_root=runs,
            run_id="r",
            provider="fake",
            max_turns=3,
        )

        self.assertEqual(artifacts.status_code, 0)
        bound = store.bind(artifacts.run_dir)
        self.assertIn("model_patch", json.loads(bound.get(RESULT_KEY)))
        header = json.loads(bound.get(TRACE_KEY).splitlines()[0])
        self.assertEqual(header["meta"]["suite"], "swebench")
        self.assertFalse(header["meta"]["in_progress"])

    def test_oracle_run_reproduces_gold_patch(self) -> None:
        """Oracle mode applies the gold patch (no model) and extract reproduces it.

        This is the suite self-check the integration doc prescribes: a tiny
        self-contained instance + its gold `patch`, driven in-process with
        `provider="oracle"`, must yield a `model_patch` equal to the gold change.
        No Docker, no network, no LLM — just the real container half end to end.
        """

        repo = _git_repo(self.root, "def f():\n    return 1\n")
        (repo / "app.py").write_text("def f():\n    return 2\n", encoding="utf-8")
        gold_patch = _git(repo, "diff")
        _git(repo, "checkout", "--", "app.py")
        self.assertIn("return 2", gold_patch)

        runs = self.root / "runs"
        store = LocalDirStore(runs)
        artifacts = run_suite_instance(
            suite=_SwebenchLikeSuite(),
            instance={
                "instance_id": "demo__repo-oracle",
                "problem_statement": "Make f return 2.",
                "language": "python",
                "patch": gold_patch,
            },
            backend=LocalProcessBackend(workspace=repo),
            store=store,
            run_root=runs,
            run_id="oracle",
            provider="oracle",
        )

        self.assertEqual(artifacts.status_code, 0)
        bound = store.bind(artifacts.run_dir)
        self.assertIn("return 2", json.loads(bound.get(RESULT_KEY))["model_patch"])
        self.assertIn("patch", _read_json(artifacts.run_dir / INSTANCE_KEY))
        self.assertTrue(json.loads(bound.get(TRACE_KEY))["meta"]["oracle"])

    def test_oracle_without_apply_hook_fails_clearly(self) -> None:
        """A suite whose container half has no apply_oracle errors, not no-ops."""

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

        artifacts = run_suite_instance(
            suite=_NoOracleSuite(),
            instance={"instance_id": "demo-1", "problem": "p"},
            backend=LocalProcessBackend(),
            store=LocalDirStore(self.root),
            run_root=self.root,
            run_id="oracle",
            provider="oracle",
        )
        self.assertEqual(artifacts.status_code, 1)
        self.assertIn("apply_oracle", artifacts.logs)

    def test_workspace_factory_isolates_concurrent_runs(self) -> None:
        """A workspace factory gives each run its own dir, so concurrency is safe."""
        ws_base = self.root / "ws"
        runs = self.root / "runs"
        instances = [
            {
                "instance_id": f"i-{n}",
                "problem_statement": "p",
                "language": "python",
            }
            for n in range(5)
        ]
        backend = LocalProcessBackend(workspace=lambda spec: ws_base / spec.instance_id)
        report = run_dataset(
            suite=_SwebenchLikeSuite(),
            instances=instances,
            backend=backend,
            store=LocalDirStore(runs),
            run_root=runs,
            run_id="batch",
            concurrency=4,
            provider="fake",
            max_turns=2,
        )
        self.assertEqual(report.summary(), {"total": 5, "ok": 5, "failed": 0})
        self.assertTrue(all((ws_base / f"i-{n}").is_dir() for n in range(5)))


class InEnvScoringTest(_TempDirTest):
    """In-environment scoring: the container-half `evaluate` hook writes the
    verdict into result.json during the run, gated on staged `eval_inputs`
    without a separate scoring driver."""

    @staticmethod
    def _reuse_module() -> str:
        """Register a throwaway container module whose `evaluate` grades gold."""

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
        mod_name = self._reuse_module()
        self.addCleanup(lambda: sys.modules.pop(mod_name, None))

        class _ReuseSuite(_DemoSuite):
            container_module = mod_name

            def eval_inputs(self, instance):  # type: ignore[no-untyped-def]
                return {"expected": "solved"}

        artifacts = run_suite_instance(
            suite=_ReuseSuite(),
            instance={"instance_id": "r-0", "gold": "solved"},
            backend=LocalProcessBackend(),
            store=LocalDirStore(self.root),
            run_root=self.root,
            run_id="reuse",
            provider="fake",
            max_turns=1,
        )
        self.assertEqual(_read_json(artifacts.run_dir / EVAL_KEY)["expected"], "solved")
        self.assertNotIn("gold", _read_json(artifacts.run_dir / INSTANCE_KEY))
        result = _read_json(artifacts.run_dir / RESULT_KEY)
        self.assertEqual((result["resolved"], result["answer"]), (True, "solved"))

    def test_hook_is_skipped_without_staged_eval_inputs(self) -> None:
        mod_name = self._reuse_module()
        self.addCleanup(lambda: sys.modules.pop(mod_name, None))

        class _NoGoldSuite(_DemoSuite):
            container_module = mod_name
            # eval_inputs inherited from _DemoSuite returns None → no staged gold.

        artifacts = run_suite_instance(
            suite=_NoGoldSuite(),
            instance={"instance_id": "r-1"},
            backend=LocalProcessBackend(),
            store=LocalDirStore(self.root),
            run_root=self.root,
            run_id="nogold",
            provider="fake",
            max_turns=1,
        )
        result = _read_json(artifacts.run_dir / RESULT_KEY)
        self.assertNotIn("resolved", result)
        self.assertEqual(result["answer"], "solved")


class RunDatasetTest(_TempDirTest):
    """The minimal controller: run many instances over a pool, aggregate outcomes."""

    def _run(
        self,
        instances: list[dict[str, Any]],
        backend: FakeBackend,
        **kwargs: Any,
    ):
        return run_dataset(
            suite=_DemoSuite(),
            instances=instances,
            backend=backend,
            store=LocalDirStore(self.root),
            run_root=self.root,
            run_id="batch",
            provider="fake",
            **kwargs,
        )

    def test_concurrent_run_with_ordering_and_callback(self) -> None:
        instances = [{"instance_id": f"i-{n}", "n": n} for n in range(6)]
        seen: list[str] = []
        backend = FakeBackend(on_run=_simulate("ok"))
        report = self._run(
            instances,
            backend,
            concurrency=4,
            on_result=lambda r: seen.append(r.instance_id),
        )
        self.assertEqual(report.summary(), {"total": 6, "ok": 6, "failed": 0})
        self.assertEqual(
            [r.instance_id for r in report.results], [f"i-{n}" for n in range(6)]
        )
        self.assertEqual(len(seen), 6)
        self.assertEqual(len(backend.runs), 6)

    def test_error_is_captured_and_retried(self) -> None:
        attempts: dict[str, int] = {}

        def flaky(spec, store) -> None:
            n = attempts.get(spec.instance_id, 0) + 1
            attempts[spec.instance_id] = n
            if spec.instance_id == "bad":
                raise RuntimeError("boom")
            _simulate("ok")(spec, store)

        report = self._run(
            [{"instance_id": "good"}, {"instance_id": "bad"}],
            FakeBackend(on_run=flaky),
            concurrency=2,
            max_attempts=3,
        )
        self.assertEqual(report.summary(), {"total": 2, "ok": 1, "failed": 1})
        bad = next(r for r in report.results if r.instance_id == "bad")
        self.assertFalse(bad.ok)
        self.assertIn("boom", bad.error)
        self.assertEqual((bad.attempts, attempts["bad"]), (3, 3))

    def test_per_instance_kwargs_override_shared_run_kwargs(self) -> None:
        instances = [
            {"instance_id": "i-0", "token": "first"},
            {"instance_id": "i-1", "token": "second"},
        ]
        backend = FakeBackend(on_run=_simulate("ok"))
        report = self._run(
            instances,
            backend,
            provider_env={"token": "shared"},
            per_instance_kwargs=lambda instance: {
                "provider_env": {"token": str(instance["token"])},
                "name": f"custom.{instance['instance_id']}",
            },
        )

        self.assertEqual(report.summary(), {"total": 2, "ok": 2, "failed": 0})
        specs = {spec.instance_id: spec for spec in backend.runs}
        self.assertEqual(
            [(specs[key].provider_env, specs[key].run_name) for key in specs],
            [
                ({"token": "first"}, "custom.i-0"),
                ({"token": "second"}, "custom.i-1"),
            ],
        )


class SubmitReconcileTest(_TempDirTest):
    """Host-reentrant batch: submit, drop all memory, reconcile from disk only."""

    def _submit(
        self,
        *,
        instances: list[Mapping[str, Any]] | None = None,
        backend: Any = None,
        suite: Any = None,
        run_id: str = "b",
        **options: Any,
    ) -> Any:
        backend = FakeBackend() if backend is None else backend
        submit_dataset(
            suite=_DemoSuite() if suite is None else suite,
            instances=instances or [{"instance_id": "x"}],
            backend=backend,
            store=LocalDirStore(self.root),
            run_root=self.root,
            run_id=run_id,
            provider=options.pop("provider", "fake"),
            **options,
        )
        return backend

    def _reconcile(self, backend: Any = None, **options: Any):
        return reconcile_dataset(
            suite=_DemoSuite(),
            backend=FakeBackend() if backend is None else backend,
            store=LocalDirStore(self.root),
            run_root=self.root,
            run_id=options.pop("run_id", "b"),
            poll_interval_s=options.pop("poll_interval_s", 0),
            **options,
        )

    def test_submit_preparation_matches_blocking_oracle_run(self) -> None:
        class _ParitySuite(_DemoSuite):
            def eval_inputs(
                self, instance: Mapping[str, Any]
            ) -> Mapping[str, Any] | None:
                return {"expected": instance["gold"]}

        instance = {"instance_id": "oracle-1", "problem": "p", "gold": "SECRET"}
        options: dict[str, Any] = {
            "provider": "oracle",
            "api_kind": "custom-api",
            "max_turns": 12,
            "wall_time_seconds": 34.5,
            "provider_env": {"TOKEN": "value"},
            "runner_module": "custom.runner",
            "install": False,
            "package_extras": ("mcp",),
            "wheelhouse_mount": "/wheelhouse",
        }
        blocking = FakeBackend()
        detached = FakeBackend()
        run_suite_instance(
            suite=_ParitySuite(),
            instance=instance,
            backend=blocking,
            store=LocalDirStore(self.root),
            run_root=self.root,
            run_id="blocking",
            **options,
        )
        self._submit(
            suite=_ParitySuite(),
            instances=[instance],
            backend=detached,
            run_id="detached",
            **options,
        )
        for run_id in ("blocking", "detached"):
            run_dir = self.root / run_id / "oracle-1"
            self.assertEqual(_read_json(run_dir / INSTANCE_KEY), instance)
            self.assertEqual(_read_json(run_dir / EVAL_KEY), {"expected": "SECRET"})

        blocking_spec = asdict(blocking.runs[0])
        detached_spec = asdict(detached.runs[0])
        for spec in (blocking_spec, detached_spec):
            spec.pop("run_name")
        self.assertEqual(blocking_spec, detached_spec)

    def test_submit_then_reconcile_from_fresh_process(self) -> None:
        instances = [{"instance_id": f"i-{n}", "problem": "p"} for n in range(4)]
        self._submit(
            instances=instances,
            backend=FakeBackend(on_run=_simulate("42")),
            run_id="batch",
        )
        self.assertTrue((self.root / "batch" / "batch.json").exists())

        seen: list[str] = []
        report = self._reconcile(
            run_id="batch", on_result=lambda r: seen.append(r.instance_id)
        )
        self.assertEqual(report.summary(), {"total": 4, "ok": 4, "failed": 0})
        self.assertEqual(len(seen), 4)
        for result in report.results:
            assert result.artifacts is not None
            self.assertEqual(
                _read_json(result.artifacts.run_dir / RESULT_KEY)["answer"], "42"
            )

    def test_reconcile_waits_on_pending_runs(self) -> None:
        self._submit(
            instances=[{"instance_id": "slow"}],
            backend=FakeBackend(on_run=_simulate("ok")),
        )
        report = self._reconcile(FakeBackend(pending_polls=2))
        self.assertEqual(report.summary(), {"total": 1, "ok": 1, "failed": 0})

    def test_submit_requires_detaching_backend(self) -> None:
        with self.assertRaises(TypeError):
            self._submit(backend=LocalProcessBackend())

    def test_mid_submit_crash_leaves_recoverable_manifest(self) -> None:
        """A crash partway through submit still records every started container."""
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
        store = LocalDirStore(self.root)
        with self.assertRaises(RuntimeError):
            self._submit(instances=instances, backend=_CrashAtThird())
        manifest = json.loads(
            _batch_store(store, self.root, "b").get(BATCH_KEY).decode()
        )
        self.assertEqual(len(manifest), 2)
        self.assertEqual(
            self._reconcile().summary(), {"total": 2, "ok": 2, "failed": 0}
        )

    def test_reconcile_uses_result_when_poll_never_reports_done(self) -> None:
        """poll() that never returns done still completes if result.json exists."""

        class _NeverDone(FakeBackend):
            def poll(self, handle):  # type: ignore[no-untyped-def]
                return None  # daemon never reports completion (e.g. already gone)

        self._submit(backend=FakeBackend(on_run=_simulate("ok")))
        report = self._reconcile(_NeverDone())
        self.assertEqual(report.summary(), {"total": 1, "ok": 1, "failed": 0})

    def test_reconcile_completes_off_result_without_instance_record(self) -> None:
        """Reconcile keys completion on result.json — decoupled from the instance.

        Reconcile no longer needs the instance record; a missing
        input/instance.json does not fail a run whose result.json landed. The
        instance re-enters only at the score phase.
        """
        self._submit(backend=FakeBackend(on_run=_simulate("ok")))
        (self.root / "b" / "x" / INSTANCE_KEY).unlink()
        self.assertEqual(
            self._reconcile().summary(), {"total": 1, "ok": 1, "failed": 0}
        )

    def test_reconcile_floors_the_idle_wait(self) -> None:
        """poll_interval_s=0 must not busy-spin: the loop sleeps a real floor."""
        slept: list[float] = []

        # Returns None a few times so the loop has to wait, then completes.
        class _SlowThenDone(FakeBackend):
            def __init__(self) -> None:
                super().__init__()
                self.n = 0

            def poll(self, handle):  # type: ignore[no-untyped-def]
                self.n += 1
                return None if self.n < 3 else RunOutcome(status_code=0)

        self._submit()
        self._reconcile(_SlowThenDone(), sleep_fn=slept.append)
        self.assertTrue(slept and all(s >= 0.5 for s in slept))


class ReviewFixesTest(_TempDirTest):
    """Docker-path guards (no real Docker): exit-code parsing, atomic put, names."""

    def test_exit_status_guards_null_and_nonint(self) -> None:
        from simple_agent_lab.evals.backends.docker_local import exit_status

        for state, expected in [
            ({"ExitCode": 0}, 0),
            ({"ExitCode": 137}, 137),
            ({"ExitCode": None}, 1),
            ({}, 1),
        ]:
            with self.subTest(state=state):
                self.assertEqual(exit_status(state), expected)

    def test_host_http_put_is_atomic(self) -> None:
        """put writes via a temp file + replace (no torn read), like LocalDirStore."""
        import os as _os

        seen_tmp: list[str] = []
        real_replace = _os.replace

        def spy_replace(src, dst):  # type: ignore[no-untyped-def]
            seen_tmp.append(str(src))
            return real_replace(src, dst)

        store = HostHttpStore(self.root, container_host="127.0.0.1")
        with mock.patch(
            "simple_agent_lab.evals.stores.host_http.os.replace", spy_replace
        ):
            with store:
                bound = store.bind(self.root / "r" / "i")
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

    def test_container_name_is_namespaced_for_both_run_modes(self) -> None:
        instance = {"instance_id": "astropy__astropy-13398"}
        run_id = "003af6184cce-9119f5eb01dc"
        for mode in ("blocking", "detached"):
            with self.subTest(mode=mode):
                names = []
                for owner in ("self_evolving", "hyperagents"):
                    root = self.root / mode / owner / "swebench_runs"
                    backend = FakeBackend()
                    common = {
                        "suite": _DemoSuite(),
                        "backend": backend,
                        "store": LocalDirStore(root),
                        "run_root": root,
                        "run_id": run_id,
                    }
                    if mode == "blocking":
                        run_suite_instance(instance=instance, **common)
                    else:
                        submit_dataset(instances=[instance], **common)
                    names.append(backend.runs[0].run_name)
                self.assertNotEqual(*names)
                self.assertTrue(
                    names[0].startswith("demo.") and names[0].endswith(f".{run_id}")
                )

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

        wheelhouse = self.root / "wheelhouse"
        wheelhouse.mkdir()
        uv = self.root / "uv"
        uv.write_bytes(b"#!/bin/sh\n")
        store_mount = {"/host/run": {"bind": "/agent/run", "mode": "rw"}}
        bound = with_local_mounts(
            ContainerBinding(mounts=dict(store_mount), env={"SAL_STORE": "localdir"}),
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

        memory_home = self.root / "memory"
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
        self.assertEqual(
            bound.env,
            {
                "SAL_STORE": "localdir",
                MEMORY_HOME_ENV: DEFAULT_MEMORY_CONTAINER_HOME,
            },
        )

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
        from simple_agent_lab.evals.backends import docker_local
        from simple_agent_lab.evals.backends.docker_local import LocalDockerBackend

        fake_client = object()
        fake_docker = SimpleNamespace(from_env=mock.Mock(return_value=fake_client))
        with (
            mock.patch.object(docker_local, "docker", fake_docker),
            mock.patch.object(docker_local, "ensure_docker_host_env") as detect_host,
        ):
            client = LocalDockerBackend(docker_timeout_s=180.0)._client()

        self.assertIs(client, fake_client)
        detect_host.assert_called_once_with()
        fake_docker.from_env.assert_called_once_with(timeout=180.0)

    def test_docker_host_probe_finds_docker_desktop_socket(self) -> None:
        from simple_agent_lab.evals.backends import docker_local

        fake_stat = mock.Mock(st_mode=1)
        home = Path("/tmp/sal-docker-home")
        with (
            mock.patch.dict("os.environ", {"HOME": str(home)}, clear=True),
            mock.patch.object(Path, "stat", return_value=fake_stat),
            mock.patch.object(docker_local.stat, "S_ISSOCK", return_value=True),
        ):
            docker_local.ensure_docker_host_env()
            resolved = docker_local.os.environ.get("DOCKER_HOST")

        self.assertEqual(resolved, f"unix://{home}/.docker/run/docker.sock")

    def test_force_existing_removes_named_container_before_start(self) -> None:
        from simple_agent_lab.evals.backends import docker_local
        from simple_agent_lab.evals.backends.docker_local import LocalDockerBackend

        existing = mock.Mock()
        containers = SimpleNamespace(get=mock.Mock(return_value=existing))
        client = SimpleNamespace(containers=containers)
        backend = LocalDockerBackend(force_existing=True)
        backend._client = mock.Mock(return_value=client)

        with mock.patch.object(docker_local, "start_container") as start:
            backend.submit(
                _run_spec(),
                store=mock.Mock(),
                binding=ContainerBinding(),
            )

        containers.get.assert_called_once_with("run-1")
        existing.remove.assert_called_once_with(force=True)
        start.assert_called_once()

    def test_remote_docker_backend_sets_client_timeout(self) -> None:
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
        from simple_agent_lab.evals.backends import docker_local
        from simple_agent_lab.evals.backends.docker_local import start_container

        class FakeImageNotFound(Exception):
            pass

        class FakeAPIError(Exception):
            pass

        class FakeStartTimeout(Exception):
            pass

        container = mock.Mock()
        container.start.side_effect = FakeStartTimeout("start timed out")
        client = SimpleNamespace(
            images=SimpleNamespace(get=mock.Mock(return_value=object())),
            containers=SimpleNamespace(create=mock.Mock(return_value=container)),
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
                    _run_spec(),
                    ContainerBinding(),
                    user="root",
                    pull="missing",
                    environment={},
                )
        container.remove.assert_called_once_with(force=True)

    def test_local_docker_run_removes_container_when_wait_times_out(self) -> None:
        """A blocking run timeout after submit() must force-remove the container."""
        from simple_agent_lab.evals.backends import docker_local
        from simple_agent_lab.evals.backends.docker_local import LocalDockerBackend
        from simple_agent_lab.evals.protocols import RunHandle

        class FakeNotFound(Exception):
            pass

        class FakeWaitTimeout(Exception):
            pass

        container = mock.Mock()
        container.wait.side_effect = FakeWaitTimeout("wait timed out")
        backend = LocalDockerBackend()
        backend.submit = mock.Mock(
            return_value=RunHandle(backend_kind="local-docker", ref="run-1", run_dir="")
        )
        backend._client = mock.Mock(
            return_value=SimpleNamespace(
                containers=SimpleNamespace(get=mock.Mock(return_value=container))
            )
        )
        store = LocalDirStore(self.root).bind(self.root / "run")
        fake_docker = SimpleNamespace(errors=SimpleNamespace(NotFound=FakeNotFound))
        with mock.patch.object(docker_local, "docker", fake_docker):
            with self.assertRaises(FakeWaitTimeout):
                backend.run(_run_spec(), store=store, binding=ContainerBinding())
        container.remove.assert_called_once_with(force=True)


class _FakeRemoteContainer:
    """Stand-in for a docker-py container: tar in (put), local FS, tar out (get)."""

    def __init__(self, root: Path) -> None:
        self.root = root  # the container's filesystem rooted here

    def put_archive(self, dest: str, tar_bytes: bytes) -> bool:
        import tarfile
        from io import BytesIO

        with tarfile.open(fileobj=BytesIO(tar_bytes), mode="r") as tar:
            target = self.root / dest.lstrip("/")
            if sys.version_info >= (3, 12):
                tar.extractall(target, filter="data")
            else:
                tar.extractall(target)
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


class RemoteDockerHostPullTest(_TempDirTest):
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

        host_root = self.root / "host"
        container_fs = self.root / "worker"
        container_fs.mkdir()
        store = LocalDirStore(host_root).bind(host_root / "run" / "inst")
        store.put(INSTANCE_KEY, b'{"instance_id":"inst"}')

        container = _FakeRemoteContainer(container_fs)
        push_inputs(container, store)
        self.assertTrue((container_fs / RUN_MOUNT.lstrip("/") / INSTANCE_KEY).exists())
        out = container_fs / RUN_MOUNT.lstrip("/") / "out"
        out.mkdir(parents=True)
        (out / "result.json").write_text('{"model_patch":"diff"}')
        (out / "trajectory.jsonl").write_text('{"meta":{"in_progress":false}}\n')

        self.assertEqual(set(pull_outputs(container, store)), {RESULT_KEY, TRACE_KEY})
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
        # Default per-container memory guardrail (docker --memory).
        self.assertEqual(launch_spec.mem_limit, "16g")

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

        spec = _run_spec(
            launch_spec=LaunchSpec(
                image="img", workdir="/w", cap_add=("SYS_PTRACE",), entrypoint=""
            ),
            max_turns=3,
        )
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

    def test_request_extra_contains_only_session_headers(self) -> None:
        from simple_agent_lab.evals.in_container import request_extra_from_env

        cases = [
            ({}, {}),
            (
                {"OPENAI_SESSION_ID": "s", "OPENAI_LOG_ID": "l"},
                {
                    "extra_headers": {
                        "extra": '{"session_id":"s"}',
                        "X-TT-logid": "l",
                    }
                },
            ),
            ({"REASONING_EFFORT": "high"}, {}),
        ]
        for env, expected in cases:
            with self.subTest(env=env):
                self.assertEqual(request_extra_from_env(env=env), expected)


class ProviderReasoningFromEnvTest(unittest.TestCase):
    """OPENAI provider picks up a normalized, validated reasoning effort that is
    independent of the API kind — the adapter maps it to the wire shape."""

    _BASE = {"OPENAI_MODEL": "gpt-x", "OPENAI_AUTH_TOKEN": "tok"}

    def _provider(self, **extra: str):
        from simple_agent_lab.evals.in_container import provider_from_env

        return provider_from_env(kind="openai", env={**self._BASE, **extra})

    def test_reasoning_effort_precedence_and_defaults(self) -> None:
        cases = [
            ({"REASONING_EFFORT": "high"}, "high"),
            ({"OPENAI_REASONING_EFFORT": "low"}, "low"),
            (
                {
                    "REASONING_EFFORT": "high",
                    "OPENAI_REASONING_EFFORT": "low",
                },
                "high",
            ),
            ({}, None),
        ]
        for env, expected in cases:
            with self.subTest(env=env):
                self.assertEqual(self._provider(**env).default_reasoning, expected)

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


class RunInContainerMemoryWiringTest(_TempDirTest):
    """memory.finish runs at SESSION_END with the suite's artifact_builder."""

    def _run(self, module: types.ModuleType, memory_home: str):
        from simple_agent_lab.evals.in_container import run_in_container

        sys.modules[module.__name__] = module
        self.addCleanup(lambda: sys.modules.pop(module.__name__, None))
        workdir = self.root / "work"
        workdir.mkdir(exist_ok=True)
        env = {
            MEMORY_HOME_ENV: memory_home,
            MEMORY_NAME_ENV: "stub-namespace",
            MEMORY_RUN_ID_ENV: "run-1",
        }
        with mock.patch.dict("os.environ", env):
            result, _state = run_in_container(
                instance={"instance_id": "stub-1"},
                container_module=module.__name__,
                provider=Provider(id="fake", api="fake", model="fake-model"),
                workdir=workdir,
                max_turns=2,
                store=LocalDirStore(self.root / "run"),
                trace_id="stub.stub-1",
                producer="suite:stub",
                suite_name="stub",
            )
        return result, workdir

    def test_memory_finish_persists_suite_artifact_via_session_end(self) -> None:
        observed: dict[str, Any] = {}
        patch_text = "diff --git a/x b/x\n+stub change\n"
        module = _memory_container_module(
            "sal_test_stub_container_mem",
            patch_text=patch_text,
            observed=observed,
            artifact=FilesystemArtifact(
                name="model_patch.diff",
                content=patch_text,
                description="Final unified git diff (model_patch).",
            ),
        )
        mem_home = self.root / "memory"
        result, workdir = self._run(module, str(mem_home))
        mem_run_dir = mem_home / "stub-namespace" / "runs" / "run-1"
        self.assertEqual(result["model_patch"], patch_text)
        self.assertEqual(observed["artifact_workspace"], workdir)
        self.assertEqual(observed["artifact_instance_id"], "stub-1")
        self.assertIn(
            "+stub change",
            (mem_run_dir / "artifacts" / "model_patch.diff").read_text(),
        )
        self.assertIn("model_patch.diff", (mem_run_dir / "artifacts.md").read_text())
        self.assertTrue((mem_home / "stub-namespace" / "INDEX.md").exists())

    def test_no_memory_home_leaves_run_unchanged(self) -> None:
        observed: dict[str, Any] = {}
        module = _memory_container_module(
            "sal_test_stub_container_nomem",
            patch_text="",
            observed=observed,
            artifact=None,
        )
        result, _workdir = self._run(module, "")
        self.assertFalse((self.root / "memory").exists())
        self.assertEqual(result["model_patch"], "")
        self.assertNotIn("collector_ran", observed)
