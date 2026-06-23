import unittest
from pathlib import Path

from simple_agent_lab.evals.suites.swebench import evolving


class EvolvingContainerTest(unittest.TestCase):
    def test_reexports_base_hooks(self):
        self.assertTrue(callable(evolving.build_task))
        self.assertTrue(callable(evolving.extract_result))
        self.assertTrue(callable(evolving.prepare))

    def test_build_agent_falls_back_when_no_package(self):
        old_builder = evolving._staged_agent_builder
        old_base = evolving._base_build_agent
        try:
            evolving._staged_agent_builder = lambda: None

            class FakeProvider:
                pass

            captured = {}

            def fake_base_build(*, spec, provider, cwd, request_extra=None, hooks=None):
                captured["used_base"] = True
                return "BASE_AGENT"

            evolving._base_build_agent = fake_base_build
            agent = evolving.build_agent(provider=FakeProvider(), cwd=Path("."))

            self.assertEqual(agent, "BASE_AGENT")
            self.assertTrue(captured["used_base"])
        finally:
            evolving._staged_agent_builder = old_builder
            evolving._base_build_agent = old_base

    def test_build_agent_uses_staged_package(self):
        old_builder = evolving._staged_agent_builder
        try:

            def fake_builder(*, provider, cwd, base_system_prompt):
                return ("PKG_AGENT", base_system_prompt)

            evolving._staged_agent_builder = lambda: fake_builder

            class FakeProvider:
                pass

            agent = evolving.build_agent(provider=FakeProvider(), cwd=Path("."))

            self.assertEqual(agent[0], "PKG_AGENT")
        finally:
            evolving._staged_agent_builder = old_builder


class EvolvingContainerMetadataTest(unittest.TestCase):
    def test_extract_result_includes_agent_package_status(self):
        old_extract = getattr(evolving, "_base_extract_result", None)
        old_status = dict(getattr(evolving, "_PACKAGE_STATUS", {}))
        try:
            evolving._PACKAGE_STATUS.clear()
            evolving._PACKAGE_STATUS.update(
                {"loaded": False, "used_fallback": True, "error": "invalid package"}
            )
            evolving._base_extract_result = lambda *args, **kwargs: {
                "model_patch": "diff --git a/x.py b/x.py\n"
            }

            result = evolving.extract_result(Path("."), {}, context={})

            self.assertTrue(result["agent_package"]["used_fallback"])
            self.assertFalse(result["agent_package"]["loaded"])
            self.assertEqual(result["agent_package"]["error"], "invalid package")
        finally:
            if old_extract is not None:
                evolving._base_extract_result = old_extract
            evolving._PACKAGE_STATUS.clear()
            evolving._PACKAGE_STATUS.update(old_status)

    def test_build_agent_records_builder_exception_as_fallback(self):
        old_builder = evolving._staged_agent_builder
        old_base = evolving._base_build_agent
        old_status = dict(evolving._PACKAGE_STATUS)
        try:

            def broken_builder(*, provider, cwd, base_system_prompt):
                raise RuntimeError("agent package boom")

            evolving._staged_agent_builder = lambda: broken_builder
            evolving._base_build_agent = lambda **kwargs: "BASE_AGENT"
            evolving._PACKAGE_STATUS.clear()

            class FakeProvider:
                pass

            agent = evolving.build_agent(provider=FakeProvider(), cwd=Path("."))

            self.assertEqual(agent, "BASE_AGENT")
            self.assertTrue(evolving._PACKAGE_STATUS["loaded"])
            self.assertTrue(evolving._PACKAGE_STATUS["used_fallback"])
            self.assertIn("agent package boom", evolving._PACKAGE_STATUS["error"])
        finally:
            evolving._staged_agent_builder = old_builder
            evolving._base_build_agent = old_base
            evolving._PACKAGE_STATUS.clear()
            evolving._PACKAGE_STATUS.update(old_status)
