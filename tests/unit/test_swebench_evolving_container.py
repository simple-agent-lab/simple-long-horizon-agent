from pathlib import Path

from simple_agent_lab.evals.suites.swebench import evolving


def test_reexports_base_hooks():
    assert callable(evolving.build_task)
    assert callable(evolving.extract_result)
    assert callable(evolving.prepare)


def test_build_agent_falls_back_when_no_package(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(evolving, "_staged_agent_builder", lambda: None)

    class FakeProvider:
        pass

    captured = {}

    def fake_base_build(*, spec, provider, cwd, request_extra=None, hooks=None):
        captured["used_base"] = True
        return "BASE_AGENT"

    monkeypatch.setattr(evolving, "_base_build_agent", fake_base_build)
    agent = evolving.build_agent(provider=FakeProvider(), cwd=tmp_path)
    assert agent == "BASE_AGENT"
    assert captured["used_base"] is True


def test_build_agent_uses_staged_package(monkeypatch, tmp_path: Path):
    def fake_builder(*, provider, cwd, base_system_prompt):
        return ("PKG_AGENT", base_system_prompt)

    monkeypatch.setattr(evolving, "_staged_agent_builder", lambda: fake_builder)

    class FakeProvider:
        pass

    agent = evolving.build_agent(provider=FakeProvider(), cwd=tmp_path)
    assert agent[0] == "PKG_AGENT"
