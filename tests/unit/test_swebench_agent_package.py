from pathlib import Path

from simple_agent_lab.evals.suites.swebench import agent_package


def test_default_package_has_entry_and_builder():
    files = agent_package.default_agent_package()
    assert agent_package.ENTRY_MODULE_FILENAME in files
    assert "def build_agent" in files[agent_package.ENTRY_MODULE_FILENAME]


def test_load_agent_package_returns_callable(tmp_path: Path):
    builder = agent_package.load_agent_package(
        agent_package.default_agent_package(), root=tmp_path
    )
    assert callable(builder)


def test_load_agent_package_returns_none_on_bad_code(tmp_path: Path):
    bad = {agent_package.ENTRY_MODULE_FILENAME: "def build_agent(:\n"}
    assert agent_package.load_agent_package(bad, root=tmp_path) is None


def test_load_agent_package_returns_none_when_builder_missing(tmp_path: Path):
    files = {agent_package.ENTRY_MODULE_FILENAME: "X = 1\n"}
    assert agent_package.load_agent_package(files, root=tmp_path) is None
