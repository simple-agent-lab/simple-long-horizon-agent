from scripts.evolution_recipes.hyperagents import whole_repo_strategy as w


def test_safe_package_edits_keeps_valid_python_under_prefix():
    edits, rejected = w.safe_package_edits(
        {"agent/agent_program.py": "x = 1\n", "agent/util.py": "y = 2\n"}
    )
    assert set(edits) == {"agent/agent_program.py", "agent/util.py"}
    assert rejected == ()


def test_safe_package_edits_rejects_outside_prefix():
    edits, rejected = w.safe_package_edits({"container.py": "x = 1\n"})
    assert edits == {}
    assert "container.py" in rejected


def test_safe_package_edits_rejects_unparseable_python():
    edits, rejected = w.safe_package_edits({"agent/agent_program.py": "def f(:\n"})
    assert edits == {}
    assert "agent/agent_program.py" in rejected


def test_safe_package_edits_allows_tombstone_none():
    edits, rejected = w.safe_package_edits({"agent/old.py": None})
    assert edits == {"agent/old.py": None}
    assert rejected == ()


def test_strategy_returns_confined_proposal(tmp_path):
    from types import SimpleNamespace
    from scripts.evolution_recipes.hyperagents import whole_repo_strategy as w

    class FakeResp:
        text = '{"note":"n","evidence":["e"],"edits":{"agent/agent_program.py":"x=1\\n","bad.py":"y=2\\n"}}'

    def fake_complete(_req):
        return FakeResp()

    class V:
        hash = "v0"
        def files(self):
            return ()
        def read(self, _n):
            return ""

    ctx = SimpleNamespace(
        current=V(), failures=(), version=lambda h: V(), workspace=tmp_path,
    )
    strat = w.make_strategy(
        tmp_path, provider=SimpleNamespace(), complete_fn=fake_complete
    )
    proposal = strat(ctx)
    assert "agent/agent_program.py" in proposal.edits
    assert "bad.py" not in proposal.edits
    assert any("discarded" in e for e in proposal.evidence)
