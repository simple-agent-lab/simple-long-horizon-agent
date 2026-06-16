from pathlib import Path

from pytest import approx

from simple_agent_lab.evolution.components.criterion import valid_when
from simple_agent_lab.evolution.kernel import log


def _scores(reward):
    return {"i1": {"reward": reward}}


def test_valid_when_admits_any_gradable_child():
    verdict = valid_when()(_scores(0.0), _scores(0.0))
    assert verdict.accepted is True
    assert verdict.deltas["valid_parent"] == 1.0


def test_valid_when_rejects_when_no_runs():
    verdict = valid_when()({}, {})
    assert verdict.accepted is False
    assert verdict.deltas["valid_parent"] == 0.0


def test_valid_when_records_reward_delta():
    verdict = valid_when()(_scores(0.5), _scores(0.4))
    assert verdict.accepted is True  # worse, but valid -> still admitted
    assert verdict.deltas["reward"] == approx(-0.1)


def test_log_append_preserves_valid_parent_on_candidate(tmp_path: Path):
    log.append(
        tmp_path,
        baseline={"hash": "p", "scores": {"reward": 0.5}},
        candidate={"hash": "c", "parent": "p", "scores": {"reward": 0.4},
                   "valid_parent": True},
        slice_=type("S", (), {"id": "s", "sha": "abc", "n": 1})(),
        verdict=type("V", (), {"accepted": True, "reason": "ok", "deltas": {}})(),
        kind="code",
        runs={"baseline": "rb", "candidate": "rc"},
    )
    rows = log.read(tmp_path)
    assert rows[-1].candidate["valid_parent"] is True
