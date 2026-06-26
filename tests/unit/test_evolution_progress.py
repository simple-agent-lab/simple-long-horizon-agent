from __future__ import annotations

import io
import threading
import time
import unittest

from simple_agent_lab.evolution.progress import ProgressReporter, mean_score


class _RaisingStream:
    def __init__(self) -> None:
        self.write_calls = 0

    def write(self, _text: str) -> int:
        self.write_calls += 1
        raise OSError("stream is closed")

    def flush(self) -> None:
        raise OSError("stream is closed")


class _SlowStream:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def write(self, text: str) -> int:
        self.parts.append(text)
        time.sleep(0.001)
        return len(text)

    def flush(self) -> None:
        return None

    def getvalue(self) -> str:
        return "".join(self.parts)


class ProgressReporterTest(unittest.TestCase):
    def test_line_formats_key_values_and_quotes_spaces(self) -> None:
        stream = io.StringIO()
        reporter = ProgressReporter(stream=stream)

        reporter.line(
            "decision",
            "accepted",
            candidate="abc123",
            baseline_reward=0.4,
            candidate_reward=0.55,
            delta=0.15,
            reason="not worse",
            missing=None,
        )

        self.assertEqual(
            stream.getvalue(),
            "[progress] decision accepted candidate=abc123 "
            "baseline_reward=0.400 candidate_reward=0.550 "
            'delta=+0.150 reason="not worse"\n',
        )

    def test_line_clips_long_error_values(self) -> None:
        stream = io.StringIO()
        reporter = ProgressReporter(stream=stream, max_value_chars=24)

        reporter.line("run", "error", error="RuntimeError: " + "x" * 80)

        self.assertEqual(
            stream.getvalue(),
            '[progress] run error error="RuntimeError: xxxxxxx..."\n',
        )

    def test_write_failures_disable_reporter_without_raising(self) -> None:
        stream = _RaisingStream()
        reporter = ProgressReporter(stream=stream)

        reporter.line("run", "start", id="demo")
        reporter.line("run", "complete", id="demo")

        self.assertEqual(stream.write_calls, 1)

    def test_concurrent_lines_remain_complete_records(self) -> None:
        stream = _SlowStream()
        reporter = ProgressReporter(stream=stream)

        threads = [
            threading.Thread(
                target=reporter.line,
                args=("rollout", "instance"),
                kwargs={"id": f"i{index}", "status": index},
            )
            for index in range(20)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 20)
        self.assertTrue(all(line.startswith("[progress] ") for line in lines))
        self.assertEqual(sum(line.count("[progress]") for line in lines), 20)

    def test_noop_reporter_is_silent(self) -> None:
        stream = io.StringIO()
        reporter = ProgressReporter(stream=stream, enabled=False)

        reporter.line("run", "start", id="demo")

        self.assertEqual(stream.getvalue(), "")

    def test_mean_score_prefers_requested_dimension(self) -> None:
        scores = {
            "i1": {"reward": 1.0, "valid_parent": 1.0},
            "i2": {"reward": 0.0, "valid_parent": 1.0},
        }

        self.assertEqual(mean_score(scores), 0.5)
        self.assertEqual(mean_score(scores, dim="valid_parent"), 1.0)
        self.assertIsNone(mean_score({}, dim="reward"))


if __name__ == "__main__":
    unittest.main()
