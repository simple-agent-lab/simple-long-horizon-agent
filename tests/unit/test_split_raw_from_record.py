"""`split_raw_from_record` externalizes the provider raw snapshot.

The provider request/response blob stashed on every assistant message's
``sidecar["raw"]`` is duplicated across a record's messages/events/model_turns
views and regrows every turn — ballooning a long run's serialized trace past
200 MB (an unopenable single-line JSON). `split_raw_from_record` lifts it into a
content-deduplicated pool, replacing each occurrence with a light ``raw_ref``
pointer that the viewer resolves on demand from a sibling ``*.raw.jsonl`` file.
"""

from __future__ import annotations

import unittest

from simple_long_horizon_agent.trace import RAW_REF_KEY, split_raw_from_record


def _raw(n: int) -> dict:
    """A provider raw blob whose size grows with the turn index `n`."""
    return {
        "request": {"model": "m", "messages": [{"role": "user"} for _ in range(n)]},
        "response": {"id": f"resp-{n}", "model": "m"},
    }


class SplitRawFromRecordTest(unittest.TestCase):
    def test_externalizes_and_replaces_with_ref(self) -> None:
        record = {
            "messages": [{"sidecar": {"raw": _raw(1)}}],
            "events": [{"message": {"sidecar": {"raw": _raw(1)}}}],
            "meta": {"model": "m"},
        }
        slim, pool = split_raw_from_record(record)

        # Every inline raw is now a {raw_ref: int} pointer.
        msg_raw = slim["messages"][0]["sidecar"]["raw"]
        evt_raw = slim["events"][0]["message"]["sidecar"]["raw"]
        self.assertEqual(set(msg_raw), {RAW_REF_KEY})
        self.assertIsInstance(msg_raw[RAW_REF_KEY], int)
        # The two identical copies dedup to ONE pool slot.
        self.assertEqual(msg_raw[RAW_REF_KEY], evt_raw[RAW_REF_KEY])
        self.assertEqual(len(pool), 1)
        # The pool entry round-trips to the original blob.
        self.assertEqual(pool[msg_raw[RAW_REF_KEY]], _raw(1))
        # Non-raw fields are untouched.
        self.assertEqual(slim["meta"], {"model": "m"})

    def test_dedups_across_views_but_keeps_distinct_turns(self) -> None:
        # turn 0 and turn 1 raw differ (history grew); each appears in 3 views.
        record = {
            "messages": [{"raw": _raw(0)}, {"raw": _raw(1)}],
            "events": [{"raw": _raw(0)}, {"raw": _raw(1)}],
            "model_turns": [
                {"sidecar": {"raw": _raw(0)}},
                {"sidecar": {"raw": _raw(1)}},
            ],
        }
        slim, pool = split_raw_from_record(record)
        # 6 occurrences (2 turns x 3 views) collapse to 2 distinct blobs.
        self.assertEqual(len(pool), 2)
        refs = {
            slim["messages"][0]["raw"][RAW_REF_KEY],
            slim["events"][0]["raw"][RAW_REF_KEY],
            slim["model_turns"][0]["sidecar"]["raw"][RAW_REF_KEY],
        }
        self.assertEqual(len(refs), 1)  # all turn-0 copies share a slot
        self.assertNotEqual(
            slim["messages"][0]["raw"][RAW_REF_KEY],
            slim["messages"][1]["raw"][RAW_REF_KEY],
        )

    def test_ignores_non_provider_raw_keys(self) -> None:
        # A "raw" key that is not a provider snapshot (no request/response) or is
        # not a dict must be left inline — only the provider blob is externalized.
        record = {
            "args": {"raw": "literal string"},
            "tool": {"raw": {"some": "payload"}},
        }
        slim, pool = split_raw_from_record(record)
        self.assertEqual(pool, [])
        self.assertEqual(slim, record)

    def test_no_raw_yields_empty_pool(self) -> None:
        record = {"messages": [{"content": "hi"}], "events": []}
        slim, pool = split_raw_from_record(record)
        self.assertEqual(pool, [])
        self.assertEqual(slim, record)
