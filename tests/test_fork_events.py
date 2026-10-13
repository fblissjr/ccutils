"""A fork's event counts, duration and decision timeline are its own.

The last part of "a fork is described by its own work" (tests/test_fork_cut.py):
`semantic_session_summary` counted the turns, compactions, API errors,
progress events and permission-mode changes of a fork's whole transcript, its
duration ran from the first entry of the conversation it inherited, and
`semantic_decisions` listed its spawner's decisions as the fork's.

Some of these entries state no time at all (permission-mode, for one). They
are placed by `fact_entry_events.derived_timestamp`, the stated time of the
entry before them in the file, which is what lets the cut apply to them.

Delete this and a fork that inherited a compaction and an API error reports
both as its own.
"""

from __future__ import annotations

import json

import pytest
from test_delegation_attribution import ROOT, _load
from test_fork_cut import _fork, _spawner

from ccutils import create_star_schema

DAY = "2026-07-17T"


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


def _system(name, clock, subtype, **fields):
    return (clock, {"type": "system", "subtype": subtype, "uuid": name, "sessionId": ROOT,
                    "timestamp": f"{DAY}{clock}Z", **fields})


def _mode(after, mode):
    """The way the source writes it: no uuid and no timestamp."""
    return (after, {"type": "permission-mode", "sessionId": ROOT, "permissionMode": mode})


def _attachment(name, clock, kind):
    return (clock, {"type": "attachment", "uuid": name, "sessionId": ROOT,
                    "timestamp": f"{DAY}{clock}Z", "attachment": {"type": kind, "files": []}})


def _progress(name, clock):
    return (clock, {"type": "progress", "uuid": name, "sessionId": ROOT,
                    "timestamp": f"{DAY}{clock}Z", "data": {"type": "hook_progress"}})


# What the spawner's history holds before the fork (the cut is 10:05:00)...
INHERITED = [
    _mode("10:00:00.000", "default"),
    _system("s-turn-old", "10:02:59.500", "turn_duration", durationMs=400, messageCount=3),
    _mode("10:03:59.000", "plan"),
    _system("s-err-old", "10:04:59.500", "api_error", error={"status": 503}, level="error"),
    _system("s-compact-old", "10:04:59.700", "compact_boundary", content="compacted",
            compactMetadata={"trigger": "auto", "preTokens": 100000}),
    _progress("p-old", "10:04:59.800"),
    _attachment("att-old", "10:04:59.900", "diagnostics"),
]
# ...and what the fork did itself.
OWN = [
    _system("s-turn-own", "10:08:00.000", "turn_duration", durationMs=700, messageCount=2),
    _mode("10:08:00.000", "acceptEdits"),
    _progress("p-own", "10:08:01.000"),
    _attachment("att-own", "10:08:02.000", "hook_success"),
]


def _with_events(path, events):
    """Insert each entry after the last line whose timestamp is not later than it."""
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    for clock, entry in events:
        at = max(i for i, line in enumerate(lines)
                 if line.get("timestamp") and line["timestamp"] <= f"{DAY}{clock}Z")
        # After anything already inserted at that position, to keep the order given.
        while at + 1 < len(lines) and not lines[at + 1].get("timestamp"):
            at += 1
        lines.insert(at + 1, entry)
    path.write_text("\n".join(json.dumps(line) for line in lines))
    return path


@pytest.fixture(params=["fork_alone", "spawner_then_fork"])
def loaded(request, conn, tmp_path):
    fork = _with_events(_fork(tmp_path), INHERITED + OWN)
    if request.param == "fork_alone":
        _load(conn, tmp_path, fork)
    else:
        _load(conn, tmp_path, _with_events(_spawner(tmp_path), INHERITED), fork)
    return conn


class TestAForksEventsAreItsOwn:
    def test_the_fixture_holds_inherited_events(self, loaded):
        """A control: the prefix really does carry an error, a compaction and a turn."""
        assert loaded.execute(
            "SELECT subtype, COUNT(*) FROM fact_system_events "
            "WHERE session_id = 'agent-f1' AND NOT is_deleted GROUP BY 1 ORDER BY 1"
        ).fetchall() == [("api_error", 1), ("compact_boundary", 1), ("turn_duration", 2)]

    def test_event_counts_start_at_the_cut(self, loaded):
        assert loaded.execute(
            "SELECT total_api_errors, total_compactions, turn_count, total_turn_durations_ms, "
            "       total_progress_events, total_attachments, total_diagnostics "
            "FROM semantic_session_summary WHERE session_id = 'agent-f1'"
        ).fetchone() == (0, 0, 1, 700, 1, 1, 0)

    def test_a_mode_change_the_fork_made_is_counted_against_what_it_inherited(self, loaded):
        """default -> plan was its spawner's. plan -> acceptEdits is the fork's own,
        and it is a change only because of the mode the fork inherited."""
        assert loaded.execute(
            "SELECT permission_mode_transition_count FROM semantic_session_summary "
            "WHERE session_id = 'agent-f1'"
        ).fetchone() == (1,)

    def test_its_duration_starts_at_its_first_own_message(self, loaded):
        """10:05:59 to 10:20:00, as its F12 facet already says. Not from 10:00:00."""
        assert loaded.execute(
            "SELECT session_duration_seconds FROM semantic_session_summary "
            "WHERE session_id = 'agent-f1'"
        ).fetchone() == (841.0,)

    def test_its_decision_timeline_is_its_own(self, loaded):
        assert loaded.execute(
            "SELECT decision_type, decision_value FROM semantic_decisions "
            "WHERE session_id = 'agent-f1' ORDER BY COALESCE(timestamp, derived_timestamp)"
        ).fetchall() == [("permission_mode_change", "acceptEdits")]

    def test_a_session_that_is_not_a_fork_keeps_every_event(self, conn, tmp_path):
        """A control: no cut, nothing withheld."""
        _load(conn, tmp_path, _with_events(_spawner(tmp_path), INHERITED))
        assert conn.execute(
            "SELECT total_api_errors, total_compactions, turn_count, total_turn_durations_ms, "
            "       total_progress_events, permission_mode_transition_count, "
            "       session_duration_seconds "
            "FROM semantic_session_summary WHERE session_id = 'agent-p1'"
        ).fetchone() == (1, 1, 1, 400, 1, 1, 3000.0)
        assert sorted(conn.execute(
            "SELECT decision_type FROM semantic_decisions WHERE session_id = 'agent-p1'"
        ).fetchall()) == [("api_error",), ("compact_boundary",), ("permission_mode_change",),
                          ("plan_revision",)]
