"""A tool call copied into a second transcript is its own row, and is marked as a replay.

A resumed, backgrounded or forked session replays earlier history into its
own file: the same `tool_use` ids, in a transcript that did not make the
calls. `fact_tool_calls` and `fact_file_operations` were keyed on
`tool_use_id` alone, so a copy could not be a second row. It took the row
over or left it, depending on load order and on whether the content hash
happened to differ, and the two facts disagreed about the same call: the file
operation kept the first transcript loaded and the tool call flipped to the
last (measured 2026-10-13, every such call of a one-project build).

Two claims, held here:

- The key is `(session_id, tool_use_id)`, stored as `tool_call_key`. Each
  transcript's copy is a row, in both facts, whatever the load order.
- `semantic_tool_call_replay` says which rows are copies. It is a VIEW over
  the whole table, never a stored per-session flag: a copy can be loaded
  before its original, and a populator cannot see rows that do not exist
  yet. A fork's calls at or before its own spawn are replays because its
  sidecar names that spawn. Among the other holders of a call the session
  that ended first is taken as the original, which is a rule and not a
  statement: two sessions that share a prefix and both continue do not say
  which came first.

Fixtures model the real shape (see tests/test_delegation_attribution.py): a
copied call keeps its id and timestamp under new entry uuids.

Delete this file and a copied call goes back to being one row that load
order assigns, or every copy is counted as the session's own work.
"""

from __future__ import annotations

import hashlib

import pytest
from test_delegation_attribution import (
    FORK_ACK,
    ROOT,
    _call,
    _launch_payload,
    _load,
    _transcript,
)

from ccutils import create_star_schema
from ccutils.schemas.star.schema import NATURAL_KEYS


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


def _read(tool_id, at, path):
    """A Read call with the payload a parent transcript states."""
    return _call(tool_id, at, name="Read", tool_input={"file_path": path},
                 text="BODY", payload={"type": "text", "file": {
                     "filePath": path, "content": "BODY", "numLines": 3,
                     "startLine": 1, "totalLines": 3}})


def _key(session_id, tool_use_id):
    return hashlib.md5(f"{session_id}|{tool_use_id}".encode()).hexdigest()


def _spawner_and_fork(tmp_path):
    """An agent reads a file, forks, and the fork reads another.

    The fork's transcript replays the spawner's history up to and including
    the call that created it, then does its own work.
    """
    spawner = _transcript(tmp_path, "agent-p1", [
        _read("tu_read", "10:02:00", "/work/a.py"),
        _call("tu_fork", "10:05:00", payload=_launch_payload("f1"), subagent_type="fork"),
        _read("tu_after", "10:10:00", "/work/c.py"),
    ], sidecar={"agentType": "general-purpose", "toolUseId": "tu_root_p", "spawnDepth": 1})
    fork = _transcript(tmp_path, "agent-f1", [
        _read("tu_read", "10:02:00", "/work/a.py"),
        _call("tu_fork", "10:05:00", text=FORK_ACK, subagent_type="fork"),
        _read("tu_native", "10:07:00", "/work/b.py"),
    ], end="10:20:00", sidecar={
        "agentType": "fork", "isFork": True, "toolUseId": "tu_fork",
        "parentAgentId": "p1", "spawnDepth": 2})
    return spawner, fork


@pytest.fixture(params=["spawner_first", "fork_first"])
def forked(request, conn, tmp_path):
    spawner, fork = _spawner_and_fork(tmp_path)
    order = (spawner, fork) if request.param == "spawner_first" else (fork, spawner)
    _load(conn, tmp_path, *order)
    return conn


class TestACopiedCallIsItsOwnRow:
    def test_the_registry_names_the_composite_key(self):
        assert NATURAL_KEYS["fact_tool_calls"] == "tool_call_key"
        assert NATURAL_KEYS["fact_file_operations"] == "tool_call_key"

    def test_each_transcript_keeps_its_own_tool_call(self, forked):
        """Two transcripts hold `tu_read`; each row is what ITS transcript stated."""
        rows = forked.execute(
            "SELECT session_id, tool_use_id, tool_call_key, result_payload_json IS NOT NULL "
            "FROM fact_tool_calls WHERE NOT is_deleted AND tool_use_id IN ('tu_read', 'tu_fork') "
            "ORDER BY tool_use_id, session_id"
        ).fetchall()
        assert rows == [
            # The fork's copy of its own spawn carries no payload; the
            # spawner's carries the launch it stated.
            ("agent-f1", "tu_fork", _key("agent-f1", "tu_fork"), False),
            ("agent-p1", "tu_fork", _key("agent-p1", "tu_fork"), True),
            ("agent-f1", "tu_read", _key("agent-f1", "tu_read"), True),
            ("agent-p1", "tu_read", _key("agent-p1", "tu_read"), True),
        ]

    def test_each_transcript_keeps_its_own_file_operation(self, forked):
        rows = forked.execute(
            "SELECT session_id, tool_use_id, tool_call_key, file_path "
            "FROM fact_file_operations WHERE NOT is_deleted ORDER BY tool_use_id, session_id"
        ).fetchall()
        assert rows == [
            ("agent-p1", "tu_after", _key("agent-p1", "tu_after"), "/work/c.py"),
            ("agent-f1", "tu_native", _key("agent-f1", "tu_native"), "/work/b.py"),
            ("agent-f1", "tu_read", _key("agent-f1", "tu_read"), "/work/a.py"),
            ("agent-p1", "tu_read", _key("agent-p1", "tu_read"), "/work/a.py"),
        ]

    def test_the_two_facts_agree_about_every_call(self, forked):
        """What was measured wrong: one call, a different session in each fact."""
        assert forked.execute(
            "SELECT COUNT(*) FROM fact_file_operations fo "
            "JOIN fact_tool_calls tc USING (tool_call_key) "
            "WHERE NOT fo.is_deleted AND NOT tc.is_deleted "
            "AND (fo.session_id <> tc.session_id OR fo.tool_use_id <> tc.tool_use_id)"
        ).fetchone()[0] == 0
        assert forked.execute(
            "SELECT COUNT(*) FROM fact_file_operations fo WHERE NOT fo.is_deleted "
            "AND NOT EXISTS (SELECT 1 FROM fact_tool_calls tc "
            "                WHERE tc.tool_call_key = fo.tool_call_key AND NOT tc.is_deleted)"
        ).fetchone()[0] == 0

    def test_reloading_a_transcript_changes_nothing(self, conn, tmp_path):
        """Idempotent, and a reload of one side does not disturb the other."""
        spawner, fork = _spawner_and_fork(tmp_path)
        _load(conn, tmp_path, spawner, fork)
        before = conn.execute(
            "SELECT tool_call_key, session_id, hash_diff FROM fact_tool_calls "
            "WHERE NOT is_deleted ORDER BY 1").fetchall()
        _load(conn, tmp_path, spawner, fork, spawner)
        after = conn.execute(
            "SELECT tool_call_key, session_id, hash_diff FROM fact_tool_calls "
            "WHERE NOT is_deleted ORDER BY 1").fetchall()
        assert after == before
        assert conn.execute(
            "SELECT COUNT(*) FROM fact_tool_calls WHERE is_deleted").fetchone()[0] == 0


def _replay(conn):
    return conn.execute(
        "SELECT session_id, tool_use_id, is_replay, replay_source "
        "FROM semantic_tool_call_replay ORDER BY tool_use_id, session_id"
    ).fetchall()


class TestAForksPrefixIsReplay:
    """Stated: the fork's sidecar names the call its own history starts after."""

    def test_calls_at_or_before_its_own_spawn_are_replays(self, forked):
        assert _replay(forked) == [
            ("agent-p1", "tu_after", False, None),
            ("agent-f1", "tu_fork", True, "fork_prefix"),
            ("agent-p1", "tu_fork", False, None),
            ("agent-f1", "tu_native", False, None),
            ("agent-f1", "tu_read", True, "fork_prefix"),
            ("agent-p1", "tu_read", False, None),
        ]

    def test_it_holds_when_the_spawner_was_never_loaded(self, conn, tmp_path):
        """No second row is needed: the sidecar alone settles it."""
        _, fork = _spawner_and_fork(tmp_path)
        _load(conn, tmp_path, fork)
        assert _replay(conn) == [
            ("agent-f1", "tu_fork", True, "fork_prefix"),
            ("agent-f1", "tu_native", False, None),
            ("agent-f1", "tu_read", True, "fork_prefix"),
        ]

    def test_a_fork_without_its_own_spawn_call_still_loses_the_tie(self, conn, tmp_path):
        """No cut to mark it by, so it falls to the rule, and a fork ranks last.

        One fork transcript on the real corpus does not hold the call that
        created it. Its shared calls are still copies. It ends BEFORE its
        spawner here, so ending first alone would call the fork the original.
        """
        spawner = _transcript(tmp_path, "agent-p1", [
            _read("tu_read", "10:02:00", "/work/a.py"),
        ], end="10:50:00", sidecar={"agentType": "general-purpose", "spawnDepth": 1})
        fork = _transcript(tmp_path, "agent-f1", [
            _read("tu_read", "10:02:00", "/work/a.py"),
        ], end="10:20:00", sidecar={
            "agentType": "fork", "isFork": True, "toolUseId": "tu_not_in_the_file",
            "parentAgentId": "p1", "spawnDepth": 2})
        _load(conn, tmp_path, spawner, fork)
        assert _replay(conn) == [
            ("agent-f1", "tu_read", True, "later_copy"),
            ("agent-p1", "tu_read", False, None),
        ]

    def test_an_agent_that_is_not_a_fork_replays_nothing(self, conn, tmp_path):
        """`toolUseId` alone is not the anchor; `isFork` is what makes it one."""
        agent = _transcript(tmp_path, "agent-c1", [
            _read("tu_own", "10:01:00", "/work/a.py"),
        ], sidecar={"agentType": "general-purpose", "toolUseId": "tu_spawned_me",
                    "spawnDepth": 1})
        _load(conn, tmp_path, agent)
        assert _replay(conn) == [("agent-c1", "tu_own", False, None)]


class TestACopyBetweenRootSessionsIsReplay:
    """Derived: nothing in a resumed or backgrounded session states its origin."""

    @pytest.fixture(params=["original_first", "copy_first"])
    def resumed(self, request, conn, tmp_path):
        original = _transcript(tmp_path, ROOT, [
            _read("tu_shared", "10:02:00", "/work/a.py"),
        ], end="10:30:00")
        # The continuation replays the call, then goes on after the original
        # stopped. Its name sorts FIRST, so a tie-break on the session id
        # alone would call it the original.
        copy = _transcript(tmp_path, "root-0000", [
            _read("tu_shared", "10:02:00", "/work/a.py"),
            _read("tu_later", "10:40:00", "/work/b.py"),
        ], end="11:00:00")
        order = (original, copy) if request.param == "original_first" else (copy, original)
        _load(conn, tmp_path, *order)
        return conn

    def test_the_session_that_ended_first_holds_the_original(self, resumed):
        assert _replay(resumed) == [
            ("root-0000", "tu_later", False, None),
            ("root-0000", "tu_shared", True, "later_copy"),
            (ROOT, "tu_shared", False, None),
        ]

    def test_a_call_held_once_is_never_a_replay(self, conn, tmp_path):
        only = _transcript(tmp_path, ROOT, [_read("tu_one", "10:02:00", "/work/a.py")])
        _load(conn, tmp_path, only)
        assert _replay(conn) == [(ROOT, "tu_one", False, None)]

    def test_the_view_has_one_row_per_live_tool_call(self, resumed):
        assert resumed.execute(
            "SELECT (SELECT COUNT(*) FROM fact_tool_calls WHERE NOT is_deleted), "
            "       (SELECT COUNT(*) FROM semantic_tool_call_replay), "
            "       (SELECT COUNT(DISTINCT tool_call_key) FROM semantic_tool_call_replay)"
        ).fetchone() == (3, 3, 3)


class TestViewsCountACallOnce:
    """Every view that counts tool calls or file operations leaves replays out."""

    def test_a_session_summary_counts_only_its_own_calls(self, forked):
        rows = dict(forked.execute(
            "SELECT session_id, total_tool_uses FROM semantic_session_summary"
        ).fetchall())
        # The spawner made three calls; the fork made one of its own.
        assert rows == {"agent-p1": 3, "agent-f1": 1}

    def test_summed_over_sessions_each_call_is_counted_once(self, forked):
        assert forked.execute(
            "SELECT SUM(total_tool_uses), "
            "       (SELECT COUNT(DISTINCT tool_use_id) FROM fact_tool_calls WHERE NOT is_deleted) "
            "FROM semantic_session_summary"
        ).fetchone() == (4, 4)

    def test_a_replayed_read_is_not_the_forks_file(self, forked):
        rows = forked.execute(
            "SELECT bsf.session_id, df.file_path, bsf.read_count "
            "FROM semantic_session_files bsf JOIN dim_file df USING (file_key) "
            "ORDER BY 1, 2"
        ).fetchall()
        assert rows == [
            ("agent-f1", "/work/b.py", 1),
            ("agent-p1", "/work/a.py", 1),
            ("agent-p1", "/work/c.py", 1),
        ]

    def test_the_behavior_view_counts_only_own_calls(self, forked):
        rows = dict(forked.execute(
            "SELECT session_id, tool_uses FROM semantic_session_behavior"
        ).fetchall())
        assert rows == {"agent-p1": 3, "agent-f1": 1}

    def test_a_replayed_transition_is_not_a_second_pattern(self, conn, tmp_path):
        """Read -> Agent happens once, in the spawner. Twice would list it."""
        spawner, fork = _spawner_and_fork(tmp_path)
        _load(conn, tmp_path, spawner, fork)
        assert conn.execute(
            "SELECT COUNT(*) FROM semantic_tool_patterns "
            "WHERE tool_name = 'Read' AND next_tool_name = 'Agent'"
        ).fetchone()[0] == 0, "the view lists a transition seen at least twice"
