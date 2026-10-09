"""Tests for the semantic_decisions view.

One unified decision timeline over facts the v0.15 ETL already
populates -- no new ETL, pure projection:
    fact_plan_revisions            -> decision_type 'plan_revision'
    fact_entry_events (permission-mode) -> decision_type 'permission_mode_change'
    fact_system_events (3 subtypes)-> 'stop_event' / 'api_error' / 'compact_boundary'
"""

from __future__ import annotations

import pytest

from ccutils import create_star_schema


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


_LINEAGE = (
    "created_by_version_key, last_updated_by_version_key, etl_run_id, "
    "record_source, hash_diff"
)
_LINEAGE_VALS = "'v', 'v', 'run1', 'test', 'h'"


def _insert_plan_revision(conn, session_id="s1", outcome="accepted"):
    conn.execute(
        f"""
        INSERT INTO fact_plan_revisions (
            {_LINEAGE}, revision_key, tool_use_id, session_id, session_key,
            revision_number, plan_timestamp, outcome, outcome_signal
        ) VALUES (
            {_LINEAGE_VALS}, 'rk1', 'tu1', '{session_id}', md5('{session_id}'),
            1, TIMESTAMP '2026-04-19 10:00:00', '{outcome}', 'structural'
        )
        """
    )


def _insert_permission_entry(conn, seq, mode, session_id="s1",
                             derived="2026-04-19 10:05:00"):
    """A permission-mode entry as the source writes it: no stated timestamp,
    ordered only by its position in the file."""
    conn.execute(
        f"""
        INSERT INTO fact_entry_events (
            {_LINEAGE}, entry_id, session_id, session_key,
            sequence_num, timestamp, derived_timestamp, entry_type, value_text
        ) VALUES (
            {_LINEAGE_VALS}, 'e-perm-{seq}', '{session_id}', md5('{session_id}'),
            {seq}, NULL, TIMESTAMP '{derived}', 'permission-mode', '{mode}'
        )
        """
    )


def _insert_permission_toggle(conn, session_id="s1", mode="plan"):
    """The session starts in `default` and changes to `mode`."""
    _insert_permission_entry(conn, 1, "default", session_id, "2026-04-19 10:04:00")
    _insert_permission_entry(conn, 5, mode, session_id, "2026-04-19 10:05:00")


def _insert_system_event(conn, subtype, session_id="s1", **cols):
    extra_cols = "".join(f", {k}" for k in cols)
    extra_vals = "".join(
        f", {v!r}" if isinstance(v, str) else f", {v}" for v in cols.values()
    )
    conn.execute(
        f"""
        INSERT INTO fact_system_events (
            {_LINEAGE}, entry_id, session_id, session_key,
            timestamp, subtype{extra_cols}
        ) VALUES (
            {_LINEAGE_VALS}, 'e-{subtype}', '{session_id}', md5('{session_id}'),
            TIMESTAMP '2026-04-19 10:10:00', '{subtype}'{extra_vals}
        )
        """
    )


class TestSemanticDecisionsView:
    def test_view_exists(self, conn):
        row = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='view' AND name='semantic_decisions'"
        ).fetchone()
        assert row is not None

    def test_plan_revision_projected(self, conn):
        _insert_plan_revision(conn, outcome="accepted")
        rows = conn.execute(
            "SELECT decision_type, decision_value, decision_signal "
            "FROM semantic_decisions"
        ).fetchall()
        assert rows == [("plan_revision", "accepted", "structural")]

    def test_permission_mode_change_projected(self, conn):
        _insert_permission_toggle(conn, mode="acceptEdits")
        rows = conn.execute(
            "SELECT decision_type, decision_value FROM semantic_decisions"
        ).fetchall()
        assert rows == [("permission_mode_change", "acceptEdits")]

    def test_a_restated_mode_is_not_a_change(self, conn):
        """Claude Code restates the current mode far more often than it
        changes it. The view once projected every permission-mode entry as a
        change; measured on the corpus that reported a change on entries
        that repeat the mode before them almost every time (the record is in
        CHANGELOG.md). A change is an entry whose mode differs from the one
        before it in file order, and the first entry is a starting mode."""
        for seq, mode in [(1, "default"), (3, "default"), (5, "plan"),
                          (7, "plan"), (9, "plan"), (11, "default")]:
            _insert_permission_entry(conn, seq, mode)
        rows = conn.execute(
            "SELECT decision_value, source_key FROM semantic_decisions "
            "ORDER BY source_key"
        ).fetchall()
        assert rows == [("default", "e-perm-11"), ("plan", "e-perm-5")]

    def test_a_change_with_no_stated_time_carries_a_derived_one(self, conn):
        """`timestamp` is stated or NULL; the placement is its own column,
        and the date follows whichever exists so a date filter keeps the row."""
        _insert_permission_toggle(conn)
        row = conn.execute(
            "SELECT timestamp, strftime(derived_timestamp, '%H:%M'), "
            "       CAST(decision_date AS VARCHAR) FROM semantic_decisions"
        ).fetchone()
        assert row == (None, "10:05", "2026-04-19")
        _insert_plan_revision(conn)
        stated = conn.execute(
            "SELECT timestamp IS NOT NULL, derived_timestamp FROM semantic_decisions "
            "WHERE decision_type = 'plan_revision'"
        ).fetchone()
        assert stated == (True, None)

    def test_other_meta_types_are_not_decisions(self, conn):
        conn.execute(
            f"""
            INSERT INTO fact_entry_events (
                {_LINEAGE}, entry_id, session_id, session_key,
                sequence_num, derived_timestamp, entry_type, value_text
            ) VALUES (
                {_LINEAGE_VALS}, 'e-title', 's1', md5('s1'),
                2, TIMESTAMP '2026-04-19 10:00:00', 'custom-title', 'My session'
            )
            """
        )
        assert conn.execute(
            "SELECT COUNT(*) FROM semantic_decisions"
        ).fetchone()[0] == 0

    def test_system_event_subtypes_projected(self, conn):
        _insert_system_event(
            conn, "stop_hook_summary", has_output=True
        )
        _insert_system_event(
            conn, "api_error", error_type="overloaded", error_status=529
        )
        _insert_system_event(
            conn, "compact_boundary",
            compact_trigger="auto", compact_pre_tokens=150000,
        )
        # turn_duration is a system event but NOT a decision
        _insert_system_event(conn, "turn_duration", duration_ms=1234)

        rows = dict(
            conn.execute(
                "SELECT decision_type, decision_value FROM semantic_decisions"
            ).fetchall()
        )
        assert rows == {
            "stop_event": None,  # stopReason is stated empty on every real row; the column is gone
            "api_error": "overloaded",
            "compact_boundary": "auto",
        }

    def test_soft_deleted_rows_excluded(self, conn):
        _insert_plan_revision(conn)
        conn.execute("UPDATE fact_plan_revisions SET is_deleted = TRUE")
        assert conn.execute(
            "SELECT COUNT(*) FROM semantic_decisions"
        ).fetchone()[0] == 0

    def test_timeline_ordering_columns_present(self, conn):
        """Every row carries session + timestamp + date for timeline queries."""
        _insert_plan_revision(conn)
        _insert_permission_toggle(conn)
        rows = conn.execute(
            """
            SELECT session_id, COALESCE(timestamp, derived_timestamp),
                   decision_date, source_table
            FROM semantic_decisions
            ORDER BY COALESCE(timestamp, derived_timestamp)
            """
        ).fetchall()
        assert len(rows) == 2
        assert all(r[0] == "s1" for r in rows)
        assert rows[0][1] < rows[1][1]
        assert str(rows[0][2]) == "2026-04-19"
        assert rows[0][3] == "fact_plan_revisions"
        assert rows[1][3] == "fact_entry_events"
