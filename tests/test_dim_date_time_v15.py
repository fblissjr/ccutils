"""Tests for dim_date / dim_time population.

Nine semantic views LEFT JOIN dim_date / dim_time for full_date and
time_of_day, but v0.15 derived date_key / time_key inline and never
populated either dim -- every view returned NULL dates. Fix:
- dim_time is seeded at DDL time (fixed 1440-row dimension, HHMM keys).
- dim_date rows are inserted during ETL for every date seen in staging.
"""

from __future__ import annotations

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.orchestrator import run_v15_etl


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


def _session_file(tmp_path, ts="2026-04-18T14:30:00Z"):
    """2026-04-18 is a Saturday (weekend, afternoon)."""
    jsonl = tmp_path / "dated.jsonl"
    jsonl.write_text(json.dumps({
        "type": "user", "uuid": "u1", "sessionId": "dated-s",
        "timestamp": ts,
        "cwd": "/p", "gitBranch": "main", "version": "2.1.114",
        "message": {"role": "user", "content": "hi"}}))
    return jsonl


class TestDimTimeSeed:
    def test_seeded_with_1440_minutes(self, conn):
        assert conn.execute("SELECT COUNT(*) FROM dim_time").fetchone()[0] == 1440

    def test_time_of_day_buckets(self, conn):
        rows = dict(conn.execute(
            "SELECT time_key, time_of_day FROM dim_time "
            "WHERE time_key IN (0, 530, 900, 1430, 1900)"
        ).fetchall())
        assert rows == {
            0: "night", 530: "night", 900: "morning",
            1430: "afternoon", 1900: "evening",
        }

    def test_seed_is_idempotent(self, tmp_path):
        db = tmp_path / "idem.duckdb"
        create_star_schema(db).close()
        conn = create_star_schema(db)
        assert conn.execute("SELECT COUNT(*) FROM dim_time").fetchone()[0] == 1440


class TestDimDatePopulation:
    def test_etl_inserts_session_dates(self, conn, tmp_path):
        run_v15_etl(conn, _session_file(tmp_path), project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT full_date, day_name, is_weekend FROM dim_date "
            "WHERE date_key = 20260418"
        ).fetchone()
        assert row is not None
        assert str(row[0]) == "2026-04-18"
        assert row[1] == "Saturday"
        assert row[2] is True

    def test_reetl_does_not_duplicate_dates(self, conn, tmp_path):
        f = _session_file(tmp_path)
        run_v15_etl(conn, f, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        run_v15_etl(conn, f, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        assert conn.execute(
            "SELECT COUNT(*) FROM dim_date WHERE date_key = 20260418"
        ).fetchone()[0] == 1


class TestSemanticViewsGetDates:
    def test_semantic_messages_full_date_and_time_of_day(self, conn, tmp_path):
        """The view-level acceptance: dates come back non-NULL."""
        run_v15_etl(conn, _session_file(tmp_path), project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT full_date, time_of_day FROM semantic_messages LIMIT 1"
        ).fetchone()
        assert row is not None
        assert str(row[0]) == "2026-04-18"
        assert row[1] == "afternoon"


class TestInsertMissingDimDatesApi:
    """The typed (table, *timestamp_cols) signature -- no raw SQL string."""

    def test_inserts_distinct_dates_from_multiple_columns(self, conn):
        from ccutils.etl.utils import insert_missing_dim_dates
        conn.execute("CREATE TABLE t (a TIMESTAMP, b TIMESTAMP)")
        conn.execute(
            "INSERT INTO t VALUES "
            "(TIMESTAMP '2026-01-02 09:00', TIMESTAMP '2026-03-04 10:00'), "
            "(TIMESTAMP '2026-01-02 11:00', NULL)"  # dup date + a NULL
        )
        insert_missing_dim_dates(conn, "t", "a", "b")
        keys = sorted(r[0] for r in conn.execute(
            "SELECT date_key FROM dim_date").fetchall())
        assert keys == [20260102, 20260304]

    def test_varchar_timestamps_are_cast(self, conn):
        from ccutils.etl.utils import insert_missing_dim_dates
        conn.execute("CREATE TABLE t (ts VARCHAR)")
        conn.execute("INSERT INTO t VALUES ('2026-05-06T12:00:00Z'), ('bad')")
        insert_missing_dim_dates(conn, "t", "ts")
        keys = [r[0] for r in conn.execute(
            "SELECT date_key FROM dim_date").fetchall()]
        assert keys == [20260506]


class TestDimTimePartialPopulation:
    def test_partial_dim_time_gets_completed(self, tmp_path):
        """A legacy warehouse with an incomplete dim_time (older ETL
        inserted only observed minutes) must be filled to 1440, not left
        partial by a whole-table emptiness guard."""
        db = tmp_path / "partial.duckdb"
        conn = create_star_schema(db)
        conn.execute("DELETE FROM dim_time WHERE time_key <> 900")  # keep 1
        assert conn.execute("SELECT COUNT(*) FROM dim_time").fetchone()[0] == 1
        conn.close()
        conn = create_star_schema(db)
        assert conn.execute(
            "SELECT COUNT(*) FROM dim_time").fetchone()[0] == 1440

    def test_time_of_day_matches_get_time_of_day_for_all_hours(self, conn):
        from ccutils.schemas.star.utils import get_time_of_day
        rows = conn.execute(
            "SELECT hour, time_of_day FROM dim_time").fetchall()
        for hour, tod in rows:
            assert tod == get_time_of_day(hour)


class TestDimDateReconcileFromSessions:
    def test_existing_sessions_get_dim_date_without_reetl(self, tmp_path):
        """A warehouse whose facts/sessions predate the dim_date fix (or
        whose JSONL was pruned) gets dim_date backfilled from dim_session
        timestamps on the next create_star_schema, no re-ETL needed."""
        db = tmp_path / "old.duckdb"
        conn = create_star_schema(db)
        conn.execute(
            "INSERT INTO dim_session (session_key, session_id, "
            "first_timestamp, last_timestamp) VALUES "
            "(md5('s'), 's', TIMESTAMP '2026-02-03 08:00', "
            "TIMESTAMP '2026-02-05 09:00')"
        )
        conn.execute("DELETE FROM dim_date")  # simulate pre-fix warehouse
        conn.close()
        conn = create_star_schema(db)
        keys = sorted(r[0] for r in conn.execute(
            "SELECT date_key FROM dim_date").fetchall())
        assert 20260203 in keys and 20260205 in keys
