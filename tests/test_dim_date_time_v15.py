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
