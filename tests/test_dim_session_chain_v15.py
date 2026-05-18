"""Tests for the v0.15 dim_session_chain populator (Phase D).

Grain: one row per slug, aggregating over dim_session. A chain is the
set of sessions that share a Claude Code slug (set by the user via
/save / continuation). chain_key = md5(slug). dim_session.chain_key
gets pointed at the chain.

Stays minimal (no lineage block) like dim_tool / dim_model. Idempotent
delete-and-reload: chains are recomputed from current dim_session each
run.
"""

from __future__ import annotations

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.dim_session_chain import populate_dim_session_chain
from ccutils.etl.lineage import EtlRun
from ccutils.etl.orchestrator import run_v15_etl


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


def _slugged_session(tmp_path, name, slug, ts_base):
    """Build a minimal session JSONL anchored on `slug`."""
    jsonl = tmp_path / f"{name}.jsonl"
    lines = [
        {"type": "user", "uuid": f"{name}-u1", "sessionId": name,
         "timestamp": f"{ts_base}:00Z", "cwd": "/p", "gitBranch": "main",
         "version": "2.1.114", "slug": slug,
         "message": {"role": "user", "content": "go"}},
        {"type": "assistant", "uuid": f"{name}-a1", "parentUuid": f"{name}-u1",
         "sessionId": name, "timestamp": f"{ts_base}:01Z",
         "requestId": f"r-{name}", "slug": slug,
         "message": {"role": "assistant", "model": "claude-opus-4-7",
                     "content": [{"type": "text", "text": "ok"}]}},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


class TestDimSessionChain:
    def test_one_row_per_distinct_slug(self, conn, tmp_path):
        # Two sessions sharing one slug + one session with another slug
        a = _slugged_session(tmp_path, "sessA", "feature-x",
                              "2026-04-19T10:00")
        b = _slugged_session(tmp_path, "sessB", "feature-x",
                              "2026-04-19T11:00")
        c = _slugged_session(tmp_path, "sessC", "bugfix-y",
                              "2026-04-19T12:00")
        for path in (a, b, c):
            run_v15_etl(conn, path, project_name="test",
                        parquet_lake_root=tmp_path / "lake")
        n = conn.execute("SELECT COUNT(*) FROM dim_session_chain").fetchone()[0]
        assert n == 2

    def test_session_count_aggregated(self, conn, tmp_path):
        a = _slugged_session(tmp_path, "sessA", "feature-x",
                              "2026-04-19T10:00")
        b = _slugged_session(tmp_path, "sessB", "feature-x",
                              "2026-04-19T11:00")
        for path in (a, b):
            run_v15_etl(conn, path, project_name="test",
                        parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT session_count FROM dim_session_chain WHERE slug = 'feature-x'"
        ).fetchone()
        assert row[0] == 2

    def test_first_and_last_session_picked_by_timestamp(self, conn, tmp_path):
        a = _slugged_session(tmp_path, "sessA", "feature-x",
                              "2026-04-19T10:00")
        b = _slugged_session(tmp_path, "sessB", "feature-x",
                              "2026-04-19T11:00")
        for path in (a, b):
            run_v15_etl(conn, path, project_name="test",
                        parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            """
            SELECT first_session_key, last_session_key, first_timestamp,
                   last_timestamp
            FROM dim_session_chain WHERE slug = 'feature-x'
            """
        ).fetchone()
        # First should be A (earlier ts), last should be B
        # session_key = md5(session_id); verify by joining back to dim_session
        first_id = conn.execute(
            "SELECT session_id FROM dim_session WHERE session_key = ?",
            [row[0]],
        ).fetchone()[0]
        last_id = conn.execute(
            "SELECT session_id FROM dim_session WHERE session_key = ?",
            [row[1]],
        ).fetchone()[0]
        assert first_id == "sessA"
        assert last_id == "sessB"

    def test_dim_session_chain_key_pointed_at_chain(self, conn, tmp_path):
        a = _slugged_session(tmp_path, "sessA", "feature-x",
                              "2026-04-19T10:00")
        b = _slugged_session(tmp_path, "sessB", "feature-x",
                              "2026-04-19T11:00")
        for path in (a, b):
            run_v15_etl(conn, path, project_name="test",
                        parquet_lake_root=tmp_path / "lake")
        # Both sessions should have chain_key set to the feature-x chain
        rows = conn.execute(
            """
            SELECT ds.session_id, dsc.slug
            FROM dim_session ds
            JOIN dim_session_chain dsc ON ds.chain_key = dsc.chain_key
            WHERE ds.session_id IN ('sessA', 'sessB')
            ORDER BY ds.session_id
            """
        ).fetchall()
        assert rows == [("sessA", "feature-x"), ("sessB", "feature-x")]

    def test_sessions_without_slug_have_no_chain(self, conn, tmp_path):
        # A session with no slug field shouldn't appear in any chain
        jsonl = tmp_path / "noslug.jsonl"
        lines = [
            {"type": "user", "uuid": "noslug-u1", "sessionId": "noslug",
             "timestamp": "2026-04-19T10:00:00Z",
             "cwd": "/p", "gitBranch": "main", "version": "2.1.114",
             "message": {"role": "user", "content": "go"}},
        ]
        jsonl.write_text("\n".join(json.dumps(d) for d in lines))
        run_v15_etl(conn, jsonl, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        n = conn.execute("SELECT COUNT(*) FROM dim_session_chain").fetchone()[0]
        assert n == 0
        chain_key = conn.execute(
            "SELECT chain_key FROM dim_session WHERE session_id = 'noslug'"
        ).fetchone()[0]
        assert chain_key is None

    def test_idempotent_rebuild(self, conn, tmp_path):
        """Re-running the populator must produce identical chain rows."""
        a = _slugged_session(tmp_path, "sessA", "feature-x",
                              "2026-04-19T10:00")
        run_v15_etl(conn, a, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        first = conn.execute(
            "SELECT * FROM dim_session_chain ORDER BY chain_key"
        ).fetchall()
        run = EtlRun.start(conn, source_path="rerun")
        populate_dim_session_chain(conn, run=run)
        second = conn.execute(
            "SELECT * FROM dim_session_chain ORDER BY chain_key"
        ).fetchall()
        assert first == second
