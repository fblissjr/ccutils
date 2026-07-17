"""Tests for the ETL run/batch/step metadata layer.

Three grains:
  - fact_etl_batch_runs: one row per CLI orchestration (BatchRun handle)
  - fact_etl_runs: one row per session ETL (EtlRun, existing) -- gains
    batch_run_id linkage, a data window (data_start_ts/data_end_ts), and
    REAL fact counts derived from its steps
  - fact_etl_steps: one row per DAG node per run -- lineage_upsert records
    one step per target fact with real DuckDB affected-row counts;
    run_v15_etl records the non-upsert stages (parquet, staging, dims)

semantic_etl_runs joins the three for run-level observability.
"""

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.lineage import BatchRun, EtlRun
from ccutils.etl.orchestrator import run_v15_etl


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


def _write_session(path, session_id, ts_base="2026-04-19T10:00"):
    lines = [
        {"type": "user", "uuid": f"{session_id}-u1", "sessionId": session_id,
         "timestamp": f"{ts_base}:00Z", "cwd": "/p",
         "message": {"role": "user", "content": "go"}},
        {"type": "assistant", "uuid": f"{session_id}-a1",
         "parentUuid": f"{session_id}-u1",
         "sessionId": session_id, "timestamp": f"{ts_base}:05Z",
         "message": {"role": "assistant", "model": "claude-opus-4-7",
                     "content": [{"type": "text", "text": "ok"}]}},
    ]
    path.write_text("\n".join(json.dumps(d) for d in lines))
    return path


class TestEtlMetadataDdl:
    def test_fact_etl_batch_runs_columns(self, conn):
        cols = {
            r[0] for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'fact_etl_batch_runs'"
            ).fetchall()
        }
        assert {
            "batch_run_id", "version_key", "started_at", "completed_at",
            "status", "source_root", "output_format",
            "sessions_seen", "sessions_succeeded", "sessions_failed",
            "rows_read", "rows_inserted", "rows_updated", "rows_soft_deleted",
            "data_start_ts", "data_end_ts", "error_message",
        } <= cols

    def test_fact_etl_steps_columns(self, conn):
        cols = {
            r[0] for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'fact_etl_steps'"
            ).fetchall()
        }
        assert {
            "step_id", "etl_run_id", "batch_run_id", "step_name",
            "step_order", "started_at", "completed_at", "status",
            "rows_read", "rows_inserted", "rows_updated",
            "rows_soft_deleted", "error_message",
        } <= cols

    def test_fact_etl_runs_gains_batch_and_window_columns(self, conn):
        cols = {
            r[0] for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'fact_etl_runs'"
            ).fetchall()
        }
        assert {"batch_run_id", "data_start_ts", "data_end_ts"} <= cols

    def test_fact_etl_runs_new_columns_are_migrations(self):
        """Adding columns to a shipped table needs _COLUMN_MIGRATIONS entries
        or pre-0.18 persistent warehouses break (CLAUDE.md DDL checklist)."""
        from ccutils.schemas.star.schema import _COLUMN_MIGRATIONS

        migrated = {
            (t, c) for t, c, _ in _COLUMN_MIGRATIONS
        }
        for col in ("batch_run_id", "data_start_ts", "data_end_ts"):
            assert ("fact_etl_runs", col) in migrated

    def test_semantic_etl_runs_view_exists(self, conn):
        conn.execute("SELECT * FROM semantic_etl_runs LIMIT 0")


class TestBatchRunLifecycle:
    def test_start_inserts_running_row(self, conn):
        batch = BatchRun.start(
            conn, source_root="/src/projects", output_format="duckdb"
        )
        row = conn.execute(
            "SELECT status, source_root, output_format FROM fact_etl_batch_runs "
            "WHERE batch_run_id = ?", [batch.batch_run_id]
        ).fetchone()
        assert row == ("running", "/src/projects", "duckdb")

    def test_complete_with_no_children_is_success_with_zeros(self, conn):
        batch = BatchRun.start(conn, source_root="/src", output_format="duckdb")
        batch.complete()
        row = conn.execute(
            "SELECT status, sessions_seen, sessions_succeeded, sessions_failed, "
            "rows_inserted, completed_at IS NOT NULL "
            "FROM fact_etl_batch_runs WHERE batch_run_id = ?",
            [batch.batch_run_id],
        ).fetchone()
        assert row == ("success", 0, 0, 0, 0, True)

    def test_fail_marks_failed_with_error(self, conn):
        batch = BatchRun.start(conn, source_root="/src", output_format="duckdb")
        batch.fail("boom")
        row = conn.execute(
            "SELECT status, error_message FROM fact_etl_batch_runs "
            "WHERE batch_run_id = ?", [batch.batch_run_id]
        ).fetchone()
        assert row == ("failed", "boom")


class TestRunLevelMetadata:
    def test_batch_run_id_stamped_on_run(self, conn, tmp_path):
        batch = BatchRun.start(conn, source_root=str(tmp_path), output_format="duckdb")
        session = _write_session(tmp_path / "s1.jsonl", "meta-s1")
        result = run_v15_etl(
            conn, session, parquet_lake_root=tmp_path / "lake",
            batch_run_id=batch.batch_run_id,
        )
        stamped = conn.execute(
            "SELECT batch_run_id FROM fact_etl_runs WHERE etl_run_id = ?",
            [result["etl_run_id"]],
        ).fetchone()[0]
        assert stamped == batch.batch_run_id

    def test_data_window_covers_session_timestamps(self, conn, tmp_path):
        session = _write_session(tmp_path / "s1.jsonl", "meta-s2")
        result = run_v15_etl(conn, session, parquet_lake_root=tmp_path / "lake")
        start, end = conn.execute(
            "SELECT data_start_ts, data_end_ts FROM fact_etl_runs "
            "WHERE etl_run_id = ?", [result["etl_run_id"]]
        ).fetchone()
        assert start is not None and end is not None
        assert str(start).startswith("2026-04-19 10:00:00")
        assert str(end).startswith("2026-04-19 10:00:05")

    def test_fact_counts_are_real_not_stubs(self, conn, tmp_path):
        session = _write_session(tmp_path / "s1.jsonl", "meta-s3")
        result = run_v15_etl(conn, session, parquet_lake_root=tmp_path / "lake")
        facts_inserted = conn.execute(
            "SELECT facts_inserted FROM fact_etl_runs WHERE etl_run_id = ?",
            [result["etl_run_id"]],
        ).fetchone()[0]
        assert facts_inserted > 0


class TestStepRecording:
    def test_upsert_steps_recorded_with_real_counts(self, conn, tmp_path):
        session = _write_session(tmp_path / "s1.jsonl", "step-s1")
        result = run_v15_etl(conn, session, parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT status, rows_read, rows_inserted, rows_updated "
            "FROM fact_etl_steps WHERE etl_run_id = ? "
            "AND step_name = 'upsert:fact_messages'",
            [result["etl_run_id"]],
        ).fetchone()
        assert row is not None
        status, rows_read, rows_inserted, rows_updated = row
        assert status == "success"
        assert rows_read == 2       # one user + one assistant message
        assert rows_inserted == 2
        assert rows_updated == 0

    def test_stage_steps_recorded(self, conn, tmp_path):
        session = _write_session(tmp_path / "s1.jsonl", "step-s2")
        result = run_v15_etl(conn, session, parquet_lake_root=tmp_path / "lake")
        names = {
            r[0] for r in conn.execute(
                "SELECT step_name FROM fact_etl_steps WHERE etl_run_id = ?",
                [result["etl_run_id"]],
            ).fetchall()
        }
        assert {"write_parquet", "load_staging", "upsert_dimensions"} <= names

    def test_step_order_unique_and_increasing(self, conn, tmp_path):
        session = _write_session(tmp_path / "s1.jsonl", "step-s3")
        result = run_v15_etl(conn, session, parquet_lake_root=tmp_path / "lake")
        orders = [
            r[0] for r in conn.execute(
                "SELECT step_order FROM fact_etl_steps WHERE etl_run_id = ? "
                "ORDER BY step_order", [result["etl_run_id"]]
            ).fetchall()
        ]
        assert orders == sorted(set(orders))
        assert len(orders) > 5

    def test_rerun_of_unchanged_session_shows_noop_counts(self, conn, tmp_path):
        session = _write_session(tmp_path / "s1.jsonl", "step-s4")
        run_v15_etl(conn, session, parquet_lake_root=tmp_path / "lake")
        second = run_v15_etl(conn, session, parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT rows_inserted, rows_updated FROM fact_etl_steps "
            "WHERE etl_run_id = ? AND step_name = 'upsert:fact_messages'",
            [second["etl_run_id"]],
        ).fetchone()
        assert row == (0, 0)

    def test_failing_step_is_marked_failed(self, conn):
        run = EtlRun.start(conn, source_path="/x")
        with pytest.raises(ValueError):
            with run.step("boom"):
                raise ValueError("kapow")
        row = conn.execute(
            "SELECT status, error_message FROM fact_etl_steps "
            "WHERE etl_run_id = ? AND step_name = 'boom'",
            [run.etl_run_id],
        ).fetchone()
        assert row[0] == "failed"
        assert "kapow" in row[1]


class TestBatchRollup:
    def test_two_sessions_roll_up(self, conn, tmp_path):
        batch = BatchRun.start(conn, source_root=str(tmp_path), output_format="duckdb")
        lake = tmp_path / "lake"
        for i, ts in ((1, "2026-04-19T10:00"), (2, "2026-04-20T11:00")):
            session = _write_session(tmp_path / f"s{i}.jsonl", f"roll-s{i}", ts)
            run_v15_etl(
                conn, session, parquet_lake_root=lake,
                batch_run_id=batch.batch_run_id,
            )
        batch.complete()
        row = conn.execute(
            "SELECT status, sessions_seen, sessions_succeeded, sessions_failed, "
            "rows_inserted, data_start_ts, data_end_ts "
            "FROM fact_etl_batch_runs WHERE batch_run_id = ?",
            [batch.batch_run_id],
        ).fetchone()
        status, seen, ok, failed, rows_inserted, start, end = row
        assert (status, seen, ok, failed) == ("success", 2, 2, 0)
        assert rows_inserted > 0
        assert str(start).startswith("2026-04-19")
        assert str(end).startswith("2026-04-20")

    def test_failed_child_makes_batch_partial(self, conn, tmp_path):
        batch = BatchRun.start(conn, source_root=str(tmp_path), output_format="duckdb")
        session = _write_session(tmp_path / "s1.jsonl", "part-s1")
        run_v15_etl(
            conn, session, parquet_lake_root=tmp_path / "lake",
            batch_run_id=batch.batch_run_id,
        )
        bad = EtlRun.start(
            conn, source_path="/bad", batch_run_id=batch.batch_run_id
        )
        bad.fail("unparseable")
        batch.complete()
        row = conn.execute(
            "SELECT status, sessions_seen, sessions_succeeded, sessions_failed "
            "FROM fact_etl_batch_runs WHERE batch_run_id = ?",
            [batch.batch_run_id],
        ).fetchone()
        assert row == ("partial", 2, 1, 1)


class TestSemanticEtlRunsView:
    def test_view_reports_run_with_step_rollup(self, conn, tmp_path):
        session = _write_session(tmp_path / "s1.jsonl", "view-s1")
        result = run_v15_etl(conn, session, parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT status, step_count, rows_inserted, batch_run_id "
            "FROM semantic_etl_runs WHERE etl_run_id = ?",
            [result["etl_run_id"]],
        ).fetchone()
        status, step_count, rows_inserted, batch_run_id = row
        assert status == "success"
        assert step_count > 5
        assert rows_inserted > 0
        assert batch_run_id is None  # no batch: LEFT JOIN keeps the run


class TestArchiveWiring:
    def test_generate_duckdb_archive_records_one_batch(self, tmp_path):
        import duckdb

        from ccutils.export.duckdb_archive import generate_duckdb_archive

        projects_dir = tmp_path / "projects"
        proj = projects_dir / "-home-user-projects-proj"
        proj.mkdir(parents=True)
        _write_session(proj / "aaa.jsonl", "arch-s1")
        _write_session(proj / "bbb.jsonl", "arch-s2", "2026-04-21T09:00")

        out = tmp_path / "out"
        generate_duckdb_archive(projects_dir, out)

        conn = duckdb.connect(str(out / "archive.duckdb"))
        try:
            batches = conn.execute(
                "SELECT batch_run_id, status, source_root, output_format, "
                "sessions_seen, sessions_succeeded "
                "FROM fact_etl_batch_runs"
            ).fetchall()
            assert len(batches) == 1
            batch_run_id, status, source_root, fmt, seen, ok = batches[0]
            assert status == "success"
            assert source_root == str(projects_dir)
            assert fmt == "duckdb"
            assert (seen, ok) == (2, 2)

            orphans = conn.execute(
                "SELECT COUNT(*) FROM fact_etl_runs WHERE batch_run_id IS NULL"
            ).fetchone()[0]
            assert orphans == 0
        finally:
            conn.close()
