"""Tests for the staging loader (Phase C chunk 1).

stg_log_entries is the bridge from Tier 1 (Parquet lake) to Tier 3
(warehouse facts). Every entry in every session lands here as one row;
fact-table populators select from this staging table to project into
their grain. Trunc-and-reload friendly: rerun overwrites the staging
rows for a session.
"""

import json
from pathlib import Path

import pytest

from ccutils import create_star_schema
from ccutils.etl.lineage import EtlRun
from ccutils.etl.staging import (
    load_session_to_staging,
    load_archive_to_staging,
    STG_LOG_ENTRIES_SCHEMA,
)
from ccutils.parsers.parquet_writer import write_session_to_parquet


@pytest.fixture
def sample_jsonl(tmp_path):
    """Synthesized session spanning multiple entry types."""
    jsonl = tmp_path / "stg-test.jsonl"
    lines = [
        {"type": "summary", "summary": "stg-test", "leafUuid": "x"},
        {"type": "user", "uuid": "u1", "sessionId": "stg-test",
         "timestamp": "2026-04-19T10:00:00Z", "cwd": "/p", "gitBranch": "main",
         "message": {"role": "user", "content": "go"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "sessionId": "stg-test", "timestamp": "2026-04-19T10:00:05Z",
         "message": {"role": "assistant", "content": [
             {"type": "text", "text": "ok"},
             {"type": "tool_use", "id": "toolu_001", "name": "Bash",
              "input": {"command": "ls"}},
         ], "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 3}}},
        {"type": "user", "uuid": "u2", "parentUuid": "a1",
         "sessionId": "stg-test", "timestamp": "2026-04-19T10:00:06Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "toolu_001", "content": "out"},
         ]},
         "toolUseResult": {"stdout": "out", "interrupted": False, "exitCode": 0}},
        {"type": "system", "subtype": "turn_duration", "uuid": "s1",
         "sessionId": "stg-test", "timestamp": "2026-04-19T10:00:07Z",
         "durationMs": 999, "messageCount": 3},
        {"type": "attachment", "uuid": "att1", "sessionId": "stg-test",
         "timestamp": "2026-04-19T10:00:08Z",
         "attachment": {"type": "diagnostics", "files": []}},
        {"type": "permission-mode", "sessionId": "stg-test", "permissionMode": "auto"},
        {"type": "progress", "sessionId": "stg-test", "toolUseID": "toolu_001",
         "data": {"type": "bash_progress", "stdout": "running"}},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


class TestStgLogEntriesDdl:
    def test_table_exists_after_create_star_schema(self, conn):
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='stg_log_entries'"
        ).fetchone()
        assert result is not None

    def test_schema_matches_constant(self, conn):
        cols = [c[0] for c in conn.execute("DESCRIBE stg_log_entries").fetchall()]
        expected = [f.name for f in STG_LOG_ENTRIES_SCHEMA]
        assert set(expected).issubset(set(cols))


class TestLoadSessionToStaging:
    def test_loads_one_session(self, conn, sample_jsonl, tmp_path):
        run = EtlRun.start(conn, source_path=str(sample_jsonl))
        # Phase A pipeline: JSONL -> Parquet first
        log_path, _ = write_session_to_parquet(
            sample_jsonl, tmp_path / "lake",
            etl_run_id=run.etl_run_id, project_slug="stg-project",
        )
        # Then Parquet -> DuckDB staging
        rows_loaded = load_session_to_staging(conn, log_path, run=run)
        assert rows_loaded == 8  # all 8 lines (including summary)

        # Every staged row carries the etl_run_id from this run
        stamped = conn.execute(
            "SELECT COUNT(*) FROM stg_log_entries WHERE etl_run_id = ?",
            [run.etl_run_id],
        ).fetchone()[0]
        assert stamped == 8

    def test_envelope_columns_populated(self, conn, sample_jsonl, tmp_path):
        run = EtlRun.start(conn, source_path=str(sample_jsonl))
        log_path, _ = write_session_to_parquet(
            sample_jsonl, tmp_path / "lake", etl_run_id=run.etl_run_id,
            project_slug="stg-project",
        )
        load_session_to_staging(conn, log_path, run=run)
        u1 = conn.execute(
            "SELECT type, session_id, cwd, git_branch FROM stg_log_entries "
            "WHERE uuid = 'u1'"
        ).fetchone()
        assert u1 == ("user", "stg-test", "/p", "main")

    def test_polymorphic_payloads_preserved_as_json(self, conn, sample_jsonl, tmp_path):
        run = EtlRun.start(conn, source_path=str(sample_jsonl))
        log_path, _ = write_session_to_parquet(
            sample_jsonl, tmp_path / "lake", etl_run_id=run.etl_run_id,
            project_slug="stg-project",
        )
        load_session_to_staging(conn, log_path, run=run)
        # User u2 has toolUseResult; assert it's available in staging
        result_json = conn.execute(
            "SELECT tool_use_result_json FROM stg_log_entries WHERE uuid = 'u2'"
        ).fetchone()[0]
        parsed = json.loads(result_json)
        assert parsed["stdout"] == "out"
        assert parsed["interrupted"] is False
        assert parsed["exitCode"] == 0

    def test_trunc_and_reload_replaces_session_rows(self, conn, sample_jsonl, tmp_path):
        run1 = EtlRun.start(conn, source_path=str(sample_jsonl))
        log_path, _ = write_session_to_parquet(
            sample_jsonl, tmp_path / "lake", etl_run_id=run1.etl_run_id,
            project_slug="stg-project",
        )
        load_session_to_staging(conn, log_path, run=run1)
        n1 = conn.execute("SELECT COUNT(*) FROM stg_log_entries").fetchone()[0]

        # Re-run on the same source -- staging should NOT double up.
        run2 = EtlRun.start(conn, source_path=str(sample_jsonl))
        log_path2, _ = write_session_to_parquet(
            sample_jsonl, tmp_path / "lake2", etl_run_id=run2.etl_run_id,
            project_slug="stg-project",
        )
        load_session_to_staging(conn, log_path2, run=run2)
        n2 = conn.execute("SELECT COUNT(*) FROM stg_log_entries").fetchone()[0]
        assert n2 == n1, "Re-loading the same session must replace its rows, not append"

        # And the etl_run_id stamped on the rows should be the LATEST run.
        latest = conn.execute(
            "SELECT DISTINCT etl_run_id FROM stg_log_entries"
        ).fetchall()
        assert latest == [(run2.etl_run_id,)]


class TestLoadArchiveToStaging:
    def test_loads_multiple_sessions(self, conn, sample_jsonl, tmp_path):
        # Write two sessions into a single Parquet lake.
        run = EtlRun.start(conn, source_path=str(tmp_path))
        lake = tmp_path / "lake"
        write_session_to_parquet(
            sample_jsonl, lake, etl_run_id=run.etl_run_id, project_slug="proj-a",
        )

        sample2 = tmp_path / "stg-test-2.jsonl"
        sample2.write_text(
            json.dumps({"type": "user", "uuid": "x1", "sessionId": "stg-test-2",
                        "message": {"role": "user", "content": "hi"}}) + "\n"
        )
        write_session_to_parquet(
            sample2, lake, etl_run_id=run.etl_run_id, project_slug="proj-b",
        )

        sessions_loaded = load_archive_to_staging(conn, lake, run=run)
        assert sessions_loaded == 2
        n = conn.execute("SELECT COUNT(*) FROM stg_log_entries").fetchone()[0]
        assert n == 8 + 1  # first fixture's 8 lines + second fixture's 1 line
