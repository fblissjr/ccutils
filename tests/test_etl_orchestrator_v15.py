"""Tests for the v0.15 ETL orchestrator (Phase C6).

run_v15_etl is the single entry point that:
  1. Parses JSONL via Pydantic
  2. Writes the per-session Parquet to the lake (Tier 1)
  3. Loads staging from Parquet (Tier 2)
  4. Populates dimensions (dim_session, dim_project, dim_model, dim_tool)
  5. Populates every v0.15 fact in dependency order
  6. Closes out the EtlRun

Idempotent end-to-end: rerunning on unchanged source produces no UPDATEs.
"""

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.orchestrator import run_v15_etl


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


@pytest.fixture
def basic_session(tmp_path):
    jsonl = tmp_path / "basic.jsonl"
    lines = [
        {"type": "user", "uuid": "u1", "sessionId": "basic-s",
         "timestamp": "2026-04-19T10:00:00Z",
         "cwd": "/p", "gitBranch": "main", "version": "2.1.114",
         "permissionMode": "default",
         "message": {"role": "user", "content": "go"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "sessionId": "basic-s", "timestamp": "2026-04-19T10:00:01Z",
         "requestId": "req_1",
         "message": {"role": "assistant", "model": "claude-opus-4-7", "content": [
             {"type": "text", "text": "ok"},
             {"type": "tool_use", "id": "tu1", "name": "Bash",
              "input": {"command": "ls"}},
         ],
         "stop_reason": "tool_use",
         "usage": {"input_tokens": 5, "output_tokens": 3,
                   "cache_read_input_tokens": 100,
                   "service_tier": "standard"}}},
        {"type": "user", "uuid": "u2", "parentUuid": "a1",
         "sessionId": "basic-s", "timestamp": "2026-04-19T10:00:02Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu1", "content": "files"},
         ]},
         "toolUseResult": {"stdout": "files", "interrupted": False, "exitCode": 0}},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


class TestRunV15Etl:
    def test_returns_etl_run_id(self, conn, basic_session, tmp_path):
        result = run_v15_etl(
            conn, basic_session, project_name="test-project",
            parquet_lake_root=tmp_path / "lake",
        )
        assert "etl_run_id" in result
        assert len(result["etl_run_id"]) == 32

    def test_populates_all_v15_facts(self, conn, basic_session, tmp_path):
        run_v15_etl(
            conn, basic_session, project_name="test-project",
            parquet_lake_root=tmp_path / "lake",
        )
        for table in (
            "fact_messages",
            "fact_tool_uses",
            "fact_tool_results",
            "fact_token_usage",
            "fact_session_summary",
        ):
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert n > 0, f"Expected rows in {table}"

    def test_records_etl_run_in_fact_etl_runs(self, conn, basic_session, tmp_path):
        result = run_v15_etl(
            conn, basic_session, project_name="test-project",
            parquet_lake_root=tmp_path / "lake",
        )
        row = conn.execute(
            "SELECT status, sessions_inserted FROM fact_etl_runs WHERE etl_run_id = ?",
            [result["etl_run_id"]],
        ).fetchone()
        assert row[0] == "success"
        # at least the one session we ETL'd
        assert row[1] >= 1

    def test_session_summary_has_one_row_for_the_session(
        self, conn, basic_session, tmp_path
    ):
        run_v15_etl(
            conn, basic_session, project_name="test-project",
            parquet_lake_root=tmp_path / "lake",
        )
        row = conn.execute(
            "SELECT total_messages, total_tool_uses, total_tool_results "
            "FROM fact_session_summary WHERE session_id = 'basic-s'"
        ).fetchone()
        # 2 user + 1 assistant = 3 messages
        assert row[0] == 3
        assert row[1] == 1  # tu1
        assert row[2] == 1  # one tool_result

    def test_idempotent_reetl(self, conn, basic_session, tmp_path):
        """Re-running ETL on unchanged source produces no UPDATEs in
        last_updated_at on any fact."""
        run_v15_etl(
            conn, basic_session, project_name="test-project",
            parquet_lake_root=tmp_path / "lake",
        )
        first = {}
        for table in ("fact_messages", "fact_tool_uses", "fact_tool_results",
                      "fact_token_usage", "fact_session_summary"):
            first[table] = conn.execute(
                f"SELECT last_updated_at FROM {table} ORDER BY 1"
            ).fetchall()

        run_v15_etl(
            conn, basic_session, project_name="test-project",
            parquet_lake_root=tmp_path / "lake",
        )
        for table in first:
            second = conn.execute(
                f"SELECT last_updated_at FROM {table} ORDER BY 1"
            ).fetchall()
            assert first[table] == second, (
                f"{table} last_updated_at changed on re-ETL of unchanged source"
            )

    def test_etl_run_fails_record_status_failed(self, conn, tmp_path):
        """If parsing fails the EtlRun should be marked failed, not left
        hanging at 'running'."""
        bad_path = tmp_path / "does_not_exist.jsonl"
        with pytest.raises(Exception):
            run_v15_etl(
                conn, bad_path, project_name="test-project",
                parquet_lake_root=tmp_path / "lake",
            )
        # The failure should have been recorded
        row = conn.execute(
            "SELECT status FROM fact_etl_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        assert row[0] == "failed"


class TestDimSessionEnrichment:
    """dim_session must carry enough context for the semantic_* views to join.

    Phase D will widen this with heuristic columns; this just covers the
    minimum so semantic_sessions / semantic_project_context / semantic_cost_analysis
    return rows immediately.
    """

    def test_dim_session_project_key_wired_to_dim_project(
        self, conn, basic_session, tmp_path
    ):
        run_v15_etl(
            conn, basic_session, project_name="test-project",
            parquet_lake_root=tmp_path / "lake",
        )
        row = conn.execute(
            """
            SELECT ds.session_id, dp.project_name
            FROM dim_session ds
            JOIN dim_project dp ON ds.project_key = dp.project_key
            """
        ).fetchone()
        assert row is not None, "dim_session.project_key must FK into dim_project"
        assert row[0] == "basic-s"

    def test_dim_session_carries_first_and_last_timestamp(
        self, conn, basic_session, tmp_path
    ):
        run_v15_etl(
            conn, basic_session, project_name="test-project",
            parquet_lake_root=tmp_path / "lake",
        )
        row = conn.execute(
            "SELECT first_timestamp, last_timestamp FROM dim_session"
        ).fetchone()
        assert row[0] is not None, "first_timestamp must be populated"
        assert row[1] is not None, "last_timestamp must be populated"
        assert row[0] <= row[1]

    def test_semantic_project_context_returns_rows(
        self, conn, basic_session, tmp_path
    ):
        run_v15_etl(
            conn, basic_session, project_name="test-project",
            parquet_lake_root=tmp_path / "lake",
        )
        rows = conn.execute(
            "SELECT session_id, project_name FROM semantic_project_context"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "basic-s"
        assert rows[0][1] is not None
