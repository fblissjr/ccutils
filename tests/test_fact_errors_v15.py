"""Tests for the v0.15 fact_errors populator (Phase D).

Grain: one row per failed tool call. Derived from fact_tool_results
where is_error = TRUE. error_type is classified by the zero-dep regex
rules in ccutils.etl.heuristics.classify_error_type.

Run AFTER populate_fact_tool_uses + populate_fact_tool_results.
"""

from __future__ import annotations

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.fact_errors import populate_fact_errors
from ccutils.etl.fact_messages import populate_fact_messages
from ccutils.etl.fact_tool_calls import (
    populate_fact_tool_results,
    populate_fact_tool_uses,
)
from ccutils.etl.heuristics import classify_error_type
from ccutils.etl.lineage import EtlRun
from ccutils.etl.staging import load_session_to_staging
from ccutils.parsers.parquet_writer import write_session_to_parquet


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


class TestClassifyErrorType:
    def test_permission_denied(self):
        assert classify_error_type("Permission denied: cannot write") == "permission_denied"

    def test_file_not_found(self):
        assert classify_error_type("ENOENT: no such file") == "file_not_found"
        assert classify_error_type("Path not found") == "file_not_found"

    def test_syntax_error(self):
        assert classify_error_type("SyntaxError on line 5") == "syntax_error"

    def test_timeout(self):
        assert classify_error_type("ETIMEDOUT after 30s") == "timeout"

    def test_import_error(self):
        assert classify_error_type("ImportError: No module named foo") == "import_error"
        assert classify_error_type("ModuleNotFoundError: bar") == "import_error"

    def test_unknown_falls_back_to_tool_error(self):
        assert classify_error_type("something weird happened") == "tool_error"

    def test_empty_returns_tool_error(self):
        assert classify_error_type("") == "tool_error"
        assert classify_error_type(None) == "tool_error"


@pytest.fixture
def errors_session(tmp_path):
    """Session with three failed tool calls of distinct error types."""
    jsonl = tmp_path / "errors.jsonl"

    def _bash_call(uid, parent_uid, ts, tool_use_id, command, request_id):
        return {
            "type": "assistant", "uuid": uid, "parentUuid": parent_uid,
            "sessionId": "err-s", "timestamp": ts, "requestId": request_id,
            "message": {"role": "assistant", "model": "claude-opus-4-7",
                        "content": [{"type": "tool_use", "id": tool_use_id,
                                     "name": "Bash",
                                     "input": {"command": command}}]},
        }

    def _bash_error(uid, parent_uid, ts, tool_use_id, content):
        return {
            "type": "user", "uuid": uid, "parentUuid": parent_uid,
            "sessionId": "err-s", "timestamp": ts,
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_use_id,
                 "content": content, "is_error": True},
            ]},
            "toolUseResult": {"stderr": content, "stdout": "",
                              "interrupted": False, "exitCode": 1},
        }

    lines = [
        {"type": "user", "uuid": "u1", "sessionId": "err-s",
         "timestamp": "2026-04-19T10:00:00Z", "cwd": "/p",
         "gitBranch": "main", "version": "2.1.114",
         "message": {"role": "user", "content": "do things"}},
        _bash_call("a1", "u1", "2026-04-19T10:00:01Z", "tu_perm",
                   "cat /root/file", "r1"),
        _bash_error("u2", "a1", "2026-04-19T10:00:02Z", "tu_perm",
                    "cat: /root/file: Permission denied"),
        _bash_call("a2", "u2", "2026-04-19T10:00:03Z", "tu_404",
                   "cat /nope", "r2"),
        _bash_error("u3", "a2", "2026-04-19T10:00:04Z", "tu_404",
                    "cat: /nope: No such file or directory"),
        _bash_call("a3", "u3", "2026-04-19T10:00:05Z", "tu_ok",
                   "echo hello", "r3"),
        # Successful (is_error=False) -- should NOT appear in fact_errors
        {"type": "user", "uuid": "u4", "parentUuid": "a3",
         "sessionId": "err-s", "timestamp": "2026-04-19T10:00:06Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_ok",
              "content": "hello", "is_error": False},
         ]},
         "toolUseResult": {"stdout": "hello", "stderr": "",
                           "interrupted": False, "exitCode": 0}},
        _bash_call("a4", "u4", "2026-04-19T10:00:07Z", "tu_syn",
                   "python -c 'def'", "r4"),
        _bash_error("u5", "a4", "2026-04-19T10:00:08Z", "tu_syn",
                    "SyntaxError: invalid syntax"),
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


def _populate(conn, jsonl_path, tmp_path):
    run = EtlRun.start(conn, source_path=str(jsonl_path))
    log_path, _ = write_session_to_parquet(
        jsonl_path, tmp_path / "lake",
        etl_run_id=run.etl_run_id, project_slug="test-project",
    )
    load_session_to_staging(conn, log_path)
    conn.execute(
        """
        INSERT INTO dim_tool (tool_key, tool_name, tool_category)
        SELECT DISTINCT md5(tool_name), tool_name, 'unknown' FROM (
            SELECT json_extract_string(b.block, '$.name') AS tool_name
            FROM stg_log_entries sle, LATERAL (
                SELECT unnest(json_extract(sle.message_json, '$.content')::JSON[]) AS block
            ) b
            WHERE sle.type = 'assistant'
              AND json_type(sle.message_json, '$.content') = 'ARRAY'
              AND json_extract_string(b.block, '$.type') = 'tool_use'
        ) WHERE tool_name IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM dim_tool dt WHERE dt.tool_key = md5(tool_name))
        """
    )
    populate_fact_messages(conn, run=run)
    populate_fact_tool_uses(conn, run=run)
    populate_fact_tool_results(conn, run=run)
    populate_fact_errors(conn, run=run)
    return run


class TestFactErrors:
    def test_one_row_per_failed_tool_call(
        self, conn, errors_session, tmp_path
    ):
        _populate(conn, errors_session, tmp_path)
        # 3 errors out of 4 tool calls; the is_error=False one is excluded.
        n = conn.execute("SELECT COUNT(*) FROM fact_errors").fetchone()[0]
        assert n == 3

    def test_error_type_classified(self, conn, errors_session, tmp_path):
        _populate(conn, errors_session, tmp_path)
        rows = conn.execute(
            """
            SELECT tool_use_id, error_type FROM fact_errors
            ORDER BY tool_use_id
            """
        ).fetchall()
        by_id = dict(rows)
        assert by_id["tu_perm"] == "permission_denied"
        assert by_id["tu_404"] == "file_not_found"
        assert by_id["tu_syn"] == "syntax_error"

    def test_error_message_captured(self, conn, errors_session, tmp_path):
        _populate(conn, errors_session, tmp_path)
        row = conn.execute(
            "SELECT error_message FROM fact_errors WHERE tool_use_id = 'tu_perm'"
        ).fetchone()
        assert "Permission denied" in row[0]

    def test_links_to_dim_tool(self, conn, errors_session, tmp_path):
        _populate(conn, errors_session, tmp_path)
        rows = conn.execute(
            """
            SELECT fe.tool_use_id, dt.tool_name
            FROM fact_errors fe JOIN dim_tool dt USING (tool_key)
            ORDER BY fe.tool_use_id
            """
        ).fetchall()
        for _tool_use_id, tool_name in rows:
            assert tool_name == "Bash"

    def test_lineage_block_populated(self, conn, errors_session, tmp_path):
        _populate(conn, errors_session, tmp_path)
        row = conn.execute(
            """
            SELECT created_at, last_updated_at, etl_run_id, record_source,
                   hash_diff, is_deleted
            FROM fact_errors LIMIT 1
            """
        ).fetchone()
        assert row[0] is not None
        assert row[1] is not None
        assert row[2] is not None
        assert row[3] == "claude_code_jsonl"
        assert row[4] is not None
        assert row[5] is False

    def test_idempotent_reetl(self, conn, errors_session, tmp_path):
        _populate(conn, errors_session, tmp_path)
        first = conn.execute(
            "SELECT last_updated_at FROM fact_errors ORDER BY tool_use_id"
        ).fetchall()
        _populate(conn, errors_session, tmp_path)
        second = conn.execute(
            "SELECT last_updated_at FROM fact_errors ORDER BY tool_use_id"
        ).fetchall()
        assert first == second
