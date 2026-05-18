"""Tests for the v0.15 bridge_session_file populator (Phase D).

Grain: one row per (session, file) the session touched. Aggregate over
fact_file_operations -- read/write/edit counts and the timestamp window
both come straight from there. Carries the v0.15 lineage block on every
row; idempotent re-builds drop-and-reload by natural key.
"""

from __future__ import annotations

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.bridge_session_file import populate_bridge_session_file
from ccutils.etl.fact_file_operations import (
    populate_dim_file,
    populate_fact_file_operations,
)
from ccutils.etl.fact_messages import populate_fact_messages
from ccutils.etl.fact_tool_calls import (
    populate_fact_tool_results,
    populate_fact_tool_uses,
)
from ccutils.etl.lineage import EtlRun
from ccutils.etl.staging import load_session_to_staging
from ccutils.parsers.parquet_writer import write_session_to_parquet


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


@pytest.fixture
def two_session_jsonls(tmp_path):
    """Two sessions touching overlapping files so the bridge is non-trivial.

    Session A: reads /work/a.py twice, writes /work/b.md once.
    Session B: edits /work/a.py once.
    """
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"

    def _read_pair(uid_prefix, parent_uuid, ts_offset, file_path):
        ts1 = f"2026-04-19T10:00:{ts_offset:02d}Z"
        ts2 = f"2026-04-19T10:00:{ts_offset+1:02d}Z"
        return [
            {"type": "assistant", "uuid": f"{uid_prefix}a",
             "parentUuid": parent_uuid, "sessionId": uid_prefix.split("-")[0],
             "timestamp": ts1, "requestId": f"r-{uid_prefix}",
             "message": {"role": "assistant", "model": "claude-opus-4-7",
                         "content": [{"type": "tool_use",
                                      "id": f"tu-{uid_prefix}",
                                      "name": "Read",
                                      "input": {"file_path": file_path}}]}},
            {"type": "user", "uuid": f"{uid_prefix}u",
             "parentUuid": f"{uid_prefix}a", "sessionId": uid_prefix.split("-")[0],
             "timestamp": ts2,
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": f"tu-{uid_prefix}",
                  "content": "data"},
             ]},
             "toolUseResult": {"type": "text",
                               "file": {"filePath": file_path, "content": "data",
                                        "numLines": 1, "startLine": 1,
                                        "totalLines": 1}}},
        ]

    # Session A
    a_lines = [
        {"type": "user", "uuid": "ustart-a", "sessionId": "sessA",
         "timestamp": "2026-04-19T09:59:59Z", "cwd": "/work",
         "gitBranch": "main", "version": "2.1.114",
         "message": {"role": "user", "content": "go"}},
    ]
    a_lines += _read_pair("sessA-r1", "ustart-a", 0, "/work/a.py")
    a_lines += _read_pair("sessA-r2", "sessA-r1u", 5, "/work/a.py")
    a_lines += [
        {"type": "assistant", "uuid": "sessA-w",
         "parentUuid": "sessA-r2u", "sessionId": "sessA",
         "timestamp": "2026-04-19T10:00:10Z", "requestId": "r-w",
         "message": {"role": "assistant", "model": "claude-opus-4-7",
                     "content": [{"type": "tool_use", "id": "tu-w",
                                  "name": "Write",
                                  "input": {"file_path": "/work/b.md",
                                            "content": "hello"}}]}},
        {"type": "user", "uuid": "sessA-wu", "parentUuid": "sessA-w",
         "sessionId": "sessA", "timestamp": "2026-04-19T10:00:11Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu-w",
              "content": "File created"},
         ]},
         "toolUseResult": {"type": "create", "filePath": "/work/b.md"}},
    ]
    a.write_text("\n".join(json.dumps(d) for d in a_lines))

    # Session B
    b_lines = [
        {"type": "user", "uuid": "ustart-b", "sessionId": "sessB",
         "timestamp": "2026-04-19T11:00:00Z", "cwd": "/work",
         "gitBranch": "main", "version": "2.1.114",
         "message": {"role": "user", "content": "go"}},
        {"type": "assistant", "uuid": "sessB-e",
         "parentUuid": "ustart-b", "sessionId": "sessB",
         "timestamp": "2026-04-19T11:00:01Z", "requestId": "r-e",
         "message": {"role": "assistant", "model": "claude-opus-4-7",
                     "content": [{"type": "tool_use", "id": "tu-e",
                                  "name": "Edit",
                                  "input": {"file_path": "/work/a.py",
                                            "old_string": "x",
                                            "new_string": "y"}}]}},
        {"type": "user", "uuid": "sessB-eu", "parentUuid": "sessB-e",
         "sessionId": "sessB", "timestamp": "2026-04-19T11:00:02Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu-e", "content": "ok"},
         ]},
         "toolUseResult": {"filePath": "/work/a.py", "userModified": False,
                           "replaceAll": False, "structuredPatch": []}},
    ]
    b.write_text("\n".join(json.dumps(d) for d in b_lines))
    return a, b


def _populate_session(conn, jsonl_path, tmp_path, project_name="test-project"):
    run = EtlRun.start(conn, source_path=str(jsonl_path))
    log_path, _ = write_session_to_parquet(
        jsonl_path, tmp_path / "lake",
        etl_run_id=run.etl_run_id, project_slug=project_name,
    )
    load_session_to_staging(conn, log_path)
    conn.execute(
        """
        INSERT INTO dim_session (session_key, session_id, project_key)
        SELECT DISTINCT md5(session_id), session_id, md5('/work')
        FROM stg_log_entries
        WHERE session_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM dim_session ds
              WHERE ds.session_key = md5(stg_log_entries.session_id)
          )
        """
    )
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
    populate_dim_file(conn, run=run)
    populate_fact_file_operations(conn, run=run)


class TestBridgeSessionFile:
    def test_aggregates_per_session_file_pair(
        self, conn, two_session_jsonls, tmp_path
    ):
        a, b = two_session_jsonls
        _populate_session(conn, a, tmp_path)
        _populate_session(conn, b, tmp_path)
        run = EtlRun.start(conn, source_path="bridge")
        populate_bridge_session_file(conn, run=run)
        # Session A touches 2 files (a.py, b.md). Session B touches 1 (a.py).
        # Bridge should have 3 rows total.
        n = conn.execute("SELECT COUNT(*) FROM bridge_session_file").fetchone()[0]
        assert n == 3

    def test_counts_per_operation_type(
        self, conn, two_session_jsonls, tmp_path
    ):
        a, b = two_session_jsonls
        _populate_session(conn, a, tmp_path)
        _populate_session(conn, b, tmp_path)
        run = EtlRun.start(conn, source_path="bridge")
        populate_bridge_session_file(conn, run=run)

        # Session A + /work/a.py: 2 reads, 0 writes, 0 edits
        row = conn.execute(
            """
            SELECT operation_count, read_count, write_count, edit_count
            FROM bridge_session_file bsf
            JOIN dim_file df USING (file_key)
            WHERE bsf.session_id = 'sessA' AND df.file_path = '/work/a.py'
            """
        ).fetchone()
        assert row == (2, 2, 0, 0)

        # Session B + /work/a.py: 0 reads, 0 writes, 1 edit
        row = conn.execute(
            """
            SELECT operation_count, read_count, write_count, edit_count
            FROM bridge_session_file bsf
            JOIN dim_file df USING (file_key)
            WHERE bsf.session_id = 'sessB' AND df.file_path = '/work/a.py'
            """
        ).fetchone()
        assert row == (1, 0, 0, 1)

    def test_lineage_block_populated(
        self, conn, two_session_jsonls, tmp_path
    ):
        a, _ = two_session_jsonls
        _populate_session(conn, a, tmp_path)
        run = EtlRun.start(conn, source_path="bridge")
        populate_bridge_session_file(conn, run=run)
        row = conn.execute(
            """
            SELECT created_at, last_updated_at, etl_run_id, record_source,
                   hash_diff, is_deleted
            FROM bridge_session_file LIMIT 1
            """
        ).fetchone()
        assert row[0] is not None
        assert row[1] is not None
        assert row[2] is not None
        assert row[3] == "claude_code_jsonl"
        assert row[4] is not None
        assert row[5] is False

    def test_idempotent_rebuild(self, conn, two_session_jsonls, tmp_path):
        """Bridge is recomputed from fact_file_operations on each run.
        Re-running on unchanged source must be a no-op."""
        a, b = two_session_jsonls
        _populate_session(conn, a, tmp_path)
        _populate_session(conn, b, tmp_path)
        run = EtlRun.start(conn, source_path="bridge")
        populate_bridge_session_file(conn, run=run)
        first = conn.execute(
            "SELECT last_updated_at FROM bridge_session_file ORDER BY 1"
        ).fetchall()
        populate_bridge_session_file(conn, run=run)
        second = conn.execute(
            "SELECT last_updated_at FROM bridge_session_file ORDER BY 1"
        ).fetchall()
        assert first == second
