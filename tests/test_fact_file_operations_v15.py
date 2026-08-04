"""Tests for the v0.15 fact_file_operations + dim_file populators (Phase D).

Grain: one row per file-touching tool call (Read / Write / Edit / MultiEdit
/ Glob / Grep / NotebookEdit / etc.). Derived from fact_tool_uses joined
to fact_tool_results -- both must be populated first.

dim_file is populated as a side effect: each distinct file_path observed
in tool inputs gets a dim_file row with parsed name/extension/directory
and a heuristic language label.

Both tables carry the v0.15 lineage block on every fact row. dim_file
stays minimal (matches dim_tool / dim_model pattern -- no lineage).
"""

from __future__ import annotations

import json

import pytest

from ccutils import create_star_schema
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
def file_op_session(tmp_path):
    """A session with one Read, one Write, one Edit, one Glob.

    Each tool_use is paired with its tool_result so fact_tool_results
    can populate too.
    """
    jsonl = tmp_path / "fileops.jsonl"
    lines = [
        {
            "type": "user", "uuid": "u1", "sessionId": "fileops-s",
            "timestamp": "2026-04-19T10:00:00Z",
            "cwd": "/work", "gitBranch": "main", "version": "2.1.114",
            "message": {"role": "user", "content": "do file work"},
        },
        # Read
        {
            "type": "assistant", "uuid": "a1", "parentUuid": "u1",
            "sessionId": "fileops-s", "timestamp": "2026-04-19T10:00:01Z",
            "requestId": "r1",
            "message": {"role": "assistant", "model": "claude-opus-4-7",
                        "content": [
                            {"type": "tool_use", "id": "tu_read",
                             "name": "Read", "input": {"file_path": "/work/a.py"}},
                        ]},
        },
        {
            "type": "user", "uuid": "u2", "parentUuid": "a1",
            "sessionId": "fileops-s", "timestamp": "2026-04-19T10:00:02Z",
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_read",
                 "content": "line1\nline2"},
            ]},
            "toolUseResult": {
                "type": "text", "file": {
                    "filePath": "/work/a.py", "content": "line1\nline2",
                    "numLines": 2, "startLine": 1, "totalLines": 2,
                },
            },
        },
        # Write
        {
            "type": "assistant", "uuid": "a2", "parentUuid": "u2",
            "sessionId": "fileops-s", "timestamp": "2026-04-19T10:00:03Z",
            "requestId": "r2",
            "message": {"role": "assistant", "model": "claude-opus-4-7",
                        "content": [
                            {"type": "tool_use", "id": "tu_write",
                             "name": "Write",
                             "input": {"file_path": "/work/b.md",
                                       "content": "# hi\nbody"}},
                        ]},
        },
        {
            "type": "user", "uuid": "u3", "parentUuid": "a2",
            "sessionId": "fileops-s", "timestamp": "2026-04-19T10:00:04Z",
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_write",
                 "content": "File created"},
            ]},
            "toolUseResult": {"type": "create", "filePath": "/work/b.md"},
        },
        # Edit
        {
            "type": "assistant", "uuid": "a3", "parentUuid": "u3",
            "sessionId": "fileops-s", "timestamp": "2026-04-19T10:00:05Z",
            "requestId": "r3",
            "message": {"role": "assistant", "model": "claude-opus-4-7",
                        "content": [
                            {"type": "tool_use", "id": "tu_edit",
                             "name": "Edit",
                             "input": {"file_path": "/work/a.py",
                                       "old_string": "line1",
                                       "new_string": "LINE1"}},
                        ]},
        },
        {
            "type": "user", "uuid": "u4", "parentUuid": "a3",
            "sessionId": "fileops-s", "timestamp": "2026-04-19T10:00:06Z",
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_edit",
                 "content": "ok"},
            ]},
            "toolUseResult": {"filePath": "/work/a.py",
                              "userModified": False, "replaceAll": False,
                              "structuredPatch": [{"oldStart": 1, "oldLines": 1,
                                                  "newStart": 1, "newLines": 1,
                                                  "lines": ["-line1", "+LINE1"]}]},
        },
        # Glob (no specific file_path but a pattern)
        {
            "type": "assistant", "uuid": "a4", "parentUuid": "u4",
            "sessionId": "fileops-s", "timestamp": "2026-04-19T10:00:07Z",
            "requestId": "r4",
            "message": {"role": "assistant", "model": "claude-opus-4-7",
                        "content": [
                            {"type": "tool_use", "id": "tu_glob",
                             "name": "Glob",
                             "input": {"pattern": "*.py"}},
                        ]},
        },
        {
            "type": "user", "uuid": "u5", "parentUuid": "a4",
            "sessionId": "fileops-s", "timestamp": "2026-04-19T10:00:08Z",
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_glob",
                 "content": "a.py"},
            ]},
            "toolUseResult": {"filenames": ["a.py"], "numFiles": 1,
                              "truncated": False},
        },
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


def _populate(conn, jsonl_path, tmp_path):
    """Run the populator chain through file ops."""
    run = EtlRun.start(conn, source_path=str(jsonl_path))
    log_path, _ = write_session_to_parquet(
        jsonl_path, tmp_path / "lake",
        etl_run_id=run.etl_run_id, project_slug="test-project",
    )
    load_session_to_staging(conn, log_path)
    # Minimal dims: dim_session + dim_tool referenced by fact_tool_uses.
    # We don't need to call _upsert_minimal_dimensions; the test exercises
    # just the file-ops populator path and the FKs are soft.
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
        """
    )
    populate_fact_messages(conn, run=run)
    populate_fact_tool_uses(conn, run=run)
    populate_fact_tool_results(conn, run=run)
    populate_dim_file(conn, run=run)
    populate_fact_file_operations(conn, run=run)
    return run


class TestDimFilePopulator:
    def test_distinct_file_paths_populated(self, conn, file_op_session, tmp_path):
        _populate(conn, file_op_session, tmp_path)
        rows = conn.execute(
            "SELECT file_path FROM dim_file ORDER BY file_path"
        ).fetchall()
        paths = [r[0] for r in rows]
        assert "/work/a.py" in paths
        assert "/work/b.md" in paths

    def test_file_name_extension_directory_parsed(
        self, conn, file_op_session, tmp_path
    ):
        _populate(conn, file_op_session, tmp_path)
        row = conn.execute(
            "SELECT file_name, file_extension, directory_path FROM dim_file "
            "WHERE file_path = '/work/a.py'"
        ).fetchone()
        assert row is not None
        assert row[0] == "a.py"
        assert row[1] == "py"
        assert row[2] == "/work"

    def test_language_inferred_from_extension(
        self, conn, file_op_session, tmp_path
    ):
        _populate(conn, file_op_session, tmp_path)
        py_lang = conn.execute(
            "SELECT language FROM dim_file WHERE file_path = '/work/a.py'"
        ).fetchone()[0]
        md_lang = conn.execute(
            "SELECT language FROM dim_file WHERE file_path = '/work/b.md'"
        ).fetchone()[0]
        assert py_lang == "python"
        assert md_lang == "markdown"

    def test_idempotent_no_duplicate_files(
        self, conn, file_op_session, tmp_path
    ):
        _populate(conn, file_op_session, tmp_path)
        _populate(conn, file_op_session, tmp_path)
        # Each distinct file_path should appear exactly once.
        rows = conn.execute(
            "SELECT file_path, COUNT(*) FROM dim_file GROUP BY file_path"
        ).fetchall()
        for path, count in rows:
            assert count == 1, f"duplicate dim_file row for {path}"


class TestFactFileOperationsPopulator:
    def test_one_row_per_file_touching_tool_use(
        self, conn, file_op_session, tmp_path
    ):
        _populate(conn, file_op_session, tmp_path)
        # Read + Write + Edit + Glob = 4 file operations.
        n = conn.execute("SELECT COUNT(*) FROM fact_file_operations").fetchone()[0]
        assert n == 4

    def test_operation_type_classified(self, conn, file_op_session, tmp_path):
        _populate(conn, file_op_session, tmp_path)
        rows = conn.execute(
            """
            SELECT dt.tool_name, ffo.operation_type
            FROM fact_file_operations ffo
            JOIN dim_tool dt USING (tool_key)
            ORDER BY ffo.timestamp
            """
        ).fetchall()
        by_tool = dict(rows)
        assert by_tool["Read"] == "read"
        assert by_tool["Write"] == "write"
        assert by_tool["Edit"] == "edit"
        assert by_tool["Glob"] == "list"

    def test_file_operations_link_to_dim_file(
        self, conn, file_op_session, tmp_path
    ):
        _populate(conn, file_op_session, tmp_path)
        # Every fact_file_operations row with a file_key must FK into dim_file.
        rows = conn.execute(
            """
            SELECT COUNT(*) FROM fact_file_operations ffo
            WHERE ffo.file_key IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM dim_file df WHERE df.file_key = ffo.file_key
              )
            """
        ).fetchone()
        assert rows[0] == 0

    def test_lineage_block_populated(self, conn, file_op_session, tmp_path):
        _populate(conn, file_op_session, tmp_path)
        row = conn.execute(
            """
            SELECT created_at, last_updated_at, created_by_version_key,
                   etl_run_id, record_source, hash_diff, is_deleted
            FROM fact_file_operations LIMIT 1
            """
        ).fetchone()
        assert row[0] is not None  # created_at
        assert row[1] is not None  # last_updated_at
        assert row[2] is not None  # version_key
        assert row[3] is not None  # etl_run_id
        assert row[4] == "claude_code_jsonl"
        assert row[5] is not None  # hash_diff
        assert row[6] is False  # is_deleted

    def test_idempotent_reetl(self, conn, file_op_session, tmp_path):
        _populate(conn, file_op_session, tmp_path)
        first = conn.execute(
            "SELECT last_updated_at FROM fact_file_operations ORDER BY 1"
        ).fetchall()
        _populate(conn, file_op_session, tmp_path)
        second = conn.execute(
            "SELECT last_updated_at FROM fact_file_operations ORDER BY 1"
        ).fetchall()
        assert first == second, "re-ETL on unchanged source must be a no-op"


class TestSoftDeletedResultsDoNotFanOut:
    """A soft-deleted fact_tool_results twin must not join into the inbound.

    The upgrade path creates exactly this state: `_repair_duplicate_natural_
    keys` soft-deletes duplicate tool_use_id rows at open, then the next
    batch run re-ETLs the session. The inbound here derives from
    fact_tool_uses JOIN fact_tool_results; without an is_deleted filter on
    the RESULTS side of the join, the repaired twin fans it out, the inbound
    carries a duplicate tool_use_id, and lineage_upsert kills the session.
    Observed on a real pre-R23 warehouse: 2 sessions failed exactly here
    (7 and 1 duplicate keys) after the open-time repair had run.
    """

    def test_repaired_twin_does_not_duplicate_the_inbound(
        self, conn, file_op_session, tmp_path
    ):
        _populate(conn, file_op_session, tmp_path)
        before = conn.execute(
            "SELECT COUNT(*) FROM fact_file_operations WHERE NOT is_deleted"
        ).fetchone()[0]

        # Plant the post-repair state: a second physical row for one
        # tool_use_id, soft-deleted, exactly as the repair leaves it.
        conn.execute(
            "INSERT INTO fact_tool_results SELECT * REPLACE ("
            "  'e_twin' AS entry_id, TRUE AS is_deleted,"
            "  current_timestamp AS deleted_at)"
            "FROM fact_tool_results ORDER BY tool_use_id LIMIT 1"
        )

        _populate(conn, file_op_session, tmp_path)  # must not raise

        after = conn.execute(
            "SELECT COUNT(*) FROM fact_file_operations WHERE NOT is_deleted"
        ).fetchone()[0]
        assert after == before
        dup = conn.execute(
            "SELECT COUNT(*) FROM (SELECT tool_use_id FROM fact_file_operations "
            "WHERE NOT is_deleted GROUP BY 1 HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        assert dup == 0
