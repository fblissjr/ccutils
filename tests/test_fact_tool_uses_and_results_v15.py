"""Tests for fact_tool_uses + fact_tool_results (Phase C chunk 3).

Splits the legacy fact_tool_calls into:
- fact_tool_uses: one row per tool_use content block
- fact_tool_results: one row per tool_result block + the entry-level
  toolUseResult structured payload, joined on tool_use_id.

Captures the 14k-entry-archive data gap (R1): structured per-tool result
payloads we previously dropped (Edit structuredPatch, Bash exit_code,
Read numLines, etc.).
"""

import json

import pytest

from ccutils import create_star_schema
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


def _stage(conn, jsonl_path, tmp_path, run):
    log_path, _ = write_session_to_parquet(
        jsonl_path, tmp_path / "lake",
        etl_run_id=run.etl_run_id, project_slug="test-project",
    )
    load_session_to_staging(conn, log_path)


@pytest.fixture
def tools_session(tmp_path):
    """Session with Bash + Edit + Read tool uses, each with structured
    toolUseResult and a tool_result block."""
    jsonl = tmp_path / "tools.jsonl"
    lines = [
        {"type": "user", "uuid": "u1", "sessionId": "tools-s",
         "timestamp": "2026-04-19T10:00:00Z",
         "message": {"role": "user", "content": "do stuff"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "sessionId": "tools-s", "timestamp": "2026-04-19T10:00:01Z",
         "message": {"role": "assistant", "model": "claude-opus-4-7", "content": [
             {"type": "text", "text": "ok"},
             {"type": "tool_use", "id": "tu_bash_1", "name": "Bash",
              "input": {"command": "ls -la"},
              "caller": {"type": "direct"}},
         ]}},
        # Bash result -- exit 0, not interrupted
        {"type": "user", "uuid": "u2", "parentUuid": "a1",
         "sessionId": "tools-s", "timestamp": "2026-04-19T10:00:02Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_bash_1",
              "content": "file1\nfile2", "is_error": False},
         ]},
         "toolUseResult": {
             "stdout": "file1\nfile2",
             "stderr": "",
             "interrupted": False,
             "isImage": False,
             "noOutputExpected": False,
             "exitCode": 0,
             "durationMs": 123,
         }},
        {"type": "assistant", "uuid": "a2", "parentUuid": "u2",
         "sessionId": "tools-s", "timestamp": "2026-04-19T10:00:03Z",
         "message": {"role": "assistant", "model": "claude-opus-4-7", "content": [
             {"type": "tool_use", "id": "tu_edit_1", "name": "Edit",
              "input": {"file_path": "/p/foo.py", "old_string": "old",
                        "new_string": "new"}},
         ]}},
        # Edit result -- structured patch
        {"type": "user", "uuid": "u3", "parentUuid": "a2",
         "sessionId": "tools-s", "timestamp": "2026-04-19T10:00:04Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_edit_1",
              "content": "edited"},
         ]},
         "toolUseResult": {
             "filePath": "/p/foo.py",
             "oldString": "old", "newString": "new",
             "originalFile": "old contents",
             "structuredPatch": [
                 {"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 1,
                  "lines": ["-old", "+new"]},
             ],
             "userModified": False, "replaceAll": False,
         }},
        {"type": "assistant", "uuid": "a3", "parentUuid": "u3",
         "sessionId": "tools-s", "timestamp": "2026-04-19T10:00:05Z",
         "message": {"role": "assistant", "model": "claude-opus-4-7", "content": [
             {"type": "tool_use", "id": "tu_read_1", "name": "Read",
              "input": {"file_path": "/p/foo.py"}},
         ]}},
        # Read result
        {"type": "user", "uuid": "u4", "parentUuid": "a3",
         "sessionId": "tools-s", "timestamp": "2026-04-19T10:00:06Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_read_1",
              "content": "  1  print('hi')"},
         ]},
         "toolUseResult": {
             "type": "text",
             "file": {
                 "filePath": "/p/foo.py",
                 "content": "print('hi')",
                 "numLines": 1,
                 "startLine": 1,
                 "totalLines": 42,
             }
         }},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def bash_interrupted_session(tmp_path):
    """Bash invocation that the user interrupted -- toolUseResult has
    interrupted=true."""
    jsonl = tmp_path / "bash_int.jsonl"
    lines = [
        {"type": "user", "uuid": "u1", "sessionId": "bi-s",
         "message": {"role": "user", "content": "run forever"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "sessionId": "bi-s",
         "message": {"role": "assistant", "model": "claude-opus-4-7", "content": [
             {"type": "tool_use", "id": "tu_x", "name": "Bash",
              "input": {"command": "sleep 9999"}},
         ]}},
        {"type": "user", "uuid": "u2", "parentUuid": "a1",
         "sessionId": "bi-s",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_x",
              "content": "interrupted", "is_error": True},
         ]},
         "toolUseResult": {
             "stdout": "", "stderr": "", "interrupted": True,
             "isImage": False, "noOutputExpected": False,
         }},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def error_string_result_session(tmp_path):
    """A tool that errored out -- toolUseResult is a plain `Error: ...` string
    rather than a structured dict. We must still create a fact_tool_results
    row with is_error=True and the error text in result_content_text."""
    jsonl = tmp_path / "err.jsonl"
    lines = [
        {"type": "user", "uuid": "u1", "sessionId": "err-s",
         "message": {"role": "user", "content": "do thing"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "sessionId": "err-s",
         "message": {"role": "assistant", "model": "claude-opus-4-7", "content": [
             {"type": "tool_use", "id": "tu_err", "name": "Read",
              "input": {"file_path": "/missing"}},
         ]}},
        {"type": "user", "uuid": "u2", "parentUuid": "a1",
         "sessionId": "err-s",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_err",
              "content": "Error: File not found", "is_error": True},
         ]},
         "toolUseResult": "Error: File not found"},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


class TestFactToolUsesDdl:
    def test_table_exists(self, conn):
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name='fact_tool_uses'"
        ).fetchone() is not None

    def test_lineage_columns(self, conn):
        cols = {c[0] for c in conn.execute("DESCRIBE fact_tool_uses").fetchall()}
        for required in (
            "created_at", "last_updated_at",
            "created_by_version_key", "last_updated_by_version_key",
            "etl_run_id", "record_source", "hash_diff",
            "is_deleted", "deleted_at",
            "entry_id", "message_id", "session_id", "tool_use_id",
        ):
            assert required in cols, f"Missing: {required}"

    def test_native_columns(self, conn):
        cols = {c[0] for c in conn.execute("DESCRIBE fact_tool_uses").fetchall()}
        for required in (
            "tool_name", "tool_key", "session_key", "project_key",
            "date_key", "time_key", "invoke_sequence_num",
            "caller_type", "input_json", "input_summary",
        ):
            assert required in cols, f"Missing: {required}"


class TestFactToolResultsDdl:
    def test_table_exists(self, conn):
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name='fact_tool_results'"
        ).fetchone() is not None

    def test_is_error_is_nullable_boolean(self, conn):
        """R16: tri-state preservation."""
        cols = dict(conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name='fact_tool_results'"
        ).fetchall())
        assert cols["is_error"] == "BOOLEAN"

    def test_per_tool_typed_columns(self, conn):
        cols = {c[0] for c in conn.execute("DESCRIBE fact_tool_results").fetchall()}
        for required in (
            # Bash
            "bash_exit_code", "bash_interrupted", "bash_stdout_bytes",
            "bash_duration_ms",
            # Edit
            "edit_user_modified", "edit_replace_all",
            "edit_structured_patch_json",
            # Read
            "read_num_lines", "read_total_lines", "read_file_path",
            # Write
            "write_type",
            # Glob
            "glob_num_files", "glob_truncated",
            # Grep
            "grep_mode", "grep_num_files",
            # WebFetch
            "webfetch_http_code", "webfetch_bytes",
            # Agent
            "agent_status", "agent_total_duration_ms",
            "agent_total_tokens", "agent_total_tool_use_count",
            "agent_was_interrupted", "agent_subagent_type",
            # Generic
            "result_payload_json", "result_content_text",
            "tool_use_id", "tool_name",
        ):
            assert required in cols, f"Missing: {required}"


class TestPopulateFactToolUses:
    def test_one_row_per_tool_use(self, conn, tools_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(tools_session))
        _stage(conn, tools_session, tmp_path, run)
        populate_fact_tool_uses(conn, run=run)
        ids = sorted(
            r[0] for r in conn.execute(
                "SELECT tool_use_id FROM fact_tool_uses"
            ).fetchall()
        )
        assert ids == ["tu_bash_1", "tu_edit_1", "tu_read_1"]

    def test_lineage_stamped(self, conn, tools_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(tools_session))
        _stage(conn, tools_session, tmp_path, run)
        populate_fact_tool_uses(conn, run=run)
        for row in conn.execute(
            "SELECT etl_run_id, record_source, hash_diff FROM fact_tool_uses"
        ).fetchall():
            assert row[0] == run.etl_run_id
            assert row[1] == "claude_code_jsonl"
            assert len(row[2]) == 32

    def test_tool_name_carried(self, conn, tools_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(tools_session))
        _stage(conn, tools_session, tmp_path, run)
        populate_fact_tool_uses(conn, run=run)
        rows = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT tool_use_id, tool_name FROM fact_tool_uses"
            ).fetchall()
        }
        assert rows["tu_bash_1"] == "Bash"
        assert rows["tu_edit_1"] == "Edit"
        assert rows["tu_read_1"] == "Read"

    def test_input_json_preserved(self, conn, tools_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(tools_session))
        _stage(conn, tools_session, tmp_path, run)
        populate_fact_tool_uses(conn, run=run)
        input_json = conn.execute(
            "SELECT input_json FROM fact_tool_uses WHERE tool_use_id = 'tu_bash_1'"
        ).fetchone()[0]
        parsed = json.loads(input_json)
        assert parsed["command"] == "ls -la"

    def test_caller_type_captured(self, conn, tools_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(tools_session))
        _stage(conn, tools_session, tmp_path, run)
        populate_fact_tool_uses(conn, run=run)
        row = conn.execute(
            "SELECT caller_type FROM fact_tool_uses WHERE tool_use_id = 'tu_bash_1'"
        ).fetchone()
        assert row[0] == "direct"

    def test_session_id_degenerate(self, conn, tools_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(tools_session))
        _stage(conn, tools_session, tmp_path, run)
        populate_fact_tool_uses(conn, run=run)
        n = conn.execute(
            "SELECT COUNT(*) FROM fact_tool_uses WHERE session_id = 'tools-s'"
        ).fetchone()[0]
        assert n == 3


class TestPopulateFactToolResults:
    def test_one_row_per_tool_result(self, conn, tools_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(tools_session))
        _stage(conn, tools_session, tmp_path, run)
        populate_fact_tool_results(conn, run=run)
        ids = sorted(
            r[0] for r in conn.execute(
                "SELECT tool_use_id FROM fact_tool_results"
            ).fetchall()
        )
        assert ids == ["tu_bash_1", "tu_edit_1", "tu_read_1"]

    def test_bash_structural_columns(self, conn, tools_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(tools_session))
        _stage(conn, tools_session, tmp_path, run)
        populate_fact_tool_results(conn, run=run)
        row = conn.execute(
            "SELECT bash_exit_code, bash_interrupted, bash_stdout_bytes, "
            "bash_duration_ms, is_error "
            "FROM fact_tool_results WHERE tool_use_id = 'tu_bash_1'"
        ).fetchone()
        assert row[0] == 0
        assert row[1] is False
        assert row[2] == len("file1\nfile2")
        assert row[3] == 123
        assert row[4] is False

    def test_edit_structured_patch_captured(self, conn, tools_session, tmp_path):
        """R1: structuredPatch is the highest-value Edit field previously dropped."""
        run = EtlRun.start(conn, source_path=str(tools_session))
        _stage(conn, tools_session, tmp_path, run)
        populate_fact_tool_results(conn, run=run)
        row = conn.execute(
            "SELECT edit_user_modified, edit_replace_all, edit_structured_patch_json "
            "FROM fact_tool_results WHERE tool_use_id = 'tu_edit_1'"
        ).fetchone()
        assert row[0] is False
        assert row[1] is False
        patch = json.loads(row[2])
        assert isinstance(patch, list)
        assert patch[0]["lines"] == ["-old", "+new"]

    def test_read_typed_columns(self, conn, tools_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(tools_session))
        _stage(conn, tools_session, tmp_path, run)
        populate_fact_tool_results(conn, run=run)
        row = conn.execute(
            "SELECT read_num_lines, read_total_lines, read_file_path "
            "FROM fact_tool_results WHERE tool_use_id = 'tu_read_1'"
        ).fetchone()
        assert row[0] == 1
        assert row[1] == 42
        assert row[2] == "/p/foo.py"

    def test_result_payload_json_preserved(self, conn, tools_session, tmp_path):
        """Full structured toolUseResult kept verbatim for tools we don't
        type-project columns for."""
        run = EtlRun.start(conn, source_path=str(tools_session))
        _stage(conn, tools_session, tmp_path, run)
        populate_fact_tool_results(conn, run=run)
        payload_json = conn.execute(
            "SELECT result_payload_json FROM fact_tool_results "
            "WHERE tool_use_id = 'tu_read_1'"
        ).fetchone()[0]
        payload = json.loads(payload_json)
        assert payload["file"]["totalLines"] == 42

    def test_bash_interrupted_captured(self, conn, bash_interrupted_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(bash_interrupted_session))
        _stage(conn, bash_interrupted_session, tmp_path, run)
        populate_fact_tool_results(conn, run=run)
        row = conn.execute(
            "SELECT bash_interrupted, is_error FROM fact_tool_results "
            "WHERE tool_use_id = 'tu_x'"
        ).fetchone()
        assert row[0] is True
        assert row[1] is True

    def test_error_string_result_captured(self, conn, error_string_result_session, tmp_path):
        """When toolUseResult is a plain `Error: ...` string (not a dict),
        the row still exists; structured columns are NULL; the error text
        lands in result_content_text."""
        run = EtlRun.start(conn, source_path=str(error_string_result_session))
        _stage(conn, error_string_result_session, tmp_path, run)
        populate_fact_tool_results(conn, run=run)
        row = conn.execute(
            "SELECT is_error, result_content_text, "
            "read_num_lines, result_payload_json "
            "FROM fact_tool_results WHERE tool_use_id = 'tu_err'"
        ).fetchone()
        assert row[0] is True
        assert "File not found" in row[1]
        assert row[2] is None
        # Plain-string payload: result_payload_json is the string wrapped
        assert "File not found" in row[3]


class TestIdempotency:
    def test_uses_reetl_is_no_op(self, conn, tools_session, tmp_path):
        run1 = EtlRun.start(conn, source_path=str(tools_session))
        _stage(conn, tools_session, tmp_path, run1)
        populate_fact_tool_uses(conn, run=run1)
        first_updates = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT tool_use_id, last_updated_at FROM fact_tool_uses"
            ).fetchall()
        }
        run2 = EtlRun.start(conn, source_path=str(tools_session))
        _stage(conn, tools_session, tmp_path, run2)
        populate_fact_tool_uses(conn, run=run2)
        second_updates = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT tool_use_id, last_updated_at FROM fact_tool_uses"
            ).fetchall()
        }
        assert first_updates == second_updates

    def test_results_reetl_is_no_op(self, conn, tools_session, tmp_path):
        run1 = EtlRun.start(conn, source_path=str(tools_session))
        _stage(conn, tools_session, tmp_path, run1)
        populate_fact_tool_results(conn, run=run1)
        first_updates = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT tool_use_id, last_updated_at FROM fact_tool_results"
            ).fetchall()
        }
        run2 = EtlRun.start(conn, source_path=str(tools_session))
        _stage(conn, tools_session, tmp_path, run2)
        populate_fact_tool_results(conn, run=run2)
        second_updates = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT tool_use_id, last_updated_at FROM fact_tool_results"
            ).fetchall()
        }
        assert first_updates == second_updates
