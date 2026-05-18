"""Tests for the v0.15 fact_diagnostics populator (Phase D).

Grain: one row per LSP diagnostic emitted during a session. Derived
from fact_attachments where attachment_type='diagnostics'. A single
attachment entry carries diagnostics for multiple files; the populator
flattens to one row per individual diagnostic.

Carries the full v0.15 lineage block. natural_key is diagnostic_id =
md5(entry_id || file_uri || range_start_line || message_prefix). Re-ETL
on unchanged source is a no-op.
"""

from __future__ import annotations

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.entry_type_facts import populate_fact_attachments
from ccutils.etl.fact_diagnostics import populate_fact_diagnostics
from ccutils.etl.fact_file_operations import populate_dim_file
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
def diagnostics_session(tmp_path):
    """Session with two diagnostics attachments:
    - First attachment: 2 diagnostics in /work/a.py
    - Second attachment: 1 diagnostic in /work/b.py
    Total: 3 individual fact_diagnostics rows expected.
    """
    jsonl = tmp_path / "diag.jsonl"
    lines = [
        {"type": "user", "uuid": "u1", "sessionId": "diag-s",
         "timestamp": "2026-04-19T10:00:00Z", "cwd": "/work",
         "gitBranch": "main", "version": "2.1.114",
         "message": {"role": "user", "content": "type-check it"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "sessionId": "diag-s", "timestamp": "2026-04-19T10:00:01Z",
         "requestId": "r1",
         "message": {"role": "assistant", "model": "claude-opus-4-7",
                     "content": [{"type": "text", "text": "ok"}]}},
        # Attachment 1: 2 diagnostics in a.py
        {"type": "attachment",
         "uuid": "att-1", "sessionId": "diag-s",
         "timestamp": "2026-04-19T10:00:02Z",
         "attachment": {
             "type": "diagnostics",
             "files": [{
                 "uri": "/work/a.py",
                 "diagnostics": [
                     {
                         "message": "Object of type None is not subscriptable",
                         "severity": "Error",
                         "range": {"start": {"line": 10, "character": 4},
                                   "end": {"line": 10, "character": 12}},
                         "source": "Pyright",
                         "code": "reportOptionalSubscript",
                     },
                     {
                         "message": "Unused import",
                         "severity": "Warning",
                         "range": {"start": {"line": 1, "character": 0},
                                   "end": {"line": 1, "character": 10}},
                         "source": "Pyright",
                         "code": "reportUnusedImport",
                     },
                 ],
             }],
         }},
        # Attachment 2: 1 diagnostic in b.py
        {"type": "attachment",
         "uuid": "att-2", "sessionId": "diag-s",
         "timestamp": "2026-04-19T10:00:03Z",
         "attachment": {
             "type": "diagnostics",
             "files": [{
                 "uri": "/work/b.py",
                 "diagnostics": [
                     {
                         "message": "Syntax error",
                         "severity": "Error",
                         "range": {"start": {"line": 5, "character": 0},
                                   "end": {"line": 5, "character": 5}},
                         "source": "Pyright",
                         "code": "reportSyntaxError",
                     },
                 ],
             }],
         }},
        # Attachment 3: not diagnostics (should be ignored)
        {"type": "attachment",
         "uuid": "att-3", "sessionId": "diag-s",
         "timestamp": "2026-04-19T10:00:04Z",
         "attachment": {"type": "hook_success", "hookName": "x",
                        "durationMs": 5}},
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
    populate_fact_messages(conn, run=run)
    populate_fact_tool_uses(conn, run=run)
    populate_fact_tool_results(conn, run=run)
    populate_fact_attachments(conn, run=run)
    populate_dim_file(conn, run=run)
    populate_fact_diagnostics(conn, run=run)
    return run


class TestFactDiagnostics:
    def test_one_row_per_diagnostic(self, conn, diagnostics_session, tmp_path):
        _populate(conn, diagnostics_session, tmp_path)
        n = conn.execute("SELECT COUNT(*) FROM fact_diagnostics").fetchone()[0]
        assert n == 3

    def test_diagnostic_fields_extracted(
        self, conn, diagnostics_session, tmp_path
    ):
        _populate(conn, diagnostics_session, tmp_path)
        rows = conn.execute(
            """
            SELECT severity, source, code, message, range_start_line
            FROM fact_diagnostics
            ORDER BY range_start_line
            """
        ).fetchall()
        # First by start line: line 1 (unused import), then 5 (syntax), then 10
        sev = [r[0] for r in rows]
        assert "Error" in sev
        assert "Warning" in sev
        sources = {r[1] for r in rows}
        assert sources == {"Pyright"}
        codes = {r[2] for r in rows}
        assert "reportOptionalSubscript" in codes
        assert "reportUnusedImport" in codes
        assert "reportSyntaxError" in codes

    def test_diagnostics_linked_to_file_path(
        self, conn, diagnostics_session, tmp_path
    ):
        _populate(conn, diagnostics_session, tmp_path)
        rows = conn.execute(
            "SELECT file_path, COUNT(*) FROM fact_diagnostics GROUP BY file_path"
        ).fetchall()
        by_file = dict(rows)
        assert by_file["/work/a.py"] == 2
        assert by_file["/work/b.py"] == 1

    def test_lineage_block_populated(self, conn, diagnostics_session, tmp_path):
        _populate(conn, diagnostics_session, tmp_path)
        row = conn.execute(
            """
            SELECT created_at, last_updated_at, etl_run_id,
                   record_source, hash_diff, is_deleted
            FROM fact_diagnostics LIMIT 1
            """
        ).fetchone()
        assert row[0] is not None
        assert row[1] is not None
        assert row[2] is not None
        assert row[3] == "claude_code_jsonl"
        assert row[4] is not None
        assert row[5] is False

    def test_idempotent_reetl(self, conn, diagnostics_session, tmp_path):
        _populate(conn, diagnostics_session, tmp_path)
        first = conn.execute(
            "SELECT last_updated_at FROM fact_diagnostics ORDER BY 1"
        ).fetchall()
        _populate(conn, diagnostics_session, tmp_path)
        second = conn.execute(
            "SELECT last_updated_at FROM fact_diagnostics ORDER BY 1"
        ).fetchall()
        assert first == second

    def test_non_diagnostics_attachments_ignored(
        self, conn, diagnostics_session, tmp_path
    ):
        """The hook_success attachment in the fixture must not produce
        a fact_diagnostics row."""
        _populate(conn, diagnostics_session, tmp_path)
        # If non-diagnostics attachments leaked through, count would be 4.
        n = conn.execute("SELECT COUNT(*) FROM fact_diagnostics").fetchone()[0]
        assert n == 3
