"""Tests for the new entry-type facts (Phase C chunk 4).

Seven new facts capture entry types the legacy ETL drops entirely or
samples sparsely:

  fact_attachments         all 23 attachment subtypes
  fact_progress_events     all 6 progress data variants (51k hook_progress
                           per-archive previously dropped)
  fact_system_events       all 7 system subtypes (5 previously dropped)
  fact_meta_events         time-series for permission-mode, custom-title,
                           agent-name, last-prompt (legacy only kept LAST
                           value of each on dim_session)
  fact_file_history_snapshots  file-history-snapshot entries (1.6k/archive
                                previously dropped entirely)
  fact_queue_operations    queue-operation entries (latency analytics)
  fact_pr_links            pr-link entries (GitHub PR <-> session linkage)

All share the C2 lineage convention.
"""

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.entry_type_facts import (
    populate_fact_attachments,
    populate_fact_file_history_snapshots,
    populate_fact_meta_events,
    populate_fact_pr_links,
    populate_fact_progress_events,
    populate_fact_queue_operations,
    populate_fact_system_events,
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
    load_session_to_staging(conn, log_path, run=run)


@pytest.fixture
def attachment_session(tmp_path):
    jsonl = tmp_path / "att.jsonl"
    lines = [
        {"type": "attachment", "uuid": "att1", "sessionId": "att-s",
         "timestamp": "2026-04-19T10:00:00Z",
         "attachment": {"type": "diagnostics", "files": [
             {"uri": "/p/x.py", "diagnostics": [
                 {"message": "unused import", "severity": 4,
                  "range": {"start": {"line": 1, "character": 0},
                            "end": {"line": 1, "character": 12}}},
             ]},
         ]}},
        {"type": "attachment", "uuid": "att2", "sessionId": "att-s",
         "timestamp": "2026-04-19T10:00:01Z",
         "attachment": {"type": "hook_success", "hookName": "ruff",
                        "hookEvent": "PostToolUse", "toolUseID": "tu_x",
                        "stdout": "ok"}},
        {"type": "attachment", "uuid": "att3", "sessionId": "att-s",
         "timestamp": "2026-04-19T10:00:02Z",
         "attachment": {"type": "invoked_skills", "skills": [
             {"name": "x", "path": "/s/x.md", "content": "..."},
         ]}},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def progress_session(tmp_path):
    jsonl = tmp_path / "prog.jsonl"
    lines = [
        {"type": "progress", "uuid": "p1", "sessionId": "prog-s",
         "timestamp": "2026-04-19T10:00:00Z",
         "toolUseID": "tu_x", "parentToolUseID": "tu_x",
         "data": {"type": "hook_progress", "hookName": "ruff",
                  "hookEvent": "PreToolUse", "command": "hooks/ruff.py"}},
        {"type": "progress", "uuid": "p2", "sessionId": "prog-s",
         "timestamp": "2026-04-19T10:00:01Z",
         "toolUseID": "tu_y", "parentToolUseID": "tu_y",
         "data": {"type": "bash_progress", "stdout": "running..."}},
        {"type": "progress", "uuid": "p3", "sessionId": "prog-s",
         "timestamp": "2026-04-19T10:00:02Z",
         "data": {"type": "agent_progress", "agentId": "ag-1"}},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def system_session(tmp_path):
    jsonl = tmp_path / "sys.jsonl"
    lines = [
        {"type": "system", "subtype": "turn_duration", "uuid": "s1",
         "sessionId": "sys-s", "timestamp": "2026-04-19T10:00:00Z",
         "durationMs": 1234, "messageCount": 3},
        {"type": "system", "subtype": "stop_hook_summary", "uuid": "s2",
         "sessionId": "sys-s", "timestamp": "2026-04-19T10:00:01Z",
         "hookCount": 1, "preventedContinuation": False,
         "stopReason": "end_turn", "hasOutput": True, "level": "suggestion"},
        {"type": "system", "subtype": "api_error", "uuid": "s3",
         "sessionId": "sys-s", "timestamp": "2026-04-19T10:00:02Z",
         "error": {"status": 503, "type": "overloaded_error"},
         "retryInMs": 1000, "retryAttempt": 1, "maxRetries": 3, "level": "error"},
        {"type": "system", "subtype": "compact_boundary", "uuid": "s4",
         "sessionId": "sys-s", "timestamp": "2026-04-19T10:00:03Z",
         "content": "Conversation compacted",
         "compactMetadata": {"trigger": "auto", "preTokens": 100000},
         "logicalParentUuid": "u1"},
        {"type": "system", "subtype": "local_command", "uuid": "s5",
         "sessionId": "sys-s", "timestamp": "2026-04-19T10:00:04Z",
         "content": "<local-command-stdout>x</local-command-stdout>"},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def meta_session(tmp_path):
    """Three permission-mode transitions, one custom-title, one agent-name."""
    jsonl = tmp_path / "meta.jsonl"
    lines = [
        {"type": "permission-mode", "sessionId": "meta-s",
         "timestamp": "2026-04-19T10:00:00Z", "permissionMode": "default"},
        {"type": "custom-title", "sessionId": "meta-s",
         "timestamp": "2026-04-19T10:00:01Z", "customTitle": "the title"},
        {"type": "permission-mode", "sessionId": "meta-s",
         "timestamp": "2026-04-19T10:01:00Z", "permissionMode": "plan"},
        {"type": "permission-mode", "sessionId": "meta-s",
         "timestamp": "2026-04-19T10:02:00Z", "permissionMode": "acceptEdits"},
        {"type": "agent-name", "sessionId": "meta-s",
         "timestamp": "2026-04-19T10:00:30Z", "agentName": "Explore"},
        {"type": "last-prompt", "sessionId": "meta-s",
         "timestamp": "2026-04-19T10:02:30Z", "lastPrompt": "make it faster"},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def file_history_session(tmp_path):
    jsonl = tmp_path / "fh.jsonl"
    lines = [
        {"type": "file-history-snapshot", "uuid": "fh1", "sessionId": "fh-s",
         "timestamp": "2026-04-19T10:00:00Z",
         "messageId": "m1", "isSnapshotUpdate": False,
         "snapshot": {"messageId": "m1", "trackedFileBackups": {},
                      "timestamp": "2026-04-19T10:00:00Z"}},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def queue_op_session(tmp_path):
    jsonl = tmp_path / "qo.jsonl"
    lines = [
        {"type": "queue-operation", "uuid": "qo1", "sessionId": "qo-s",
         "timestamp": "2026-04-19T10:00:00Z",
         "operation": "enqueue", "content": "queued prompt"},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def pr_session(tmp_path):
    jsonl = tmp_path / "pr.jsonl"
    lines = [
        {"type": "pr-link", "uuid": "pr1", "sessionId": "pr-s",
         "timestamp": "2026-04-19T10:00:00Z",
         "prNumber": 42, "prUrl": "https://github.com/o/r/pull/42",
         "prRepository": "o/r"},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


# --- DDL tests ---


class TestNewFactDdl:
    """Every new fact must have the standard lineage block + degenerate dims."""

    REQUIRED_LINEAGE = (
        "created_at", "last_updated_at",
        "created_by_version_key", "last_updated_by_version_key",
        "etl_run_id", "record_source", "hash_diff",
        "is_deleted", "deleted_at",
        "entry_id", "session_id",
    )

    @pytest.mark.parametrize("table", [
        "fact_attachments", "fact_progress_events", "fact_system_events",
        "fact_meta_events", "fact_file_history_snapshots",
        "fact_queue_operations", "fact_pr_links",
    ])
    def test_table_exists_with_lineage(self, conn, table):
        result = conn.execute(
            f"SELECT name FROM sqlite_master WHERE name='{table}'"
        ).fetchone()
        assert result is not None, f"Table missing: {table}"
        cols = {c[0] for c in conn.execute(f"DESCRIBE {table}").fetchall()}
        for required in self.REQUIRED_LINEAGE:
            assert required in cols, f"{table} missing {required}"


# --- Populator tests ---


class TestFactAttachments:
    def test_one_row_per_attachment_entry(self, conn, attachment_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(attachment_session))
        _stage(conn, attachment_session, tmp_path, run)
        populate_fact_attachments(conn, run=run)
        n = conn.execute("SELECT COUNT(*) FROM fact_attachments").fetchone()[0]
        assert n == 3

    def test_attachment_type_carried(self, conn, attachment_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(attachment_session))
        _stage(conn, attachment_session, tmp_path, run)
        populate_fact_attachments(conn, run=run)
        types = sorted(
            r[0] for r in conn.execute(
                "SELECT attachment_type FROM fact_attachments"
            ).fetchall()
        )
        assert types == ["diagnostics", "hook_success", "invoked_skills"]

    def test_attachment_payload_preserved(self, conn, attachment_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(attachment_session))
        _stage(conn, attachment_session, tmp_path, run)
        populate_fact_attachments(conn, run=run)
        payload = conn.execute(
            "SELECT attachment_json FROM fact_attachments WHERE entry_id IN "
            "(SELECT entry_id FROM fact_attachments WHERE attachment_type = 'hook_success')"
        ).fetchone()[0]
        parsed = json.loads(payload)
        assert parsed["hookName"] == "ruff"


class TestFactProgressEvents:
    def test_all_three_progress_variants_loaded(self, conn, progress_session, tmp_path):
        """The legacy ETL only kept agent_progress; we now keep all 6."""
        run = EtlRun.start(conn, source_path=str(progress_session))
        _stage(conn, progress_session, tmp_path, run)
        populate_fact_progress_events(conn, run=run)
        types = sorted(
            r[0] for r in conn.execute(
                "SELECT data_type FROM fact_progress_events"
            ).fetchall()
        )
        assert types == ["agent_progress", "bash_progress", "hook_progress"]

    def test_hook_event_extracted(self, conn, progress_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(progress_session))
        _stage(conn, progress_session, tmp_path, run)
        populate_fact_progress_events(conn, run=run)
        row = conn.execute(
            "SELECT hook_name, hook_event "
            "FROM fact_progress_events WHERE data_type = 'hook_progress'"
        ).fetchone()
        assert row[0] == "ruff"
        assert row[1] == "PreToolUse"


class TestFactSystemEvents:
    def test_all_five_subtypes_loaded(self, conn, system_session, tmp_path):
        """Legacy ETL only kept turn_duration + stop_hook_summary;
        we now keep all 5 distinct subtypes in the fixture."""
        run = EtlRun.start(conn, source_path=str(system_session))
        _stage(conn, system_session, tmp_path, run)
        populate_fact_system_events(conn, run=run)
        subtypes = sorted(
            r[0] for r in conn.execute(
                "SELECT subtype FROM fact_system_events"
            ).fetchall()
        )
        assert subtypes == [
            "api_error", "compact_boundary", "local_command",
            "stop_hook_summary", "turn_duration",
        ]

    def test_typed_columns_per_subtype(self, conn, system_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(system_session))
        _stage(conn, system_session, tmp_path, run)
        populate_fact_system_events(conn, run=run)
        # turn_duration: durationMs + messageCount
        td = conn.execute(
            "SELECT duration_ms, message_count "
            "FROM fact_system_events WHERE subtype = 'turn_duration'"
        ).fetchone()
        assert td[0] == 1234
        assert td[1] == 3
        # api_error: retry fields
        ae = conn.execute(
            "SELECT retry_in_ms, retry_attempt, max_retries "
            "FROM fact_system_events WHERE subtype = 'api_error'"
        ).fetchone()
        assert ae[0] == 1000.0
        assert ae[1] == 1
        assert ae[2] == 3
        # compact_boundary: trigger + preTokens
        cb = conn.execute(
            "SELECT compact_trigger, compact_pre_tokens "
            "FROM fact_system_events WHERE subtype = 'compact_boundary'"
        ).fetchone()
        assert cb[0] == "auto"
        assert cb[1] == 100000


class TestFactMetaEvents:
    def test_all_meta_entries_become_rows(self, conn, meta_session, tmp_path):
        """Time series: every permission-mode toggle is its own row, NOT
        just the last value on dim_session."""
        run = EtlRun.start(conn, source_path=str(meta_session))
        _stage(conn, meta_session, tmp_path, run)
        populate_fact_meta_events(conn, run=run)
        n = conn.execute("SELECT COUNT(*) FROM fact_meta_events").fetchone()[0]
        assert n == 6

    def test_permission_mode_values_recorded_in_order(self, conn, meta_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(meta_session))
        _stage(conn, meta_session, tmp_path, run)
        populate_fact_meta_events(conn, run=run)
        rows = conn.execute(
            "SELECT meta_value FROM fact_meta_events "
            "WHERE meta_type = 'permission-mode' "
            "ORDER BY timestamp"
        ).fetchall()
        assert [r[0] for r in rows] == ["default", "plan", "acceptEdits"]


class TestFactFileHistorySnapshots:
    def test_row_created(self, conn, file_history_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(file_history_session))
        _stage(conn, file_history_session, tmp_path, run)
        populate_fact_file_history_snapshots(conn, run=run)
        n = conn.execute("SELECT COUNT(*) FROM fact_file_history_snapshots").fetchone()[0]
        assert n == 1

    def test_snapshot_payload_preserved(self, conn, file_history_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(file_history_session))
        _stage(conn, file_history_session, tmp_path, run)
        populate_fact_file_history_snapshots(conn, run=run)
        row = conn.execute(
            "SELECT message_id_link, is_snapshot_update, snapshot_json "
            "FROM fact_file_history_snapshots"
        ).fetchone()
        assert row[0] == "m1"
        assert row[1] is False
        snapshot = json.loads(row[2])
        assert snapshot["messageId"] == "m1"


class TestFactQueueOperations:
    def test_row_created(self, conn, queue_op_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(queue_op_session))
        _stage(conn, queue_op_session, tmp_path, run)
        populate_fact_queue_operations(conn, run=run)
        row = conn.execute(
            "SELECT operation, content FROM fact_queue_operations"
        ).fetchone()
        assert row[0] == "enqueue"
        assert row[1] == "queued prompt"


class TestFactPrLinks:
    def test_row_created(self, conn, pr_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(pr_session))
        _stage(conn, pr_session, tmp_path, run)
        populate_fact_pr_links(conn, run=run)
        row = conn.execute(
            "SELECT pr_number, pr_url, pr_repository FROM fact_pr_links"
        ).fetchone()
        assert row[0] == 42
        assert row[1] == "https://github.com/o/r/pull/42"
        assert row[2] == "o/r"


class TestIdempotency:
    """All seven populators must be idempotent under re-ETL."""

    @pytest.mark.parametrize("fixture_name,populator_name", [
        ("attachment_session", "populate_fact_attachments"),
        ("progress_session", "populate_fact_progress_events"),
        ("system_session", "populate_fact_system_events"),
        ("meta_session", "populate_fact_meta_events"),
        ("file_history_session", "populate_fact_file_history_snapshots"),
        ("queue_op_session", "populate_fact_queue_operations"),
        ("pr_session", "populate_fact_pr_links"),
    ])
    def test_reetl_does_not_bump_last_updated_at(
        self, request, conn, tmp_path, fixture_name, populator_name,
    ):
        from ccutils.etl import entry_type_facts
        populator = getattr(entry_type_facts, populator_name)
        fixture = request.getfixturevalue(fixture_name)

        run1 = EtlRun.start(conn, source_path=str(fixture))
        _stage(conn, fixture, tmp_path, run1)
        populator(conn, run=run1)
        table = populator_name.replace("populate_", "")
        first = sorted(
            (r[0], r[1])
            for r in conn.execute(
                f"SELECT entry_id, last_updated_at FROM {table}"
            ).fetchall()
        )

        run2 = EtlRun.start(conn, source_path=str(fixture))
        _stage(conn, fixture, tmp_path, run2)
        populator(conn, run=run2)
        second = sorted(
            (r[0], r[1])
            for r in conn.execute(
                f"SELECT entry_id, last_updated_at FROM {table}"
            ).fetchall()
        )
        assert first == second
