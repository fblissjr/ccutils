"""Tests for fact_session_summary (Phase C5b).

Session-level aggregate over all the v0.15 entry-type facts. Replaces the
legacy fact_session_summary that joined to the old fact_messages /
fact_tool_calls shapes.

Note on Kimball "facts don't join to facts": the aggregation happens IN
the populator (joining facts at ETL time to derive per-session rollups),
not in queries. Consumers get one row per session with no joins required
for common analytics -- so the rule is honored at the query layer.
"""

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.entry_type_facts import (
    populate_fact_attachments,
    populate_fact_file_history_snapshots,
    populate_fact_meta_events,
    populate_fact_progress_events,
    populate_fact_system_events,
)
from ccutils.etl.fact_messages import populate_fact_messages
from ccutils.etl.fact_session_summary import populate_fact_session_summary
from ccutils.etl.fact_token_usage import populate_fact_token_usage
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


def _populate_everything(conn, run):
    """Run every fact populator so the summary has something to roll up."""
    populate_fact_messages(conn, run=run)
    populate_fact_tool_uses(conn, run=run)
    populate_fact_tool_results(conn, run=run)
    populate_fact_token_usage(conn, run=run)
    populate_fact_attachments(conn, run=run)
    populate_fact_progress_events(conn, run=run)
    populate_fact_system_events(conn, run=run)
    populate_fact_meta_events(conn, run=run)
    populate_fact_file_history_snapshots(conn, run=run)
    populate_fact_session_summary(conn, run=run)


@pytest.fixture
def rich_session(tmp_path):
    """A session that touches every v0.15 fact so summary aggregates can
    be sanity-checked end-to-end."""
    jsonl = tmp_path / "rich.jsonl"
    lines = [
        # 1. User
        {"type": "user", "uuid": "u1", "sessionId": "rich-s",
         "timestamp": "2026-04-19T10:00:00Z", "cwd": "/p",
         "permissionMode": "default",
         "message": {"role": "user", "content": "do stuff"}},
        # 2. Assistant with usage + tool_use (Bash)
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "sessionId": "rich-s", "timestamp": "2026-04-19T10:00:01Z",
         "requestId": "req_1",
         "message": {"role": "assistant", "model": "claude-opus-4-7", "content": [
             {"type": "text", "text": "ok"},
             {"type": "tool_use", "id": "tu_bash", "name": "Bash",
              "input": {"command": "ls"}},
         ],
         "stop_reason": "tool_use",
         "usage": {
             "input_tokens": 10, "output_tokens": 5,
             "cache_creation_input_tokens": 100,
             "cache_read_input_tokens": 500,
             "cache_creation": {
                 "ephemeral_5m_input_tokens": 80,
                 "ephemeral_1h_input_tokens": 20,
             },
             "service_tier": "standard",
         }}},
        # 3. User with tool_result (success)
        {"type": "user", "uuid": "u2", "parentUuid": "a1",
         "sessionId": "rich-s", "timestamp": "2026-04-19T10:00:02Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_bash", "content": "files"},
         ]},
         "toolUseResult": {"stdout": "files", "interrupted": False, "exitCode": 0}},
        # 4. Assistant with usage + another tool_use that ERRORS
        {"type": "assistant", "uuid": "a2", "parentUuid": "u2",
         "sessionId": "rich-s", "timestamp": "2026-04-19T10:00:03Z",
         "requestId": "req_2",
         "message": {"role": "assistant", "model": "claude-opus-4-7", "content": [
             {"type": "tool_use", "id": "tu_read", "name": "Read",
              "input": {"file_path": "/missing"}},
         ],
         "stop_reason": "tool_use",
         "usage": {
             "input_tokens": 3, "output_tokens": 4,
             "cache_read_input_tokens": 700,
             "service_tier": "standard",
         }}},
        # 5. User with errored tool_result
        {"type": "user", "uuid": "u3", "parentUuid": "a2",
         "sessionId": "rich-s", "timestamp": "2026-04-19T10:00:04Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_read",
              "content": "Error: not found", "is_error": True},
         ]},
         "toolUseResult": "Error: not found"},
        # 6. System: turn_duration
        {"type": "system", "subtype": "turn_duration", "uuid": "s1",
         "sessionId": "rich-s", "timestamp": "2026-04-19T10:00:05Z",
         "durationMs": 500, "messageCount": 3},
        # 7. System: api_error
        {"type": "system", "subtype": "api_error", "uuid": "s2",
         "sessionId": "rich-s", "timestamp": "2026-04-19T10:00:06Z",
         "error": {"status": 503}, "level": "error"},
        # 8. System: compact_boundary
        {"type": "system", "subtype": "compact_boundary", "uuid": "s3",
         "sessionId": "rich-s", "timestamp": "2026-04-19T10:00:07Z",
         "content": "compacted", "compactMetadata": {"trigger": "auto", "preTokens": 100000}},
        # 9. System: stop_hook_summary
        {"type": "system", "subtype": "stop_hook_summary", "uuid": "s4",
         "sessionId": "rich-s", "timestamp": "2026-04-19T10:00:08Z",
         "hookCount": 1, "preventedContinuation": True,
         "stopReason": "end_turn"},
        # 10. Attachment: diagnostics
        {"type": "attachment", "uuid": "att1", "sessionId": "rich-s",
         "timestamp": "2026-04-19T10:00:09Z",
         "attachment": {"type": "diagnostics", "files": []}},
        # 11. Attachment: hook_success
        {"type": "attachment", "uuid": "att2", "sessionId": "rich-s",
         "timestamp": "2026-04-19T10:00:10Z",
         "attachment": {"type": "hook_success", "hookName": "ruff",
                        "hookEvent": "PostToolUse", "toolUseID": "tu_bash"}},
        # 12. Progress (hook_progress)
        {"type": "progress", "uuid": "p1", "sessionId": "rich-s",
         "timestamp": "2026-04-19T10:00:11Z",
         "data": {"type": "hook_progress", "hookName": "ruff"}},
        # 13. Progress (bash_progress)
        {"type": "progress", "uuid": "p2", "sessionId": "rich-s",
         "timestamp": "2026-04-19T10:00:12Z",
         "data": {"type": "bash_progress", "stdout": "running"}},
        # 14. Permission-mode transition
        {"type": "permission-mode", "sessionId": "rich-s",
         "timestamp": "2026-04-19T10:00:13Z", "permissionMode": "acceptEdits"},
        {"type": "permission-mode", "sessionId": "rich-s",
         "timestamp": "2026-04-19T10:00:14Z", "permissionMode": "plan"},
        # 16. File history snapshot
        {"type": "file-history-snapshot", "uuid": "fh1", "sessionId": "rich-s",
         "timestamp": "2026-04-19T10:00:15Z",
         "messageId": "u1", "isSnapshotUpdate": False,
         "snapshot": {"messageId": "u1", "trackedFileBackups": {}}},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


class TestFactSessionSummaryDdl:
    def test_table_exists(self, conn):
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name='fact_session_summary'"
        ).fetchone() is not None

    def test_lineage_columns(self, conn):
        cols = {c[0] for c in conn.execute("DESCRIBE fact_session_summary").fetchall()}
        for required in (
            "created_at", "last_updated_at",
            "created_by_version_key", "last_updated_by_version_key",
            "etl_run_id", "record_source", "hash_diff",
            "is_deleted", "deleted_at",
            "session_id",
        ):
            assert required in cols, f"Missing: {required}"

    def test_message_rollups(self, conn):
        cols = {c[0] for c in conn.execute("DESCRIBE fact_session_summary").fetchall()}
        for required in (
            "total_messages", "user_messages", "assistant_messages",
            "total_thinking_blocks",
        ):
            assert required in cols, f"Missing: {required}"

    def test_token_rollups_split_by_tier(self, conn):
        """R11: cache_creation rollups split per pricing tier."""
        cols = {c[0] for c in conn.execute("DESCRIBE fact_session_summary").fetchall()}
        for required in (
            "total_input_tokens", "total_output_tokens",
            "total_cache_creation_5m_tokens",
            "total_cache_creation_1h_tokens",
            "total_cache_creation_total_tokens",
            "total_cache_read_tokens",
            "total_uncached_equivalent_tokens",
            "api_response_count",
        ):
            assert required in cols, f"Missing: {required}"

    def test_tool_rollups(self, conn):
        cols = {c[0] for c in conn.execute("DESCRIBE fact_session_summary").fetchall()}
        for required in (
            "total_tool_uses", "unique_tools_used",
            "total_tool_results", "total_tool_errors",
            "total_bash_interrupted",
        ):
            assert required in cols, f"Missing: {required}"

    def test_system_rollups(self, conn):
        cols = {c[0] for c in conn.execute("DESCRIBE fact_session_summary").fetchall()}
        for required in (
            "total_api_errors", "total_compactions",
            "total_turn_durations_ms", "turn_count",
            "total_stop_events", "total_prevented_continuations",
        ):
            assert required in cols, f"Missing: {required}"

    def test_other_rollups(self, conn):
        cols = {c[0] for c in conn.execute("DESCRIBE fact_session_summary").fetchall()}
        for required in (
            "total_progress_events", "total_hook_progress_events",
            "total_bash_progress_events",
            "total_attachments", "total_diagnostics", "total_hook_successes",
            "permission_mode_transition_count", "current_permission_mode",
            "total_file_history_snapshots",
        ):
            assert required in cols, f"Missing: {required}"


class TestPopulateFactSessionSummary:
    def test_one_row_per_session(self, conn, rich_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(rich_session))
        _stage(conn, rich_session, tmp_path, run)
        _populate_everything(conn, run)
        n = conn.execute("SELECT COUNT(*) FROM fact_session_summary").fetchone()[0]
        assert n == 1

    def test_message_counts(self, conn, rich_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(rich_session))
        _stage(conn, rich_session, tmp_path, run)
        _populate_everything(conn, run)
        row = conn.execute(
            "SELECT total_messages, user_messages, assistant_messages "
            "FROM fact_session_summary WHERE session_id = 'rich-s'"
        ).fetchone()
        # 3 user (u1, u2, u3) + 2 assistant (a1, a2) = 5
        assert row[0] == 5
        assert row[1] == 3
        assert row[2] == 2

    def test_token_rollups(self, conn, rich_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(rich_session))
        _stage(conn, rich_session, tmp_path, run)
        _populate_everything(conn, run)
        row = conn.execute(
            "SELECT total_input_tokens, total_output_tokens, "
            "total_cache_creation_5m_tokens, total_cache_creation_1h_tokens, "
            "total_cache_creation_total_tokens, total_cache_read_tokens, "
            "total_uncached_equivalent_tokens, api_response_count "
            "FROM fact_session_summary WHERE session_id = 'rich-s'"
        ).fetchone()
        # a1: 10 input, 5 output, 80 5m, 20 1h, 500 read
        # a2: 3 input, 4 output, 0 5m, 0 1h, 700 read
        assert row[0] == 13
        assert row[1] == 9
        assert row[2] == 80
        assert row[3] == 20
        assert row[4] == 100
        assert row[5] == 1200
        # total = 13 + 100 + 1200 = 1313
        assert row[6] == 1313
        assert row[7] == 2

    def test_tool_rollups(self, conn, rich_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(rich_session))
        _stage(conn, rich_session, tmp_path, run)
        _populate_everything(conn, run)
        row = conn.execute(
            "SELECT total_tool_uses, unique_tools_used, total_tool_results, "
            "total_tool_errors, total_bash_interrupted "
            "FROM fact_session_summary WHERE session_id = 'rich-s'"
        ).fetchone()
        # 2 tool_use blocks (tu_bash, tu_read)
        assert row[0] == 2
        # Bash + Read = 2 unique
        assert row[1] == 2
        # 2 tool_result events
        assert row[2] == 2
        # tu_read errored (is_error=True)
        assert row[3] == 1
        # No interrupted bash in this fixture
        assert row[4] == 0

    def test_system_rollups(self, conn, rich_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(rich_session))
        _stage(conn, rich_session, tmp_path, run)
        _populate_everything(conn, run)
        row = conn.execute(
            "SELECT total_api_errors, total_compactions, "
            "total_turn_durations_ms, turn_count, "
            "total_stop_events, total_prevented_continuations "
            "FROM fact_session_summary WHERE session_id = 'rich-s'"
        ).fetchone()
        assert row[0] == 1   # api_error
        assert row[1] == 1   # compact_boundary
        assert row[2] == 500 # one turn_duration of 500ms
        assert row[3] == 1
        assert row[4] == 1   # stop_hook_summary
        assert row[5] == 1   # preventedContinuation=true

    def test_other_rollups(self, conn, rich_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(rich_session))
        _stage(conn, rich_session, tmp_path, run)
        _populate_everything(conn, run)
        row = conn.execute(
            "SELECT total_progress_events, total_hook_progress_events, "
            "total_bash_progress_events, "
            "total_attachments, total_diagnostics, total_hook_successes, "
            "permission_mode_transition_count, current_permission_mode, "
            "total_file_history_snapshots "
            "FROM fact_session_summary WHERE session_id = 'rich-s'"
        ).fetchone()
        assert row[0] == 2  # 2 progress events
        assert row[1] == 1  # 1 hook_progress
        assert row[2] == 1  # 1 bash_progress
        assert row[3] == 2  # 2 attachments
        assert row[4] == 1  # diagnostics
        assert row[5] == 1  # hook_success
        assert row[6] == 2  # 2 permission-mode transitions
        # current_permission_mode is the LAST timestamp value -> "plan"
        assert row[7] == "plan"
        assert row[8] == 1  # 1 file_history_snapshot

    def test_session_duration(self, conn, rich_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(rich_session))
        _stage(conn, rich_session, tmp_path, run)
        _populate_everything(conn, run)
        row = conn.execute(
            "SELECT first_timestamp, last_timestamp, session_duration_seconds "
            "FROM fact_session_summary WHERE session_id = 'rich-s'"
        ).fetchone()
        assert row[0] is not None
        assert row[1] is not None
        assert row[2] > 0

    def test_lineage_stamped(self, conn, rich_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(rich_session))
        _stage(conn, rich_session, tmp_path, run)
        _populate_everything(conn, run)
        row = conn.execute(
            "SELECT etl_run_id, record_source, hash_diff "
            "FROM fact_session_summary WHERE session_id = 'rich-s'"
        ).fetchone()
        assert row[0] == run.etl_run_id
        assert row[1] == "claude_code_jsonl"
        assert len(row[2]) == 32


class TestIdempotency:
    def test_reetl_does_not_bump_last_updated_at(self, conn, rich_session, tmp_path):
        run1 = EtlRun.start(conn, source_path=str(rich_session))
        _stage(conn, rich_session, tmp_path, run1)
        _populate_everything(conn, run1)
        first = conn.execute(
            "SELECT session_id, last_updated_at FROM fact_session_summary"
        ).fetchall()
        run2 = EtlRun.start(conn, source_path=str(rich_session))
        _stage(conn, rich_session, tmp_path, run2)
        _populate_everything(conn, run2)
        second = conn.execute(
            "SELECT session_id, last_updated_at FROM fact_session_summary"
        ).fetchall()
        assert first == second
