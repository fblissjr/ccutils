"""Round-trip tests for the typed reader (Phase A5).

Property: writing a JSONL session through Parquet and reading it back as
typed entries must produce equivalent typed entries to parsing the JSONL
directly. If the round-trip drops a field, Tier 1 is lossy and downstream
analytics will be silently wrong.
"""

import json
from pathlib import Path

import pytest

from ccutils.parsers.models import (
    AssistantEntry,
    AttachmentEntry,
    PermissionModeEntry,
    ProgressEntry,
    SystemEntry,
    UserEntry,
    iter_typed_entries,
)
from ccutils.parsers.parquet_writer import write_session_to_parquet
from ccutils.parsers.typed_reader import (
    iter_log_entries_raw,
    iter_log_entries_typed,
    read_session_meta,
)


@pytest.fixture
def sample_jsonl(tmp_path):
    """Comprehensive single-session fixture spanning every entry type."""
    jsonl = tmp_path / "session-rt.jsonl"
    lines = [
        {"type": "summary", "summary": "rt-test", "leafUuid": "x"},
        {"type": "user", "uuid": "u1", "sessionId": "session-rt", "timestamp": "2026-04-19T10:00:00Z",
         "cwd": "/p", "gitBranch": "main", "version": "2.1.114",
         "message": {"role": "user", "content": "go"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1", "sessionId": "session-rt",
         "timestamp": "2026-04-19T10:00:05Z", "requestId": "req_xyz",
         "message": {"role": "assistant", "content": [
             {"type": "text", "text": "ok"},
             {"type": "tool_use", "id": "toolu_001", "name": "Bash", "input": {"command": "ls"}},
         ], "stop_reason": "tool_use", "usage": {"input_tokens": 5, "output_tokens": 3}}},
        {"type": "user", "uuid": "u2", "parentUuid": "a1", "sessionId": "session-rt",
         "timestamp": "2026-04-19T10:00:06Z",
         "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_001", "content": "out"}]},
         "toolUseResult": {"stdout": "out", "interrupted": False, "exitCode": 0}},
        {"type": "system", "subtype": "turn_duration", "uuid": "s1",
         "sessionId": "session-rt", "timestamp": "2026-04-19T10:00:07Z",
         "durationMs": 999, "messageCount": 3},
        {"type": "attachment", "uuid": "att1", "sessionId": "session-rt",
         "timestamp": "2026-04-19T10:00:08Z",
         "attachment": {"type": "diagnostics", "files": []}},
        {"type": "permission-mode", "sessionId": "session-rt", "permissionMode": "auto"},
        {"type": "progress", "sessionId": "session-rt", "toolUseID": "toolu_001",
         "data": {"type": "bash_progress", "stdout": "running"}},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def written_parquet(sample_jsonl, tmp_path):
    """Write the sample JSONL through the writer and return the log_entries path."""
    log_path, meta_path = write_session_to_parquet(
        sample_jsonl, tmp_path / "lake", project_slug="rt",
    )
    return log_path, meta_path


class TestRoundTripEquivalence:
    """JSONL -> Parquet -> typed entries must be equivalent to JSONL -> typed entries."""

    def test_entry_count_matches(self, sample_jsonl, written_parquet):
        log_path, _ = written_parquet
        from_jsonl = list(iter_typed_entries(sample_jsonl))
        from_parquet = list(iter_log_entries_typed(log_path))
        assert len(from_jsonl) == len(from_parquet)

    def test_entry_types_match_in_order(self, sample_jsonl, written_parquet):
        log_path, _ = written_parquet
        from_jsonl_types = [e.type for e in iter_typed_entries(sample_jsonl)]
        from_parquet_types = [e.type for e in iter_log_entries_typed(log_path)]
        assert from_jsonl_types == from_parquet_types

    def test_user_entry_round_trips_with_tool_use_result(self, written_parquet):
        log_path, _ = written_parquet
        entries = list(iter_log_entries_typed(log_path))
        user_entries = [e for e in entries if isinstance(e, UserEntry) and e.tool_use_result is not None]
        assert len(user_entries) == 1
        assert user_entries[0].tool_use_result == {"stdout": "out", "interrupted": False, "exitCode": 0}

    def test_assistant_entry_preserves_message_with_stop_reason_and_usage(self, written_parquet):
        log_path, _ = written_parquet
        entries = list(iter_log_entries_typed(log_path))
        a1 = next(e for e in entries if isinstance(e, AssistantEntry) and e.uuid == "a1")
        assert a1.request_id == "req_xyz"
        assert a1.message["stop_reason"] == "tool_use"
        assert a1.message["usage"]["input_tokens"] == 5

    def test_system_entry_preserves_subtype_payload(self, written_parquet):
        log_path, _ = written_parquet
        entries = list(iter_log_entries_typed(log_path))
        s1 = next(e for e in entries if isinstance(e, SystemEntry))
        assert s1.subtype == "turn_duration"

    def test_attachment_entry_preserves_attachment(self, written_parquet):
        log_path, _ = written_parquet
        entries = list(iter_log_entries_typed(log_path))
        att = next(e for e in entries if isinstance(e, AttachmentEntry))
        assert att.attachment["type"] == "diagnostics"

    def test_permission_mode_entry_round_trips(self, written_parquet):
        log_path, _ = written_parquet
        entries = list(iter_log_entries_typed(log_path))
        pm = next(e for e in entries if isinstance(e, PermissionModeEntry))
        assert pm.permission_mode == "auto"

    def test_progress_entry_round_trips_data(self, written_parquet):
        log_path, _ = written_parquet
        entries = list(iter_log_entries_typed(log_path))
        p = next(e for e in entries if isinstance(e, ProgressEntry))
        assert p.data["type"] == "bash_progress"
        assert p.data["stdout"] == "running"


class TestRawReader:
    def test_raw_iterator_yields_lineage_columns(self, written_parquet):
        log_path, _ = written_parquet
        rows = list(iter_log_entries_raw(log_path))
        assert len(rows) > 0
        for row in rows:
            assert row["etl_run_id"]
            assert row["parser_version"]
            assert row["record_source"]
            assert row["entry_id"]


class TestSessionMetaReader:
    def test_reads_session_meta(self, written_parquet):
        _, meta_path = written_parquet
        meta = read_session_meta(meta_path)
        assert meta["session_id"] == "session-rt"
        assert meta["entry_count"] == 8
        assert meta["project_slug"] == "rt"
