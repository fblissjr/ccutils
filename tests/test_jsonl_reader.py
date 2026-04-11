"""Tests for the shared JSONL session reader."""

import json
import tempfile
from pathlib import Path

import pytest

from ccutils.parsers.jsonl_reader import (
    SessionAttachment,
    SessionEntry,
    SessionMetaEntry,
    SessionMetaHeader,
    SessionSystemEntry,
    iter_all_session_entries,
    iter_loglines,
    iter_session_entries,
    parse_session_header,
)


def _write_jsonl(lines: list[dict]) -> Path:
    """Write a list of dicts as JSONL to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False)
    for line in lines:
        tmp.write(json.dumps(line) + "\n")
    tmp.close()
    return Path(tmp.name)


class TestParseSessionHeader:
    def test_extracts_basic_fields(self):
        path = _write_jsonl(
            [
                {
                    "type": "user",
                    "sessionId": "abc123",
                    "cwd": "/home/user/project",
                    "gitBranch": "main",
                    "version": "1.0.0",
                    "slug": "my-slug",
                    "timestamp": "2025-01-01T00:00:00.000Z",
                    "message": {"content": "hello"},
                }
            ]
        )
        header = parse_session_header(path)
        assert header is not None
        assert header.session_id == path.stem
        assert header.cwd == "/home/user/project"
        assert header.git_branch == "main"
        assert header.version == "1.0.0"
        assert header.slug == "my-slug"
        assert header.is_agent is False
        assert header.agent_id is None

    def test_detects_agent_session(self):
        path = _write_jsonl(
            [
                {
                    "type": "user",
                    "sessionId": "parent-123",
                    "agentId": "agent-456",
                    "isSidechain": True,
                    "message": {"content": "hello"},
                }
            ]
        )
        header = parse_session_header(path)
        assert header is not None
        assert header.is_agent is True
        assert header.agent_id == "agent-456"
        assert header.parent_session_id == "parent-123"
        assert header.is_sidechain is True

    def test_returns_none_for_empty_file(self):
        path = _write_jsonl([])
        assert parse_session_header(path) is None

    def test_returns_none_for_missing_file(self, tmp_path):
        assert parse_session_header(tmp_path / "nonexistent.jsonl") is None

    def test_skips_malformed_lines(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False)
        tmp.write("not json\n")
        tmp.write(json.dumps({"type": "user", "cwd": "/test", "message": {}}) + "\n")
        tmp.close()
        header = parse_session_header(Path(tmp.name))
        assert header is not None
        assert header.cwd == "/test"


class TestIterSessionEntries:
    def test_yields_user_and_assistant(self):
        path = _write_jsonl(
            [
                {
                    "type": "user",
                    "uuid": "u1",
                    "timestamp": "2025-01-01T00:00:00.000Z",
                    "message": {"content": "hello"},
                },
                {
                    "type": "assistant",
                    "uuid": "a1",
                    "parentUuid": "u1",
                    "timestamp": "2025-01-01T00:00:01.000Z",
                    "message": {"content": "hi there", "model": "claude-opus-4-6"},
                },
            ]
        )
        entries = list(iter_session_entries(path))
        assert len(entries) == 2
        assert entries[0].entry_type == "user"
        assert entries[0].uuid == "u1"
        assert entries[0].content == "hello"
        assert entries[1].entry_type == "assistant"
        assert entries[1].model == "claude-opus-4-6"
        assert entries[1].parent_uuid == "u1"

    def test_skips_non_message_types(self):
        path = _write_jsonl(
            [
                {"type": "summary", "summary": "test summary"},
                {
                    "type": "user",
                    "uuid": "u1",
                    "message": {"content": "hello"},
                },
                {"type": "system", "data": "something"},
            ]
        )
        entries = list(iter_session_entries(path))
        assert len(entries) == 1
        assert entries[0].entry_type == "user"

    def test_yields_progress_records(self):
        path = _write_jsonl(
            [
                {
                    "type": "user",
                    "uuid": "u1",
                    "message": {"content": "hello"},
                },
                {
                    "type": "progress",
                    "parentToolUseID": "tool-123",
                    "data": {"agentId": "agent-456"},
                },
            ]
        )
        entries = list(iter_session_entries(path))
        assert len(entries) == 2
        assert entries[1].entry_type == "progress"
        assert entries[1].progress_parent_tool_id == "tool-123"
        assert entries[1].progress_agent_id == "agent-456"

    def test_parses_timestamps(self):
        path = _write_jsonl(
            [
                {
                    "type": "user",
                    "uuid": "u1",
                    "timestamp": "2025-06-15T14:30:00.000Z",
                    "message": {"content": "test"},
                },
            ]
        )
        entries = list(iter_session_entries(path))
        assert entries[0].timestamp is not None
        assert entries[0].timestamp.year == 2025
        assert entries[0].timestamp.month == 6
        assert entries[0].timestamp.hour == 14
        assert entries[0].timestamp_raw == "2025-06-15T14:30:00.000Z"

    def test_handles_missing_timestamp(self):
        path = _write_jsonl(
            [
                {
                    "type": "user",
                    "uuid": "u1",
                    "message": {"content": "test"},
                },
            ]
        )
        entries = list(iter_session_entries(path))
        assert entries[0].timestamp is None
        assert entries[0].timestamp_raw == ""

    def test_preserves_content_types(self):
        """Content can be a string or a list of blocks."""
        path = _write_jsonl(
            [
                {
                    "type": "user",
                    "uuid": "u1",
                    "message": {"content": "simple string"},
                },
                {
                    "type": "assistant",
                    "uuid": "a1",
                    "message": {
                        "content": [
                            {"type": "text", "text": "hello"},
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "Bash",
                                "input": {},
                            },
                        ]
                    },
                },
            ]
        )
        entries = list(iter_session_entries(path))
        assert entries[0].content == "simple string"
        assert isinstance(entries[1].content, list)
        assert len(entries[1].content) == 2

    def test_preserves_compact_summary_flag(self):
        path = _write_jsonl(
            [
                {
                    "type": "user",
                    "uuid": "u1",
                    "isCompactSummary": True,
                    "message": {"content": "continued"},
                },
            ]
        )
        entries = list(iter_session_entries(path))
        assert entries[0].is_compact_summary is True

    def test_preserves_is_meta_flag(self):
        path = _write_jsonl(
            [
                {
                    "type": "user",
                    "uuid": "u1",
                    "isMeta": True,
                    "message": {"content": "meta"},
                },
            ]
        )
        entries = list(iter_session_entries(path))
        assert entries[0].is_meta is True

    def test_preserves_sidechain_flag(self):
        path = _write_jsonl(
            [
                {
                    "type": "assistant",
                    "uuid": "a1",
                    "isSidechain": True,
                    "message": {"content": "from agent"},
                },
            ]
        )
        entries = list(iter_session_entries(path))
        assert entries[0].is_sidechain is True

    def test_raw_dict_accessible(self):
        path = _write_jsonl(
            [
                {
                    "type": "user",
                    "uuid": "u1",
                    "cwd": "/test",
                    "gitBranch": "main",
                    "message": {"content": "hello"},
                },
            ]
        )
        entries = list(iter_session_entries(path))
        assert entries[0].raw["cwd"] == "/test"
        assert entries[0].raw["gitBranch"] == "main"

    def test_skips_malformed_json_lines(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False)
        tmp.write("not json\n")
        tmp.write(
            json.dumps({"type": "user", "uuid": "u1", "message": {"content": "hello"}})
            + "\n"
        )
        tmp.write("{broken\n")
        tmp.close()
        entries = list(iter_session_entries(Path(tmp.name)))
        assert len(entries) == 1

    def test_empty_file(self):
        path = _write_jsonl([])
        entries = list(iter_session_entries(path))
        assert entries == []

    def test_progress_without_agent_id_skipped(self):
        """Progress records without both parentToolUseID and agentId are skipped."""
        path = _write_jsonl(
            [
                {
                    "type": "progress",
                    "parentToolUseID": "tool-123",
                    "data": {},  # no agentId
                },
                {
                    "type": "progress",
                    "data": {"agentId": "agent-456"},  # no parentToolUseID
                },
            ]
        )
        entries = list(iter_session_entries(path))
        assert len(entries) == 0


class TestIterLoglines:
    """Tests for iter_loglines() which converts pre-parsed logline dicts to SessionEntry."""

    def test_converts_user_and_assistant(self):
        loglines = [
            {
                "type": "user",
                "uuid": "u1",
                "timestamp": "2025-01-01T00:00:00.000Z",
                "message": {"content": [{"type": "text", "text": "hello"}]},
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "timestamp": "2025-01-01T00:00:01.000Z",
                "message": {
                    "content": [{"type": "text", "text": "hi"}],
                    "model": "claude-opus-4-6",
                },
            },
        ]
        entries = list(iter_loglines(loglines))
        assert len(entries) == 2
        assert entries[0].entry_type == "user"
        assert entries[0].uuid == "u1"
        assert entries[0].content == [{"type": "text", "text": "hello"}]
        assert entries[1].entry_type == "assistant"
        assert entries[1].model == "claude-opus-4-6"

    def test_parses_timestamps(self):
        loglines = [
            {
                "type": "user",
                "uuid": "u1",
                "timestamp": "2025-06-15T14:30:00.000Z",
                "message": {"content": "test"},
            },
        ]
        entries = list(iter_loglines(loglines))
        assert entries[0].timestamp is not None
        assert entries[0].timestamp.year == 2025
        assert entries[0].timestamp_raw == "2025-06-15T14:30:00.000Z"

    def test_skips_non_message_types(self):
        loglines = [
            {"type": "summary", "data": "ignored"},
            {"type": "user", "uuid": "u1", "message": {"content": "hello"}},
        ]
        entries = list(iter_loglines(loglines))
        assert len(entries) == 1

    def test_empty_list(self):
        assert list(iter_loglines([])) == []

    def test_sets_defaults_for_missing_fields(self):
        loglines = [
            {"type": "user", "message": {"content": "test"}},
        ]
        entries = list(iter_loglines(loglines))
        assert entries[0].uuid == ""
        assert entries[0].parent_uuid is None
        assert entries[0].model is None
        assert entries[0].is_sidechain is False

    def test_preserves_raw_dict(self):
        loglines = [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-123",
                "message": {"content": "test"},
            },
        ]
        entries = list(iter_loglines(loglines))
        assert entries[0].raw["sessionId"] == "sess-123"


class TestIterAllSessionEntries:
    """Tests for iter_all_session_entries() which yields all entry types."""

    def test_yields_user_and_assistant(self):
        """Still yields SessionEntry for user/assistant messages."""
        path = _write_jsonl(
            [
                {
                    "type": "user",
                    "uuid": "u1",
                    "timestamp": "2025-01-01T00:00:00.000Z",
                    "message": {"content": "hello"},
                },
                {
                    "type": "assistant",
                    "uuid": "a1",
                    "timestamp": "2025-01-01T00:00:01.000Z",
                    "message": {"content": "hi", "model": "claude-opus-4-6"},
                },
            ]
        )
        entries = list(iter_all_session_entries(path))
        session_entries = [e for e in entries if isinstance(e, SessionEntry)]
        assert len(session_entries) == 2
        assert session_entries[0].entry_type == "user"
        assert session_entries[1].entry_type == "assistant"

    def test_yields_system_entries(self):
        path = _write_jsonl(
            [
                {
                    "type": "system",
                    "subtype": "turn_duration",
                    "durationMs": 45000,
                    "messageCount": 12,
                    "timestamp": "2025-01-01T00:01:00.000Z",
                    "uuid": "s1",
                    "isSidechain": False,
                },
                {
                    "type": "system",
                    "subtype": "stop_hook_summary",
                    "stopReason": "end_turn",
                    "hookCount": 1,
                    "hasOutput": True,
                    "preventedContinuation": False,
                    "hookInfos": [{"command": "test.py", "durationMs": 40}],
                    "hookErrors": [],
                    "timestamp": "2025-01-01T00:02:00.000Z",
                    "uuid": "s2",
                    "isSidechain": False,
                },
            ]
        )
        entries = list(iter_all_session_entries(path))
        system_entries = [e for e in entries if isinstance(e, SessionSystemEntry)]
        assert len(system_entries) == 2
        assert system_entries[0].subtype == "turn_duration"
        assert system_entries[0].data["durationMs"] == 45000
        assert system_entries[0].data["messageCount"] == 12
        assert system_entries[1].subtype == "stop_hook_summary"
        assert system_entries[1].data["stopReason"] == "end_turn"
        assert system_entries[1].data["preventedContinuation"] is False

    def test_yields_attachment_entries(self):
        path = _write_jsonl(
            [
                {
                    "type": "attachment",
                    "attachment": {
                        "type": "diagnostics",
                        "files": [
                            {
                                "uri": "/dev/workspace/project/app.py",
                                "diagnostics": [
                                    {
                                        "message": "Undefined variable",
                                        "severity": "Error",
                                        "range": {
                                            "start": {"line": 10, "character": 5},
                                            "end": {"line": 10, "character": 15},
                                        },
                                        "source": "Pyright",
                                        "code": "reportUndefinedVariable",
                                    }
                                ],
                            }
                        ],
                        "isNew": True,
                    },
                    "timestamp": "2025-01-01T00:01:00.000Z",
                    "uuid": "att1",
                    "isSidechain": False,
                },
                {
                    "type": "attachment",
                    "attachment": {
                        "type": "hook_success",
                        "hookName": "PreToolUse:Bash",
                        "durationMs": 35,
                        "exitCode": 0,
                    },
                    "timestamp": "2025-01-01T00:01:01.000Z",
                    "uuid": "att2",
                    "isSidechain": False,
                },
            ]
        )
        entries = list(iter_all_session_entries(path))
        attachments = [e for e in entries if isinstance(e, SessionAttachment)]
        assert len(attachments) == 2
        assert attachments[0].attachment_type == "diagnostics"
        assert "files" in attachments[0].data
        assert attachments[1].attachment_type == "hook_success"

    def test_yields_meta_entries(self):
        path = _write_jsonl(
            [
                {
                    "type": "custom-title",
                    "customTitle": "my-session-title",
                    "sessionId": "sess-123",
                },
                {
                    "type": "agent-name",
                    "agentName": "code-reviewer",
                    "sessionId": "sess-123",
                },
                {
                    "type": "permission-mode",
                    "permissionMode": "plan",
                    "sessionId": "sess-123",
                },
            ]
        )
        entries = list(iter_all_session_entries(path))
        meta_entries = [e for e in entries if isinstance(e, SessionMetaEntry)]
        assert len(meta_entries) == 3
        assert meta_entries[0].meta_type == "custom-title"
        assert meta_entries[0].value == "my-session-title"
        assert meta_entries[1].meta_type == "agent-name"
        assert meta_entries[1].value == "code-reviewer"
        assert meta_entries[2].meta_type == "permission-mode"
        assert meta_entries[2].value == "plan"

    def test_extracts_entrypoint(self):
        path = _write_jsonl(
            [
                {
                    "type": "user",
                    "uuid": "u1",
                    "entrypoint": "cli",
                    "timestamp": "2025-01-01T00:00:00.000Z",
                    "message": {"content": "hello"},
                },
            ]
        )
        entries = list(iter_all_session_entries(path))
        session_entries = [e for e in entries if isinstance(e, SessionEntry)]
        assert session_entries[0].entrypoint == "cli"

    def test_extracts_usage_from_assistant(self):
        path = _write_jsonl(
            [
                {
                    "type": "assistant",
                    "uuid": "a1",
                    "timestamp": "2025-01-01T00:00:01.000Z",
                    "message": {
                        "content": "hi",
                        "model": "claude-opus-4-6",
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 50,
                            "cache_creation_input_tokens": 500,
                            "cache_read_input_tokens": 200,
                            "service_tier": "standard",
                            "speed": "standard",
                            "cache_creation": {
                                "ephemeral_1h_input_tokens": 500,
                                "ephemeral_5m_input_tokens": 0,
                            },
                        },
                    },
                },
            ]
        )
        entries = list(iter_all_session_entries(path))
        session_entries = [e for e in entries if isinstance(e, SessionEntry)]
        assert session_entries[0].usage is not None
        assert session_entries[0].usage["input_tokens"] == 100
        assert session_entries[0].usage["output_tokens"] == 50
        assert session_entries[0].usage["cache_read_input_tokens"] == 200

    def test_backward_compatible_iter_session_entries(self):
        """iter_session_entries still only yields user/assistant/progress."""
        path = _write_jsonl(
            [
                {
                    "type": "user",
                    "uuid": "u1",
                    "message": {"content": "hello"},
                },
                {
                    "type": "system",
                    "subtype": "turn_duration",
                    "durationMs": 1000,
                    "messageCount": 2,
                },
                {
                    "type": "attachment",
                    "attachment": {"type": "hook_success"},
                },
                {
                    "type": "custom-title",
                    "customTitle": "test",
                },
                {
                    "type": "assistant",
                    "uuid": "a1",
                    "message": {"content": "hi"},
                },
            ]
        )
        entries = list(iter_session_entries(path))
        assert len(entries) == 2
        assert entries[0].entry_type == "user"
        assert entries[1].entry_type == "assistant"

    def test_system_entry_timestamps(self):
        path = _write_jsonl(
            [
                {
                    "type": "system",
                    "subtype": "turn_duration",
                    "durationMs": 5000,
                    "messageCount": 3,
                    "timestamp": "2025-06-15T14:30:00.000Z",
                    "uuid": "s1",
                    "isSidechain": False,
                },
            ]
        )
        entries = list(iter_all_session_entries(path))
        system_entries = [e for e in entries if isinstance(e, SessionSystemEntry)]
        assert system_entries[0].timestamp is not None
        assert system_entries[0].timestamp.year == 2025

    def test_mixed_entry_types_ordering(self):
        """All entry types are yielded in file order."""
        path = _write_jsonl(
            [
                {
                    "type": "user",
                    "uuid": "u1",
                    "message": {"content": "hello"},
                },
                {
                    "type": "system",
                    "subtype": "turn_duration",
                    "durationMs": 1000,
                    "messageCount": 2,
                    "timestamp": "2025-01-01T00:00:01.000Z",
                    "uuid": "s1",
                    "isSidechain": False,
                },
                {
                    "type": "custom-title",
                    "customTitle": "test",
                },
                {
                    "type": "assistant",
                    "uuid": "a1",
                    "message": {"content": "hi"},
                },
            ]
        )
        entries = list(iter_all_session_entries(path))
        types = [type(e).__name__ for e in entries]
        assert types == [
            "SessionEntry",
            "SessionSystemEntry",
            "SessionMetaEntry",
            "SessionEntry",
        ]

    def test_header_extracts_entrypoint(self):
        path = _write_jsonl(
            [
                {
                    "type": "user",
                    "sessionId": "abc123",
                    "cwd": "/dev/workspace/project",
                    "entrypoint": "web",
                    "message": {"content": "hello"},
                },
            ]
        )
        header = parse_session_header(path)
        assert header is not None
        assert header.entrypoint == "web"
