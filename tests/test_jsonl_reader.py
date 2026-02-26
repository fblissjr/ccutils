"""Tests for the shared JSONL session reader."""

import json
import tempfile
from pathlib import Path

import pytest

from ccutils.parsers.jsonl_reader import (
    SessionEntry,
    SessionMetaHeader,
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
