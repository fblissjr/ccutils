"""Tests for JSON export functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from ccutils import (
    resolve_schema_format,
    export_sessions_to_json,
)


class TestResolveSchemaFormat:
    """Tests for schema/format resolution logic."""

    def test_simple_duckdb_inferred(self):
        """Test that duckdb format infers simple schema."""
        schema, fmt = resolve_schema_format("duckdb")
        assert schema == "simple"
        assert fmt == "duckdb"

    def test_star_duckdb_inferred(self):
        """Test that duckdb-star format infers star schema."""
        schema, fmt = resolve_schema_format("duckdb-star")
        assert schema == "star"
        assert fmt == "duckdb"

    def test_simple_json_inferred(self):
        """Test that json format infers simple schema."""
        schema, fmt = resolve_schema_format("json")
        assert schema == "simple"
        assert fmt == "json"

    def test_star_json_inferred(self):
        """Test that json-star format infers star schema."""
        schema, fmt = resolve_schema_format("json-star")
        assert schema == "star"
        assert fmt == "json"

    def test_html_infers_simple(self):
        """Test that html format infers simple schema."""
        schema, fmt = resolve_schema_format("html")
        assert schema == "simple"
        assert fmt == "html"


@pytest.fixture
def sample_session_file():
    """Create a sample JSONL session file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # User message
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-001",
                    "parentUuid": None,
                    "sessionId": "session-123",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "cwd": "/home/user/project",
                    "gitBranch": "main",
                    "version": "2.0.0",
                    "message": {
                        "role": "user",
                        "content": "Help me write a hello world program",
                    },
                }
            )
            + "\n"
        )
        # Assistant message with tool_use
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-001",
                    "parentUuid": "user-001",
                    "sessionId": "session-123",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-opus-4-5-20251101",
                        "content": [
                            {"type": "text", "text": "I'll create that for you."},
                            {
                                "type": "tool_use",
                                "id": "tool-001",
                                "name": "Write",
                                "input": {
                                    "file_path": "/home/user/project/hello.py",
                                    "content": "print('Hello, World!')",
                                },
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        # User message with tool_result
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-002",
                    "parentUuid": "asst-001",
                    "sessionId": "session-123",
                    "timestamp": "2025-01-01T10:00:10.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-001",
                                "content": "File written successfully",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        # Assistant with thinking
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-002",
                    "parentUuid": "user-002",
                    "sessionId": "session-123",
                    "timestamp": "2025-01-01T10:00:15.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-opus-4-5-20251101",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "The file was created successfully.",
                            },
                            {
                                "type": "text",
                                "text": "Done! I've created hello.py.",
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        f.flush()
        yield Path(f.name)


@pytest.fixture
def output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestExportSessionsToJson:
    """Tests for simple schema JSON export."""

    def test_creates_json_file(self, sample_session_file, output_dir):
        """Test that JSON file is created."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)
        assert output_path.exists()

    def test_output_is_valid_json(self, sample_session_file, output_dir):
        """Test that output is valid JSON."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_has_schema_type(self, sample_session_file, output_dir):
        """Test that output has schema_type field."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)
        assert data.get("schema_type") == "simple"

    def test_has_version(self, sample_session_file, output_dir):
        """Test that output has version field."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)
        assert "version" in data

    def test_has_tables_object(self, sample_session_file, output_dir):
        """Test that output has tables object."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)
        assert "tables" in data
        assert isinstance(data["tables"], dict)

    def test_has_sessions_table(self, sample_session_file, output_dir):
        """Test that tables contains sessions."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)
        assert "sessions" in data["tables"]
        assert len(data["tables"]["sessions"]) == 1

    def test_has_messages_table(self, sample_session_file, output_dir):
        """Test that tables contains messages."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)
        assert "messages" in data["tables"]
        assert len(data["tables"]["messages"]) == 4  # 2 user + 2 assistant

    def test_has_tool_calls_table(self, sample_session_file, output_dir):
        """Test that tables contains tool_calls."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)
        assert "tool_calls" in data["tables"]
        assert len(data["tables"]["tool_calls"]) == 1  # One Write tool call

    def test_has_thinking_table(self, sample_session_file, output_dir):
        """Test that tables contains thinking (empty by default)."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)
        assert "thinking" in data["tables"]
        # By default, thinking is not included
        assert len(data["tables"]["thinking"]) == 0

    def test_includes_thinking_when_enabled(self, sample_session_file, output_dir):
        """Test that thinking is included when enabled."""
        output_path = output_dir / "export.json"
        export_sessions_to_json(
            [sample_session_file], output_path, include_thinking=True
        )

        with open(output_path) as f:
            data = json.load(f)
        assert len(data["tables"]["thinking"]) == 1

    def test_session_has_required_fields(self, sample_session_file, output_dir):
        """Test that session records have required fields."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)

        session = data["tables"]["sessions"][0]
        assert "session_id" in session
        assert "project_name" in session
        assert "cwd" in session
        assert "git_branch" in session
        assert "message_count" in session

    def test_message_has_required_fields(self, sample_session_file, output_dir):
        """Test that message records have required fields."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)

        message = data["tables"]["messages"][0]
        assert "id" in message
        assert "session_id" in message
        assert "type" in message
        assert "timestamp" in message
        assert "content" in message

    def test_tool_call_has_required_fields(self, sample_session_file, output_dir):
        """Test that tool_call records have required fields."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)

        tool_call = data["tables"]["tool_calls"][0]
        assert "tool_use_id" in tool_call
        assert "session_id" in tool_call
        assert "tool_name" in tool_call
        assert "input_json" in tool_call

    def test_multiple_sessions(self, sample_session_file, output_dir):
        """Test exporting multiple sessions."""
        # Create a second session file
        second_session = output_dir / "session2.jsonl"
        second_session.write_text(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-101",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-02T10:00:00.000Z",
                    "cwd": "/home/user/other",
                    "message": {"role": "user", "content": "Another session"},
                }
            )
            + "\n"
        )

        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file, second_session], output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert len(data["tables"]["sessions"]) == 2

    def test_output_to_directory_path(self, sample_session_file, output_dir):
        """Test that exporting to a directory path creates sessions.json inside."""
        # Use the directory itself as output (simulates -o .)
        export_sessions_to_json([sample_session_file], output_dir / "sessions.json")

        # Should create sessions.json in the directory
        assert (output_dir / "sessions.json").exists()
        with open(output_dir / "sessions.json") as f:
            data = json.load(f)
        assert data["schema_type"] == "simple"



class TestSimpleJsonTokenEstimation:
    """Tests for token estimation in simple JSON export."""

    def test_simple_json_includes_estimated_tokens(
        self, sample_session_file, output_dir
    ):
        """Test that simple JSON export includes estimated_tokens in session metadata."""
        json_path = output_dir / "sessions.json"
        export_sessions_to_json([sample_session_file], json_path)

        with open(json_path) as f:
            data = json.load(f)

        sessions = data["tables"]["sessions"]
        assert len(sessions) == 1
        session = sessions[0]
        assert "estimated_tokens" in session
        assert session["estimated_tokens"] > 0

    def test_simple_json_tokens_include_all_content_types(self, output_dir):
        """Test that estimated_tokens includes text, thinking, and tool I/O."""
        # Create session with thinking and tool use
        session_file = output_dir / "test_session.jsonl"
        session_file.write_text(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "u1",
                    "parentUuid": None,
                    "sessionId": "s1",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "cwd": "/tmp/test",
                    "message": {"role": "user", "content": "Fix the bug please"},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "assistant",
                    "uuid": "a1",
                    "parentUuid": "u1",
                    "sessionId": "s1",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-sonnet-4-20250514",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "Let me analyze this bug carefully and think about solutions",
                            },
                            {"type": "text", "text": "I found the issue."},
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "Edit",
                                "input": {
                                    "file_path": "/tmp/test/main.py",
                                    "old_string": "x",
                                    "new_string": "y",
                                },
                            },
                        ],
                    },
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "user",
                    "uuid": "u2",
                    "parentUuid": "a1",
                    "sessionId": "s1",
                    "timestamp": "2025-01-01T10:00:10.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": "File edited successfully with the new changes applied",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )

        json_path = output_dir / "sessions.json"
        export_sessions_to_json([session_file], json_path)

        with open(json_path) as f:
            data = json.load(f)

        session = data["tables"]["sessions"][0]
        # Should have tokens from: user text, thinking, assistant text, tool input, tool result
        assert session["estimated_tokens"] > 0

        # Now test WITHOUT thinking/tool (plain text only)
        text_only_file = output_dir / "text_only.jsonl"
        text_only_file.write_text(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "u1",
                    "parentUuid": None,
                    "sessionId": "s2",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "cwd": "/tmp/test",
                    "message": {"role": "user", "content": "Fix the bug please"},
                }
            )
            + "\n"
        )

        json_path2 = output_dir / "text_only.json"
        export_sessions_to_json([text_only_file], json_path2)

        with open(json_path2) as f:
            data2 = json.load(f)

        text_only_tokens = data2["tables"]["sessions"][0]["estimated_tokens"]
        # Session with thinking+tools should have more tokens than text-only
        assert session["estimated_tokens"] > text_only_tokens


class TestConvertCommandFormats:
    """Tests for the convert command with different output formats."""

    def test_convert_command_json_simple(self, output_dir):
        """Test convert command with --format json produces valid simple JSON."""
        from click.testing import CliRunner
        from ccutils.cli import cli

        jsonl_file = output_dir / "test.jsonl"
        jsonl_file.write_text(
            '{"type": "user", "uuid": "u1", "parentUuid": null, "sessionId": "s1", "timestamp": "2025-01-01T10:00:00.000Z", "cwd": "/tmp", "message": {"role": "user", "content": "Hello"}}\n'
            '{"type": "assistant", "uuid": "a1", "parentUuid": "u1", "sessionId": "s1", "timestamp": "2025-01-01T10:00:05.000Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]}}\n'
        )

        json_output = output_dir / "json_output"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "convert",
                str(jsonl_file),
                "--format",
                "json",
                "-o",
                str(json_output),
            ],
        )

        assert result.exit_code == 0
        # Should create sessions.json inside the output dir
        json_path = json_output / "sessions.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert data["schema_type"] == "simple"
        assert len(data["tables"]["sessions"]) == 1

    def test_convert_command_json_star(self, output_dir):
        """Test convert command with --format json-star produces star schema JSON."""
        from click.testing import CliRunner
        from ccutils.cli import cli

        jsonl_file = output_dir / "test.jsonl"
        jsonl_file.write_text(
            '{"type": "user", "uuid": "u1", "parentUuid": null, "sessionId": "s1", "timestamp": "2025-01-01T10:00:00.000Z", "cwd": "/tmp", "message": {"role": "user", "content": "Hello"}}\n'
            '{"type": "assistant", "uuid": "a1", "parentUuid": "u1", "sessionId": "s1", "timestamp": "2025-01-01T10:00:05.000Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]}}\n'
        )

        star_output = output_dir / "star_output"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "convert",
                str(jsonl_file),
                "--format",
                "json-star",
                "-o",
                str(star_output),
            ],
        )

        assert result.exit_code == 0
        assert (star_output / "meta.json").exists()
        assert (star_output / "dimensions").is_dir()
        assert (star_output / "facts").is_dir()

        with open(star_output / "meta.json") as f:
            meta = json.load(f)
        assert meta["schema_type"] == "star"

    def test_convert_command_duckdb(self, output_dir):
        """Test convert command with --format duckdb produces a DuckDB file."""
        from click.testing import CliRunner
        from ccutils.cli import cli

        jsonl_file = output_dir / "test.jsonl"
        jsonl_file.write_text(
            '{"type": "user", "uuid": "u1", "parentUuid": null, "sessionId": "s1", "timestamp": "2025-01-01T10:00:00.000Z", "cwd": "/tmp", "message": {"role": "user", "content": "Hello"}}\n'
            '{"type": "assistant", "uuid": "a1", "parentUuid": "u1", "sessionId": "s1", "timestamp": "2025-01-01T10:00:05.000Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]}}\n'
        )

        db_output = output_dir / "test.duckdb"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "convert",
                str(jsonl_file),
                "--format",
                "duckdb",
                "-o",
                str(db_output),
            ],
        )

        assert result.exit_code == 0
        assert db_output.exists()
