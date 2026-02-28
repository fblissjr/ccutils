"""Tests for PathSanitizer privacy-preserving path sanitization."""

import json
import tempfile
from pathlib import Path

import duckdb
import pytest

from ccutils.sanitize import PathSanitizer
from ccutils.schemas.simple import create_duckdb_schema, export_session_to_duckdb
from ccutils.schemas.simple.etl import _extract_session_data


class TestPathSanitizer:
    """Core path sanitization behavior."""

    def test_paths_under_cwd_become_relative(self):
        san = PathSanitizer("/Users/fred/workspace/project")
        assert (
            san.sanitize_path("/Users/fred/workspace/project/src/main.py")
            == "src/main.py"
        )

    def test_deeply_nested_path_under_cwd(self):
        san = PathSanitizer("/Users/fred/workspace/project")
        result = san.sanitize_path("/Users/fred/workspace/project/a/b/c/d.txt")
        assert result == "a/b/c/d.txt"

    def test_cwd_itself_becomes_dot(self):
        san = PathSanitizer("/Users/fred/workspace/project")
        assert san.sanitize_path("/Users/fred/workspace/project") == "."

    def test_cwd_with_trailing_slash(self):
        san = PathSanitizer("/Users/fred/workspace/project/")
        assert (
            san.sanitize_path("/Users/fred/workspace/project/src/main.py")
            == "src/main.py"
        )

    def test_home_dir_paths_get_tilde(self):
        san = PathSanitizer("/Users/fred/workspace/project", home_dir="/Users/fred")
        result = san.sanitize_path("/Users/fred/.claude/settings.json")
        assert result == "~/.claude/settings.json"

    def test_home_dir_itself_becomes_tilde(self):
        san = PathSanitizer("/Users/fred/workspace/project", home_dir="/Users/fred")
        assert san.sanitize_path("/Users/fred") == "~"

    def test_system_paths_unchanged(self):
        san = PathSanitizer("/Users/fred/workspace/project", home_dir="/Users/fred")
        assert san.sanitize_path("/usr/local/bin/python") == "/usr/local/bin/python"

    def test_system_paths_var(self):
        san = PathSanitizer("/Users/fred/workspace/project", home_dir="/Users/fred")
        assert san.sanitize_path("/var/log/syslog") == "/var/log/syslog"

    def test_none_cwd_is_noop(self):
        san = PathSanitizer(None)
        path = "/Users/fred/workspace/project/src/main.py"
        assert san.sanitize_path(path) == path

    def test_none_cwd_text_is_noop(self):
        san = PathSanitizer(None)
        text = "Reading /Users/fred/workspace/project/src/main.py"
        assert san.sanitize_text(text) == text

    def test_empty_string_path(self):
        san = PathSanitizer("/Users/fred/workspace/project")
        assert san.sanitize_path("") == ""

    def test_none_path(self):
        san = PathSanitizer("/Users/fred/workspace/project")
        assert san.sanitize_path(None) is None

    def test_cwd_priority_over_home(self):
        """Paths under cwd should be made relative, not ~/..."""
        san = PathSanitizer("/Users/fred/workspace/project", home_dir="/Users/fred")
        result = san.sanitize_path("/Users/fred/workspace/project/src/main.py")
        assert result == "src/main.py"
        assert not result.startswith("~")


class TestSanitizeCwd:
    """sanitize_cwd returns project directory name."""

    def test_returns_dot(self):
        san = PathSanitizer("/Users/fred/workspace/project")
        assert san.sanitize_cwd() == "."

    def test_none_cwd_returns_none(self):
        san = PathSanitizer(None)
        assert san.sanitize_cwd() is None


class TestSanitizeProjectPath:
    """sanitize_project_path strips to basename."""

    def test_strips_to_basename(self):
        san = PathSanitizer("/Users/fred/workspace/project")
        result = san.sanitize_project_path(
            "/Users/fred/.claude/projects/-Users-fred-workspace-project"
        )
        assert result == "-Users-fred-workspace-project"

    def test_none_cwd_returns_original(self):
        san = PathSanitizer(None)
        original = "/Users/fred/.claude/projects/-Users-fred-workspace-project"
        assert san.sanitize_project_path(original) == original

    def test_none_path_returns_none(self):
        san = PathSanitizer("/Users/fred/workspace/project")
        assert san.sanitize_project_path(None) is None

    def test_simple_path(self):
        san = PathSanitizer("/Users/fred/workspace/project")
        assert san.sanitize_project_path("/some/path/session.jsonl") == "session.jsonl"


class TestSanitizeText:
    """sanitize_text does ordered string replacement in free text."""

    def test_replaces_cwd_in_commands(self):
        san = PathSanitizer("/Users/fred/workspace/project")
        text = "cd /Users/fred/workspace/project && ls"
        assert san.sanitize_text(text) == "cd . && ls"

    def test_replaces_cwd_slash_prefix(self):
        san = PathSanitizer("/Users/fred/workspace/project")
        text = "cat /Users/fred/workspace/project/README.md"
        assert san.sanitize_text(text) == "cat README.md"

    def test_replaces_home_dir(self):
        san = PathSanitizer("/Users/fred/workspace/project", home_dir="/Users/fred")
        text = "cat /Users/fred/.bashrc"
        assert san.sanitize_text(text) == "cat ~/.bashrc"

    def test_cwd_replaced_before_home(self):
        """cwd/ is more specific and should be replaced first."""
        san = PathSanitizer("/Users/fred/workspace/project", home_dir="/Users/fred")
        text = "Reading /Users/fred/workspace/project/src/main.py"
        result = san.sanitize_text(text)
        assert result == "Reading src/main.py"
        assert "~" not in result

    def test_multiple_paths_in_text(self):
        san = PathSanitizer("/Users/fred/workspace/project", home_dir="/Users/fred")
        text = (
            "Editing /Users/fred/workspace/project/a.py and "
            "/Users/fred/.config/settings.json"
        )
        result = san.sanitize_text(text)
        assert result == "Editing a.py and ~/.config/settings.json"

    def test_none_text(self):
        san = PathSanitizer("/Users/fred/workspace/project")
        assert san.sanitize_text(None) is None


class TestSanitizeJsonString:
    """sanitize_json_string applies text replacement to JSON strings."""

    def test_replaces_paths_in_json(self):
        san = PathSanitizer("/Users/fred/workspace/project")
        json_str = '{"file_path": "/Users/fred/workspace/project/src/main.py"}'
        result = san.sanitize_json_string(json_str)
        assert '"file_path": "src/main.py"' in result

    def test_replaces_home_in_json(self):
        san = PathSanitizer("/Users/fred/workspace/project", home_dir="/Users/fred")
        json_str = '{"path": "/Users/fred/.claude/config"}'
        result = san.sanitize_json_string(json_str)
        assert '"path": "~/.claude/config"' in result

    def test_none_json_returns_none(self):
        san = PathSanitizer("/Users/fred/workspace/project")
        assert san.sanitize_json_string(None) is None

    def test_none_cwd_noop(self):
        san = PathSanitizer(None)
        json_str = '{"file_path": "/Users/fred/workspace/project/src/main.py"}'
        assert san.sanitize_json_string(json_str) == json_str


class TestAutoDetectHome:
    """When home_dir is not provided, auto-detect from cwd."""

    def test_auto_detect_from_users_prefix(self):
        san = PathSanitizer("/Users/fred/workspace/project")
        # Should auto-detect /Users/fred as home
        result = san.sanitize_path("/Users/fred/.claude/settings.json")
        assert result == "~/.claude/settings.json"

    def test_auto_detect_from_home_prefix(self):
        san = PathSanitizer("/home/fred/workspace/project")
        result = san.sanitize_path("/home/fred/.bashrc")
        assert result == "~/.bashrc"


# -- Integration tests with simple schema ETL --


@pytest.fixture
def private_session_file():
    """JSONL session with absolute paths that should be sanitized."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "u1",
                    "parentUuid": None,
                    "sessionId": "sess-1",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "cwd": "/home/fred/workspace/myproject",
                    "gitBranch": "main",
                    "version": "2.0.0",
                    "message": {"role": "user", "content": "fix the bug"},
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "a1",
                    "parentUuid": "u1",
                    "sessionId": "sess-1",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-opus-4-5-20251101",
                        "content": [
                            {"type": "text", "text": "I'll fix it."},
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "Read",
                                "input": {
                                    "file_path": "/home/fred/workspace/myproject/src/main.py"
                                },
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "u2",
                    "parentUuid": "a1",
                    "sessionId": "sess-1",
                    "timestamp": "2025-01-01T10:00:10.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": "Contents of /home/fred/workspace/myproject/src/main.py:\ndef main(): pass",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "a2",
                    "parentUuid": "u2",
                    "sessionId": "sess-1",
                    "timestamp": "2025-01-01T10:00:15.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-opus-4-5-20251101",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "t2",
                                "name": "Bash",
                                "input": {
                                    "command": "cd /home/fred/workspace/myproject && python src/main.py"
                                },
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "u3",
                    "parentUuid": "a2",
                    "sessionId": "sess-1",
                    "timestamp": "2025-01-01T10:00:20.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t2",
                                "content": "ran in /home/fred/workspace/myproject",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        f.flush()
        yield Path(f.name)


class TestSimpleEtlPrivate:
    """Integration: private=True sanitizes paths in simple schema DuckDB export."""

    def test_cwd_sanitized_in_sessions(self, private_session_file):
        conn = create_duckdb_schema(":memory:")
        export_session_to_duckdb(conn, private_session_file, "myproject", private=True)
        row = conn.execute("SELECT cwd FROM sessions").fetchone()
        assert row[0] == "."
        conn.close()

    def test_project_path_sanitized_in_sessions(self, private_session_file):
        conn = create_duckdb_schema(":memory:")
        export_session_to_duckdb(conn, private_session_file, "myproject", private=True)
        row = conn.execute("SELECT project_path FROM sessions").fetchone()
        # Should be basename only, not full path
        assert "/home/fred" not in row[0]
        conn.close()

    def test_tool_input_file_path_sanitized(self, private_session_file):
        conn = create_duckdb_schema(":memory:")
        export_session_to_duckdb(conn, private_session_file, "myproject", private=True)
        rows = conn.execute(
            "SELECT input_json FROM tool_calls WHERE tool_name = 'Read'"
        ).fetchall()
        for row in rows:
            assert "/home/fred" not in row[0]
            assert "src/main.py" in row[0]
        conn.close()

    def test_tool_input_command_sanitized(self, private_session_file):
        conn = create_duckdb_schema(":memory:")
        export_session_to_duckdb(conn, private_session_file, "myproject", private=True)
        rows = conn.execute(
            "SELECT input_json FROM tool_calls WHERE tool_name = 'Bash'"
        ).fetchall()
        for row in rows:
            assert "/home/fred" not in row[0]
        conn.close()

    def test_output_text_sanitized(self, private_session_file):
        conn = create_duckdb_schema(":memory:")
        export_session_to_duckdb(conn, private_session_file, "myproject", private=True)
        rows = conn.execute("SELECT output_text FROM tool_calls").fetchall()
        for row in rows:
            if row[0]:
                assert "/home/fred" not in row[0]
        conn.close()

    def test_private_false_keeps_absolute_paths(self, private_session_file):
        conn = create_duckdb_schema(":memory:")
        export_session_to_duckdb(conn, private_session_file, "myproject", private=False)
        row = conn.execute("SELECT cwd FROM sessions").fetchone()
        assert row[0] == "/home/fred/workspace/myproject"
        conn.close()


class TestExtractSessionDataPrivate:
    """Integration: _extract_session_data with private=True."""

    def test_session_cwd_sanitized(self, private_session_file):
        data = _extract_session_data(private_session_file, private=True)
        assert data["session"]["cwd"] == "."

    def test_tool_calls_input_sanitized(self, private_session_file):
        data = _extract_session_data(private_session_file, private=True)
        for tc in data["tool_calls"]:
            input_json = json.dumps(tc["input_json"])
            assert "/home/fred" not in input_json

    def test_tool_calls_output_sanitized(self, private_session_file):
        data = _extract_session_data(private_session_file, private=True)
        for tc in data["tool_calls"]:
            if tc.get("output_text"):
                assert "/home/fred" not in tc["output_text"]
