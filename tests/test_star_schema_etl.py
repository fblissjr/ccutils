"""Tests for star schema ETL -- population and data extraction tests."""

import json
import tempfile
from pathlib import Path

import pytest

from ccutils import (
    create_star_schema,
    run_star_schema_etl,
)


@pytest.fixture
def granular_session_file():
    """Create a session file with rich content for granular testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # User message asking to read and modify a file
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-001",
                    "parentUuid": None,
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:30:00.000Z",
                    "cwd": "/home/user/myproject",
                    "gitBranch": "feature/auth",
                    "version": "2.1.0",
                    "message": {
                        "role": "user",
                        "content": "Read the auth.py file and fix the login bug",
                    },
                }
            )
            + "\n"
        )
        # Assistant reads file
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-001",
                    "parentUuid": "user-001",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:30:05.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-opus-4-5-20251101",
                        "content": [
                            {"type": "text", "text": "Let me read the auth file."},
                            {
                                "type": "tool_use",
                                "id": "tool-read-001",
                                "name": "Read",
                                "input": {
                                    "file_path": "/home/user/myproject/src/auth.py"
                                },
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        # Tool result with Python code
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-002",
                    "parentUuid": "asst-001",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:30:10.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-read-001",
                                "content": """def login(username, password):
    # Bug: not checking password correctly
    if username == 'admin':
        return True
    return False""",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        # Assistant analyzes and uses Bash
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-002",
                    "parentUuid": "user-002",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:30:20.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-opus-4-5-20251101",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "I see the bug - password is not being validated. Need to fix this.",
                            },
                            {
                                "type": "text",
                                "text": "I found the bug. Let me run the tests first:\n\n```python\ndef login(username, password):\n    # Fixed: now validates password\n    return validate_credentials(username, password)\n```",
                            },
                            {
                                "type": "tool_use",
                                "id": "tool-bash-001",
                                "name": "Bash",
                                "input": {
                                    "command": "cd /home/user/myproject && python -m pytest tests/"
                                },
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        # Bash result
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-003",
                    "parentUuid": "asst-002",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:30:30.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-bash-001",
                                "content": "FAILED tests/test_auth.py::test_login - AssertionError",
                                "is_error": True,
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        # Assistant edits file
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-003",
                    "parentUuid": "user-003",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:30:40.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-sonnet-4-20250514",
                        "content": [
                            {"type": "text", "text": "Let me fix the auth file."},
                            {
                                "type": "tool_use",
                                "id": "tool-edit-001",
                                "name": "Edit",
                                "input": {
                                    "file_path": "/home/user/myproject/src/auth.py",
                                    "old_string": "if username == 'admin':",
                                    "new_string": "if verify_password(username, password):",
                                },
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        # Edit result
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-004",
                    "parentUuid": "asst-003",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:30:45.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-edit-001",
                                "content": "File edited successfully",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        # Assistant uses Grep
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-004",
                    "parentUuid": "user-004",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:30:50.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-sonnet-4-20250514",
                        "content": [
                            {
                                "type": "text",
                                "text": "Let me search for related files.",
                            },
                            {
                                "type": "tool_use",
                                "id": "tool-grep-001",
                                "name": "Grep",
                                "input": {
                                    "pattern": "verify_password",
                                    "path": "/home/user/myproject",
                                },
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        # Grep result
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-005",
                    "parentUuid": "asst-004",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:30:55.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-grep-001",
                                "content": "/home/user/myproject/src/utils.py:15:def verify_password(username, password):",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        # Final assistant message
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-005",
                    "parentUuid": "user-005",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:31:00.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-sonnet-4-20250514",
                        "content": [
                            {
                                "type": "text",
                                "text": "Done! The auth.py file has been fixed to use proper password verification.",
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        f.flush()
        yield Path(f.name)


class TestRunStarSchemaETL:
    """Tests for the ETL process that populates the star schema."""

    def test_etl_populates_dim_tool(self, sample_session_file, output_dir):
        """Test that ETL populates dim_tool with tools from session."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT tool_name FROM dim_tool ORDER BY tool_name"
        ).fetchall()
        tool_names = [r[0] for r in result]
        assert "Write" in tool_names
        assert "Read" in tool_names
        conn.close()

    def test_etl_populates_dim_model(self, sample_session_file, output_dir):
        """Test that ETL populates dim_model with models from session."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT model_name FROM dim_model ORDER BY model_name"
        ).fetchall()
        model_names = [r[0] for r in result]
        assert "claude-opus-4-5-20251101" in model_names
        assert "claude-sonnet-4-20250514" in model_names
        conn.close()

    def test_etl_populates_dim_project(self, sample_session_file, output_dir):
        """Test that ETL populates dim_project."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute("SELECT project_name FROM dim_project").fetchone()
        assert result[0] == "test-project"
        conn.close()

    def test_etl_populates_dim_session(self, sample_session_file, output_dir):
        """Test that ETL populates dim_session."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT session_id, cwd, git_branch FROM dim_session"
        ).fetchone()
        assert result[1] == "/home/user/project"
        assert result[2] == "main"
        conn.close()

    def test_etl_populates_dim_date(self, sample_session_file, output_dir):
        """Test that ETL populates dim_date for dates in session."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT date_key, year, month, day FROM dim_date WHERE date_key = 20250115"
        ).fetchone()
        assert result is not None
        assert result[1] == 2025
        assert result[2] == 1
        assert result[3] == 15
        conn.close()

    def test_etl_populates_fact_messages(self, sample_session_file, output_dir):
        """Test that ETL populates fact_messages."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute("SELECT COUNT(*) FROM fact_messages").fetchone()
        assert result[0] == 6  # 3 user + 3 assistant messages
        conn.close()

    def test_etl_populates_fact_tool_calls(self, sample_session_file, output_dir):
        """Test that ETL populates fact_tool_calls."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute("SELECT COUNT(*) FROM fact_tool_calls").fetchone()
        assert result[0] == 2  # Write and Read tools
        conn.close()

    def test_etl_populates_fact_content_blocks(self, sample_session_file, output_dir):
        """Test that ETL populates fact_content_blocks."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, sample_session_file, "test-project", include_thinking=True
        )

        result = conn.execute("SELECT COUNT(*) FROM fact_content_blocks").fetchone()
        # Count all content blocks: text blocks, tool_use, tool_result, thinking
        assert result[0] > 0
        conn.close()

    def test_etl_populates_fact_session_summary(self, sample_session_file, output_dir):
        """Test that ETL populates fact_session_summary."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            """SELECT total_messages, user_messages, assistant_messages,
                      total_tool_calls, session_duration_seconds
               FROM fact_session_summary"""
        ).fetchone()
        assert result[0] == 6  # total messages
        assert result[1] == 3  # user messages
        assert result[2] == 3  # assistant messages
        assert result[3] == 2  # tool calls (Write and Read)
        assert result[4] == 25  # duration in seconds (10:00:00 to 10:00:25)
        conn.close()

    def test_etl_assigns_tool_categories(self, sample_session_file, output_dir):
        """Test that ETL assigns correct tool categories."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT tool_name, tool_category FROM dim_tool ORDER BY tool_name"
        ).fetchall()
        tool_dict = {r[0]: r[1] for r in result}
        assert tool_dict["Write"] == "file_operations"
        assert tool_dict["Read"] == "file_operations"
        conn.close()

    def test_etl_assigns_model_families(self, sample_session_file, output_dir):
        """Test that ETL assigns correct model families."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT model_name, model_family FROM dim_model ORDER BY model_name"
        ).fetchall()
        model_dict = {r[0]: r[1] for r in result}
        assert model_dict["claude-opus-4-5-20251101"] == "opus"
        assert model_dict["claude-sonnet-4-20250514"] == "sonnet"
        conn.close()

    def test_etl_links_tool_calls_to_dimensions(self, sample_session_file, output_dir):
        """Test that fact_tool_calls correctly links to dim_tool."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            """SELECT dt.tool_name, ft.input_char_count
               FROM fact_tool_calls ft
               JOIN dim_tool dt ON ft.tool_key = dt.tool_key
               ORDER BY dt.tool_name"""
        ).fetchall()
        assert len(result) == 2
        tool_names = [r[0] for r in result]
        assert "Read" in tool_names
        assert "Write" in tool_names
        conn.close()

    def test_etl_links_messages_to_date_dimension(
        self, sample_session_file, output_dir
    ):
        """Test that fact_messages correctly links to dim_date."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            """SELECT dd.year, dd.month, dd.day, COUNT(*) as msg_count
               FROM fact_messages fm
               JOIN dim_date dd ON fm.date_key = dd.date_key
               GROUP BY dd.year, dd.month, dd.day"""
        ).fetchone()
        assert result[0] == 2025
        assert result[1] == 1
        assert result[2] == 15
        assert result[3] == 6  # All 6 messages on same day
        conn.close()


class TestContentBlockGranularity:
    """Tests for granular content block tracking."""

    def test_text_blocks_tracked(self, sample_session_file, output_dir):
        """Test that text content blocks are tracked individually."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, sample_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT COUNT(*) FROM fact_content_blocks
               WHERE block_type = 'text'"""
        ).fetchone()
        # At least 3 text blocks from assistant messages
        assert result[0] >= 3
        conn.close()

    def test_tool_use_blocks_tracked(self, sample_session_file, output_dir):
        """Test that tool_use content blocks are tracked."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, sample_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT COUNT(*) FROM fact_content_blocks
               WHERE block_type = 'tool_use'"""
        ).fetchone()
        assert result[0] == 2  # Write and Read tool_use blocks
        conn.close()

    def test_thinking_blocks_tracked_when_enabled(
        self, sample_session_file, output_dir
    ):
        """Test that thinking blocks are tracked when include_thinking=True."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, sample_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT COUNT(*) FROM fact_content_blocks
               WHERE block_type = 'thinking'"""
        ).fetchone()
        assert result[0] == 1  # One thinking block
        conn.close()

    def test_block_index_tracks_position(self, sample_session_file, output_dir):
        """Test that block_index tracks position within message."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, sample_session_file, "test-project", include_thinking=True
        )

        # The assistant message asst-002 has: thinking (0), text (1), tool_use (2)
        result = conn.execute(
            """SELECT block_index, block_type
               FROM fact_content_blocks
               WHERE message_id = 'asst-002'
               ORDER BY block_index"""
        ).fetchall()
        assert len(result) == 3
        assert result[0][1] == "thinking"
        assert result[1][1] == "text"
        assert result[2][1] == "tool_use"
        conn.close()


class TestGranularDimensions:
    """Tests for granular dimension tables."""

    def test_creates_dim_file_table(self, output_dir):
        """Test that dim_file dimension table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_file'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_dim_file_has_required_columns(self, output_dir):
        """Test that dim_file has all required columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_file").fetchall()
        column_names = [c[0] for c in columns]
        assert "file_key" in column_names
        assert "file_path" in column_names
        assert "file_name" in column_names
        assert "file_extension" in column_names
        assert "directory_path" in column_names
        conn.close()

    def test_dim_file_has_language_column(self, output_dir):
        """Test that dim_file has language column (replaces dim_programming_language)."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_file").fetchall()
        column_names = [c[0] for c in columns]
        assert "language" in column_names
        conn.close()


class TestGranularFactTables:
    """Tests for granular fact tables."""

    def test_creates_fact_file_operations_table(self, output_dir):
        """Test that fact_file_operations table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_file_operations'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_fact_file_operations_has_required_columns(self, output_dir):
        """Test that fact_file_operations has all required columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_file_operations").fetchall()
        column_names = [c[0] for c in columns]
        assert "file_operation_id" in column_names
        assert "tool_call_id" in column_names
        assert "session_key" in column_names
        assert "file_key" in column_names
        assert "tool_key" in column_names
        assert "operation_type" in column_names  # read, write, edit, etc.
        assert "file_size_chars" in column_names
        conn.close()

    def test_creates_fact_code_blocks_table(self, output_dir):
        """Test that fact_code_blocks table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_code_blocks'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_fact_code_blocks_has_required_columns(self, output_dir):
        """Test that fact_code_blocks has all required columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_code_blocks").fetchall()
        column_names = [c[0] for c in columns]
        assert "code_block_id" in column_names
        assert "message_id" in column_names
        assert "session_key" in column_names
        assert "language" in column_names
        assert "line_count" in column_names
        assert "char_count" in column_names
        assert "code_text" in column_names
        conn.close()

    def test_creates_fact_errors_table(self, output_dir):
        """Test that fact_errors table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_errors'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_fact_messages_has_token_columns(self, output_dir):
        """Test that fact_messages has token tracking columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_messages").fetchall()
        column_names = [c[0] for c in columns]
        assert "estimated_tokens" in column_names
        assert "word_count" in column_names
        conn.close()


class TestGranularETL:
    """Tests for granular ETL processing."""

    def test_etl_populates_dim_file(self, granular_session_file, output_dir):
        """Test that ETL extracts files from tool calls."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            "SELECT file_name, file_extension FROM dim_file ORDER BY file_name"
        ).fetchall()
        file_names = [r[0] for r in result]
        # Should have auth.py and utils.py from the tool calls
        assert "auth.py" in file_names
        conn.close()

    def test_etl_populates_fact_file_operations(
        self, granular_session_file, output_dir
    ):
        """Test that ETL creates file operation records."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT ffo.operation_type, df.file_name
               FROM fact_file_operations ffo
               JOIN dim_file df ON ffo.file_key = df.file_key
               ORDER BY df.file_name, ffo.operation_type"""
        ).fetchall()
        # Should have read and edit operations on auth.py
        operations = [(r[0], r[1]) for r in result]
        assert ("read", "auth.py") in operations
        assert ("edit", "auth.py") in operations
        conn.close()

    def test_etl_extracts_code_blocks(self, granular_session_file, output_dir):
        """Test that ETL extracts code blocks from messages."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT language, line_count
               FROM fact_code_blocks"""
        ).fetchall()
        # Should detect Python code blocks
        languages = [r[0] for r in result]
        assert "python" in languages
        conn.close()

    def test_etl_tracks_errors(self, granular_session_file, output_dir):
        """Test that ETL tracks tool errors."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT fe.error_message, dt.tool_name
               FROM fact_errors fe
               JOIN dim_tool dt ON fe.tool_key = dt.tool_key"""
        ).fetchall()
        # Should have the pytest failure error
        assert len(result) >= 1
        conn.close()

    def test_etl_estimates_tokens(self, granular_session_file, output_dir):
        """Test that ETL estimates token counts."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            "SELECT estimated_tokens, word_count FROM fact_messages WHERE estimated_tokens > 0"
        ).fetchall()
        assert len(result) > 0
        # Token estimate should be reasonable (roughly 1.3x word count)
        for tokens, words in result:
            if words > 0:
                assert tokens >= words  # Tokens should be >= words
        conn.close()


class TestSessionHeuristicColumns:
    """Tests for dim_session heuristic classification columns (intent, complexity, outcome, domain)."""

    def test_dim_session_has_heuristic_columns(self, output_dir):
        """Test that dim_session has intent, complexity, outcome, domain columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_session").fetchall()
        column_names = [c[0] for c in columns]
        assert "intent" in column_names
        assert "complexity" in column_names
        assert "outcome" in column_names
        assert "domain" in column_names
        conn.close()

    def test_heuristic_columns_populated_by_etl(self, sample_session_file, output_dir):
        """Test that heuristic columns are populated during ETL."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT intent, complexity, outcome, domain FROM dim_session"
        ).fetchone()
        # Heuristic columns should be populated (not all NULL)
        assert result is not None
        # intent should be a string value
        assert result[0] is not None
        conn.close()

    def test_first_user_message_populated(self, sample_session_file, output_dir):
        """Test that first_user_message is populated from the first user message."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute("SELECT first_user_message FROM dim_session").fetchone()
        assert result is not None
        assert result[0] is not None
        assert "hello world" in result[0].lower()
        conn.close()

    def test_last_assistant_message_populated(self, sample_session_file, output_dir):
        """Test that last_assistant_message is populated from the last assistant message."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT last_assistant_message FROM dim_session"
        ).fetchone()
        assert result is not None
        assert result[0] is not None
        # Last assistant message is "Done! I've created hello.py..."
        assert "hello.py" in result[0].lower()
        conn.close()

    def test_message_columns_truncated_to_500_chars(self, output_dir):
        """Test that message columns are truncated to 500 chars."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "type": "user",
                        "uuid": "user-001",
                        "parentUuid": None,
                        "sessionId": "session-trunc",
                        "timestamp": "2025-01-15T10:00:00.000Z",
                        "cwd": "/home/user/project",
                        "message": {
                            "role": "user",
                            "content": "x" * 1000,
                        },
                    }
                )
                + "\n"
            )
            f.flush()
            session_path = Path(f.name)

        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, session_path, "test-project")

        result = conn.execute("SELECT first_user_message FROM dim_session").fetchone()
        assert result is not None
        assert len(result[0]) == 500
        conn.close()

    def test_dim_session_no_hierarchy_keys(self, output_dir):
        """Test that dim_session no longer has goal_key, task_key, attempt_key."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_session").fetchall()
        column_names = [c[0] for c in columns]
        assert "goal_key" not in column_names
        assert "task_key" not in column_names
        assert "attempt_key" not in column_names
        conn.close()


class TestFactSessionSummaryTimeKeyETL:
    """Tests for time_key population on fact_session_summary."""

    def test_time_key_populated_from_first_timestamp(
        self, sample_session_file, output_dir
    ):
        """Test that time_key is computed from the session's first timestamp."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute("SELECT time_key FROM fact_session_summary").fetchone()
        assert result is not None
        assert result[0] is not None
        # sample_session_file has timestamps at 10:00 AM -> time_key = 1000
        assert result[0] == 1000
        conn.close()

    def test_time_key_matches_dim_time(self, sample_session_file, output_dir):
        """Test that fact_session_summary.time_key has a matching dim_time row."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            """SELECT dt.time_of_day
               FROM fact_session_summary fss
               JOIN dim_time dt ON fss.time_key = dt.time_key"""
        ).fetchone()
        assert result is not None
        assert result[0] == "morning"
        conn.close()


class TestOrphanToolUsesStar:
    """Tests for tool_use blocks with no matching tool_result in star schema."""

    def test_star_includes_orphan_tool_calls(
        self, interrupted_session_file, output_dir
    ):
        """Star schema fact_tool_calls should include orphan tool uses."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, interrupted_session_file, "test-project")

        rows = conn.execute(
            "SELECT tool_call_id, result_message_id, output_text "
            "FROM fact_tool_calls ORDER BY tool_call_id"
        ).fetchall()

        assert (
            len(rows) == 2
        ), f"Expected 2 tool calls (1 matched + 1 orphan), got {len(rows)}"

        orphan = [r for r in rows if r[0] == "tool-orphan"][0]
        matched = [r for r in rows if r[0] == "tool-matched"][0]

        assert orphan[1] is None  # result_message_id NULL
        assert orphan[2] is None  # output_text NULL
        assert matched[1] is not None
        assert matched[2] is not None
        conn.close()

    def test_orphan_not_counted_as_error(self, interrupted_session_file, output_dir):
        """Orphan tool uses should not appear in fact_errors."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, interrupted_session_file, "test-project")

        error_count = conn.execute("SELECT COUNT(*) FROM fact_errors").fetchone()[0]
        assert error_count == 0
        conn.close()


class TestTokenEstimation:
    """Tests for comprehensive token estimation including thinking and tool I/O."""

    def test_total_estimated_tokens_includes_thinking(
        self, granular_session_file, output_dir
    ):
        """Test that total_estimated_tokens includes thinking block tokens."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            "SELECT total_estimated_tokens, total_thinking_tokens FROM fact_session_summary"
        ).fetchone()
        total_tokens, thinking_tokens = result
        assert total_tokens > 0
        assert thinking_tokens > 0
        # Thinking tokens should be part of the total
        assert total_tokens >= thinking_tokens
        conn.close()

    def test_total_estimated_tokens_includes_tool_io(
        self, granular_session_file, output_dir
    ):
        """Test that total_estimated_tokens includes tool input/output tokens."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, granular_session_file, "test-project")

        result = conn.execute(
            "SELECT total_estimated_tokens, total_tool_io_tokens FROM fact_session_summary"
        ).fetchone()
        total_tokens, tool_io_tokens = result
        assert total_tokens > 0
        assert tool_io_tokens > 0
        # Tool I/O tokens should be part of the total
        assert total_tokens >= tool_io_tokens
        conn.close()

    def test_token_breakdown_components_less_than_total(
        self, granular_session_file, output_dir
    ):
        """Test that thinking and tool_io are each strictly less than total."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT total_estimated_tokens, total_thinking_tokens,
                      total_tool_io_tokens
               FROM fact_session_summary"""
        ).fetchone()
        total, thinking, tool_io = result
        # Each component must be less than total (session has text + thinking + tools)
        assert thinking < total
        assert tool_io < total
        # Components can't exceed total
        assert thinking + tool_io <= total
        conn.close()

    def test_message_tokens_sum_to_session_total(
        self, granular_session_file, output_dir
    ):
        """SUM(fact_messages.estimated_tokens) must equal session total."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        session_total = conn.execute(
            "SELECT total_estimated_tokens FROM fact_session_summary"
        ).fetchone()[0]
        message_sum = conn.execute(
            "SELECT SUM(estimated_tokens) FROM fact_messages"
        ).fetchone()[0]
        assert message_sum == session_total
        conn.close()

    def test_message_tokens_include_thinking(self, granular_session_file, output_dir):
        """Messages with thinking blocks should have higher token counts."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        # Get a message that has thinking
        thinking_msg = conn.execute(
            "SELECT estimated_tokens, word_count FROM fact_messages WHERE has_thinking = TRUE"
        ).fetchone()
        if thinking_msg:
            # estimated_tokens should exceed word-based text estimate
            # because it includes thinking content
            assert thinking_msg[0] > thinking_msg[1]
        conn.close()

    def test_session_summary_has_token_breakdown_columns(self, output_dir):
        """Test that fact_session_summary has the new token breakdown columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_session_summary").fetchall()
        column_names = [c[0] for c in columns]
        assert "total_estimated_tokens" in column_names
        assert "total_thinking_tokens" in column_names
        assert "total_tool_io_tokens" in column_names
        conn.close()


class TestTokenUsageETL:
    """Tests for actual API token usage extraction."""

    def test_populates_fact_token_usage(self, new_format_session_file, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, new_format_session_file, "test-project")

        rows = conn.execute("SELECT * FROM fact_token_usage").fetchall()
        assert len(rows) == 2  # Two assistant messages with usage
        conn.close()

    def test_actual_tokens_on_fact_messages(self, new_format_session_file, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, new_format_session_file, "test-project")

        rows = conn.execute(
            """SELECT message_type, actual_input_tokens, actual_output_tokens, cache_read_tokens
               FROM fact_messages
               WHERE actual_input_tokens IS NOT NULL
               ORDER BY timestamp"""
        ).fetchall()
        assert len(rows) == 2
        # First assistant: 1500 input, 200 output, 500 cache read
        assert rows[0][1] == 1500
        assert rows[0][2] == 200
        assert rows[0][3] == 500
        # Second assistant: 2000 input, 300 output, 1000 cache read
        assert rows[1][1] == 2000
        assert rows[1][2] == 300
        assert rows[1][3] == 1000
        conn.close()

    def test_session_summary_actual_token_totals(
        self, new_format_session_file, output_dir
    ):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, new_format_session_file, "test-project")

        result = conn.execute(
            """SELECT actual_input_tokens, actual_output_tokens,
                      cache_creation_tokens, cache_read_tokens
               FROM fact_session_summary"""
        ).fetchone()
        assert result[0] == 3500  # 1500 + 2000
        assert result[1] == 500   # 200 + 300
        assert result[2] == 3000  # 3000 + 0
        assert result[3] == 1500  # 500 + 1000
        conn.close()

    def test_token_usage_has_cache_breakdown(
        self, new_format_session_file, output_dir
    ):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, new_format_session_file, "test-project")

        row = conn.execute(
            """SELECT cache_ephemeral_1h_tokens, cache_ephemeral_5m_tokens,
                      service_tier, speed
               FROM fact_token_usage
               ORDER BY timestamp
               LIMIT 1"""
        ).fetchone()
        assert row[0] == 3000  # ephemeral_1h
        assert row[1] == 0     # ephemeral_5m
        assert row[2] == "standard"
        assert row[3] == "standard"
        conn.close()


class TestTurnDurationETL:
    """Tests for turn duration extraction."""

    def test_populates_fact_turn_durations(self, new_format_session_file, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, new_format_session_file, "test-project")

        rows = conn.execute("SELECT * FROM fact_turn_durations").fetchall()
        assert len(rows) == 1
        conn.close()

    def test_turn_duration_values(self, new_format_session_file, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, new_format_session_file, "test-project")

        row = conn.execute(
            "SELECT duration_ms, message_count FROM fact_turn_durations"
        ).fetchone()
        assert row[0] == 45000
        assert row[1] == 12
        conn.close()

    def test_session_summary_turn_duration(self, new_format_session_file, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, new_format_session_file, "test-project")

        result = conn.execute(
            "SELECT total_turn_duration_ms, turn_count FROM fact_session_summary"
        ).fetchone()
        assert result[0] == 45000
        assert result[1] == 1
        conn.close()


class TestDiagnosticsETL:
    """Tests for LSP diagnostics extraction."""

    def test_populates_fact_diagnostics(self, new_format_session_file, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, new_format_session_file, "test-project")

        rows = conn.execute("SELECT * FROM fact_diagnostics").fetchall()
        assert len(rows) == 1
        conn.close()

    def test_diagnostic_values(self, new_format_session_file, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, new_format_session_file, "test-project")

        row = conn.execute(
            """SELECT severity, source, code, message,
                      range_start_line, range_start_col
               FROM fact_diagnostics"""
        ).fetchone()
        assert row[0] == "Error"
        assert row[1] == "Pyright"
        assert row[2] == "reportUndefinedVariable"
        assert "Undefined variable" in row[3]
        assert row[4] == 42
        assert row[5] == 8
        conn.close()

    def test_diagnostics_linked_to_dim_file(self, new_format_session_file, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, new_format_session_file, "test-project")

        row = conn.execute(
            """SELECT df.file_name
               FROM fact_diagnostics fd
               JOIN dim_file df ON fd.file_key = df.file_key"""
        ).fetchone()
        assert row is not None
        assert row[0] == "auth.py"
        conn.close()

    def test_session_summary_counts_diagnostics(
        self, new_format_session_file, output_dir
    ):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, new_format_session_file, "test-project")

        result = conn.execute(
            "SELECT total_diagnostics FROM fact_session_summary"
        ).fetchone()
        assert result[0] == 1
        conn.close()


class TestStopEventsETL:
    """Tests for stop event extraction."""

    def test_populates_fact_stop_events(self, new_format_session_file, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, new_format_session_file, "test-project")

        rows = conn.execute("SELECT * FROM fact_stop_events").fetchall()
        assert len(rows) == 1
        conn.close()

    def test_stop_event_values(self, new_format_session_file, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, new_format_session_file, "test-project")

        row = conn.execute(
            """SELECT stop_reason, hook_count, has_output,
                      prevented_continuation, hook_total_duration_ms
               FROM fact_stop_events"""
        ).fetchone()
        assert row[0] == "end_turn"
        assert row[1] == 2
        assert row[2] is True
        assert row[3] is False
        assert row[4] == 59  # 47 + 12
        conn.close()

    def test_session_summary_counts_stops(self, new_format_session_file, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, new_format_session_file, "test-project")

        result = conn.execute(
            "SELECT stop_count, prevented_continuations FROM fact_session_summary"
        ).fetchone()
        assert result[0] == 1
        assert result[1] == 0  # preventedContinuation was False
        conn.close()


class TestSessionMetadataETL:
    """Tests for new session-level metadata extraction."""

    def test_entrypoint_on_dim_session(self, new_format_session_file, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, new_format_session_file, "test-project")

        result = conn.execute("SELECT entrypoint FROM dim_session").fetchone()
        assert result[0] == "cli"
        conn.close()

    def test_custom_title_on_dim_session(self, new_format_session_file, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, new_format_session_file, "test-project")

        result = conn.execute("SELECT custom_title FROM dim_session").fetchone()
        assert result[0] == "fix-auth-bug"
        conn.close()

    def test_permission_mode_on_dim_session(self, new_format_session_file, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, new_format_session_file, "test-project")

        result = conn.execute("SELECT permission_mode FROM dim_session").fetchone()
        assert result[0] == "normal"
        conn.close()

    def test_hook_runs_counted(self, new_format_session_file, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, new_format_session_file, "test-project")

        result = conn.execute(
            "SELECT total_hook_runs FROM fact_session_summary"
        ).fetchone()
        assert result[0] == 2  # Two hook_success attachments
        conn.close()


class TestBackwardCompatibility:
    """Tests that old-format sessions still work with new schema."""

    def test_old_format_still_works(self, sample_session_file, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        # Should still create a session
        result = conn.execute("SELECT COUNT(*) FROM dim_session").fetchone()
        assert result[0] == 1
        # Should still have messages
        result = conn.execute("SELECT COUNT(*) FROM fact_messages").fetchone()
        assert result[0] > 0
        conn.close()

    def test_new_columns_null_for_old_sessions(self, sample_session_file, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            """SELECT entrypoint, custom_title, permission_mode,
                      actual_input_tokens, actual_output_tokens
               FROM dim_session
               CROSS JOIN fact_session_summary"""
        ).fetchone()
        # Old sessions don't have these fields
        assert result[0] is None  # entrypoint
        assert result[1] is None  # custom_title
        assert result[2] is None  # permission_mode
        assert result[3] is None  # actual_input_tokens
        assert result[4] is None  # actual_output_tokens
        conn.close()
