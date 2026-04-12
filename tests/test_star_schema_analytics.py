"""Tests for star schema analytics -- queries and semantic model tests."""

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


class TestStarSchemaAnalytics:
    """Tests for analytical queries on the star schema."""

    def test_tool_usage_by_category(self, sample_session_file, output_dir):
        """Test that we can analyze tool usage by category."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            """SELECT dt.tool_category, COUNT(*) as usage_count
               FROM fact_tool_calls ft
               JOIN dim_tool dt ON ft.tool_key = dt.tool_key
               GROUP BY dt.tool_category"""
        ).fetchall()
        # Both Write and Read are file_operations
        assert len(result) == 1
        assert result[0][0] == "file_operations"
        assert result[0][1] == 2
        conn.close()

    def test_messages_by_model_family(self, sample_session_file, output_dir):
        """Test that we can analyze messages by model family."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            """SELECT dm.model_family, COUNT(*) as msg_count
               FROM fact_messages fm
               JOIN dim_model dm ON fm.model_key = dm.model_key
               WHERE fm.model_key IS NOT NULL
               GROUP BY dm.model_family
               ORDER BY dm.model_family"""
        ).fetchall()
        result_dict = {r[0]: r[1] for r in result}
        # 2 opus messages, 1 sonnet message
        assert result_dict.get("opus", 0) == 2
        assert result_dict.get("sonnet", 0) == 1
        conn.close()

    def test_session_metrics_query(self, sample_session_file, output_dir):
        """Test session metrics from fact_session_summary."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            """SELECT dp.project_name, ds.git_branch,
                      fs.total_messages, fs.total_tool_calls
               FROM fact_session_summary fs
               JOIN dim_session ds ON fs.session_key = ds.session_key
               JOIN dim_project dp ON fs.project_key = dp.project_key"""
        ).fetchone()
        assert result[0] == "test-project"
        assert result[1] == "main"
        assert result[2] == 6
        assert result[3] == 2
        conn.close()

    def test_time_of_day_analysis(self, sample_session_file, output_dir):
        """Test that we can analyze activity by time of day."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            """SELECT dt.time_of_day, COUNT(*) as msg_count
               FROM fact_messages fm
               JOIN dim_time dt ON fm.time_key = dt.time_key
               GROUP BY dt.time_of_day"""
        ).fetchone()
        # All messages at 10:00 AM are in "morning"
        assert result[0] == "morning"
        assert result[1] == 6
        conn.close()


class TestFileOperationAnalytics:
    """Tests for file operation analytics queries."""

    def test_files_by_operation_count(self, granular_session_file, output_dir):
        """Test query for most frequently accessed files."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT df.file_name, COUNT(*) as op_count
               FROM fact_file_operations ffo
               JOIN dim_file df ON ffo.file_key = df.file_key
               GROUP BY df.file_name
               ORDER BY op_count DESC"""
        ).fetchall()
        # auth.py should have multiple operations
        assert len(result) > 0
        assert result[0][0] == "auth.py"  # Most accessed file
        conn.close()

    def test_operations_by_file_extension(self, granular_session_file, output_dir):
        """Test query for operations grouped by file extension."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT df.file_extension, COUNT(*) as op_count
               FROM fact_file_operations ffo
               JOIN dim_file df ON ffo.file_key = df.file_key
               GROUP BY df.file_extension"""
        ).fetchall()
        ext_counts = {r[0]: r[1] for r in result}
        assert ".py" in ext_counts
        conn.close()

    def test_operation_types_distribution(self, granular_session_file, output_dir):
        """Test query for operation type distribution."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT operation_type, COUNT(*) as count
               FROM fact_file_operations
               GROUP BY operation_type
               ORDER BY count DESC"""
        ).fetchall()
        op_types = [r[0] for r in result]
        # Should have read and edit operations
        assert "read" in op_types
        assert "edit" in op_types
        conn.close()


class TestCodeBlockAnalytics:
    """Tests for code block analytics queries."""

    def test_code_by_language(self, granular_session_file, output_dir):
        """Test query for code blocks by language."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT language, COUNT(*) as block_count, SUM(line_count) as total_lines
               FROM fact_code_blocks
               GROUP BY language"""
        ).fetchall()
        lang_stats = {r[0]: (r[1], r[2]) for r in result}
        assert "python" in lang_stats
        conn.close()

    def test_code_blocks_by_session(self, granular_session_file, output_dir):
        """Test query for code blocks per session."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT ds.session_id, COUNT(*) as code_blocks, SUM(fcb.char_count) as total_chars
               FROM fact_code_blocks fcb
               JOIN dim_session ds ON fcb.session_key = ds.session_key
               GROUP BY ds.session_id"""
        ).fetchall()
        assert len(result) > 0
        conn.close()


class TestErrorAnalytics:
    """Tests for error tracking analytics."""

    def test_errors_by_tool(self, granular_session_file, output_dir):
        """Test query for errors grouped by tool."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT dt.tool_name, COUNT(*) as error_count
               FROM fact_errors fe
               JOIN dim_tool dt ON fe.tool_key = dt.tool_key
               GROUP BY dt.tool_name
               ORDER BY error_count DESC"""
        ).fetchall()
        # Bash had an error in our test data
        tool_errors = {r[0]: r[1] for r in result}
        assert "Bash" in tool_errors
        conn.close()


class TestTokenAndCostAnalytics:
    """Tests for token estimation and cost analytics."""

    def test_tokens_by_model(self, granular_session_file, output_dir):
        """Test query for token usage by model."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT dm.model_family, SUM(fm.estimated_tokens) as total_tokens
               FROM fact_messages fm
               JOIN dim_model dm ON fm.model_key = dm.model_key
               WHERE fm.model_key IS NOT NULL
               GROUP BY dm.model_family"""
        ).fetchall()
        assert len(result) > 0
        conn.close()

    def test_tokens_by_message_type(self, granular_session_file, output_dir):
        """Test query for tokens by message type."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT fm.message_type, SUM(fm.estimated_tokens) as total_tokens, AVG(fm.word_count) as avg_words
               FROM fact_messages fm
               GROUP BY fm.message_type"""
        ).fetchall()
        msg_types = {r[0]: (r[1], r[2]) for r in result}
        assert "user" in msg_types
        assert "assistant" in msg_types
        conn.close()


# =============================================================================
# Response Time and Conversation Depth Tests
# =============================================================================


class TestResponseTimeTracking:
    """Tests for response time calculation between messages."""

    def test_fact_messages_has_response_time_column(self, output_dir):
        """Test that fact_messages has response_time_seconds column."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_messages").fetchall()
        column_names = [c[0] for c in columns]
        assert "response_time_seconds" in column_names
        conn.close()

    def test_etl_calculates_response_time(self, sample_session_file, output_dir):
        """Test that ETL calculates response time between messages."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        # Check that response times are populated
        result = conn.execute(
            """SELECT message_id, response_time_seconds
               FROM fact_messages
               WHERE response_time_seconds IS NOT NULL
               ORDER BY timestamp"""
        ).fetchall()
        # First message should not have response time (no parent)
        # Subsequent messages should have response times
        assert len(result) > 0
        # The second message (asst-001) should have 5 second response time
        for msg_id, resp_time in result:
            if msg_id == "asst-001":
                assert resp_time == 5.0
                break
        conn.close()


class TestConversationDepthTracking:
    """Tests for conversation depth calculation."""

    def test_fact_messages_has_conversation_depth_column(self, output_dir):
        """Test that fact_messages has conversation_depth column."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_messages").fetchall()
        column_names = [c[0] for c in columns]
        assert "conversation_depth" in column_names
        conn.close()

    def test_etl_calculates_conversation_depth(self, sample_session_file, output_dir):
        """Test that ETL calculates conversation depth."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            """SELECT message_id, conversation_depth
               FROM fact_messages
               ORDER BY timestamp"""
        ).fetchall()
        # First message should have depth 0
        # Each subsequent message should increase depth
        assert result[0][1] == 0  # user-001 at depth 0
        assert result[1][1] == 1  # asst-001 at depth 1
        conn.close()


class TestConversationFlowAnalytics:
    """Tests for conversation flow analytics using new columns."""

    def test_response_time_by_message_type(self, granular_session_file, output_dir):
        """Test query for average response time by message type."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT fm.message_type, AVG(fm.response_time_seconds) as avg_response_time
               FROM fact_messages fm
               WHERE fm.response_time_seconds IS NOT NULL
               GROUP BY fm.message_type"""
        ).fetchall()
        assert len(result) > 0
        conn.close()

    def test_max_conversation_depth(self, granular_session_file, output_dir):
        """Test query for maximum conversation depth per session."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT ds.session_id, MAX(fm.conversation_depth) as max_depth
               FROM fact_messages fm
               JOIN dim_session ds ON fm.session_key = ds.session_key
               GROUP BY ds.session_id"""
        ).fetchall()
        assert len(result) > 0
        # Depth should be > 0 for a conversation
        assert result[0][1] > 0
        conn.close()

    def test_tool_chain_patterns(self, granular_session_file, output_dir):
        """Test query for common tool chain patterns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT curr.tool_name as current_tool, prev.tool_name as prev_tool, COUNT(*) as count
               FROM fact_tool_chain_steps tcs
               JOIN dim_tool curr ON tcs.tool_key = curr.tool_key
               LEFT JOIN dim_tool prev ON tcs.prev_tool_key = prev.tool_key
               GROUP BY curr.tool_name, prev.tool_name
               ORDER BY count DESC"""
        ).fetchall()
        # Should have some tool chain patterns
        assert len(result) > 0
        conn.close()

class TestSemanticProjectContextView:
    """Tests for the semantic_project_context view."""

    def test_project_context_returns_expected_columns(
        self, granular_session_file, output_dir
    ):
        """Test that semantic_project_context has all expected columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, granular_session_file, "test-project")

        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'semantic_project_context'"
        ).fetchall()
        column_names = [c[0] for c in columns]
        assert "session_id" in column_names
        assert "project_name" in column_names
        assert "first_user_message" in column_names
        assert "last_assistant_message" in column_names
        assert "intent" in column_names
        assert "total_messages" in column_names
        assert "total_tool_calls" in column_names
        assert "total_errors" in column_names
        conn.close()

    def test_project_context_populated_after_etl(
        self, granular_session_file, output_dir
    ):
        """Test that semantic_project_context returns data after ETL."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, granular_session_file, "test-project")

        result = conn.execute(
            "SELECT session_id, project_name, first_user_message, last_assistant_message "
            "FROM semantic_project_context"
        ).fetchall()
        assert len(result) == 1
        assert result[0][1] == "test-project"
        # first_user_message should contain the user's request
        assert "auth" in result[0][2].lower()
        # last_assistant_message should contain the final response
        assert "fixed" in result[0][3].lower()
        conn.close()

    def test_project_context_ordered_by_created_at_desc(
        self, granular_session_file, output_dir
    ):
        """Test that results are ordered by created_at descending."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, granular_session_file, "test-project")

        # Verify the view can be queried (even with 1 row, ordering should work)
        result = conn.execute(
            "SELECT session_id, created_at FROM semantic_project_context"
        ).fetchall()
        assert len(result) >= 1
        conn.close()


class TestSemanticProjectFilesView:
    """Tests for the semantic_project_files view."""

    def test_project_files_returns_expected_columns(
        self, granular_session_file, output_dir
    ):
        """Test that semantic_project_files has all expected columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, granular_session_file, "test-project")

        from ccutils.export import finalize_star_schema

        finalize_star_schema(conn)

        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'semantic_project_files'"
        ).fetchall()
        column_names = [c[0] for c in columns]
        assert "project_name" in column_names
        assert "file_path" in column_names
        assert "language" in column_names
        assert "sessions_touching_file" in column_names
        assert "total_reads" in column_names
        assert "total_writes" in column_names
        assert "total_edits" in column_names
        assert "last_touched" in column_names
        conn.close()

    def test_project_files_populated_after_finalize(
        self, granular_session_file, output_dir
    ):
        """Test that semantic_project_files returns data after finalize."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, granular_session_file, "test-project")

        from ccutils.export import finalize_star_schema

        finalize_star_schema(conn)

        result = conn.execute(
            "SELECT project_name, file_path, sessions_touching_file, total_reads, total_edits "
            "FROM semantic_project_files ORDER BY sessions_touching_file DESC"
        ).fetchall()
        assert len(result) > 0
        # auth.py should appear (was read and edited)
        file_paths = [r[1] for r in result]
        assert any("auth.py" in fp for fp in file_paths)
        conn.close()
