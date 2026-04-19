"""Tests for star schema DDL -- schema creation and column validation."""

import hashlib

from ccutils import (
    create_star_schema,
    generate_dimension_key,
)


class TestGenerateDimensionKey:
    """Tests for dimension key generation."""

    def test_generates_md5_hash(self):
        """Test that dimension keys are MD5 hashes."""
        key = generate_dimension_key("Write")
        assert len(key) == 32  # MD5 produces 32 hex characters
        assert all(c in "0123456789abcdef" for c in key)

    def test_same_input_produces_same_key(self):
        """Test that same input always produces same key."""
        key1 = generate_dimension_key("Write")
        key2 = generate_dimension_key("Write")
        assert key1 == key2

    def test_different_inputs_produce_different_keys(self):
        """Test that different inputs produce different keys."""
        key1 = generate_dimension_key("Write")
        key2 = generate_dimension_key("Read")
        assert key1 != key2

    def test_handles_multiple_natural_keys(self):
        """Test composite key generation from multiple values."""
        key = generate_dimension_key("project", "/home/user")
        expected = hashlib.md5("project|/home/user".encode()).hexdigest()
        assert key == expected

    def test_handles_none_values(self):
        """Test that None values are handled gracefully."""
        key = generate_dimension_key(None)
        assert key is not None
        assert len(key) == 32


class TestCreateStarSchema:
    """Tests for star schema creation."""

    def test_creates_dim_tool_table(self, output_dir):
        """Test that dim_tool dimension table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_tool'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_dim_model_table(self, output_dir):
        """Test that dim_model dimension table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_model'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_dim_project_table(self, output_dir):
        """Test that dim_project dimension table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_project'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_dim_session_table(self, output_dir):
        """Test that dim_session dimension table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_session'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_dim_date_table(self, output_dir):
        """Test that dim_date dimension table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_date'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_dim_time_table(self, output_dir):
        """Test that dim_time dimension table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_time'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_fact_messages_table(self, output_dir):
        """Test that fact_messages table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_messages'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_fact_content_blocks_table(self, output_dir):
        """Test that fact_content_blocks table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_content_blocks'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_fact_tool_calls_table(self, output_dir):
        """Test that fact_tool_calls table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_tool_calls'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_fact_session_summary_table(self, output_dir):
        """Test that fact_session_summary table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_session_summary'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_semantic_sessions_view(self, output_dir):
        """Test that semantic_sessions view is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name='semantic_sessions'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_semantic_messages_view(self, output_dir):
        """Test that semantic_messages view is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name='semantic_messages'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_semantic_tool_calls_view(self, output_dir):
        """Test that semantic_tool_calls view is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name='semantic_tool_calls'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_semantic_file_operations_view(self, output_dir):
        """Test that semantic_file_operations view is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name='semantic_file_operations'"
        ).fetchone()
        assert result is not None
        conn.close()


class TestDimToolTable:
    """Tests for dim_tool dimension table."""

    def test_dim_tool_has_tool_key(self, output_dir):
        """Test that dim_tool has tool_key column."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_tool").fetchall()
        column_names = [c[0] for c in columns]
        assert "tool_key" in column_names
        conn.close()

    def test_dim_tool_has_tool_name(self, output_dir):
        """Test that dim_tool has tool_name column."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_tool").fetchall()
        column_names = [c[0] for c in columns]
        assert "tool_name" in column_names
        conn.close()

    def test_dim_tool_has_tool_category(self, output_dir):
        """Test that dim_tool has tool_category column."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_tool").fetchall()
        column_names = [c[0] for c in columns]
        assert "tool_category" in column_names
        conn.close()


class TestDimModelTable:
    """Tests for dim_model dimension table."""

    def test_dim_model_has_required_columns(self, output_dir):
        """Test that dim_model has all required columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_model").fetchall()
        column_names = [c[0] for c in columns]
        assert "model_key" in column_names
        assert "model_name" in column_names
        assert "model_family" in column_names
        conn.close()


class TestDimDateTable:
    """Tests for dim_date dimension table."""

    def test_dim_date_has_required_columns(self, output_dir):
        """Test that dim_date has all required columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_date").fetchall()
        column_names = [c[0] for c in columns]
        assert "date_key" in column_names
        assert "full_date" in column_names
        assert "year" in column_names
        assert "month" in column_names
        assert "day" in column_names
        assert "day_of_week" in column_names
        assert "day_name" in column_names
        assert "month_name" in column_names
        assert "quarter" in column_names
        assert "is_weekend" in column_names
        conn.close()


class TestDimTimeTable:
    """Tests for dim_time dimension table."""

    def test_dim_time_has_required_columns(self, output_dir):
        """Test that dim_time has all required columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_time").fetchall()
        column_names = [c[0] for c in columns]
        assert "time_key" in column_names
        assert "hour" in column_names
        assert "minute" in column_names
        assert "time_of_day" in column_names
        conn.close()


class TestFactMessagesTable:
    """Tests for fact_messages table."""

    def test_fact_messages_has_dimension_keys(self, output_dir):
        """Test that fact_messages has foreign keys to dimensions."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_messages").fetchall()
        column_names = [c[0] for c in columns]
        assert "session_key" in column_names
        assert "project_key" in column_names
        assert "message_type" in column_names
        assert "model_key" in column_names
        assert "date_key" in column_names
        assert "time_key" in column_names
        conn.close()

    def test_fact_messages_has_measures(self, output_dir):
        """Test that fact_messages has measure columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_messages").fetchall()
        column_names = [c[0] for c in columns]
        assert "content_length" in column_names
        assert "content_block_count" in column_names
        assert "has_tool_use" in column_names
        assert "has_tool_result" in column_names
        assert "has_thinking" in column_names
        conn.close()


class TestFactToolCallsTable:
    """Tests for fact_tool_calls table."""

    def test_fact_tool_calls_has_dimension_keys(self, output_dir):
        """Test that fact_tool_calls has foreign keys to dimensions."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_tool_calls").fetchall()
        column_names = [c[0] for c in columns]
        assert "session_key" in column_names
        assert "tool_key" in column_names
        assert "date_key" in column_names
        assert "time_key" in column_names
        conn.close()

    def test_fact_tool_calls_has_measures(self, output_dir):
        """Test that fact_tool_calls has measure columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_tool_calls").fetchall()
        column_names = [c[0] for c in columns]
        assert "input_char_count" in column_names
        assert "output_char_count" in column_names
        assert "is_error" in column_names
        conn.close()


class TestNoHardConstraints:
    """Tests verifying soft business rules instead of hard constraints."""

    def test_no_primary_key_constraint_on_dimensions(self, output_dir):
        """Test that dimension tables don't have hard PK constraints."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        # Should be able to insert duplicate keys (soft constraint)
        conn.execute(
            "INSERT INTO dim_tool (tool_key, tool_name, tool_category) VALUES ('abc', 'Test', 'test')"
        )
        conn.execute(
            "INSERT INTO dim_tool (tool_key, tool_name, tool_category) VALUES ('abc', 'Test2', 'test')"
        )
        result = conn.execute(
            "SELECT COUNT(*) FROM dim_tool WHERE tool_key = 'abc'"
        ).fetchone()
        assert result[0] == 2  # Both rows inserted
        conn.close()

    def test_no_foreign_key_constraint_on_facts(self, output_dir):
        """Test that fact tables don't have hard FK constraints."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        # Should be able to insert with non-existent dimension key
        conn.execute(
            """INSERT INTO fact_messages
               (message_id, session_key, project_key, message_type, model_key,
                date_key, time_key, timestamp, content_length, content_block_count,
                has_tool_use, has_tool_result, has_thinking)
               VALUES ('test-001', 'nonexistent', 'nonexistent', 'user', 'nonexistent',
                       99999999, 9999, '2025-01-01', 100, 1, false, false, false)"""
        )
        result = conn.execute(
            "SELECT COUNT(*) FROM fact_messages WHERE message_id = 'test-001'"
        ).fetchone()
        assert result[0] == 1
        conn.close()


class TestDimSessionMessageColumns:
    """Tests for first_user_message and last_assistant_message on dim_session."""

    def test_dim_session_has_first_user_message(self, output_dir):
        """Test that dim_session has first_user_message column."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_session").fetchall()
        column_names = [c[0] for c in columns]
        assert "first_user_message" in column_names
        conn.close()

    def test_dim_session_has_last_assistant_message(self, output_dir):
        """Test that dim_session has last_assistant_message column."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_session").fetchall()
        column_names = [c[0] for c in columns]
        assert "last_assistant_message" in column_names
        conn.close()


class TestFactSessionSummaryTimeKey:
    """Tests for time_key on fact_session_summary."""

    def test_fact_session_summary_has_time_key(self, output_dir):
        """Test that fact_session_summary has time_key column."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_session_summary").fetchall()
        column_names = [c[0] for c in columns]
        assert "time_key" in column_names
        conn.close()


class TestViewDateTimeColumns:
    """Tests for date/time columns on semantic views."""

    def test_semantic_sessions_has_session_datetime(self, output_dir):
        """Test that semantic_sessions has session_datetime column."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'semantic_sessions'"
        ).fetchall()
        column_names = [c[0] for c in columns]
        assert "session_datetime" in column_names
        conn.close()

    def test_semantic_sessions_has_time_of_day(self, output_dir):
        """Test that semantic_sessions has time_of_day from dim_time."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'semantic_sessions'"
        ).fetchall()
        column_names = [c[0] for c in columns]
        assert "time_of_day" in column_names
        conn.close()

    def test_semantic_file_operations_has_full_date(self, output_dir):
        """Test that semantic_file_operations has full_date from dim_date."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'semantic_file_operations'"
        ).fetchall()
        column_names = [c[0] for c in columns]
        assert "full_date" in column_names
        conn.close()

    def test_semantic_file_operations_has_time_of_day(self, output_dir):
        """Test that semantic_file_operations has time_of_day from dim_time."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'semantic_file_operations'"
        ).fetchall()
        column_names = [c[0] for c in columns]
        assert "time_of_day" in column_names
        conn.close()

    def test_semantic_session_chains_has_chain_start_date(self, output_dir):
        """Test that semantic_session_chains has chain_start_date."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'semantic_session_chains'"
        ).fetchall()
        column_names = [c[0] for c in columns]
        assert "chain_start_date" in column_names
        conn.close()

    def test_semantic_agent_delegations_has_delegation_date(self, output_dir):
        """Test that semantic_agent_delegations has delegation_date."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'semantic_agent_delegations'"
        ).fetchall()
        column_names = [c[0] for c in columns]
        assert "delegation_date" in column_names
        conn.close()

    def test_semantic_agent_delegations_has_time_of_day(self, output_dir):
        """Test that semantic_agent_delegations has time_of_day."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'semantic_agent_delegations'"
        ).fetchall()
        column_names = [c[0] for c in columns]
        assert "time_of_day" in column_names
        conn.close()

    def test_semantic_file_evolution_has_date_columns(self, output_dir):
        """Test that semantic_file_evolution has first_seen_date and last_seen_date."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'semantic_file_evolution'"
        ).fetchall()
        column_names = [c[0] for c in columns]
        assert "first_seen_date" in column_names
        assert "last_seen_date" in column_names
        conn.close()

    def test_semantic_project_context_has_session_date(self, output_dir):
        """Test that semantic_project_context has session_date."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'semantic_project_context'"
        ).fetchall()
        column_names = [c[0] for c in columns]
        assert "session_date" in column_names
        conn.close()

    def test_semantic_project_context_has_time_of_day(self, output_dir):
        """Test that semantic_project_context has time_of_day."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'semantic_project_context'"
        ).fetchall()
        column_names = [c[0] for c in columns]
        assert "time_of_day" in column_names
        conn.close()

    def test_semantic_project_files_has_last_touched_date(self, output_dir):
        """Test that semantic_project_files has last_touched_date."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'semantic_project_files'"
        ).fetchall()
        column_names = [c[0] for c in columns]
        assert "last_touched_date" in column_names
        conn.close()


class TestProjectContextViews:
    """Tests for semantic_project_context and semantic_project_files views."""

    def test_creates_semantic_project_context_view(self, output_dir):
        """Test that semantic_project_context view is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name='semantic_project_context'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_semantic_project_files_view(self, output_dir):
        """Test that semantic_project_files view is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name='semantic_project_files'"
        ).fetchone()
        assert result is not None
        conn.close()


class TestDimSessionNewColumns:
    """Tests for new columns on dim_session (entrypoint, custom_title, etc.)."""

    def test_dim_session_has_entrypoint(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE dim_session").fetchall()]
        assert "entrypoint" in columns
        conn.close()

    def test_dim_session_has_custom_title(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE dim_session").fetchall()]
        assert "custom_title" in columns
        conn.close()

    def test_dim_session_has_permission_mode(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE dim_session").fetchall()]
        assert "permission_mode" in columns
        conn.close()

    def test_dim_session_has_agent_type(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE dim_session").fetchall()]
        assert "agent_type" in columns
        conn.close()

    def test_dim_session_has_agent_description(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE dim_session").fetchall()]
        assert "agent_description" in columns
        conn.close()


class TestFactSessionSummaryNewColumns:
    """Tests for new token/duration columns on fact_session_summary."""

    def test_has_actual_input_tokens(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE fact_session_summary").fetchall()]
        assert "actual_input_tokens" in columns
        conn.close()

    def test_has_actual_output_tokens(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE fact_session_summary").fetchall()]
        assert "actual_output_tokens" in columns
        conn.close()

    def test_has_cache_creation_tokens(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE fact_session_summary").fetchall()]
        assert "cache_creation_tokens" in columns
        conn.close()

    def test_has_cache_read_tokens(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE fact_session_summary").fetchall()]
        assert "cache_read_tokens" in columns
        conn.close()

    def test_has_total_turn_duration_ms(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE fact_session_summary").fetchall()]
        assert "total_turn_duration_ms" in columns
        conn.close()

    def test_has_turn_count(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE fact_session_summary").fetchall()]
        assert "turn_count" in columns
        conn.close()


class TestFactMessagesNewColumns:
    """Tests for actual token columns on fact_messages."""

    def test_has_actual_input_tokens(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE fact_messages").fetchall()]
        assert "actual_input_tokens" in columns
        conn.close()

    def test_has_actual_output_tokens(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE fact_messages").fetchall()]
        assert "actual_output_tokens" in columns
        conn.close()

    def test_has_cache_read_tokens(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE fact_messages").fetchall()]
        assert "cache_read_tokens" in columns
        conn.close()


class TestNewFactTables:
    """Tests for new fact tables (token_usage, turn_durations, diagnostics, stop_events)."""

    def test_creates_fact_token_usage_table(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_token_usage'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_fact_token_usage_has_required_columns(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE fact_token_usage").fetchall()]
        for col in [
            "usage_id", "session_key", "date_key", "time_key", "model_key",
            "input_tokens", "output_tokens", "cache_creation_input_tokens",
            "cache_read_input_tokens", "service_tier", "speed", "timestamp",
        ]:
            assert col in columns, f"Missing column: {col}"
        conn.close()

    def test_creates_fact_turn_durations_table(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_turn_durations'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_fact_turn_durations_has_required_columns(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE fact_turn_durations").fetchall()]
        for col in ["turn_id", "session_key", "date_key", "time_key", "duration_ms", "message_count", "timestamp"]:
            assert col in columns, f"Missing column: {col}"
        conn.close()

    def test_creates_fact_diagnostics_table(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_diagnostics'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_fact_diagnostics_has_required_columns(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE fact_diagnostics").fetchall()]
        for col in [
            "diagnostic_id", "session_key", "file_key", "severity", "source",
            "code", "message", "range_start_line", "timestamp",
        ]:
            assert col in columns, f"Missing column: {col}"
        conn.close()

    def test_creates_fact_stop_events_table(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_stop_events'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_fact_stop_events_has_required_columns(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE fact_stop_events").fetchall()]
        for col in [
            "stop_event_id", "session_key", "stop_reason", "hook_count",
            "has_output", "prevented_continuation", "timestamp",
        ]:
            assert col in columns, f"Missing column: {col}"
        conn.close()


class TestDimPromptTable:
    """Tests for dim_prompt table (history.jsonl data)."""

    def test_creates_dim_prompt_table(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_prompt'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_dim_prompt_has_required_columns(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE dim_prompt").fetchall()]
        for col in [
            "prompt_key", "session_key", "project_path", "project_name",
            "display_text", "timestamp", "date_key", "time_key",
            "has_pasted_content",
        ]:
            assert col in columns, f"Missing column: {col}"
        conn.close()


class TestSemanticPromptHistoryView:
    """Tests for semantic_prompt_history view."""

    def test_creates_view(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name='semantic_prompt_history'"
        ).fetchone()
        assert result is not None
        conn.close()


class TestNewSemanticViews:
    """Tests for new semantic views."""

    def test_creates_semantic_token_usage_view(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name='semantic_token_usage'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_semantic_cost_analysis_view(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name='semantic_cost_analysis'"
        ).fetchone()
        assert result is not None
        conn.close()


class TestFactPlanRevisions:
    """Tests for fact_plan_revisions table (ExitPlanMode revision chain)."""

    def test_creates_fact_plan_revisions_table(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_plan_revisions'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_fact_plan_revisions_has_required_columns(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        columns = [c[0] for c in conn.execute("DESCRIBE fact_plan_revisions").fetchall()]
        for col in [
            "revision_key",
            "session_key",
            "project_key",
            "date_key",
            "time_key",
            "tool_call_id",
            "invoke_message_id",
            "result_message_id",
            "revision_number",
            "parent_revision_key",
            "plan_text",
            "plan_char_count",
            "plan_estimated_tokens",
            "outcome",
            "outcome_signal",
            "user_feedback_message_id",
            "user_feedback_text",
            "plan_timestamp",
            "resolved_timestamp",
            "seconds_to_resolution",
        ]:
            assert col in columns, f"Missing column: {col}"
        conn.close()

    def test_creates_semantic_plan_revisions_view(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name='semantic_plan_revisions'"
        ).fetchone()
        assert result is not None
        conn.close()
