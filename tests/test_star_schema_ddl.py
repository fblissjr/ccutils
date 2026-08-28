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

    def test_creates_fact_tool_uses_and_results_tables(self, output_dir):
        """v0.15 replaces legacy fact_tool_calls with fact_tool_uses + fact_tool_results."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        for tbl in ("fact_tool_uses", "fact_tool_results"):
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                [tbl],
            ).fetchone()
            assert result is not None, f"missing {tbl}"
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


class TestFactToolUsesAndResultsTables:
    """v0.15 split fact_tool_calls into fact_tool_uses + fact_tool_results."""

    def test_fact_tool_uses_has_dimension_keys(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = [c[0] for c in conn.execute("DESCRIBE fact_tool_uses").fetchall()]
        for col in ("session_key", "tool_key", "date_key", "time_key"):
            assert col in columns
        conn.close()

    def test_fact_tool_results_has_is_error_tri_state(self, output_dir):
        """R16: is_error is nullable BOOLEAN so we can preserve missing-vs-false."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        info = conn.execute("DESCRIBE fact_tool_results").fetchall()
        is_error_row = [r for r in info if r[0] == "is_error"]
        assert is_error_row, "fact_tool_results missing is_error"
        # DuckDB DESCRIBE row: (column_name, column_type, null, key, default, extra)
        assert is_error_row[0][2] == "YES", "is_error must be nullable for tri-state"
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

        # Should be able to insert with non-existent dimension keys
        conn.execute(
            """INSERT INTO fact_messages
               (created_by_version_key, last_updated_by_version_key, etl_run_id,
                record_source, hash_diff,
                entry_id, message_id, session_id,
                session_key, project_key, model_key, message_type, timestamp)
               VALUES ('v', 'v', 'run',
                       'claude_code_jsonl', 'h',
                       'e-001', 'test-001', 'sess-001',
                       'nonexistent', 'nonexistent', 'nonexistent', 'user',
                       '2025-01-01 00:00:00')"""
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
        # v0.15 R11: cache_creation split into 5m / 1h pricing tiers.
        for col in [
            "session_key", "session_id", "date_key", "time_key", "model_key",
            "input_tokens", "output_tokens",
            "cache_creation_5m_tokens", "cache_creation_1h_tokens",
            "cache_read_tokens", "total_uncached_equivalent_tokens",
            "service_tier", "timestamp",
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
        # v0.15: tool_call_id -> tool_use_id (matching the rename across
        # fact_tool_uses / fact_tool_results); invoke_message_id and
        # result_message_id dropped (derivable via tool_use_id join);
        # plan_estimated_tokens dropped (plan_char_count is enough --
        # the v0.15 schema treats word-count token estimates as a
        # presentation concern, not a fact column).
        for col in [
            "revision_key",
            "tool_use_id",
            "session_key",
            "project_key",
            "date_key",
            "time_key",
            "revision_number",
            "parent_revision_key",
            "plan_text",
            "plan_char_count",
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


class TestColumnMigrations:
    """Every entry in _COLUMN_MIGRATIONS must actually re-add its column.

    The warehouse is persistent and all tables are CREATE TABLE IF NOT
    EXISTS, so a column added to a table that already shipped only reaches
    existing warehouses via _COLUMN_MIGRATIONS. A forgotten/typo'd entry is
    invisible on fresh-DB tests otherwise; this drops each registered
    column and asserts create_star_schema restores it.
    """

    def test_every_migration_readds_its_column(self, output_dir):
        from ccutils.schemas.star.schema import _COLUMN_MIGRATIONS

        db_path = output_dir / "mig.duckdb"
        conn = create_star_schema(db_path)
        for table, column, _type in _COLUMN_MIGRATIONS:
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
        conn.close()

        conn = create_star_schema(db_path)
        for table, column, _type in _COLUMN_MIGRATIONS:
            cols = {
                r[0] for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = ?", [table]
                ).fetchall()
            }
            assert column in cols, f"{table}.{column} not re-added by migration"
        conn.close()


class TestCreateTableIsSelfSufficient:
    """Every migrated column must ALSO exist in its table's CREATE.

    Migrations run after the CREATEs on every `create_star_schema()` call, so
    a column that lives only in `_COLUMN_MIGRATIONS` is invisibly fine today:
    fresh warehouses get it from the ALTER, existing ones get it from the
    ALTER, and nothing distinguishes the two.

    1.0.0 deletes `_COLUMN_MIGRATIONS` wholesale under the no-migration rule.
    Any column that exists only there disappears from every fresh warehouse
    at that moment. Five did when this test was written --
    `fact_agent_delegations.agent_resolved_model`, `.agent_is_async`,
    `.completion_state`, `.agent_derived_io_tokens`, and
    `fact_etl_runs.run_kind` -- and `completion_state` is the column the
    entire delegation reconciliation pass keys on, so the failure would have
    been "delegation outcomes silently stop working" rather than an error.

    Delete this test and that trap can quietly grow again between now and the
    deletion. The rule it encodes outlives `_COLUMN_MIGRATIONS`: a CREATE
    statement should be a complete description of its table.
    """

    def _create_body(self, source: str, table: str) -> str:
        marker = f"CREATE TABLE IF NOT EXISTS {table} ("
        start = source.index(marker) + len(marker)
        depth = 1
        for i, ch in enumerate(source[start:], start):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return source[start:i]
        raise AssertionError(f"unbalanced CREATE for {table}")

    def test_every_migrated_column_is_also_in_its_create(self):
        import re
        from pathlib import Path

        import ccutils.schemas.star.schema as schema_mod

        source = Path(schema_mod.__file__).read_text()
        missing = []
        for table, column, _type in schema_mod._COLUMN_MIGRATIONS:
            body = self._create_body(source, table)
            if not re.search(rf"^\s*{re.escape(column)}\s+\w", body, re.M):
                missing.append(f"{table}.{column}")
        assert not missing, (
            "these columns exist ONLY in _COLUMN_MIGRATIONS and would vanish "
            f"from fresh warehouses when it is deleted: {missing}. Add each "
            "to its CREATE TABLE body."
        )

    def test_the_check_can_actually_fail(self):
        """The oracle detects a column absent from a CREATE body."""
        body = self._create_body(
            "CREATE TABLE IF NOT EXISTS t (\n    a VARCHAR,\n    b INTEGER\n)", "t"
        )
        import re

        assert re.search(r"^\s*a\s+\w", body, re.M)
        assert not re.search(r"^\s*zzz_absent\s+\w", body, re.M)
