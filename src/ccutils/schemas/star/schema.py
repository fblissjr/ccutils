"""Star schema DDL - creates the dimensional model tables.

28 tables + 14 views. Tiny lookup dimensions (message_type, content_block_type,
error_type, entity_type, programming_language) replaced by degenerate VARCHAR
columns on fact tables. LLM enrichment tables removed entirely -- replaced by
heuristic classification columns on dim_session.
"""

import duckdb


def create_star_schema(db_path):
    """Create DuckDB database with star schema for transcript analytics.

    This creates a dimensional model with:
    - 6 core dimensions (session, project, tool, model, date, time)
    - 6 core facts (messages, tool_calls, session_summary, file_operations, errors, tool_chain_steps)
    - 2 granular dimensions (file, session_chain)
    - 3 granular facts (content_blocks, code_blocks, entity_mentions)
    - 4 new facts (token_usage, turn_durations, diagnostics, stop_events)
    - 3 agent/bridge/staging tables
    - 2 optional tables (embeddings, tool_input_params)
    - 1 prompt history table (dim_prompt from history.jsonl)
    - 13 semantic views

    No hard PK/FK constraints - relies on soft business rules.

    Args:
        db_path: Path to the DuckDB database file

    Returns:
        duckdb.Connection to the database
    """
    conn = duckdb.connect(str(db_path))

    # =========================================================================
    # Lineage + Meta Tables (Phase B of v0.15 rethink)
    # =========================================================================
    # dim_etl_version is the catalog of (ccutils_version, business_rules_version)
    # tuples; every fact row references one created_by_version_key and one
    # last_updated_by_version_key. fact_etl_runs records every batch.
    # meta_schema_version tracks DDL-level migrations applied to this database.

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_etl_version (
            version_key VARCHAR,           -- MD5(ccutils_version || business_rules_version)
            ccutils_version VARCHAR NOT NULL,
            business_rules_version VARCHAR NOT NULL DEFAULT '1',
            description VARCHAR,           -- e.g., "0.15.0 -- Pydantic parser, structured toolUseResult capture"
            first_seen_at TIMESTAMP NOT NULL DEFAULT current_timestamp
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_etl_runs (
            etl_run_id VARCHAR NOT NULL,        -- UUID4 hex per run
            version_key VARCHAR,                -- FK dim_etl_version
            started_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            completed_at TIMESTAMP,
            status VARCHAR NOT NULL DEFAULT 'running',  -- running | success | failed
            source_path VARCHAR,                -- archive root or single session path
            sessions_seen INTEGER DEFAULT 0,
            sessions_inserted INTEGER DEFAULT 0,
            sessions_updated INTEGER DEFAULT 0,
            sessions_unchanged INTEGER DEFAULT 0,
            sessions_soft_deleted INTEGER DEFAULT 0,
            facts_inserted INTEGER DEFAULT 0,
            facts_updated INTEGER DEFAULT 0,
            error_message VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE meta_schema_version (
            migration_id VARCHAR NOT NULL,      -- e.g., '20260419_0001_initial'
            applied_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            description VARCHAR,
            ccutils_version VARCHAR
        )
    """
    )

    # =========================================================================
    # Staging Tables
    # =========================================================================
    # stg_log_entries: bridge from Tier 1 (Parquet lake) to Tier 3 (warehouse
    # facts). One row per JSONL line. Fact-table populators select from here
    # to project into their grain. Trunc-and-reload friendly: reloading a
    # session replaces its rows by source_path.

    conn.execute(
        """
        CREATE OR REPLACE TABLE stg_log_entries (
            etl_run_id VARCHAR NOT NULL,
            parsed_at TIMESTAMP NOT NULL,
            parser_version VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            entry_id VARCHAR NOT NULL,
            source_path VARCHAR NOT NULL,
            sequence_num INTEGER NOT NULL,
            type VARCHAR NOT NULL,
            uuid VARCHAR,
            parent_uuid VARCHAR,
            session_id VARCHAR,
            timestamp VARCHAR,
            cwd VARCHAR,
            git_branch VARCHAR,
            slug VARCHAR,
            version VARCHAR,
            user_type VARCHAR,
            entrypoint VARCHAR,
            is_sidechain BOOLEAN,
            is_meta BOOLEAN,
            agent_id VARCHAR,
            message_json VARCHAR,
            tool_use_result_json VARCHAR,
            attachment_json VARCHAR,
            progress_data_json VARCHAR,
            system_subtype VARCHAR,
            system_payload_json VARCHAR,
            meta_payload_json VARCHAR,
            extras_json VARCHAR,
            raw_json VARCHAR
        )
        """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE stg_task_agent_map (
            tool_use_id VARCHAR,
            agent_id VARCHAR,
            session_key VARCHAR
        )
    """
    )

    # =========================================================================
    # Core Dimension Tables (6)
    # =========================================================================

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_tool (
            tool_key VARCHAR,
            tool_name VARCHAR,
            tool_category VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_model (
            model_key VARCHAR,
            model_name VARCHAR,
            model_family VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_project (
            project_key VARCHAR,
            project_path VARCHAR,
            project_name VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_session (
            session_key VARCHAR,
            session_id VARCHAR,
            project_key VARCHAR,
            cwd VARCHAR,
            git_branch VARCHAR,
            version VARCHAR,
            slug VARCHAR,
            first_timestamp TIMESTAMP,
            last_timestamp TIMESTAMP,
            is_agent BOOLEAN DEFAULT FALSE,
            agent_id VARCHAR,
            parent_session_key VARCHAR,
            depth_level INTEGER DEFAULT 0,
            chain_key VARCHAR,
            intent VARCHAR,
            complexity VARCHAR,
            outcome VARCHAR,
            domain VARCHAR,
            first_user_message VARCHAR,
            last_assistant_message VARCHAR,
            entrypoint VARCHAR,
            custom_title VARCHAR,
            permission_mode VARCHAR,
            agent_type VARCHAR,
            agent_description VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_date (
            date_key INTEGER,
            full_date DATE,
            year INTEGER,
            month INTEGER,
            day INTEGER,
            day_of_week INTEGER,
            day_name VARCHAR,
            month_name VARCHAR,
            quarter INTEGER,
            is_weekend BOOLEAN,
            week_of_year INTEGER
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_time (
            time_key INTEGER,
            hour INTEGER,
            minute INTEGER,
            time_of_day VARCHAR
        )
    """
    )

    # =========================================================================
    # Core Fact Tables (6)
    # =========================================================================

    # Grain: one row per user or assistant entry in a session.
    # See docs/STAR_SCHEMA.md for the lineage column convention shared by
    # every fact table in v0.15+.
    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_messages (
            -- Lineage (every fact in v0.15+ carries this block)
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            created_by_version_key VARCHAR NOT NULL,
            last_updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            last_updated_by_version_key VARCHAR NOT NULL,
            etl_run_id VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            hash_diff VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP,

            -- Degenerate dimensions on every row
            entry_id VARCHAR NOT NULL,
            message_id VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,

            -- Dimension FKs
            session_key VARCHAR,
            project_key VARCHAR,
            model_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,

            -- Native columns
            message_type VARCHAR NOT NULL,
            parent_message_id VARCHAR,
            timestamp TIMESTAMP,
            sequence_num INTEGER,
            is_sidechain BOOLEAN DEFAULT FALSE,
            is_meta BOOLEAN DEFAULT FALSE,
            is_compact_summary BOOLEAN DEFAULT FALSE,
            is_api_error_message BOOLEAN DEFAULT FALSE,
            stop_reason VARCHAR,
            permission_mode_at_send VARCHAR,
            prompt_id VARCHAR,
            request_id VARCHAR,
            api_error_text VARCHAR,

            -- Tokens (R11 cache-arithmetic fix: cache_creation split per TTL)
            input_tokens INTEGER,
            output_tokens INTEGER,
            cache_creation_5m_tokens INTEGER,
            cache_creation_1h_tokens INTEGER,
            cache_read_tokens INTEGER,
            total_uncached_equivalent_tokens INTEGER,

            -- Counters / derived
            content_length INTEGER,
            content_block_count INTEGER,
            has_tool_use BOOLEAN DEFAULT FALSE,
            has_tool_result BOOLEAN DEFAULT FALSE,
            has_thinking BOOLEAN DEFAULT FALSE,
            word_count INTEGER,
            estimated_tokens INTEGER,
            response_time_seconds FLOAT,
            conversation_depth INTEGER,
            content_text VARCHAR
        )
    """
    )

    # Grain: one row per tool_use content block emitted by the assistant.
    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_tool_uses (
            -- Lineage
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            created_by_version_key VARCHAR NOT NULL,
            last_updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            last_updated_by_version_key VARCHAR NOT NULL,
            etl_run_id VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            hash_diff VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP,

            -- Degenerate dims
            entry_id VARCHAR NOT NULL,
            message_id VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,
            tool_use_id VARCHAR NOT NULL,

            -- Dimension FKs
            session_key VARCHAR,
            project_key VARCHAR,
            tool_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,

            -- Native
            tool_name VARCHAR NOT NULL,
            invoke_sequence_num INTEGER,
            caller_type VARCHAR,
            input_json VARCHAR,
            input_summary VARCHAR,
            timestamp TIMESTAMP
        )
    """
    )

    # =========================================================================
    # New entry-type facts (Phase C4)
    # =========================================================================
    # Each captures one Claude Code top-level entry type that the legacy ETL
    # either dropped entirely or sampled sparsely. All follow the v0.15
    # lineage + degenerate-dim convention.

    # Grain: one row per attachment entry attached to a user message.
    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_attachments (
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            created_by_version_key VARCHAR NOT NULL,
            last_updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            last_updated_by_version_key VARCHAR NOT NULL,
            etl_run_id VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            hash_diff VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP,

            entry_id VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,
            session_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            timestamp TIMESTAMP,
            attachment_type VARCHAR NOT NULL,
            attachment_json VARCHAR
        )
        """
    )

    # Grain: one row per progress entry emitted during tool/hook execution.
    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_progress_events (
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            created_by_version_key VARCHAR NOT NULL,
            last_updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            last_updated_by_version_key VARCHAR NOT NULL,
            etl_run_id VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            hash_diff VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP,

            entry_id VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,
            session_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            timestamp TIMESTAMP,
            data_type VARCHAR NOT NULL,
            tool_use_id VARCHAR,
            parent_tool_use_id VARCHAR,
            hook_name VARCHAR,
            hook_event VARCHAR,
            agent_id VARCHAR,
            data_json VARCHAR
        )
        """
    )

    # Grain: one row per system entry. Discriminated by `subtype`. Typed
    # columns for the 7 documented subtypes plus a JSON catch-all.
    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_system_events (
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            created_by_version_key VARCHAR NOT NULL,
            last_updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            last_updated_by_version_key VARCHAR NOT NULL,
            etl_run_id VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            hash_diff VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP,

            entry_id VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,
            session_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            timestamp TIMESTAMP,
            subtype VARCHAR NOT NULL,
            level VARCHAR,

            -- turn_duration
            duration_ms INTEGER,
            message_count INTEGER,
            -- stop_hook_summary
            hook_count INTEGER,
            prevented_continuation BOOLEAN,
            stop_reason VARCHAR,
            has_output BOOLEAN,
            -- api_error
            error_status INTEGER,
            error_type VARCHAR,
            retry_in_ms FLOAT,
            retry_attempt INTEGER,
            max_retries INTEGER,
            -- compact_boundary
            compact_trigger VARCHAR,
            compact_pre_tokens INTEGER,
            logical_parent_uuid VARCHAR,
            -- local_command / away_summary / bridge_status (text content)
            content VARCHAR,
            -- bridge_status
            bridge_url VARCHAR,

            payload_json VARCHAR
        )
        """
    )

    # Grain: one row per meta entry (custom-title, agent-name, permission-mode,
    # last-prompt) AT THE MOMENT IT OCCURRED. Time-series, NOT last-value-only.
    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_meta_events (
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            created_by_version_key VARCHAR NOT NULL,
            last_updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            last_updated_by_version_key VARCHAR NOT NULL,
            etl_run_id VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            hash_diff VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP,

            entry_id VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,
            session_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            timestamp TIMESTAMP,
            meta_type VARCHAR NOT NULL,
            meta_value VARCHAR
        )
        """
    )

    # Grain: one row per file-history-snapshot entry; carries trackedFileBackups
    # JSON for restore-point analysis.
    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_file_history_snapshots (
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            created_by_version_key VARCHAR NOT NULL,
            last_updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            last_updated_by_version_key VARCHAR NOT NULL,
            etl_run_id VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            hash_diff VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP,

            entry_id VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,
            session_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            timestamp TIMESTAMP,
            message_id_link VARCHAR,
            is_snapshot_update BOOLEAN,
            snapshot_json VARCHAR
        )
        """
    )

    # Grain: one row per queue-operation entry (user prompt enqueue/dequeue
    # mid-turn).
    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_queue_operations (
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            created_by_version_key VARCHAR NOT NULL,
            last_updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            last_updated_by_version_key VARCHAR NOT NULL,
            etl_run_id VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            hash_diff VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP,

            entry_id VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,
            session_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            timestamp TIMESTAMP,
            operation VARCHAR,
            content VARCHAR
        )
        """
    )

    # Grain: one row per pr-link entry binding a session to a GitHub PR.
    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_pr_links (
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            created_by_version_key VARCHAR NOT NULL,
            last_updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            last_updated_by_version_key VARCHAR NOT NULL,
            etl_run_id VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            hash_diff VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP,

            entry_id VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,
            session_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            timestamp TIMESTAMP,
            pr_number INTEGER,
            pr_url VARCHAR,
            pr_repository VARCHAR
        )
        """
    )

    # Grain: one row per tool_result event -- combines the tool_result content
    # block (truncated text) with the entry-level toolUseResult structured
    # payload (typed per-tool columns + JSON catch-all).
    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_tool_results (
            -- Lineage
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            created_by_version_key VARCHAR NOT NULL,
            last_updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            last_updated_by_version_key VARCHAR NOT NULL,
            etl_run_id VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            hash_diff VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP,

            -- Degenerate dims
            entry_id VARCHAR NOT NULL,
            message_id VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,
            tool_use_id VARCHAR NOT NULL,

            -- Dimension FKs
            session_key VARCHAR,
            project_key VARCHAR,
            tool_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,

            -- Native
            tool_name VARCHAR,
            timestamp TIMESTAMP,
            -- R16: tri-state nullable BOOLEAN
            is_error BOOLEAN,
            result_content_text VARCHAR,
            result_payload_json VARCHAR,

            -- Per-tool typed projections (NULL for tools without these fields)
            -- Bash / BashOutput
            bash_exit_code INTEGER,
            bash_interrupted BOOLEAN,
            bash_stdout_bytes INTEGER,
            bash_duration_ms FLOAT,
            -- Edit / MultiEdit
            edit_user_modified BOOLEAN,
            edit_replace_all BOOLEAN,
            edit_structured_patch_json VARCHAR,
            -- Read
            read_num_lines INTEGER,
            read_total_lines INTEGER,
            read_file_path VARCHAR,
            -- Write
            write_type VARCHAR,
            -- Glob
            glob_num_files INTEGER,
            glob_truncated BOOLEAN,
            -- Grep
            grep_mode VARCHAR,
            grep_num_files INTEGER,
            -- WebFetch
            webfetch_http_code INTEGER,
            webfetch_bytes INTEGER,
            -- Agent / Task (subagent rollup)
            agent_status VARCHAR,
            agent_total_duration_ms FLOAT,
            agent_total_tokens INTEGER,
            agent_total_tool_use_count INTEGER,
            agent_was_interrupted BOOLEAN,
            agent_subagent_type VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_session_summary (
            session_key VARCHAR,
            project_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            total_messages INTEGER,
            user_messages INTEGER,
            assistant_messages INTEGER,
            total_tool_calls INTEGER,
            total_thinking_blocks INTEGER,
            total_content_blocks INTEGER,
            total_errors INTEGER,
            unique_tools_used INTEGER,
            unique_files_touched INTEGER,
            max_conversation_depth INTEGER,
            total_estimated_tokens INTEGER,
            total_thinking_tokens INTEGER,
            total_tool_io_tokens INTEGER,
            session_duration_seconds INTEGER,
            first_timestamp TIMESTAMP,
            last_timestamp TIMESTAMP,
            total_estimated_tokens_incl_agents INTEGER,
            total_tool_calls_incl_agents INTEGER,
            total_errors_incl_agents INTEGER,
            total_duration_incl_agents INTEGER,
            actual_input_tokens BIGINT,
            actual_output_tokens BIGINT,
            cache_creation_tokens BIGINT,
            cache_read_tokens BIGINT,
            total_turn_duration_ms BIGINT,
            turn_count INTEGER,
            total_diagnostics INTEGER,
            total_hook_runs INTEGER,
            stop_count INTEGER,
            prevented_continuations INTEGER
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_file_operations (
            file_operation_id VARCHAR,
            tool_call_id VARCHAR,
            session_key VARCHAR,
            file_key VARCHAR,
            tool_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            operation_type VARCHAR,
            file_size_chars INTEGER,
            timestamp TIMESTAMP
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_errors (
            error_id VARCHAR,
            tool_call_id VARCHAR,
            session_key VARCHAR,
            tool_key VARCHAR,
            error_type VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            error_message TEXT,
            timestamp TIMESTAMP
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_tool_chain_steps (
            chain_step_id VARCHAR,
            session_key VARCHAR,
            chain_id VARCHAR,
            tool_call_id VARCHAR,
            tool_key VARCHAR,
            step_position INTEGER,
            prev_tool_key VARCHAR,
            next_tool_key VARCHAR,
            is_error BOOLEAN,
            time_since_prev_seconds FLOAT
        )
    """
    )

    # =========================================================================
    # Granular Dimensions (2)
    # =========================================================================

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_file (
            file_key VARCHAR,
            file_path VARCHAR,
            file_name VARCHAR,
            file_extension VARCHAR,
            directory_path VARCHAR,
            language VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_session_chain (
            chain_key VARCHAR,
            slug VARCHAR,
            project_key VARCHAR,
            first_session_key VARCHAR,
            last_session_key VARCHAR,
            session_count INTEGER,
            first_timestamp TIMESTAMP,
            last_timestamp TIMESTAMP,
            total_duration_seconds INTEGER
        )
    """
    )

    # =========================================================================
    # Granular Fact Tables (3)
    # =========================================================================

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_content_blocks (
            content_block_id VARCHAR,
            message_id VARCHAR,
            session_key VARCHAR,
            block_type VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            block_index INTEGER,
            content_length INTEGER,
            content_text TEXT,
            content_json JSON
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_code_blocks (
            code_block_id VARCHAR,
            message_id VARCHAR,
            session_key VARCHAR,
            language VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            block_index INTEGER,
            line_count INTEGER,
            char_count INTEGER,
            code_text TEXT
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_entity_mentions (
            mention_id VARCHAR,
            message_id VARCHAR,
            session_key VARCHAR,
            entity_type VARCHAR,
            entity_text VARCHAR,
            entity_normalized VARCHAR,
            context_snippet TEXT,
            position_start INTEGER,
            position_end INTEGER
        )
    """
    )

    # =========================================================================
    # Agent Delegation Tracking
    # =========================================================================

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_agent_delegations (
            delegation_key VARCHAR,
            parent_session_key VARCHAR,
            agent_session_key VARCHAR,
            task_tool_call_id VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            task_description TEXT,
            task_prompt TEXT,
            subagent_type VARCHAR,
            agent_output TEXT,
            completion_status VARCHAR,
            delegation_timestamp TIMESTAMP,
            completion_timestamp TIMESTAMP,
            match_confidence FLOAT,
            agent_tool_calls INTEGER,
            agent_errors INTEGER,
            agent_duration_seconds INTEGER,
            agent_estimated_tokens INTEGER
        )
    """
    )

    # =========================================================================
    # Plan Revision Tracking (ExitPlanMode chain)
    # =========================================================================

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_plan_revisions (
            revision_key VARCHAR,
            session_key VARCHAR,
            project_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            tool_call_id VARCHAR,
            invoke_message_id VARCHAR,
            result_message_id VARCHAR,
            revision_number INTEGER,
            parent_revision_key VARCHAR,
            plan_text TEXT,
            plan_char_count INTEGER,
            plan_estimated_tokens INTEGER,
            outcome VARCHAR,
            outcome_signal VARCHAR,
            user_feedback_message_id VARCHAR,
            user_feedback_text TEXT,
            plan_timestamp TIMESTAMP,
            resolved_timestamp TIMESTAMP,
            seconds_to_resolution DOUBLE
        )
    """
    )

    # =========================================================================
    # Cross-Session Bridge Table
    # =========================================================================

    conn.execute(
        """
        CREATE OR REPLACE TABLE bridge_session_file (
            session_file_key VARCHAR,
            session_key VARCHAR,
            file_key VARCHAR,
            first_operation_timestamp TIMESTAMP,
            last_operation_timestamp TIMESTAMP,
            operation_count INTEGER,
            read_count INTEGER,
            write_count INTEGER,
            edit_count INTEGER,
            total_chars_written INTEGER
        )
    """
    )

    # =========================================================================
    # New Fact Tables (token usage, turn durations, diagnostics, stop events)
    # =========================================================================

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_token_usage (
            usage_id VARCHAR,
            session_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            model_key VARCHAR,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cache_creation_input_tokens INTEGER,
            cache_read_input_tokens INTEGER,
            cache_ephemeral_1h_tokens INTEGER,
            cache_ephemeral_5m_tokens INTEGER,
            service_tier VARCHAR,
            speed VARCHAR,
            timestamp TIMESTAMP
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_turn_durations (
            turn_id VARCHAR,
            session_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            duration_ms INTEGER,
            message_count INTEGER,
            timestamp TIMESTAMP
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_diagnostics (
            diagnostic_id VARCHAR,
            session_key VARCHAR,
            file_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            severity VARCHAR,
            source VARCHAR,
            code VARCHAR,
            message TEXT,
            range_start_line INTEGER,
            range_start_col INTEGER,
            range_end_line INTEGER,
            range_end_col INTEGER,
            timestamp TIMESTAMP
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_stop_events (
            stop_event_id VARCHAR,
            session_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            stop_reason VARCHAR,
            hook_count INTEGER,
            has_output BOOLEAN,
            prevented_continuation BOOLEAN,
            hook_total_duration_ms INTEGER,
            hook_error_count INTEGER,
            timestamp TIMESTAMP
        )
    """
    )

    # =========================================================================
    # Prompt History (from ~/.claude/history.jsonl)
    # =========================================================================

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_prompt (
            prompt_key VARCHAR,
            session_key VARCHAR,
            project_path VARCHAR,
            project_name VARCHAR,
            display_text TEXT,
            timestamp TIMESTAMP,
            date_key INTEGER,
            time_key INTEGER,
            has_pasted_content BOOLEAN
        )
    """
    )

    # =========================================================================
    # Optional Tables (require pylate)
    # =========================================================================

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_session_embeddings (
            embedding_key VARCHAR,
            session_key VARCHAR,
            content_type VARCHAR,
            embedding_model VARCHAR,
            embedding_dim INTEGER,
            mean_embedding FLOAT[64],
            embedded_at TIMESTAMP,
            content_hash VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_tool_input_params (
            param_id VARCHAR,
            tool_call_id VARCHAR,
            session_key VARCHAR,
            param_key VARCHAR,
            param_value_text VARCHAR,
            param_value_number FLOAT,
            param_value_bool BOOLEAN
        )
    """
    )

    # =========================================================================
    # Semantic Views (10)
    # =========================================================================

    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_sessions AS
        SELECT
            ds.session_id,
            ds.cwd,
            ds.git_branch,
            ds.version,
            ds.first_timestamp AS session_datetime,
            ds.last_timestamp,
            ds.intent,
            ds.complexity,
            ds.outcome,
            ds.domain,
            dp.project_name,
            dp.project_path,
            fss.total_messages,
            fss.user_messages,
            fss.assistant_messages,
            fss.total_tool_calls,
            fss.total_thinking_blocks,
            fss.total_errors,
            fss.unique_tools_used,
            fss.unique_files_touched,
            fss.max_conversation_depth,
            fss.total_estimated_tokens,
            fss.total_estimated_tokens_incl_agents,
            fss.total_tool_calls_incl_agents,
            fss.total_errors_incl_agents,
            fss.total_duration_incl_agents,
            fss.session_duration_seconds,
            dd.full_date,
            dd.day_name,
            dd.month_name,
            dd.year,
            dd.is_weekend,
            dti.hour,
            dti.time_of_day
        FROM fact_session_summary fss
        JOIN dim_session ds ON fss.session_key = ds.session_key
        JOIN dim_project dp ON fss.project_key = dp.project_key
        LEFT JOIN dim_date dd ON fss.date_key = dd.date_key
        LEFT JOIN dim_time dti ON fss.time_key = dti.time_key
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_messages AS
        SELECT
            fm.message_id,
            fm.timestamp,
            fm.content_text,
            fm.content_length,
            fm.word_count,
            fm.estimated_tokens,
            fm.has_tool_use,
            fm.has_thinking,
            fm.response_time_seconds,
            fm.conversation_depth,
            fm.message_type,
            dm.model_name,
            dm.model_family,
            ds.session_id,
            ds.cwd,
            dp.project_name,
            dd.full_date,
            dd.day_name,
            dt.hour,
            dt.time_of_day
        FROM fact_messages fm
        LEFT JOIN dim_model dm ON fm.model_key = dm.model_key
        JOIN dim_session ds ON fm.session_key = ds.session_key
        LEFT JOIN dim_project dp ON fm.project_key = dp.project_key
        LEFT JOIN dim_date dd ON fm.date_key = dd.date_key
        LEFT JOIN dim_time dt ON fm.time_key = dt.time_key
    """
    )

    # Backwards-compat view over the v0.15 fact_tool_uses + fact_tool_results
    # split. Old queries that select tool_use_id, tool_name, is_error etc.
    # keep working. New analytics should query the two facts directly.
    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_tool_calls AS
        SELECT
            ftu.tool_use_id,
            ftu.tool_name,
            dt.tool_category,
            ftu.input_summary,
            ftu.input_json,
            ftu.caller_type,
            ftu.timestamp AS invoke_timestamp,
            ftr.timestamp AS result_timestamp,
            ftr.is_error,
            ftr.result_content_text,
            ftr.result_payload_json,
            ftr.bash_exit_code,
            ftr.bash_interrupted,
            ftr.bash_duration_ms,
            ftr.edit_user_modified,
            ftr.read_num_lines,
            ftr.read_total_lines,
            ftr.read_file_path,
            ftr.webfetch_http_code,
            ftr.agent_status,
            ftr.agent_total_duration_ms,
            ds.session_id,
            ds.cwd,
            dp.project_name,
            dd.full_date,
            dti.hour,
            dti.time_of_day
        FROM fact_tool_uses ftu
        LEFT JOIN fact_tool_results ftr ON ftr.tool_use_id = ftu.tool_use_id
        LEFT JOIN dim_tool dt ON ftu.tool_key = dt.tool_key
        LEFT JOIN dim_session ds ON ftu.session_key = ds.session_key
        LEFT JOIN dim_project dp ON ds.project_key = dp.project_key
        LEFT JOIN dim_date dd ON ftu.date_key = dd.date_key
        LEFT JOIN dim_time dti ON ftu.time_key = dti.time_key
        WHERE ftu.is_deleted = FALSE
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_file_operations AS
        SELECT
            ffo.operation_type,
            ffo.file_size_chars,
            ffo.timestamp,
            df.file_path,
            df.file_name,
            df.file_extension,
            df.directory_path,
            df.language,
            dt.tool_name,
            dt.tool_category,
            ds.session_id,
            dp.project_name,
            dd.full_date,
            dti.time_of_day
        FROM fact_file_operations ffo
        JOIN dim_file df ON ffo.file_key = df.file_key
        JOIN dim_tool dt ON ffo.tool_key = dt.tool_key
        JOIN dim_session ds ON ffo.session_key = ds.session_key
        LEFT JOIN dim_project dp ON ds.project_key = dp.project_key
        LEFT JOIN dim_date dd ON ffo.date_key = dd.date_key
        LEFT JOIN dim_time dti ON ffo.time_key = dti.time_key
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_session_chains AS
        SELECT
            dsc.chain_key,
            dsc.slug,
            dsc.session_count,
            CAST(dsc.first_timestamp AS DATE) AS chain_start_date,
            dsc.first_timestamp AS chain_first_timestamp,
            dsc.last_timestamp AS chain_last_timestamp,
            dsc.total_duration_seconds,
            ds.session_id,
            ds.session_key,
            ds.is_agent,
            ds.depth_level,
            dp.project_name,
            fss.total_messages,
            fss.total_tool_calls,
            fss.session_duration_seconds
        FROM dim_session_chain dsc
        JOIN dim_session ds ON ds.chain_key = dsc.chain_key
        LEFT JOIN dim_project dp ON ds.project_key = dp.project_key
        LEFT JOIN fact_session_summary fss ON ds.session_key = fss.session_key
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_agent_delegations AS
        SELECT
            fad.delegation_key,
            fad.task_description,
            fad.task_prompt,
            fad.subagent_type,
            fad.completion_status,
            fad.match_confidence,
            dd.full_date AS delegation_date,
            fad.delegation_timestamp,
            fad.completion_timestamp,
            dti.time_of_day,
            fad.agent_tool_calls,
            fad.agent_errors,
            fad.agent_duration_seconds,
            fad.agent_estimated_tokens,
            ps.session_id AS parent_session_id,
            ps.cwd AS parent_cwd,
            ags.session_id AS agent_session_id,
            ags.depth_level AS agent_depth_level,
            dp.project_name
        FROM fact_agent_delegations fad
        JOIN dim_session ps ON fad.parent_session_key = ps.session_key
        JOIN dim_session ags ON fad.agent_session_key = ags.session_key
        LEFT JOIN dim_project dp ON ps.project_key = dp.project_key
        LEFT JOIN dim_date dd ON fad.date_key = dd.date_key
        LEFT JOIN dim_time dti ON fad.time_key = dti.time_key
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_plan_revisions AS
        SELECT
            fpr.revision_key,
            fpr.revision_number,
            fpr.parent_revision_key,
            fpr.plan_text,
            fpr.plan_char_count,
            fpr.plan_estimated_tokens,
            fpr.outcome,
            fpr.outcome_signal,
            fpr.user_feedback_text,
            fpr.plan_timestamp,
            fpr.resolved_timestamp,
            fpr.seconds_to_resolution,
            dd.full_date AS plan_date,
            dti.time_of_day,
            ds.session_id,
            ds.cwd,
            ds.depth_level,
            ds.agent_type,
            dp.project_name
        FROM fact_plan_revisions fpr
        JOIN dim_session ds ON fpr.session_key = ds.session_key
        LEFT JOIN dim_project dp ON fpr.project_key = dp.project_key
        LEFT JOIN dim_date dd ON fpr.date_key = dd.date_key
        LEFT JOIN dim_time dti ON fpr.time_key = dti.time_key
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_file_evolution AS
        SELECT
            df.file_path,
            df.file_name,
            df.file_extension,
            df.directory_path,
            df.language,
            COUNT(DISTINCT bsf.session_key) AS session_count,
            SUM(bsf.operation_count) AS total_operations,
            SUM(bsf.read_count) AS total_reads,
            SUM(bsf.write_count) AS total_writes,
            SUM(bsf.edit_count) AS total_edits,
            SUM(bsf.total_chars_written) AS total_chars_written,
            CAST(MIN(bsf.first_operation_timestamp) AS DATE) AS first_seen_date,
            MIN(bsf.first_operation_timestamp) AS first_seen,
            CAST(MAX(bsf.last_operation_timestamp) AS DATE) AS last_seen_date,
            MAX(bsf.last_operation_timestamp) AS last_seen
        FROM bridge_session_file bsf
        JOIN dim_file df ON bsf.file_key = df.file_key
        GROUP BY df.file_path, df.file_name, df.file_extension, df.directory_path, df.language
        HAVING COUNT(DISTINCT bsf.session_key) > 1
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_tool_patterns AS
        SELECT
            dt1.tool_name AS tool_name,
            dt2.tool_name AS next_tool_name,
            COUNT(*) AS frequency,
            AVG(ftcs.time_since_prev_seconds) AS avg_time_between,
            SUM(CASE WHEN ftcs.is_error THEN 1 ELSE 0 END) AS error_count,
            ROUND(SUM(CASE WHEN ftcs.is_error THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS error_rate_pct
        FROM fact_tool_chain_steps ftcs
        JOIN dim_tool dt1 ON ftcs.tool_key = dt1.tool_key
        LEFT JOIN dim_tool dt2 ON ftcs.next_tool_key = dt2.tool_key
        WHERE ftcs.next_tool_key IS NOT NULL
        GROUP BY dt1.tool_name, dt2.tool_name
        HAVING COUNT(*) >= 2
        ORDER BY frequency DESC
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_project_context AS
        SELECT
            ds.session_id,
            dp.project_name,
            ds.slug,
            CAST(ds.first_timestamp AS DATE) AS session_date,
            ds.first_timestamp AS created_at,
            dti.time_of_day,
            ds.git_branch,
            ds.intent,
            ds.complexity,
            ds.outcome,
            ds.domain,
            ds.first_user_message,
            ds.last_assistant_message,
            fss.total_messages,
            fss.user_messages,
            fss.assistant_messages,
            fss.total_tool_calls,
            fss.unique_tools_used,
            fss.total_errors,
            fss.total_estimated_tokens,
            fss.total_estimated_tokens_incl_agents,
            fss.total_tool_calls_incl_agents,
            fss.total_errors_incl_agents
        FROM dim_session ds
        JOIN dim_project dp ON ds.project_key = dp.project_key
        LEFT JOIN fact_session_summary fss ON ds.session_key = fss.session_key
        LEFT JOIN dim_time dti ON fss.time_key = dti.time_key
        ORDER BY ds.first_timestamp DESC
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_project_files AS
        SELECT
            dp.project_name,
            df.file_path,
            df.file_extension,
            df.language,
            COUNT(DISTINCT ds.session_id) AS sessions_touching_file,
            SUM(bsf.read_count) AS total_reads,
            SUM(bsf.write_count) AS total_writes,
            SUM(bsf.edit_count) AS total_edits,
            CAST(MAX(ds.first_timestamp) AS DATE) AS last_touched_date,
            MAX(ds.first_timestamp) AS last_touched
        FROM bridge_session_file bsf
        JOIN dim_file df ON bsf.file_key = df.file_key
        JOIN dim_session ds ON bsf.session_key = ds.session_key
        JOIN dim_project dp ON ds.project_key = dp.project_key
        GROUP BY dp.project_name, df.file_path, df.file_extension, df.language
        ORDER BY sessions_touching_file DESC
    """
    )

    # =========================================================================
    # New Semantic Views (token usage, cost analysis)
    # =========================================================================

    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_token_usage AS
        SELECT
            ftu.usage_id,
            ftu.input_tokens,
            ftu.output_tokens,
            ftu.cache_creation_input_tokens,
            ftu.cache_read_input_tokens,
            ftu.service_tier,
            ftu.speed,
            ftu.timestamp,
            dm.model_name,
            dm.model_family,
            ds.session_id,
            ds.cwd,
            dp.project_name,
            dd.full_date,
            dti.time_of_day
        FROM fact_token_usage ftu
        LEFT JOIN dim_model dm ON ftu.model_key = dm.model_key
        JOIN dim_session ds ON ftu.session_key = ds.session_key
        LEFT JOIN dim_project dp ON ds.project_key = dp.project_key
        LEFT JOIN dim_date dd ON ftu.date_key = dd.date_key
        LEFT JOIN dim_time dti ON ftu.time_key = dti.time_key
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_cost_analysis AS
        SELECT
            ds.session_id,
            dp.project_name,
            ds.entrypoint,
            ds.custom_title,
            ds.intent,
            ds.complexity,
            dd.full_date,
            fss.session_duration_seconds,
            fss.actual_input_tokens,
            fss.actual_output_tokens,
            fss.cache_creation_tokens,
            fss.cache_read_tokens,
            fss.total_turn_duration_ms,
            fss.turn_count,
            CASE WHEN fss.cache_read_tokens > 0
                      AND (COALESCE(fss.actual_input_tokens, 0) + fss.cache_read_tokens) > 0
                 THEN ROUND(100.0 * fss.cache_read_tokens
                      / (COALESCE(fss.actual_input_tokens, 0) + fss.cache_read_tokens), 1)
                 ELSE 0 END AS cache_hit_rate_pct,
            fss.total_estimated_tokens,
            fss.total_messages,
            fss.total_tool_calls
        FROM fact_session_summary fss
        JOIN dim_session ds ON fss.session_key = ds.session_key
        JOIN dim_project dp ON fss.project_key = dp.project_key
        LEFT JOIN dim_date dd ON fss.date_key = dd.date_key
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_prompt_history AS
        SELECT
            dp.prompt_key,
            dp.display_text,
            dp.timestamp,
            dp.project_name,
            dp.project_path,
            dp.has_pasted_content,
            ds.session_id,
            ds.intent,
            ds.complexity,
            ds.custom_title,
            ds.entrypoint,
            fss.total_messages,
            fss.actual_input_tokens,
            fss.actual_output_tokens,
            dd.full_date
        FROM dim_prompt dp
        LEFT JOIN dim_session ds ON dp.session_key = ds.session_key
        LEFT JOIN fact_session_summary fss ON ds.session_key = fss.session_key
        LEFT JOIN dim_date dd ON dp.date_key = dd.date_key
    """
    )

    return conn
