# path-privacy: skip-file -- references universal Claude Code data paths (not personal)
"""Star schema DDL - creates the dimensional model tables.

42 tables + 15 views. Tiny lookup dimensions (message_type, content_block_type,
error_type, entity_type, programming_language) replaced by degenerate VARCHAR
columns on fact tables. LLM enrichment tables removed entirely -- replaced by
heuristic classification columns on dim_session.
"""

import duckdb

from ccutils.schemas.star.utils import get_time_of_day


def create_star_schema(db_path):
    """Create DuckDB database with star schema for transcript analytics.

    Authoritative table inventory lives in CLAUDE.md ("Star Schema Tables")
    and docs/STAR_SCHEMA.md; per-populator wiring in etl/orchestrator.py.
    Highlights:
    - Core dimensions: session, project, tool, model, file, session_chain,
      prompt, facet_type. dim_time is seeded here (1440 rows); dim_date rows
      are inserted during ETL for every staged calendar date.
    - v0.15 facts: messages, tool_uses + tool_results, token_usage, the
      entry-type facts (attachments/progress/system/meta/...), file
      operations, plan revisions, agent delegations, errors, chain steps,
      session facets, session summary.
    - Staging: stg_log_entries (Tier 2 of the four-tier pipeline).
    - 15 semantic views (semantic_*), created after _apply_column_migrations
      so views can reference migrated columns on pre-existing warehouses.

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
        CREATE TABLE IF NOT EXISTS dim_etl_version (
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
        CREATE TABLE IF NOT EXISTS fact_etl_runs (
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
        CREATE TABLE IF NOT EXISTS meta_schema_version (
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
        CREATE TABLE IF NOT EXISTS stg_log_entries (
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

    # =========================================================================
    # Core Dimension Tables (6)
    # =========================================================================

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dim_tool (
            tool_key VARCHAR,
            tool_name VARCHAR,
            tool_category VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dim_model (
            model_key VARCHAR,
            model_name VARCHAR,
            model_family VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dim_project (
            project_key VARCHAR,
            project_path VARCHAR,
            project_name VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dim_session (
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
        CREATE TABLE IF NOT EXISTS dim_date (
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
        CREATE TABLE IF NOT EXISTS dim_time (
            time_key INTEGER,
            hour INTEGER,
            minute INTEGER,
            time_of_day VARCHAR
        )
    """
    )

    # Fixed-cardinality dimension: seed the 1440 minutes not already
    # present (per-minute anti-join, NOT a whole-table emptiness guard --
    # a legacy warehouse with a PARTIALLY populated dim_time would else
    # never be completed). time_of_day comes from get_time_of_day, the
    # single source of truth, so SQL and Python labels can't drift.
    existing_time_keys = {
        r[0] for r in conn.execute("SELECT time_key FROM dim_time").fetchall()
    }
    missing_time_rows = [
        (h * 100 + m, h, m, get_time_of_day(h))
        for h in range(24)
        for m in range(60)
        if h * 100 + m not in existing_time_keys
    ]
    if missing_time_rows:
        conn.executemany(
            "INSERT INTO dim_time VALUES (?, ?, ?, ?)", missing_time_rows
        )

    # =========================================================================
    # Core Fact Tables (6)
    # =========================================================================

    # Grain: one row per user or assistant entry in a session.
    # See docs/STAR_SCHEMA.md for the lineage column convention shared by
    # every fact table in v0.15+.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_messages (
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
        CREATE TABLE IF NOT EXISTS fact_tool_uses (
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
        CREATE TABLE IF NOT EXISTS fact_attachments (
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
        CREATE TABLE IF NOT EXISTS fact_progress_events (
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
        CREATE TABLE IF NOT EXISTS fact_system_events (
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
        CREATE TABLE IF NOT EXISTS fact_meta_events (
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
        CREATE TABLE IF NOT EXISTS fact_file_history_snapshots (
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
        CREATE TABLE IF NOT EXISTS fact_queue_operations (
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
        CREATE TABLE IF NOT EXISTS fact_pr_links (
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
        CREATE TABLE IF NOT EXISTS fact_tool_results (
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
            agent_subagent_type VARCHAR,
            agent_id VARCHAR
        )
    """
    )

    # Grain: one row per session. Pre-aggregated rollups over the v0.15
    # entry-type facts. Note on Kimball "facts don't join to facts": the
    # aggregation joins happen IN the populator (ETL time); query consumers
    # see one self-contained row per session and never join facts to facts.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_session_summary (
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            created_by_version_key VARCHAR NOT NULL,
            last_updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            last_updated_by_version_key VARCHAR NOT NULL,
            etl_run_id VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            hash_diff VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP,

            session_id VARCHAR NOT NULL,
            session_key VARCHAR,
            project_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            first_timestamp TIMESTAMP,
            last_timestamp TIMESTAMP,
            session_duration_seconds DOUBLE,

            -- From fact_messages
            total_messages INTEGER,
            user_messages INTEGER,
            assistant_messages INTEGER,
            total_thinking_blocks INTEGER,

            -- From fact_token_usage (R11 split by pricing tier)
            total_input_tokens BIGINT,
            total_output_tokens BIGINT,
            total_cache_creation_5m_tokens BIGINT,
            total_cache_creation_1h_tokens BIGINT,
            total_cache_creation_total_tokens BIGINT,
            total_cache_read_tokens BIGINT,
            total_uncached_equivalent_tokens BIGINT,
            api_response_count INTEGER,

            -- From fact_tool_uses / fact_tool_results
            total_tool_uses INTEGER,
            unique_tools_used INTEGER,
            total_tool_results INTEGER,
            total_tool_errors INTEGER,
            total_bash_interrupted INTEGER,

            -- From fact_system_events
            total_api_errors INTEGER,
            total_compactions INTEGER,
            total_turn_durations_ms BIGINT,
            turn_count INTEGER,
            total_stop_events INTEGER,
            total_prevented_continuations INTEGER,

            -- From fact_progress_events / fact_attachments
            total_progress_events INTEGER,
            total_hook_progress_events INTEGER,
            total_bash_progress_events INTEGER,
            total_attachments INTEGER,
            total_diagnostics INTEGER,
            total_hook_successes INTEGER,

            -- From fact_meta_events
            permission_mode_transition_count INTEGER,
            current_permission_mode VARCHAR,

            -- From fact_file_history_snapshots
            total_file_history_snapshots INTEGER
        )
    """
    )

    conn.execute(
        """
        -- Grain: one row per file-touching tool call (Read/Write/Edit/MultiEdit
        -- /Glob/Grep/NotebookEdit/etc.). Derived from fact_tool_uses joined
        -- to fact_tool_results on tool_use_id.
        CREATE TABLE IF NOT EXISTS fact_file_operations (
            -- Lineage (every v0.15 fact carries this block)
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
            tool_use_id VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,

            -- Dimension FKs
            session_key VARCHAR,
            file_key VARCHAR,
            tool_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,

            -- Native
            operation_type VARCHAR,
            file_path VARCHAR,
            file_size_chars INTEGER,
            timestamp TIMESTAMP
        )
    """
    )

    conn.execute(
        """
        -- Grain: one row per failed tool call. Derived from
        -- fact_tool_results where is_error = TRUE. error_type is
        -- classified by zero-dep regex rules in
        -- ccutils.etl.heuristics.classify_error_type.
        CREATE TABLE IF NOT EXISTS fact_errors (
            -- Lineage (every v0.15 fact carries this block)
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            created_by_version_key VARCHAR NOT NULL,
            last_updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            last_updated_by_version_key VARCHAR NOT NULL,
            etl_run_id VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            hash_diff VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP,

            -- Degenerate
            error_id VARCHAR NOT NULL,
            tool_use_id VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,

            -- Dimension FKs
            session_key VARCHAR,
            tool_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,

            -- Native
            error_type VARCHAR,
            error_message TEXT,
            timestamp TIMESTAMP
        )
    """
    )

    conn.execute(
        """
        -- Grain: one row per (session, tool_use, step_position) tuple
        -- recording tool-sequence patterns. A "chain" is the contiguous
        -- block of tool_uses emitted by a single assistant turn (one
        -- message_id). prev_tool_key / next_tool_key let queries like
        -- "after I Read, do I usually Edit?" work without window functions
        -- on every query.
        CREATE TABLE IF NOT EXISTS fact_tool_chain_steps (
            -- Lineage (every v0.15 fact carries this block)
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            created_by_version_key VARCHAR NOT NULL,
            last_updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            last_updated_by_version_key VARCHAR NOT NULL,
            etl_run_id VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            hash_diff VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP,

            -- Degenerate
            chain_step_id VARCHAR NOT NULL,
            tool_use_id VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,

            -- Dimension FKs
            session_key VARCHAR,
            tool_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,

            -- Chain context
            chain_id VARCHAR,
            step_position INTEGER,
            prev_tool_key VARCHAR,
            next_tool_key VARCHAR,

            -- Outcome of this tool
            is_error BOOLEAN,
            time_since_prev_seconds FLOAT,
            timestamp TIMESTAMP
        )
    """
    )

    # =========================================================================
    # Granular Dimensions (2)
    # =========================================================================

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dim_file (
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
        CREATE TABLE IF NOT EXISTS dim_session_chain (
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
        CREATE TABLE IF NOT EXISTS fact_content_blocks (
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
        CREATE TABLE IF NOT EXISTS fact_code_blocks (
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
        CREATE TABLE IF NOT EXISTS fact_entity_mentions (
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
        -- Grain: one row per Task tool_use (parent-side agent spawn).
        -- Joins fact_tool_uses(Task) to fact_tool_results to get the
        -- agent rollup metrics from the v0.15 toolUseResult capture (R1).
        --
        -- agent_session_key / parent_session_key are NULL for now -- the
        -- cross-session subagent linkage (reading .meta.json sidecars
        -- to mark dim_session.is_agent / parent_session_key) is a
        -- separate Phase D follow-up. session_id on this fact = parent
        -- session that did the delegating.
        CREATE TABLE IF NOT EXISTS fact_agent_delegations (
            -- Lineage (every v0.15 fact carries this block)
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            created_by_version_key VARCHAR NOT NULL,
            last_updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            last_updated_by_version_key VARCHAR NOT NULL,
            etl_run_id VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            hash_diff VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP,

            -- Degenerate
            delegation_key VARCHAR NOT NULL,
            tool_use_id VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,

            -- Dimension FKs (session_key = parent session)
            session_key VARCHAR,
            parent_session_key VARCHAR,
            agent_session_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,

            -- Task input (from fact_tool_uses.input_json)
            task_description TEXT,
            task_prompt TEXT,
            subagent_type VARCHAR,

            -- Agent rollup (from fact_tool_results.agent_* columns)
            agent_status VARCHAR,
            agent_total_duration_ms FLOAT,
            agent_total_tokens INTEGER,
            agent_total_tool_use_count INTEGER,
            agent_was_interrupted BOOLEAN,
            agent_output_text TEXT,

            -- Timing
            timestamp TIMESTAMP,
            delegation_timestamp TIMESTAMP,
            completion_timestamp TIMESTAMP,
            seconds_to_completion DOUBLE
        )
    """
    )

    # =========================================================================
    # Plan Revision Tracking (ExitPlanMode chain)
    # =========================================================================

    conn.execute(
        """
        -- Grain: one row per ExitPlanMode tool_use. Outcome is classified
        -- from the v0.15 structural signal (fact_tool_results.is_error,
        -- tri-state nullable BOOLEAN) instead of string-matching against
        -- truncated tool_result content -- the original rethink driver.
        --
        -- outcome:
        --   'superseded' -- a later ExitPlanMode exists in the same session
        --   'accepted'   -- tool_result.is_error = FALSE
        --   'rejected'   -- tool_result.is_error = TRUE
        --   'pending'    -- no tool_result yet (session in flight)
        --   'unknown'    -- tool_result present but is_error is NULL
        --
        -- parent_revision_key chains revisions within a session by timestamp.
        CREATE TABLE IF NOT EXISTS fact_plan_revisions (
            -- Lineage (every v0.15 fact carries this block)
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            created_by_version_key VARCHAR NOT NULL,
            last_updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            last_updated_by_version_key VARCHAR NOT NULL,
            etl_run_id VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            hash_diff VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP,

            -- Degenerate
            revision_key VARCHAR NOT NULL,
            tool_use_id VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,

            -- Dimension FKs
            session_key VARCHAR,
            project_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,

            -- Chain
            revision_number INTEGER,
            parent_revision_key VARCHAR,

            -- Plan content + timing
            plan_text TEXT,
            plan_file_path VARCHAR,
            plan_char_count INTEGER,
            plan_timestamp TIMESTAMP,
            resolved_timestamp TIMESTAMP,
            seconds_to_resolution DOUBLE,

            -- Outcome (structural classification from fact_tool_results.is_error)
            outcome VARCHAR,
            outcome_signal VARCHAR,
            -- Feedback message that followed a rejection (next user text)
            user_feedback_message_id VARCHAR,
            user_feedback_text TEXT,
            -- Mirror for query convenience
            timestamp TIMESTAMP
        )
    """
    )

    # =========================================================================
    # Cross-Session Bridge Table
    # =========================================================================

    conn.execute(
        """
        -- Grain: one row per (session, file) touched together. Aggregate
        -- over fact_file_operations. Idempotent re-builds drop-and-reload.
        CREATE TABLE IF NOT EXISTS bridge_session_file (
            -- Lineage (every v0.15 fact carries this block)
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            created_by_version_key VARCHAR NOT NULL,
            last_updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            last_updated_by_version_key VARCHAR NOT NULL,
            etl_run_id VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            hash_diff VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP,

            -- Degenerate
            session_id VARCHAR NOT NULL,

            -- Routing keys
            session_file_key VARCHAR NOT NULL,
            session_key VARCHAR,
            file_key VARCHAR,
            -- Derived from first_operation_timestamp for date/time-of-day filtering
            date_key INTEGER,
            time_key INTEGER,

            -- Aggregate measures
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

    # Grain: one row per assistant API response that carried `usage` data.
    # R11 cache-arithmetic fix: cache_creation split into _5m and _1h tiers
    # (1.25x and 2x pricing respectively); total_uncached_equivalent_tokens
    # is the "what would this have cost with no caching" derivation.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_token_usage (
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
            project_key VARCHAR,
            model_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            timestamp TIMESTAMP,

            -- Anthropic Messages API `usage` shape (R11/R18 corrections applied)
            input_tokens INTEGER,
            output_tokens INTEGER,
            cache_creation_5m_tokens INTEGER,
            cache_creation_1h_tokens INTEGER,
            cache_creation_total_tokens INTEGER,
            cache_read_tokens INTEGER,
            total_uncached_equivalent_tokens INTEGER,

            -- Pricing/tier metadata
            service_tier VARCHAR,
            speed VARCHAR,
            inference_geo VARCHAR,
            server_tool_use_web_search_requests INTEGER,
            server_tool_use_web_fetch_requests INTEGER
        )
    """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_turn_durations (
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
        -- Grain: one row per LSP diagnostic emitted during a session.
        -- Derived from fact_attachments.attachment_type='diagnostics'; the
        -- attachment_json carries a list of diagnostic objects that get
        -- flattened here. natural_key is diagnostic_id = md5(entry_id || index).
        CREATE TABLE IF NOT EXISTS fact_diagnostics (
            -- Lineage (every v0.15 fact carries this block)
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            created_by_version_key VARCHAR NOT NULL,
            last_updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            last_updated_by_version_key VARCHAR NOT NULL,
            etl_run_id VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            hash_diff VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP,

            -- Degenerate
            diagnostic_id VARCHAR NOT NULL,
            entry_id VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,

            -- Dimension FKs
            session_key VARCHAR,
            file_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,

            -- Native
            file_path VARCHAR,
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
        CREATE TABLE IF NOT EXISTS fact_stop_events (
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
        CREATE TABLE IF NOT EXISTS dim_prompt (
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
        CREATE TABLE IF NOT EXISTS fact_session_embeddings (
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
        CREATE TABLE IF NOT EXISTS fact_tool_input_params (
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
    # Facet & Cluster Pipeline (docs/FACET_CLUSTER_PIPELINE.md)
    # =========================================================================
    # dim_facet_type        - registry of facet definitions (Tier 1/2/3)
    # fact_session_facets   - one row per (session, facet, prompt_version);
    #                         typed value columns; NO embedding here.
    # fact_facet_embeddings - one row per (session, facet, model, model_version);
    #                         FLOAT[384] so DuckDB array_cosine_similarity works
    #                         natively.
    #
    # Embeddings are split off into their own table on purpose: keeps the EAV
    # facet table lean for SQL scans, lets DuckDB native array ops work without
    # a vector DB, and absorbs future model-version coexistence as new rows
    # rather than destructive overwrites of structured-value facets.

    # IMPORTANT: dim_facet_type uses CREATE IF NOT EXISTS, not CREATE OR
    # REPLACE. The registry holds historical prompt_version rows that
    # fact_session_facets references by facet_type_key -- wiping them on
    # every create_star_schema() call (the CLI path) would orphan fact rows.
    # Tier 1 seeds below use INSERT ... ON CONFLICT DO NOTHING for the same
    # reason: existing seed rows from a prior run survive untouched.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dim_facet_type (
            facet_type_key VARCHAR PRIMARY KEY,
            facet_id VARCHAR NOT NULL,
            facet_name VARCHAR NOT NULL,
            tier INTEGER NOT NULL,
            method VARCHAR NOT NULL,
            output_type VARCHAR NOT NULL,
            prompt_text VARCHAR,
            prompt_version VARCHAR,
            embedding_model VARCHAR,
            notes VARCHAR,
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
        )
        """
    )

    # Seed Tier 1 facets F01..F19. Names + output types mirror
    # FACET_CLUSTER_PIPELINE.md §3 "Tier 1" exactly. Tier 1 is computed by SQL
    # off existing facts so prompt_text / prompt_version stay NULL.
    # `notes` carries data-level caveats so future analytical queries can see
    # them (F16's UTC-hour limitation is the current example).
    _tier1_seeds = [
        ("F01", "session_intent", "enum", None),
        ("F02", "session_complexity", "enum", None),
        ("F03", "session_outcome", "enum", None),
        ("F04", "session_domain", "enum", None),
        ("F05", "error_signature", "json", None),
        ("F06", "tool_mix", "json", None),
        ("F07", "tool_bigram_top3", "json", None),
        ("F08", "loc_delta", "int",
         "Proxy: count of write+edit operations. Literal LOC requires "
         "unpacking fact_tool_results.edit_structured_patch_json."),
        ("F09", "file_extensions_touched", "json", None),
        ("F10", "repo_slug", "text", None),
        ("F11", "model_mix", "json", None),
        ("F12", "duration_seconds", "int", None),
        ("F13", "agent_depth", "int", None),
        ("F14", "human_message_count", "int", None),
        ("F15", "tokens_in", "int", None),
        ("F16", "local_hour", "int",
         "Hour-of-day in UTC; user-local TZ not preserved at capture time."),
        ("F17", "had_subagents", "bool", None),
        ("F18", "pr_referenced", "bool", None),
        ("F19", "had_plan_revision", "bool", None),
    ]
    conn.executemany(
        """
        INSERT INTO dim_facet_type
            (facet_type_key, facet_id, facet_name, tier, method, output_type,
             notes)
        VALUES (md5(? || '|' || ''), ?, ?, 1, 'computed', ?, ?)
        ON CONFLICT (facet_type_key) DO NOTHING
        """,
        [(fid, fid, fname, otype, notes)
         for (fid, fname, otype, notes) in _tier1_seeds],
    )

    # Seed Tier 2 facets from the catalog module. Each FacetSpec produces
    # one row keyed by md5(facet_id || '|' || prompt_version) -- bumping
    # prompt_version on a FacetSpec adds a new row rather than overwriting
    # (ON CONFLICT DO NOTHING preserves the historical row that existing
    # fact_session_facets rows reference). The description text becomes
    # prompt_text in the registry so it's queryable / auditable post-hoc.
    from ccutils.etl.facets.catalog import FACET_SPECS as _TIER2_SPECS

    conn.executemany(
        """
        INSERT INTO dim_facet_type
            (facet_type_key, facet_id, facet_name, tier, method, output_type,
             prompt_text, prompt_version)
        VALUES (md5(? || '|' || ?), ?, ?, 2, 'llm', ?, ?, ?)
        ON CONFLICT (facet_type_key) DO NOTHING
        """,
        [
            (
                spec.facet_id, spec.prompt_version,
                spec.facet_id, spec.facet_name, spec.output_type,
                spec.description, spec.prompt_version,
            )
            for spec in _TIER2_SPECS
        ],
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_session_facets (
            -- Lineage envelope (v0.15 convention)
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            created_by_version_key VARCHAR NOT NULL,
            last_updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            last_updated_by_version_key VARCHAR NOT NULL,
            etl_run_id VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            hash_diff VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP,

            -- Natural key (synthesized, single-column for lineage_upsert):
            -- md5(session_id || '|' || facet_id || '|' || COALESCE(prompt_version, ''))
            facet_row_key VARCHAR NOT NULL,

            -- Natural key parts + degenerate dims
            session_key VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,
            facet_type_key VARCHAR NOT NULL,
            prompt_version VARCHAR,

            -- Typed value columns (one populated per row depending on output_type)
            value_text VARCHAR,
            value_json JSON,
            value_numeric DOUBLE,
            value_bool BOOLEAN,

            -- Tier 2 QA aids. NULL / FALSE for Tier 1 rows.
            -- is_fallback distinguishes "model said it couldn't extract"
            -- (genuine null) from "we couldn't parse the response"
            -- (parse-fail null). extraction_metadata_json holds raw model
            -- response + retry/cache bookkeeping for QA inspection.
            is_fallback BOOLEAN NOT NULL DEFAULT FALSE,
            extraction_metadata_json JSON,

            date_key INTEGER,
            time_key INTEGER,
            extracted_at TIMESTAMP NOT NULL DEFAULT current_timestamp
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_facet_embeddings (
            -- Lineage envelope
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            created_by_version_key VARCHAR NOT NULL,
            last_updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            last_updated_by_version_key VARCHAR NOT NULL,
            etl_run_id VARCHAR NOT NULL,
            record_source VARCHAR NOT NULL,
            hash_diff VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP,

            -- Natural key (synthesized): md5(session_id || facet_type_key ||
            -- embedding_model || embedding_model_version)
            embedding_row_key VARCHAR NOT NULL,

            -- Natural key parts + degenerate dims
            session_key VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,
            facet_type_key VARCHAR NOT NULL,
            embedding_model VARCHAR NOT NULL,
            embedding_model_version VARCHAR NOT NULL,

            -- (embedding_model, embedding_model_version) uniquely determines
            -- dim, so no embedding_dim column is carried.
            embedding FLOAT[384],

            date_key INTEGER,
            time_key INTEGER,
            embedded_at TIMESTAMP NOT NULL DEFAULT current_timestamp
        )
        """
    )

    # Column migrations must run after every CREATE TABLE and before the
    # views: a view can reference a migrated column, and on a pre-existing
    # warehouse the CREATE TABLE IF NOT EXISTS above did not add it.
    _apply_column_migrations(conn)

    # Reconcile dim_date from every session already in the warehouse.
    # Per-session ETL only inserts dim_date for the sessions it re-stages;
    # this repairs a pre-0.17 warehouse (and sessions whose JSONL Claude
    # Code has since pruned, which can never be re-staged) so their
    # semantic views stop returning NULL dates. No-op on a fresh DB.
    from ccutils.etl.utils import insert_missing_dim_dates

    insert_missing_dim_dates(
        conn, "dim_session", "first_timestamp", "last_timestamp"
    )

    # =========================================================================
    # Semantic Views (15)
    # =========================================================================

    # Updated for v0.15 fact_session_summary shape.
    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_sessions AS
        SELECT
            fss.session_id,
            ds.cwd,
            ds.git_branch,
            ds.version,
            fss.first_timestamp AS session_datetime,
            fss.last_timestamp,
            ds.intent,
            ds.complexity,
            ds.outcome,
            ds.domain,
            dp.project_name,
            dp.project_path,
            fss.total_messages,
            fss.user_messages,
            fss.assistant_messages,
            fss.total_tool_uses,
            fss.total_tool_results,
            fss.total_thinking_blocks,
            fss.total_tool_errors,
            fss.unique_tools_used,
            fss.total_input_tokens,
            fss.total_output_tokens,
            fss.total_cache_creation_total_tokens,
            fss.total_cache_read_tokens,
            fss.total_uncached_equivalent_tokens,
            fss.api_response_count,
            fss.total_api_errors,
            fss.total_compactions,
            fss.total_turn_durations_ms,
            fss.turn_count,
            fss.permission_mode_transition_count,
            fss.current_permission_mode,
            fss.session_duration_seconds,
            dd.full_date,
            dd.day_name,
            dd.month_name,
            dd.year,
            dd.is_weekend,
            dti.hour,
            dti.time_of_day
        FROM fact_session_summary fss
        LEFT JOIN dim_session ds ON fss.session_key = ds.session_key
        LEFT JOIN dim_project dp ON fss.project_key = dp.project_key
        LEFT JOIN dim_date dd ON fss.date_key = dd.date_key
        LEFT JOIN dim_time dti ON fss.time_key = dti.time_key
        WHERE fss.is_deleted = FALSE
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
            fss.total_tool_uses,
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
            fad.tool_use_id,
            fad.task_description,
            fad.task_prompt,
            fad.subagent_type,
            fad.agent_status,
            dd.full_date AS delegation_date,
            fad.delegation_timestamp,
            fad.completion_timestamp,
            fad.seconds_to_completion,
            dti.time_of_day,
            fad.agent_total_tool_use_count,
            fad.agent_was_interrupted,
            fad.agent_total_duration_ms,
            fad.agent_total_tokens,
            fad.agent_output_text,
            ps.session_id AS parent_session_id,
            ps.cwd AS parent_cwd,
            ags.session_id AS agent_session_id,
            ags.depth_level AS agent_depth_level,
            dp.project_name
        FROM fact_agent_delegations fad
        JOIN dim_session ps ON fad.session_key = ps.session_key
        LEFT JOIN dim_session ags ON fad.agent_session_key = ags.session_key
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
            fpr.plan_file_path,
            fpr.plan_char_count,
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

    # One unified decision timeline over already-populated facts (the
    # "fact_decisions backbone" of the ETL-rethink proposal, reduced to a
    # projection: every structural signal it wanted already lands in a
    # v0.15 fact, so no new ETL or physical table is needed).
    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_decisions AS
        SELECT
            session_id,
            session_key,
            'plan_revision' AS decision_type,
            outcome AS decision_value,
            outcome_signal AS decision_signal,
            user_feedback_text AS decision_detail,
            plan_timestamp AS timestamp,
            CAST(plan_timestamp AS DATE) AS decision_date,
            revision_key AS source_key,
            'fact_plan_revisions' AS source_table
        FROM fact_plan_revisions
        WHERE is_deleted = FALSE

        UNION ALL

        SELECT
            session_id,
            session_key,
            'permission_mode_change' AS decision_type,
            meta_value AS decision_value,
            NULL AS decision_signal,
            NULL AS decision_detail,
            timestamp,
            CAST(timestamp AS DATE) AS decision_date,
            entry_id AS source_key,
            'fact_meta_events' AS source_table
        FROM fact_meta_events
        WHERE meta_type = 'permission-mode' AND is_deleted = FALSE

        UNION ALL

        SELECT
            session_id,
            session_key,
            CASE subtype
                WHEN 'stop_hook_summary' THEN 'stop_event'
                ELSE subtype
            END AS decision_type,
            CASE subtype
                WHEN 'stop_hook_summary' THEN stop_reason
                WHEN 'api_error' THEN error_type
                WHEN 'compact_boundary' THEN compact_trigger
            END AS decision_value,
            NULL AS decision_signal,
            CASE subtype
                WHEN 'stop_hook_summary'
                    THEN 'prevented_continuation='
                         || COALESCE(CAST(prevented_continuation AS VARCHAR), 'NULL')
                WHEN 'api_error'
                    THEN 'status=' || COALESCE(CAST(error_status AS VARCHAR), 'NULL')
                WHEN 'compact_boundary'
                    THEN 'pre_tokens='
                         || COALESCE(CAST(compact_pre_tokens AS VARCHAR), 'NULL')
            END AS decision_detail,
            timestamp,
            CAST(timestamp AS DATE) AS decision_date,
            entry_id AS source_key,
            'fact_system_events' AS source_table
        FROM fact_system_events
        WHERE subtype IN ('stop_hook_summary', 'api_error', 'compact_boundary')
          AND is_deleted = FALSE
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
            fss.total_tool_uses,
            fss.unique_tools_used,
            fss.total_tool_errors,
            fss.total_input_tokens,
            fss.total_output_tokens,
            fss.total_cache_read_tokens,
            fss.total_uncached_equivalent_tokens
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

    # Updated for the v0.15 fact_token_usage shape: cache_creation split
    # per TTL tier; total_uncached_equivalent_tokens derived; service_tier
    # / speed / inference_geo + server-tool counts surfaced.
    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_token_usage AS
        SELECT
            ftu.entry_id,
            ftu.input_tokens,
            ftu.output_tokens,
            ftu.cache_creation_5m_tokens,
            ftu.cache_creation_1h_tokens,
            ftu.cache_creation_total_tokens,
            ftu.cache_read_tokens,
            ftu.total_uncached_equivalent_tokens,
            ftu.service_tier,
            ftu.speed,
            ftu.inference_geo,
            ftu.server_tool_use_web_search_requests,
            ftu.server_tool_use_web_fetch_requests,
            ftu.timestamp,
            dm.model_name,
            dm.model_family,
            ftu.session_id,
            ds.cwd,
            dp.project_name,
            dd.full_date,
            dti.time_of_day
        FROM fact_token_usage ftu
        LEFT JOIN dim_model dm ON ftu.model_key = dm.model_key
        LEFT JOIN dim_session ds ON ftu.session_key = ds.session_key
        LEFT JOIN dim_project dp ON ds.project_key = dp.project_key
        LEFT JOIN dim_date dd ON ftu.date_key = dd.date_key
        LEFT JOIN dim_time dti ON ftu.time_key = dti.time_key
        WHERE ftu.is_deleted = FALSE
    """
    )

    # R11: cache_hit_rate_pct denominator now includes cache_creation_total
    # (legacy view used input + read only, over-stating the hit rate).
    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_cost_analysis AS
        SELECT
            fss.session_id,
            dp.project_name,
            ds.entrypoint,
            ds.custom_title,
            ds.intent,
            ds.complexity,
            dd.full_date,
            fss.session_duration_seconds,
            fss.total_input_tokens,
            fss.total_output_tokens,
            fss.total_cache_creation_5m_tokens,
            fss.total_cache_creation_1h_tokens,
            fss.total_cache_creation_total_tokens,
            fss.total_cache_read_tokens,
            fss.total_uncached_equivalent_tokens,
            fss.total_turn_durations_ms,
            fss.turn_count,
            CASE WHEN fss.total_uncached_equivalent_tokens > 0
                 THEN ROUND(100.0 * fss.total_cache_read_tokens
                      / fss.total_uncached_equivalent_tokens, 1)
                 ELSE 0 END AS cache_hit_rate_pct,
            fss.total_messages,
            fss.total_tool_uses,
            fss.api_response_count,
            fss.total_api_errors,
            fss.total_compactions
        FROM fact_session_summary fss
        LEFT JOIN dim_session ds ON fss.session_key = ds.session_key
        LEFT JOIN dim_project dp ON fss.project_key = dp.project_key
        LEFT JOIN dim_date dd ON fss.date_key = dd.date_key
        WHERE fss.is_deleted = FALSE
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
            fss.total_input_tokens,
            fss.total_output_tokens,
            dd.full_date
        FROM dim_prompt dp
        LEFT JOIN dim_session ds ON dp.session_key = ds.session_key
        LEFT JOIN fact_session_summary fss ON ds.session_key = fss.session_key
        LEFT JOIN dim_date dd ON dp.date_key = dd.date_key
    """
    )

    return conn


# Columns added AFTER a table first shipped in the persistent warehouse
# (0.17.0+). CREATE TABLE IF NOT EXISTS never widens an existing table,
# so every later column addition needs an entry here or old warehouses
# break on the populator's INSERT. Append-only.
_COLUMN_MIGRATIONS = [
    ("fact_plan_revisions", "plan_file_path", "VARCHAR"),
]


def _apply_column_migrations(conn) -> None:
    for table, column, col_type in _COLUMN_MIGRATIONS:
        conn.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_type}"
        )
