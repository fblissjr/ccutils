"""Star schema DDL - creates the dimensional model tables.

22 tables + 10 views. Tiny lookup dimensions (message_type, content_block_type,
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
    - 3 agent/bridge/staging tables
    - 2 optional tables (embeddings, tool_input_params)
    - 10 semantic views

    No hard PK/FK constraints - relies on soft business rules.

    Args:
        db_path: Path to the DuckDB database file

    Returns:
        duckdb.Connection to the database
    """
    conn = duckdb.connect(str(db_path))

    # =========================================================================
    # Staging Tables
    # =========================================================================

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
            last_assistant_message VARCHAR
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

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_messages (
            message_id VARCHAR,
            session_key VARCHAR,
            project_key VARCHAR,
            message_type VARCHAR,
            model_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            parent_message_id VARCHAR,
            timestamp TIMESTAMP,
            content_length INTEGER,
            content_block_count INTEGER,
            has_tool_use BOOLEAN,
            has_tool_result BOOLEAN,
            has_thinking BOOLEAN,
            word_count INTEGER,
            estimated_tokens INTEGER,
            response_time_seconds FLOAT,
            conversation_depth INTEGER,
            content_text TEXT,
            content_json JSON,
            is_sidechain BOOLEAN DEFAULT FALSE
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_tool_calls (
            tool_call_id VARCHAR,
            session_key VARCHAR,
            tool_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            invoke_message_id VARCHAR,
            result_message_id VARCHAR,
            timestamp TIMESTAMP,
            input_char_count INTEGER,
            output_char_count INTEGER,
            is_error BOOLEAN,
            duration_seconds FLOAT,
            input_json JSON,
            input_summary TEXT,
            output_text TEXT,
            file_path VARCHAR,
            command VARCHAR,
            pattern VARCHAR,
            query_text VARCHAR
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
            last_timestamp TIMESTAMP
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
            agent_duration_seconds INTEGER
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

    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_tool_calls AS
        SELECT
            ftc.tool_call_id,
            ftc.timestamp,
            ftc.input_char_count,
            ftc.output_char_count,
            ftc.is_error,
            ftc.duration_seconds,
            ftc.input_summary,
            ftc.output_text,
            ftc.file_path,
            ftc.command,
            ftc.pattern,
            ftc.query_text,
            dt.tool_name,
            dt.tool_category,
            ds.session_id,
            ds.cwd,
            dp.project_name,
            dd.full_date,
            dti.hour,
            dti.time_of_day
        FROM fact_tool_calls ftc
        JOIN dim_tool dt ON ftc.tool_key = dt.tool_key
        JOIN dim_session ds ON ftc.session_key = ds.session_key
        LEFT JOIN dim_project dp ON ds.project_key = dp.project_key
        LEFT JOIN dim_date dd ON ftc.date_key = dd.date_key
        LEFT JOIN dim_time dti ON ftc.time_key = dti.time_key
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
            fss.total_estimated_tokens
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

    return conn
