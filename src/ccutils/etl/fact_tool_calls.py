"""Populate fact_tool_uses + fact_tool_results from stg_log_entries (Phase C3).

Two facts replace the legacy fact_tool_calls. Pure SQL projection, mirroring
the C2 pattern: project staging into an _inbound temp table, compute hash_diff,
INSERT new rows, UPDATE diff'd rows, soft-delete missing rows.

The key new capture vs the legacy ETL: per-tool typed columns from the
entry-level `toolUseResult` field on user entries. Linked by tool_use_id.
"""

from __future__ import annotations

from ccutils.etl.lineage import EtlRun


# Columns hashed for change detection on fact_tool_uses. IDs, FKs, lineage
# excluded (not "content").
_USES_HASH_COLS = [
    "tool_name", "invoke_sequence_num", "caller_type", "input_json",
    "input_summary", "timestamp",
]

# Columns hashed for change detection on fact_tool_results.
_RESULTS_HASH_COLS = [
    "tool_name", "timestamp", "is_error",
    "result_content_text", "result_payload_json",
    "bash_exit_code", "bash_interrupted", "bash_stdout_bytes", "bash_duration_ms",
    "edit_user_modified", "edit_replace_all", "edit_structured_patch_json",
    "read_num_lines", "read_total_lines", "read_file_path",
    "write_type",
    "glob_num_files", "glob_truncated",
    "grep_mode", "grep_num_files",
    "webfetch_http_code", "webfetch_bytes",
    "agent_status", "agent_total_duration_ms", "agent_total_tokens",
    "agent_total_tool_use_count", "agent_was_interrupted", "agent_subagent_type",
]


def _hash_expr(cols: list[str]) -> str:
    return "md5(" + " || '|' || ".join(
        f"COALESCE(CAST({c} AS VARCHAR), '')" for c in cols
    ) + ")"


# fact_tool_uses: explode every assistant message's content[*] for tool_use blocks.
_PROJECT_USES_SQL = """
WITH assistant_entries AS (
    SELECT
        sle.entry_id,
        sle.uuid AS message_id,
        sle.session_id,
        sle.source_path,
        TRY_CAST(sle.timestamp AS TIMESTAMP) AS timestamp,
        sle.message_json
    FROM stg_log_entries sle
    WHERE sle.type = 'assistant'
      AND json_type(sle.message_json, '$.content') = 'ARRAY'
),
exploded AS (
    SELECT
        ae.*,
        b.block,
        b.block_idx
    FROM assistant_entries ae,
    LATERAL (
        SELECT unnest(json_extract(ae.message_json, '$.content')::JSON[]) AS block,
               generate_subscripts(
                   json_extract(ae.message_json, '$.content')::JSON[], 1
               ) AS block_idx
    ) b
    WHERE json_extract_string(b.block, '$.type') = 'tool_use'
)
SELECT
    -- entry_id of the source entry is used as the partition key for upsert;
    -- but multiple tool_uses can share an entry, so the natural key for
    -- this fact is tool_use_id. We still carry entry_id for lineage.
    entry_id,
    message_id,
    session_id,
    source_path,
    timestamp,
    json_extract_string(block, '$.id') AS tool_use_id,
    json_extract_string(block, '$.name') AS tool_name,
    block_idx AS invoke_sequence_num,
    json_extract_string(block, '$.caller.type') AS caller_type,
    CAST(json_extract(block, '$.input') AS VARCHAR) AS input_json
FROM exploded
"""


# fact_tool_results: explode user messages' content[*] for tool_result blocks,
# then JOIN to the user entry's toolUseResult payload.
_PROJECT_RESULTS_SQL = """
WITH user_entries AS (
    SELECT
        sle.entry_id,
        sle.uuid AS message_id,
        sle.session_id,
        sle.source_path,
        TRY_CAST(sle.timestamp AS TIMESTAMP) AS timestamp,
        sle.message_json,
        sle.tool_use_result_json
    FROM stg_log_entries sle
    WHERE sle.type = 'user'
      AND json_type(sle.message_json, '$.content') = 'ARRAY'
),
exploded AS (
    SELECT
        ue.*,
        b.block
    FROM user_entries ue,
    LATERAL (
        SELECT unnest(json_extract(ue.message_json, '$.content')::JSON[]) AS block
    ) b
    WHERE json_extract_string(b.block, '$.type') = 'tool_result'
),
tool_name_map AS (
    -- Resolve tool_name by joining tool_use_id back to the assistant's
    -- tool_use blocks in staging. Decoupled from fact_tool_uses so this
    -- populator can run before, after, or independently.
    SELECT
        json_extract_string(b.block, '$.id') AS tool_use_id,
        json_extract_string(b.block, '$.name') AS tool_name
    FROM stg_log_entries sle,
    LATERAL (
        SELECT unnest(json_extract(sle.message_json, '$.content')::JSON[]) AS block
    ) b
    WHERE sle.type = 'assistant'
      AND json_type(sle.message_json, '$.content') = 'ARRAY'
      AND json_extract_string(b.block, '$.type') = 'tool_use'
),
with_tool_name AS (
    SELECT
        e.*,
        json_extract_string(e.block, '$.tool_use_id') AS tool_use_id,
        tnm.tool_name AS tool_name
    FROM exploded e
    LEFT JOIN tool_name_map tnm
      ON tnm.tool_use_id = json_extract_string(e.block, '$.tool_use_id')
)
SELECT
    entry_id, message_id, session_id, source_path, timestamp,
    tool_use_id, tool_name,

    -- is_error tri-state (R16): NULL when the block omits it
    CASE
        WHEN json_type(block, '$.is_error') = 'NULL' THEN NULL
        ELSE json_extract(block, '$.is_error')::BOOLEAN
    END AS is_error,

    -- result_content_text: the tool_result block's `content` (often truncated
    -- preview of the structured payload).
    CASE
        WHEN json_type(block, '$.content') = 'VARCHAR'
        THEN json_extract_string(block, '$.content')
        WHEN json_type(block, '$.content') = 'ARRAY'
        THEN list_aggregate(
            list_filter(
                list_transform(
                    COALESCE(json_extract(block, '$.content[*]')::JSON[],
                             CAST([] AS JSON[])),
                    bb -> CASE WHEN json_extract_string(bb, '$.type') = 'text'
                               THEN json_extract_string(bb, '$.text') END
                ),
                t -> t IS NOT NULL
            ),
            'string_agg', ' '
        )
        ELSE NULL
    END AS result_content_text,

    -- result_payload_json: the entry-level toolUseResult (full untruncated,
    -- structured payload). For string-shape toolUseResult (errors), the
    -- string is preserved as JSON.
    tool_use_result_json AS result_payload_json,

    -- Per-tool typed projections. NULL when the tool doesn't match.
    -- Bash / BashOutput
    CASE WHEN tool_name IN ('Bash', 'BashOutput')
         THEN json_extract(tool_use_result_json, '$.exitCode')::INTEGER END
        AS bash_exit_code,
    CASE WHEN tool_name IN ('Bash', 'BashOutput')
         THEN json_extract(tool_use_result_json, '$.interrupted')::BOOLEAN END
        AS bash_interrupted,
    CASE WHEN tool_name IN ('Bash', 'BashOutput')
         THEN length(json_extract_string(tool_use_result_json, '$.stdout')) END
        AS bash_stdout_bytes,
    CASE WHEN tool_name IN ('Bash', 'BashOutput')
         THEN json_extract(tool_use_result_json, '$.durationMs')::FLOAT END
        AS bash_duration_ms,

    -- Edit / MultiEdit
    CASE WHEN tool_name IN ('Edit', 'MultiEdit')
         THEN json_extract(tool_use_result_json, '$.userModified')::BOOLEAN END
        AS edit_user_modified,
    CASE WHEN tool_name IN ('Edit', 'MultiEdit')
         THEN json_extract(tool_use_result_json, '$.replaceAll')::BOOLEAN END
        AS edit_replace_all,
    CASE WHEN tool_name IN ('Edit', 'MultiEdit')
         THEN CAST(json_extract(tool_use_result_json, '$.structuredPatch') AS VARCHAR) END
        AS edit_structured_patch_json,

    -- Read
    CASE WHEN tool_name = 'Read'
         THEN json_extract(tool_use_result_json, '$.file.numLines')::INTEGER END
        AS read_num_lines,
    CASE WHEN tool_name = 'Read'
         THEN json_extract(tool_use_result_json, '$.file.totalLines')::INTEGER END
        AS read_total_lines,
    CASE WHEN tool_name = 'Read'
         THEN json_extract_string(tool_use_result_json, '$.file.filePath') END
        AS read_file_path,

    -- Write
    CASE WHEN tool_name = 'Write'
         THEN json_extract_string(tool_use_result_json, '$.type') END
        AS write_type,

    -- Glob
    CASE WHEN tool_name = 'Glob'
         THEN json_extract(tool_use_result_json, '$.numFiles')::INTEGER END
        AS glob_num_files,
    CASE WHEN tool_name = 'Glob'
         THEN json_extract(tool_use_result_json, '$.truncated')::BOOLEAN END
        AS glob_truncated,

    -- Grep
    CASE WHEN tool_name = 'Grep'
         THEN json_extract_string(tool_use_result_json, '$.mode') END
        AS grep_mode,
    CASE WHEN tool_name = 'Grep'
         THEN json_extract(tool_use_result_json, '$.numFiles')::INTEGER END
        AS grep_num_files,

    -- WebFetch
    CASE WHEN tool_name = 'WebFetch'
         THEN json_extract(tool_use_result_json, '$.code')::INTEGER END
        AS webfetch_http_code,
    CASE WHEN tool_name = 'WebFetch'
         THEN json_extract(tool_use_result_json, '$.bytes')::INTEGER END
        AS webfetch_bytes,

    -- Agent / Task subagent rollup
    CASE WHEN tool_name IN ('Agent', 'Task', 'TaskCreate')
         THEN json_extract_string(tool_use_result_json, '$.status') END
        AS agent_status,
    CASE WHEN tool_name IN ('Agent', 'Task', 'TaskCreate')
         THEN json_extract(tool_use_result_json, '$.totalDurationMs')::FLOAT END
        AS agent_total_duration_ms,
    CASE WHEN tool_name IN ('Agent', 'Task', 'TaskCreate')
         THEN json_extract(tool_use_result_json, '$.totalTokens')::INTEGER END
        AS agent_total_tokens,
    CASE WHEN tool_name IN ('Agent', 'Task', 'TaskCreate')
         THEN json_extract(tool_use_result_json, '$.totalToolUseCount')::INTEGER END
        AS agent_total_tool_use_count,
    CASE WHEN tool_name IN ('Agent', 'Task', 'TaskCreate')
         THEN json_extract(tool_use_result_json, '$.wasInterrupted')::BOOLEAN END
        AS agent_was_interrupted,
    CASE WHEN tool_name IN ('Agent', 'Task', 'TaskCreate')
         THEN json_extract_string(tool_use_result_json, '$.agentType') END
        AS agent_subagent_type
FROM with_tool_name
"""


def populate_fact_tool_uses(conn, *, run: EtlRun) -> None:
    """Project staging tool_use blocks into fact_tool_uses."""
    conn.execute("DROP TABLE IF EXISTS _inbound_tool_uses")
    conn.execute(f"CREATE TEMP TABLE _inbound_tool_uses AS {_PROJECT_USES_SQL}")

    # Derive input_summary (first 200 chars), session_key, project_key,
    # tool_key, date_key, time_key.
    for ddl in (
        "ALTER TABLE _inbound_tool_uses ADD COLUMN input_summary VARCHAR",
        "ALTER TABLE _inbound_tool_uses ADD COLUMN session_key VARCHAR",
        "ALTER TABLE _inbound_tool_uses ADD COLUMN project_key VARCHAR",
        "ALTER TABLE _inbound_tool_uses ADD COLUMN tool_key VARCHAR",
        "ALTER TABLE _inbound_tool_uses ADD COLUMN date_key INTEGER",
        "ALTER TABLE _inbound_tool_uses ADD COLUMN time_key INTEGER",
        "ALTER TABLE _inbound_tool_uses ADD COLUMN hash_diff VARCHAR",
    ):
        conn.execute(ddl)

    conn.execute("UPDATE _inbound_tool_uses SET input_summary = substring(input_json, 1, 200)")
    conn.execute("UPDATE _inbound_tool_uses SET session_key = md5(session_id)")
    conn.execute("UPDATE _inbound_tool_uses SET tool_key = md5(tool_name)")
    conn.execute(
        "UPDATE _inbound_tool_uses "
        "SET date_key = CAST(strftime(timestamp, '%Y%m%d') AS INTEGER), "
        "    time_key = CAST(strftime(timestamp, '%H%M') AS INTEGER) "
        "WHERE timestamp IS NOT NULL"
    )
    conn.execute(
        "UPDATE _inbound_tool_uses "
        "SET project_key = md5(regexp_replace(source_path, '/[^/]+$', ''))"
    )
    conn.execute(f"UPDATE _inbound_tool_uses SET hash_diff = {_hash_expr(_USES_HASH_COLS)}")

    # UPDATE existing rows whose hash_diff changed
    conn.execute(
        f"""
        UPDATE fact_tool_uses ftu
        SET
            last_updated_at = current_timestamp,
            last_updated_by_version_key = '{run.version_key}',
            etl_run_id = '{run.etl_run_id}',
            hash_diff = im.hash_diff,
            entry_id = im.entry_id,
            message_id = im.message_id,
            session_key = im.session_key,
            project_key = im.project_key,
            tool_key = im.tool_key,
            date_key = im.date_key,
            time_key = im.time_key,
            tool_name = im.tool_name,
            invoke_sequence_num = im.invoke_sequence_num,
            caller_type = im.caller_type,
            input_json = im.input_json,
            input_summary = im.input_summary,
            timestamp = im.timestamp,
            is_deleted = FALSE,
            deleted_at = NULL
        FROM _inbound_tool_uses im
        WHERE ftu.tool_use_id = im.tool_use_id
          AND ftu.hash_diff IS DISTINCT FROM im.hash_diff
        """
    )

    # INSERT new
    conn.execute(
        f"""
        INSERT INTO fact_tool_uses (
            created_by_version_key, last_updated_by_version_key,
            etl_run_id, record_source, hash_diff,
            entry_id, message_id, session_id, tool_use_id,
            session_key, project_key, tool_key, date_key, time_key,
            tool_name, invoke_sequence_num, caller_type,
            input_json, input_summary, timestamp
        )
        SELECT
            '{run.version_key}', '{run.version_key}',
            '{run.etl_run_id}', 'claude_code_jsonl', im.hash_diff,
            im.entry_id, im.message_id, im.session_id, im.tool_use_id,
            im.session_key, im.project_key, im.tool_key, im.date_key, im.time_key,
            im.tool_name, im.invoke_sequence_num, im.caller_type,
            im.input_json, im.input_summary, im.timestamp
        FROM _inbound_tool_uses im
        WHERE NOT EXISTS (
            SELECT 1 FROM fact_tool_uses ftu WHERE ftu.tool_use_id = im.tool_use_id
        )
        """
    )

    # Soft-delete missing
    conn.execute(
        f"""
        UPDATE fact_tool_uses ftu
        SET is_deleted = TRUE,
            deleted_at = current_timestamp,
            last_updated_at = current_timestamp,
            last_updated_by_version_key = '{run.version_key}',
            etl_run_id = '{run.etl_run_id}'
        WHERE ftu.is_deleted = FALSE
          AND ftu.session_id IN (SELECT DISTINCT session_id FROM _inbound_tool_uses)
          AND ftu.tool_use_id NOT IN (SELECT tool_use_id FROM _inbound_tool_uses)
        """
    )

    conn.execute("DROP TABLE _inbound_tool_uses")


def populate_fact_tool_results(conn, *, run: EtlRun) -> None:
    """Project staging tool_result blocks (joined with toolUseResult payload)
    into fact_tool_results. Requires fact_tool_uses to be populated first
    (uses it to resolve tool_name from tool_use_id).
    """
    conn.execute("DROP TABLE IF EXISTS _inbound_tool_results")
    conn.execute(f"CREATE TEMP TABLE _inbound_tool_results AS {_PROJECT_RESULTS_SQL}")

    for ddl in (
        "ALTER TABLE _inbound_tool_results ADD COLUMN session_key VARCHAR",
        "ALTER TABLE _inbound_tool_results ADD COLUMN project_key VARCHAR",
        "ALTER TABLE _inbound_tool_results ADD COLUMN tool_key VARCHAR",
        "ALTER TABLE _inbound_tool_results ADD COLUMN date_key INTEGER",
        "ALTER TABLE _inbound_tool_results ADD COLUMN time_key INTEGER",
        "ALTER TABLE _inbound_tool_results ADD COLUMN hash_diff VARCHAR",
    ):
        conn.execute(ddl)

    conn.execute("UPDATE _inbound_tool_results SET session_key = md5(session_id)")
    conn.execute("UPDATE _inbound_tool_results SET tool_key = md5(tool_name) WHERE tool_name IS NOT NULL")
    conn.execute(
        "UPDATE _inbound_tool_results "
        "SET date_key = CAST(strftime(timestamp, '%Y%m%d') AS INTEGER), "
        "    time_key = CAST(strftime(timestamp, '%H%M') AS INTEGER) "
        "WHERE timestamp IS NOT NULL"
    )
    conn.execute(
        "UPDATE _inbound_tool_results "
        "SET project_key = md5(regexp_replace(source_path, '/[^/]+$', ''))"
    )
    conn.execute(f"UPDATE _inbound_tool_results SET hash_diff = {_hash_expr(_RESULTS_HASH_COLS)}")

    # UPDATE
    update_cols = [
        "tool_name", "session_key", "project_key", "tool_key",
        "date_key", "time_key", "timestamp",
        "is_error", "result_content_text", "result_payload_json",
        "bash_exit_code", "bash_interrupted", "bash_stdout_bytes", "bash_duration_ms",
        "edit_user_modified", "edit_replace_all", "edit_structured_patch_json",
        "read_num_lines", "read_total_lines", "read_file_path",
        "write_type",
        "glob_num_files", "glob_truncated",
        "grep_mode", "grep_num_files",
        "webfetch_http_code", "webfetch_bytes",
        "agent_status", "agent_total_duration_ms", "agent_total_tokens",
        "agent_total_tool_use_count", "agent_was_interrupted", "agent_subagent_type",
        "entry_id", "message_id",
    ]
    set_clause = ",\n            ".join(f"{c} = im.{c}" for c in update_cols)
    conn.execute(
        f"""
        UPDATE fact_tool_results ftr
        SET
            last_updated_at = current_timestamp,
            last_updated_by_version_key = '{run.version_key}',
            etl_run_id = '{run.etl_run_id}',
            hash_diff = im.hash_diff,
            {set_clause},
            is_deleted = FALSE,
            deleted_at = NULL
        FROM _inbound_tool_results im
        WHERE ftr.tool_use_id = im.tool_use_id
          AND ftr.hash_diff IS DISTINCT FROM im.hash_diff
        """
    )

    # INSERT new
    insert_cols = [
        "entry_id", "message_id", "session_id", "tool_use_id",
        "session_key", "project_key", "tool_key", "date_key", "time_key",
        "tool_name", "timestamp",
        "is_error", "result_content_text", "result_payload_json",
        "bash_exit_code", "bash_interrupted", "bash_stdout_bytes", "bash_duration_ms",
        "edit_user_modified", "edit_replace_all", "edit_structured_patch_json",
        "read_num_lines", "read_total_lines", "read_file_path",
        "write_type",
        "glob_num_files", "glob_truncated",
        "grep_mode", "grep_num_files",
        "webfetch_http_code", "webfetch_bytes",
        "agent_status", "agent_total_duration_ms", "agent_total_tokens",
        "agent_total_tool_use_count", "agent_was_interrupted", "agent_subagent_type",
    ]
    insert_col_list = ", ".join(insert_cols)
    select_col_list = ", ".join(f"im.{c}" for c in insert_cols)
    conn.execute(
        f"""
        INSERT INTO fact_tool_results (
            created_by_version_key, last_updated_by_version_key,
            etl_run_id, record_source, hash_diff,
            {insert_col_list}
        )
        SELECT
            '{run.version_key}', '{run.version_key}',
            '{run.etl_run_id}', 'claude_code_jsonl', im.hash_diff,
            {select_col_list}
        FROM _inbound_tool_results im
        WHERE NOT EXISTS (
            SELECT 1 FROM fact_tool_results ftr WHERE ftr.tool_use_id = im.tool_use_id
        )
        """
    )

    # Soft-delete missing
    conn.execute(
        f"""
        UPDATE fact_tool_results ftr
        SET is_deleted = TRUE,
            deleted_at = current_timestamp,
            last_updated_at = current_timestamp,
            last_updated_by_version_key = '{run.version_key}',
            etl_run_id = '{run.etl_run_id}'
        WHERE ftr.is_deleted = FALSE
          AND ftr.session_id IN (SELECT DISTINCT session_id FROM _inbound_tool_results)
          AND ftr.tool_use_id NOT IN (SELECT tool_use_id FROM _inbound_tool_results)
        """
    )

    conn.execute("DROP TABLE _inbound_tool_results")
