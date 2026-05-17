"""Populate fact_messages from stg_log_entries (Phase C2).

Pure SQL projection. The staging table has every JSONL line; this module
filters to user/assistant rows and projects the typed columns the warehouse
exposes.

Idempotent via hash_diff: rows whose mutable content didn't change between
runs are not UPDATEd (so last_updated_at stays a precise temporal signal,
not a "last ETL touch" timestamp). New rows INSERT; missing rows soft-delete.
"""

from __future__ import annotations

from ccutils.etl.lineage import EtlRun


# Mutable-content columns hashed for change detection. Lineage cols, IDs,
# and FKs are excluded -- they don't represent "content" for the purpose
# of "did anything meaningful change?"
_HASH_DIFF_COLUMNS = [
    "message_type",
    "parent_message_id",
    "timestamp",
    "is_sidechain", "is_meta", "is_compact_summary", "is_api_error_message",
    "stop_reason", "permission_mode_at_send", "prompt_id", "request_id",
    "api_error_text",
    "input_tokens", "output_tokens",
    "cache_creation_5m_tokens", "cache_creation_1h_tokens",
    "cache_read_tokens", "total_uncached_equivalent_tokens",
    "content_block_count",
    "has_tool_use", "has_tool_result", "has_thinking",
    # content_length and word_count are derived from content_text -- hashing
    # content_text alone is enough. response_time_seconds and conversation_depth
    # are derived post-load (not in the staging projection); not hashed.
    "content_text",
]


# DuckDB SQL: project staging rows into the fact-messages shape.
#
# Notes:
#   - We use json_extract / json_extract_string against the message_json
#     column to pull typed fields out (DuckDB native).
#   - cache_creation TTL split comes from message.usage.cache_creation.
#     Older sessions without that nested dict get NULLs.
#   - response_time_seconds derived via window function (parent -> child).
#     Skipped for now -- needs a self-join through staging that's cleaner
#     as a post-step. Marked TODO; for now it's NULL.
#   - conversation_depth requires recursive parent walk. Same TODO; NULL.
#   - content_text and content_length are flattened text via json_extract.
#     For multi-block content (list-of-blocks), we concat text-block .text
#     fields; tool_use and tool_result blocks contribute 0 chars to
#     content_length.
_PROJECT_SQL = """
SELECT
    sle.entry_id,
    sle.uuid AS message_id,
    sle.session_id,
    sle.type AS message_type,
    sle.parent_uuid AS parent_message_id,
    TRY_CAST(sle.timestamp AS TIMESTAMP) AS timestamp,
    sle.sequence_num,
    sle.is_sidechain,
    sle.is_meta,
    -- user-only flags
    CASE WHEN sle.type = 'user'
         THEN COALESCE(json_extract(sle.raw_json, '$.isCompactSummary')::BOOLEAN, FALSE)
         ELSE FALSE END AS is_compact_summary,
    -- assistant-only flags
    CASE WHEN sle.type = 'assistant'
         THEN COALESCE(json_extract(sle.raw_json, '$.isApiErrorMessage')::BOOLEAN, FALSE)
         ELSE FALSE END AS is_api_error_message,
    CASE WHEN sle.type = 'assistant'
         THEN json_extract_string(sle.message_json, '$.stop_reason')
         END AS stop_reason,
    CASE WHEN sle.type = 'user'
         THEN json_extract_string(sle.raw_json, '$.permissionMode')
         END AS permission_mode_at_send,
    CASE WHEN sle.type = 'user'
         THEN json_extract_string(sle.raw_json, '$.promptId')
         END AS prompt_id,
    CASE WHEN sle.type = 'assistant'
         THEN json_extract_string(sle.raw_json, '$.requestId')
         END AS request_id,
    CASE WHEN sle.type = 'assistant'
         THEN json_extract_string(sle.raw_json, '$.apiError')
         END AS api_error_text,

    -- Tokens (assistant-only; from message.usage)
    CASE WHEN sle.type = 'assistant'
         THEN json_extract(sle.message_json, '$.usage.input_tokens')::INTEGER
         END AS input_tokens,
    CASE WHEN sle.type = 'assistant'
         THEN json_extract(sle.message_json, '$.usage.output_tokens')::INTEGER
         END AS output_tokens,
    CASE WHEN sle.type = 'assistant'
         THEN json_extract(sle.message_json, '$.usage.cache_creation.ephemeral_5m_input_tokens')::INTEGER
         END AS cache_creation_5m_tokens,
    CASE WHEN sle.type = 'assistant'
         THEN json_extract(sle.message_json, '$.usage.cache_creation.ephemeral_1h_input_tokens')::INTEGER
         END AS cache_creation_1h_tokens,
    CASE WHEN sle.type = 'assistant'
         THEN json_extract(sle.message_json, '$.usage.cache_read_input_tokens')::INTEGER
         END AS cache_read_tokens,
    -- Derived: total = cache_read + creation_5m + creation_1h + input_tokens
    CASE WHEN sle.type = 'assistant'
         THEN COALESCE(json_extract(sle.message_json, '$.usage.input_tokens')::INTEGER, 0)
            + COALESCE(json_extract(sle.message_json, '$.usage.cache_creation.ephemeral_5m_input_tokens')::INTEGER, 0)
            + COALESCE(json_extract(sle.message_json, '$.usage.cache_creation.ephemeral_1h_input_tokens')::INTEGER, 0)
            + COALESCE(json_extract(sle.message_json, '$.usage.cache_read_input_tokens')::INTEGER, 0)
         END AS total_uncached_equivalent_tokens,

    -- Content flags. Extract every block's .type into a JSON[] and
    -- list_contains. For string-shape content there are no blocks so all
    -- three flags are FALSE.
    list_contains(
        COALESCE(json_extract(sle.message_json, '$.content[*].type')::JSON[],
                 CAST([] AS JSON[])),
        '"tool_use"'::JSON
    ) AS has_tool_use,
    list_contains(
        COALESCE(json_extract(sle.message_json, '$.content[*].type')::JSON[],
                 CAST([] AS JSON[])),
        '"tool_result"'::JSON
    ) AS has_tool_result,
    (
        list_contains(
            COALESCE(json_extract(sle.message_json, '$.content[*].type')::JSON[],
                     CAST([] AS JSON[])),
            '"thinking"'::JSON
        )
        OR list_contains(
            COALESCE(json_extract(sle.message_json, '$.content[*].type')::JSON[],
                     CAST([] AS JSON[])),
            '"redacted_thinking"'::JSON
        )
    ) AS has_thinking,

    -- Content text: for VARCHAR content, the string; for ARRAY content,
    -- the concatenated text-block .text fields (space-joined).
    CASE
        WHEN json_type(sle.message_json, '$.content') = 'VARCHAR'
        THEN json_extract_string(sle.message_json, '$.content')
        WHEN json_type(sle.message_json, '$.content') = 'ARRAY'
        THEN list_aggregate(
            list_filter(
                list_transform(
                    COALESCE(json_extract(sle.message_json, '$.content[*]')::JSON[],
                             CAST([] AS JSON[])),
                    b -> CASE WHEN json_extract_string(b, '$.type') = 'text'
                              THEN json_extract_string(b, '$.text') END
                ),
                t -> t IS NOT NULL
            ),
            'string_agg', ' '
        )
        ELSE NULL
    END AS content_text,

    CASE
        WHEN json_type(sle.message_json, '$.content') = 'ARRAY'
        THEN json_array_length(json_extract(sle.message_json, '$.content'))
        WHEN json_type(sle.message_json, '$.content') = 'VARCHAR'
        THEN 1
        ELSE 0
    END AS content_block_count
FROM stg_log_entries sle
WHERE sle.type IN ('user', 'assistant')
"""


def populate_fact_messages(conn, *, run: EtlRun) -> None:
    """Project staging into fact_messages with idempotent hash-diff upsert.

    Algorithm:
      1. Build a temporary table of inbound rows projected from staging.
      2. Compute hash_diff and content_length on inbound rows.
      3. INSERT new rows (where entry_id not in fact_messages).
      4. UPDATE rows where hash_diff differs.
      5. Soft-delete rows that existed for the loaded sessions but are
         no longer present in staging.
    """
    # Step 1: project to a temp table.
    conn.execute("DROP TABLE IF EXISTS _inbound_messages")
    conn.execute(f"CREATE TEMP TABLE _inbound_messages AS {_PROJECT_SQL}")

    # Step 2: compute hash_diff + content_length on the inbound table.
    # hash_diff is MD5 over a delimiter-joined concat of the mutable columns
    # (mirrors what Python hash_diff() does, but computed in SQL so we don't
    # round-trip every row through Python).
    hash_parts_sql = " || '|' || ".join(
        f"COALESCE(CAST({col} AS VARCHAR), '')" for col in _HASH_DIFF_COLUMNS
    )
    conn.execute(
        f"ALTER TABLE _inbound_messages ADD COLUMN hash_diff VARCHAR"
    )
    conn.execute(
        f"UPDATE _inbound_messages SET hash_diff = md5({hash_parts_sql})"
    )
    conn.execute(
        f"ALTER TABLE _inbound_messages ADD COLUMN content_length INTEGER"
    )
    conn.execute(
        "UPDATE _inbound_messages SET content_length = "
        "COALESCE(length(content_text), 0)"
    )

    # Step 3: derive session_key and project_key. session_key is MD5(session_id);
    # project_key is MD5(parent dir path) -- staging carries cwd but the
    # project_path natural key historically is the JSONL parent dir.
    conn.execute("ALTER TABLE _inbound_messages ADD COLUMN session_key VARCHAR")
    conn.execute("ALTER TABLE _inbound_messages ADD COLUMN project_key VARCHAR")
    conn.execute("ALTER TABLE _inbound_messages ADD COLUMN model_key VARCHAR")
    conn.execute("ALTER TABLE _inbound_messages ADD COLUMN date_key INTEGER")
    conn.execute("ALTER TABLE _inbound_messages ADD COLUMN time_key INTEGER")

    # Populate session_key + date/time keys using DuckDB scalar funcs.
    conn.execute("UPDATE _inbound_messages SET session_key = md5(session_id)")
    conn.execute(
        "UPDATE _inbound_messages "
        "SET date_key = CAST(strftime(timestamp, '%Y%m%d') AS INTEGER), "
        "    time_key = CAST(strftime(timestamp, '%H%M') AS INTEGER) "
        "WHERE timestamp IS NOT NULL"
    )

    # project_key requires the source JSONL parent dir. Pull from staging.
    conn.execute(
        """
        UPDATE _inbound_messages im
        SET project_key = md5(
            regexp_replace(sle.source_path, '/[^/]+$', '')
        )
        FROM stg_log_entries sle
        WHERE sle.entry_id = im.entry_id
        """
    )

    # model_key from message.usage's model field (assistant only). Pull
    # via json_extract from staging.
    conn.execute(
        """
        UPDATE _inbound_messages im
        SET model_key = md5(json_extract_string(sle.message_json, '$.model'))
        FROM stg_log_entries sle
        WHERE sle.entry_id = im.entry_id
          AND im.message_type = 'assistant'
          AND json_extract_string(sle.message_json, '$.model') IS NOT NULL
        """
    )

    # Step 4: UPDATE existing rows where hash_diff changed.
    conn.execute(
        f"""
        UPDATE fact_messages fm
        SET
            last_updated_at = current_timestamp,
            last_updated_by_version_key = '{run.version_key}',
            etl_run_id = '{run.etl_run_id}',
            hash_diff = im.hash_diff,
            session_key = im.session_key,
            project_key = im.project_key,
            model_key = im.model_key,
            date_key = im.date_key,
            time_key = im.time_key,
            message_type = im.message_type,
            parent_message_id = im.parent_message_id,
            timestamp = im.timestamp,
            sequence_num = im.sequence_num,
            is_sidechain = im.is_sidechain,
            is_meta = im.is_meta,
            is_compact_summary = im.is_compact_summary,
            is_api_error_message = im.is_api_error_message,
            stop_reason = im.stop_reason,
            permission_mode_at_send = im.permission_mode_at_send,
            prompt_id = im.prompt_id,
            request_id = im.request_id,
            api_error_text = im.api_error_text,
            input_tokens = im.input_tokens,
            output_tokens = im.output_tokens,
            cache_creation_5m_tokens = im.cache_creation_5m_tokens,
            cache_creation_1h_tokens = im.cache_creation_1h_tokens,
            cache_read_tokens = im.cache_read_tokens,
            total_uncached_equivalent_tokens = im.total_uncached_equivalent_tokens,
            content_length = im.content_length,
            content_block_count = im.content_block_count,
            has_tool_use = im.has_tool_use,
            has_tool_result = im.has_tool_result,
            has_thinking = im.has_thinking,
            content_text = im.content_text,
            is_deleted = FALSE,
            deleted_at = NULL
        FROM _inbound_messages im
        WHERE fm.entry_id = im.entry_id
          AND fm.hash_diff IS DISTINCT FROM im.hash_diff
        """
    )

    # Step 5: INSERT new rows.
    conn.execute(
        f"""
        INSERT INTO fact_messages (
            created_by_version_key, last_updated_by_version_key,
            etl_run_id, record_source, hash_diff,
            entry_id, message_id, session_id,
            session_key, project_key, model_key, date_key, time_key,
            message_type, parent_message_id, timestamp, sequence_num,
            is_sidechain, is_meta, is_compact_summary, is_api_error_message,
            stop_reason, permission_mode_at_send, prompt_id, request_id,
            api_error_text,
            input_tokens, output_tokens,
            cache_creation_5m_tokens, cache_creation_1h_tokens,
            cache_read_tokens, total_uncached_equivalent_tokens,
            content_length, content_block_count,
            has_tool_use, has_tool_result, has_thinking,
            content_text
        )
        SELECT
            '{run.version_key}', '{run.version_key}',
            '{run.etl_run_id}', 'claude_code_jsonl', im.hash_diff,
            im.entry_id, im.message_id, im.session_id,
            im.session_key, im.project_key, im.model_key, im.date_key, im.time_key,
            im.message_type, im.parent_message_id, im.timestamp, im.sequence_num,
            im.is_sidechain, im.is_meta, im.is_compact_summary, im.is_api_error_message,
            im.stop_reason, im.permission_mode_at_send, im.prompt_id, im.request_id,
            im.api_error_text,
            im.input_tokens, im.output_tokens,
            im.cache_creation_5m_tokens, im.cache_creation_1h_tokens,
            im.cache_read_tokens, im.total_uncached_equivalent_tokens,
            im.content_length, im.content_block_count,
            im.has_tool_use, im.has_tool_result, im.has_thinking,
            im.content_text
        FROM _inbound_messages im
        WHERE NOT EXISTS (
            SELECT 1 FROM fact_messages fm WHERE fm.entry_id = im.entry_id
        )
        """
    )

    # Step 6: Soft-delete rows for sessions present in inbound, but whose
    # entry_id is not in inbound. Scope by session_id to avoid wiping facts
    # for sessions not part of this batch.
    conn.execute(
        f"""
        UPDATE fact_messages fm
        SET is_deleted = TRUE,
            deleted_at = current_timestamp,
            last_updated_at = current_timestamp,
            last_updated_by_version_key = '{run.version_key}',
            etl_run_id = '{run.etl_run_id}'
        WHERE fm.is_deleted = FALSE
          AND fm.session_id IN (SELECT DISTINCT session_id FROM _inbound_messages)
          AND fm.entry_id NOT IN (SELECT entry_id FROM _inbound_messages)
        """
    )

    conn.execute("DROP TABLE _inbound_messages")
