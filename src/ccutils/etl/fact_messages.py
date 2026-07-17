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
from ccutils.etl.upsert import lineage_upsert
from ccutils.etl.utils import project_key_sql


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


# Columns copied onto fact_messages by the lineage upsert. EXCLUDES the
# natural key (entry_id), session_id, and the derived keys
# (session_key, date_key, time_key) which the helper handles.
_PAYLOAD_COLUMNS = [
    "message_id",
    "project_key", "model_key",
    "message_type", "parent_message_id", "timestamp", "sequence_num",
    "is_sidechain", "is_meta", "is_compact_summary", "is_api_error_message",
    "stop_reason", "permission_mode_at_send", "prompt_id", "request_id",
    "api_error_text",
    "input_tokens", "output_tokens",
    "cache_creation_5m_tokens", "cache_creation_1h_tokens",
    "cache_read_tokens", "total_uncached_equivalent_tokens",
    "content_length", "content_block_count",
    "has_tool_use", "has_tool_result", "has_thinking",
    "content_text",
]


def populate_fact_messages(conn, *, run: EtlRun) -> None:
    """Project staging into fact_messages with idempotent hash-diff upsert."""
    conn.execute("DROP TABLE IF EXISTS _inbound_messages")
    conn.execute(f"CREATE TEMP TABLE _inbound_messages AS {_PROJECT_SQL}")

    # Derive content_length, project_key, model_key in SQL on the temp table.
    # (session_key/date_key/time_key/hash_diff are added by lineage_upsert.)
    conn.execute(
        "ALTER TABLE _inbound_messages ADD COLUMN content_length INTEGER"
    )
    conn.execute(
        "UPDATE _inbound_messages "
        "SET content_length = COALESCE(length(content_text), 0)"
    )

    conn.execute("ALTER TABLE _inbound_messages ADD COLUMN project_key VARCHAR")
    conn.execute("ALTER TABLE _inbound_messages ADD COLUMN model_key VARCHAR")
    conn.execute(
        f"""
        UPDATE _inbound_messages im
        SET project_key = {project_key_sql("sle.source_path")}
        FROM stg_log_entries sle
        WHERE sle.entry_id = im.entry_id
        """
    )
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

    lineage_upsert(
        conn,
        run=run,
        table="fact_messages",
        inbound_table="_inbound_messages",
        natural_key="entry_id",
        payload_cols=_PAYLOAD_COLUMNS,
        hash_cols=_HASH_DIFF_COLUMNS,
    )
