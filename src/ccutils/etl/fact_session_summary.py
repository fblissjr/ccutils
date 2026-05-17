"""Populate fact_session_summary from the v0.15 entry-type facts (Phase C5b).

One row per session with pre-aggregated rollups. The Kimball "facts don't
join to facts" rule applies at the query layer -- the populator joins facts
at ETL time to derive per-session counts and sums, but consumers see one
self-contained row per session and never need to join facts to facts.

Sourced from:
  fact_messages              user_messages, assistant_messages, total_thinking_blocks
  fact_token_usage           token tier rollups, api_response_count
  fact_tool_uses             total_tool_uses, unique_tools_used
  fact_tool_results          total_tool_results, total_tool_errors,
                             total_bash_interrupted
  fact_system_events         api_errors, compactions, turn_durations,
                             stop_events, prevented_continuations
  fact_progress_events       totals + hook/bash variant counts
  fact_attachments           totals + diagnostics + hook_successes
  fact_meta_events           permission_mode transition count + current mode
  fact_file_history_snapshots total

All counts/sums are NULL-safe via COALESCE so a session that lacks one
entry type gets 0 there rather than NULL.
"""

from __future__ import annotations

from ccutils.etl.lineage import EtlRun
from ccutils.etl.upsert import lineage_upsert


_PAYLOAD_COLS = [
    "project_key",
    "first_timestamp", "last_timestamp", "session_duration_seconds",
    "total_messages", "user_messages", "assistant_messages",
    "total_thinking_blocks",
    "total_input_tokens", "total_output_tokens",
    "total_cache_creation_5m_tokens", "total_cache_creation_1h_tokens",
    "total_cache_creation_total_tokens", "total_cache_read_tokens",
    "total_uncached_equivalent_tokens", "api_response_count",
    "total_tool_uses", "unique_tools_used",
    "total_tool_results", "total_tool_errors", "total_bash_interrupted",
    "total_api_errors", "total_compactions",
    "total_turn_durations_ms", "turn_count",
    "total_stop_events", "total_prevented_continuations",
    "total_progress_events",
    "total_hook_progress_events", "total_bash_progress_events",
    "total_attachments", "total_diagnostics", "total_hook_successes",
    "permission_mode_transition_count", "current_permission_mode",
    "total_file_history_snapshots",
]
# Hash includes every payload column; content-bearing.
_HASH_COLS = _PAYLOAD_COLS


# The aggregate SQL. Each subquery rolls up one fact table; LEFT JOINs from
# the message-grain base ensure a session with no rows in (say) fact_attachments
# still gets a summary row with zeros there.
_PROJECT_SQL = """
WITH session_envelope AS (
    SELECT
        sle.session_id,
        MAX(sle.source_path) AS source_path,
        MIN(TRY_CAST(sle.timestamp AS TIMESTAMP)) AS first_timestamp,
        MAX(TRY_CAST(sle.timestamp AS TIMESTAMP)) AS last_timestamp
    FROM stg_log_entries sle
    GROUP BY sle.session_id
),
msg_rollup AS (
    SELECT
        session_id,
        COUNT(*) AS total_messages,
        SUM(CASE WHEN message_type = 'user' THEN 1 ELSE 0 END) AS user_messages,
        SUM(CASE WHEN message_type = 'assistant' THEN 1 ELSE 0 END) AS assistant_messages,
        SUM(CASE WHEN has_thinking THEN 1 ELSE 0 END) AS total_thinking_blocks
    FROM fact_messages
    WHERE is_deleted = FALSE
    GROUP BY session_id
),
token_rollup AS (
    SELECT
        session_id,
        SUM(COALESCE(input_tokens, 0)) AS total_input_tokens,
        SUM(COALESCE(output_tokens, 0)) AS total_output_tokens,
        SUM(COALESCE(cache_creation_5m_tokens, 0)) AS total_cache_creation_5m_tokens,
        SUM(COALESCE(cache_creation_1h_tokens, 0)) AS total_cache_creation_1h_tokens,
        SUM(COALESCE(cache_creation_total_tokens, 0)) AS total_cache_creation_total_tokens,
        SUM(COALESCE(cache_read_tokens, 0)) AS total_cache_read_tokens,
        SUM(COALESCE(total_uncached_equivalent_tokens, 0)) AS total_uncached_equivalent_tokens,
        COUNT(*) AS api_response_count
    FROM fact_token_usage
    WHERE is_deleted = FALSE
    GROUP BY session_id
),
tool_use_rollup AS (
    SELECT
        session_id,
        COUNT(*) AS total_tool_uses,
        COUNT(DISTINCT tool_name) AS unique_tools_used
    FROM fact_tool_uses
    WHERE is_deleted = FALSE
    GROUP BY session_id
),
tool_result_rollup AS (
    SELECT
        session_id,
        COUNT(*) AS total_tool_results,
        SUM(CASE WHEN is_error THEN 1 ELSE 0 END) AS total_tool_errors,
        SUM(CASE WHEN bash_interrupted THEN 1 ELSE 0 END) AS total_bash_interrupted
    FROM fact_tool_results
    WHERE is_deleted = FALSE
    GROUP BY session_id
),
system_rollup AS (
    SELECT
        session_id,
        SUM(CASE WHEN subtype = 'api_error' THEN 1 ELSE 0 END) AS total_api_errors,
        SUM(CASE WHEN subtype = 'compact_boundary' THEN 1 ELSE 0 END) AS total_compactions,
        SUM(CASE WHEN subtype = 'turn_duration' THEN COALESCE(duration_ms, 0) ELSE 0 END)
            AS total_turn_durations_ms,
        SUM(CASE WHEN subtype = 'turn_duration' THEN 1 ELSE 0 END) AS turn_count,
        SUM(CASE WHEN subtype = 'stop_hook_summary' THEN 1 ELSE 0 END) AS total_stop_events,
        SUM(CASE WHEN subtype = 'stop_hook_summary' AND prevented_continuation THEN 1 ELSE 0 END)
            AS total_prevented_continuations
    FROM fact_system_events
    WHERE is_deleted = FALSE
    GROUP BY session_id
),
progress_rollup AS (
    SELECT
        session_id,
        COUNT(*) AS total_progress_events,
        SUM(CASE WHEN data_type = 'hook_progress' THEN 1 ELSE 0 END) AS total_hook_progress_events,
        SUM(CASE WHEN data_type = 'bash_progress' THEN 1 ELSE 0 END) AS total_bash_progress_events
    FROM fact_progress_events
    WHERE is_deleted = FALSE
    GROUP BY session_id
),
attachment_rollup AS (
    SELECT
        session_id,
        COUNT(*) AS total_attachments,
        SUM(CASE WHEN attachment_type = 'diagnostics' THEN 1 ELSE 0 END) AS total_diagnostics,
        SUM(CASE WHEN attachment_type = 'hook_success' THEN 1 ELSE 0 END) AS total_hook_successes
    FROM fact_attachments
    WHERE is_deleted = FALSE
    GROUP BY session_id
),
meta_rollup AS (
    SELECT
        session_id,
        SUM(CASE WHEN meta_type = 'permission-mode' THEN 1 ELSE 0 END)
            AS permission_mode_transition_count
    FROM fact_meta_events
    WHERE is_deleted = FALSE
    GROUP BY session_id
),
meta_current_mode AS (
    -- Last permission-mode value in timestamp order.
    SELECT
        session_id,
        meta_value AS current_permission_mode
    FROM (
        SELECT session_id, meta_value,
               row_number() OVER (PARTITION BY session_id ORDER BY timestamp DESC) AS rn
        FROM fact_meta_events
        WHERE is_deleted = FALSE AND meta_type = 'permission-mode'
    )
    WHERE rn = 1
),
file_history_rollup AS (
    SELECT session_id, COUNT(*) AS total_file_history_snapshots
    FROM fact_file_history_snapshots
    WHERE is_deleted = FALSE
    GROUP BY session_id
)
SELECT
    se.session_id,
    se.source_path,
    se.first_timestamp,
    se.last_timestamp,
    EXTRACT(EPOCH FROM (se.last_timestamp - se.first_timestamp))::DOUBLE
        AS session_duration_seconds,
    -- timestamp surrogate for date_key/time_key derivation in lineage_upsert
    se.first_timestamp AS timestamp,

    COALESCE(mr.total_messages, 0) AS total_messages,
    COALESCE(mr.user_messages, 0) AS user_messages,
    COALESCE(mr.assistant_messages, 0) AS assistant_messages,
    COALESCE(mr.total_thinking_blocks, 0) AS total_thinking_blocks,

    COALESCE(tr.total_input_tokens, 0) AS total_input_tokens,
    COALESCE(tr.total_output_tokens, 0) AS total_output_tokens,
    COALESCE(tr.total_cache_creation_5m_tokens, 0) AS total_cache_creation_5m_tokens,
    COALESCE(tr.total_cache_creation_1h_tokens, 0) AS total_cache_creation_1h_tokens,
    COALESCE(tr.total_cache_creation_total_tokens, 0) AS total_cache_creation_total_tokens,
    COALESCE(tr.total_cache_read_tokens, 0) AS total_cache_read_tokens,
    COALESCE(tr.total_uncached_equivalent_tokens, 0) AS total_uncached_equivalent_tokens,
    COALESCE(tr.api_response_count, 0) AS api_response_count,

    COALESCE(tur.total_tool_uses, 0) AS total_tool_uses,
    COALESCE(tur.unique_tools_used, 0) AS unique_tools_used,
    COALESCE(trr.total_tool_results, 0) AS total_tool_results,
    COALESCE(trr.total_tool_errors, 0) AS total_tool_errors,
    COALESCE(trr.total_bash_interrupted, 0) AS total_bash_interrupted,

    COALESCE(sr.total_api_errors, 0) AS total_api_errors,
    COALESCE(sr.total_compactions, 0) AS total_compactions,
    COALESCE(sr.total_turn_durations_ms, 0) AS total_turn_durations_ms,
    COALESCE(sr.turn_count, 0) AS turn_count,
    COALESCE(sr.total_stop_events, 0) AS total_stop_events,
    COALESCE(sr.total_prevented_continuations, 0) AS total_prevented_continuations,

    COALESCE(pr.total_progress_events, 0) AS total_progress_events,
    COALESCE(pr.total_hook_progress_events, 0) AS total_hook_progress_events,
    COALESCE(pr.total_bash_progress_events, 0) AS total_bash_progress_events,
    COALESCE(ar.total_attachments, 0) AS total_attachments,
    COALESCE(ar.total_diagnostics, 0) AS total_diagnostics,
    COALESCE(ar.total_hook_successes, 0) AS total_hook_successes,

    COALESCE(metar.permission_mode_transition_count, 0) AS permission_mode_transition_count,
    mcm.current_permission_mode,

    COALESCE(fhr.total_file_history_snapshots, 0) AS total_file_history_snapshots
FROM session_envelope se
LEFT JOIN msg_rollup mr ON mr.session_id = se.session_id
LEFT JOIN token_rollup tr ON tr.session_id = se.session_id
LEFT JOIN tool_use_rollup tur ON tur.session_id = se.session_id
LEFT JOIN tool_result_rollup trr ON trr.session_id = se.session_id
LEFT JOIN system_rollup sr ON sr.session_id = se.session_id
LEFT JOIN progress_rollup pr ON pr.session_id = se.session_id
LEFT JOIN attachment_rollup ar ON ar.session_id = se.session_id
LEFT JOIN meta_rollup metar ON metar.session_id = se.session_id
LEFT JOIN meta_current_mode mcm ON mcm.session_id = se.session_id
LEFT JOIN file_history_rollup fhr ON fhr.session_id = se.session_id
"""


def populate_fact_session_summary(conn, *, run: EtlRun) -> None:
    """One row per session aggregating every v0.15 fact.

    Natural key: session_id. lineage_upsert handles the standard
    UPDATE/INSERT/soft-delete with hash_diff change detection.

    Should be the LAST populator called in an ETL run -- everything else
    has to be populated for the aggregates to be correct.
    """
    conn.execute("DROP TABLE IF EXISTS _inbound_session_summary")
    conn.execute(f"CREATE TEMP TABLE _inbound_session_summary AS {_PROJECT_SQL}")

    # Derive project_key from the per-session source_path (every row of one
    # session shares one source_path).
    conn.execute("ALTER TABLE _inbound_session_summary ADD COLUMN project_key VARCHAR")
    conn.execute(
        "UPDATE _inbound_session_summary "
        "SET project_key = md5(regexp_replace(source_path, '/[^/]+$', ''))"
    )

    lineage_upsert(
        conn,
        run=run,
        table="fact_session_summary",
        inbound_table="_inbound_session_summary",
        natural_key="session_id",
        payload_cols=_PAYLOAD_COLS,
        hash_cols=_HASH_COLS,
    )
