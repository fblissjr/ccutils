"""Populate the seven new entry-type facts (Phase C4).

Each populator builds a per-entry-type inbound temp table from staging,
then delegates to lineage_upsert() for the UPDATE/INSERT/soft-delete
choreography.
"""

from __future__ import annotations

from ccutils.etl.lineage import EtlRun
from ccutils.etl.upsert import lineage_upsert


# --------------------------------------------------------------------------
# fact_attachments
# --------------------------------------------------------------------------

_ATTACH_PAYLOAD_COLS = ["timestamp", "attachment_type", "attachment_json"]
_ATTACH_HASH_COLS = ["timestamp", "attachment_type", "attachment_json"]


def populate_fact_attachments(conn, *, run: EtlRun) -> None:
    conn.execute("DROP TABLE IF EXISTS _inbound_attachments")
    conn.execute(
        """
        CREATE TEMP TABLE _inbound_attachments AS
        SELECT
            sle.entry_id,
            sle.session_id,
            TRY_CAST(sle.timestamp AS TIMESTAMP) AS timestamp,
            json_extract_string(sle.attachment_json, '$.type') AS attachment_type,
            sle.attachment_json
        FROM stg_log_entries sle
        WHERE sle.type = 'attachment'
        """
    )
    lineage_upsert(
        conn, run=run,
        table="fact_attachments",
        inbound_table="_inbound_attachments",
        natural_key="entry_id",
        payload_cols=_ATTACH_PAYLOAD_COLS,
        hash_cols=_ATTACH_HASH_COLS,
    )


# --------------------------------------------------------------------------
# fact_progress_events
# --------------------------------------------------------------------------

_PROG_PAYLOAD_COLS = [
    "timestamp", "data_type", "tool_use_id", "parent_tool_use_id",
    "hook_name", "hook_event", "agent_id", "data_json",
]
_PROG_HASH_COLS = [
    "timestamp", "data_type", "tool_use_id", "parent_tool_use_id",
    "hook_name", "hook_event", "agent_id", "data_json",
]


def populate_fact_progress_events(conn, *, run: EtlRun) -> None:
    conn.execute("DROP TABLE IF EXISTS _inbound_progress")
    conn.execute(
        """
        CREATE TEMP TABLE _inbound_progress AS
        SELECT
            sle.entry_id,
            sle.session_id,
            TRY_CAST(sle.timestamp AS TIMESTAMP) AS timestamp,
            json_extract_string(sle.progress_data_json, '$.type') AS data_type,
            json_extract_string(sle.raw_json, '$.toolUseID') AS tool_use_id,
            json_extract_string(sle.raw_json, '$.parentToolUseID') AS parent_tool_use_id,
            json_extract_string(sle.progress_data_json, '$.hookName') AS hook_name,
            json_extract_string(sle.progress_data_json, '$.hookEvent') AS hook_event,
            json_extract_string(sle.progress_data_json, '$.agentId') AS agent_id,
            sle.progress_data_json AS data_json
        FROM stg_log_entries sle
        WHERE sle.type = 'progress'
        """
    )
    lineage_upsert(
        conn, run=run,
        table="fact_progress_events",
        inbound_table="_inbound_progress",
        natural_key="entry_id",
        payload_cols=_PROG_PAYLOAD_COLS,
        hash_cols=_PROG_HASH_COLS,
    )


# --------------------------------------------------------------------------
# fact_system_events
# --------------------------------------------------------------------------

_SYS_PAYLOAD_COLS = [
    "timestamp", "subtype", "level",
    "duration_ms", "message_count",
    "hook_count", "prevented_continuation", "stop_reason", "has_output",
    "error_status", "error_type",
    "retry_in_ms", "retry_attempt", "max_retries",
    "compact_trigger", "compact_pre_tokens", "logical_parent_uuid",
    "content", "bridge_url",
    "payload_json",
]
_SYS_HASH_COLS = _SYS_PAYLOAD_COLS  # all of payload is content-bearing


def populate_fact_system_events(conn, *, run: EtlRun) -> None:
    conn.execute("DROP TABLE IF EXISTS _inbound_system")
    conn.execute(
        """
        CREATE TEMP TABLE _inbound_system AS
        SELECT
            sle.entry_id,
            sle.session_id,
            TRY_CAST(sle.timestamp AS TIMESTAMP) AS timestamp,
            sle.system_subtype AS subtype,
            json_extract_string(sle.system_payload_json, '$.level') AS level,
            -- turn_duration
            json_extract(sle.system_payload_json, '$.durationMs')::INTEGER AS duration_ms,
            json_extract(sle.system_payload_json, '$.messageCount')::INTEGER AS message_count,
            -- stop_hook_summary
            json_extract(sle.system_payload_json, '$.hookCount')::INTEGER AS hook_count,
            json_extract(sle.system_payload_json, '$.preventedContinuation')::BOOLEAN AS prevented_continuation,
            json_extract_string(sle.system_payload_json, '$.stopReason') AS stop_reason,
            json_extract(sle.system_payload_json, '$.hasOutput')::BOOLEAN AS has_output,
            -- api_error
            json_extract(sle.system_payload_json, '$.error.status')::INTEGER AS error_status,
            json_extract_string(sle.system_payload_json, '$.error.type') AS error_type,
            json_extract(sle.system_payload_json, '$.retryInMs')::FLOAT AS retry_in_ms,
            json_extract(sle.system_payload_json, '$.retryAttempt')::INTEGER AS retry_attempt,
            json_extract(sle.system_payload_json, '$.maxRetries')::INTEGER AS max_retries,
            -- compact_boundary
            json_extract_string(sle.system_payload_json, '$.compactMetadata.trigger') AS compact_trigger,
            json_extract(sle.system_payload_json, '$.compactMetadata.preTokens')::INTEGER AS compact_pre_tokens,
            json_extract_string(sle.system_payload_json, '$.logicalParentUuid') AS logical_parent_uuid,
            -- local_command / away_summary / bridge_status (text)
            json_extract_string(sle.system_payload_json, '$.content') AS content,
            json_extract_string(sle.system_payload_json, '$.url') AS bridge_url,
            sle.system_payload_json AS payload_json
        FROM stg_log_entries sle
        WHERE sle.type = 'system'
        """
    )
    lineage_upsert(
        conn, run=run,
        table="fact_system_events",
        inbound_table="_inbound_system",
        natural_key="entry_id",
        payload_cols=_SYS_PAYLOAD_COLS,
        hash_cols=_SYS_HASH_COLS,
    )


# --------------------------------------------------------------------------
# fact_meta_events
# --------------------------------------------------------------------------

_META_PAYLOAD_COLS = ["timestamp", "meta_type", "meta_value"]
_META_HASH_COLS = _META_PAYLOAD_COLS


def populate_fact_meta_events(conn, *, run: EtlRun) -> None:
    """Time-series for permission-mode, custom-title, agent-name, last-prompt.

    Critically: each entry is its own row -- NOT the last value on
    dim_session as the legacy ETL kept.
    """
    conn.execute("DROP TABLE IF EXISTS _inbound_meta")
    conn.execute(
        """
        CREATE TEMP TABLE _inbound_meta AS
        SELECT
            sle.entry_id,
            sle.session_id,
            TRY_CAST(sle.timestamp AS TIMESTAMP) AS timestamp,
            sle.type AS meta_type,
            CASE sle.type
                WHEN 'permission-mode' THEN json_extract_string(sle.meta_payload_json, '$.permission_mode')
                WHEN 'custom-title' THEN json_extract_string(sle.meta_payload_json, '$.customTitle')
                WHEN 'agent-name' THEN json_extract_string(sle.meta_payload_json, '$.agentName')
                WHEN 'last-prompt' THEN json_extract_string(sle.meta_payload_json, '$.lastPrompt')
            END AS meta_value
        FROM stg_log_entries sle
        WHERE sle.type IN ('permission-mode', 'custom-title', 'agent-name', 'last-prompt')
        """
    )
    lineage_upsert(
        conn, run=run,
        table="fact_meta_events",
        inbound_table="_inbound_meta",
        natural_key="entry_id",
        payload_cols=_META_PAYLOAD_COLS,
        hash_cols=_META_HASH_COLS,
    )


# --------------------------------------------------------------------------
# fact_file_history_snapshots
# --------------------------------------------------------------------------

_FHS_PAYLOAD_COLS = [
    "timestamp", "message_id_link", "is_snapshot_update", "snapshot_json",
]
_FHS_HASH_COLS = _FHS_PAYLOAD_COLS


def populate_fact_file_history_snapshots(conn, *, run: EtlRun) -> None:
    conn.execute("DROP TABLE IF EXISTS _inbound_fhs")
    conn.execute(
        """
        CREATE TEMP TABLE _inbound_fhs AS
        SELECT
            sle.entry_id,
            sle.session_id,
            TRY_CAST(sle.timestamp AS TIMESTAMP) AS timestamp,
            json_extract_string(sle.meta_payload_json, '$.messageId') AS message_id_link,
            json_extract(sle.meta_payload_json, '$.isSnapshotUpdate')::BOOLEAN AS is_snapshot_update,
            CAST(json_extract(sle.meta_payload_json, '$.snapshot') AS VARCHAR) AS snapshot_json
        FROM stg_log_entries sle
        WHERE sle.type = 'file-history-snapshot'
        """
    )
    lineage_upsert(
        conn, run=run,
        table="fact_file_history_snapshots",
        inbound_table="_inbound_fhs",
        natural_key="entry_id",
        payload_cols=_FHS_PAYLOAD_COLS,
        hash_cols=_FHS_HASH_COLS,
    )


# --------------------------------------------------------------------------
# fact_queue_operations
# --------------------------------------------------------------------------

_QO_PAYLOAD_COLS = ["timestamp", "operation", "content"]
_QO_HASH_COLS = _QO_PAYLOAD_COLS


def populate_fact_queue_operations(conn, *, run: EtlRun) -> None:
    conn.execute("DROP TABLE IF EXISTS _inbound_qo")
    conn.execute(
        """
        CREATE TEMP TABLE _inbound_qo AS
        SELECT
            sle.entry_id,
            sle.session_id,
            TRY_CAST(sle.timestamp AS TIMESTAMP) AS timestamp,
            json_extract_string(sle.meta_payload_json, '$.operation') AS operation,
            json_extract_string(sle.meta_payload_json, '$.content') AS content
        FROM stg_log_entries sle
        WHERE sle.type = 'queue-operation'
        """
    )
    lineage_upsert(
        conn, run=run,
        table="fact_queue_operations",
        inbound_table="_inbound_qo",
        natural_key="entry_id",
        payload_cols=_QO_PAYLOAD_COLS,
        hash_cols=_QO_HASH_COLS,
    )


# --------------------------------------------------------------------------
# fact_pr_links
# --------------------------------------------------------------------------

_PR_PAYLOAD_COLS = ["timestamp", "pr_number", "pr_url", "pr_repository"]
_PR_HASH_COLS = _PR_PAYLOAD_COLS


def populate_fact_pr_links(conn, *, run: EtlRun) -> None:
    conn.execute("DROP TABLE IF EXISTS _inbound_pr")
    conn.execute(
        """
        CREATE TEMP TABLE _inbound_pr AS
        SELECT
            sle.entry_id,
            sle.session_id,
            TRY_CAST(sle.timestamp AS TIMESTAMP) AS timestamp,
            TRY_CAST(json_extract_string(sle.meta_payload_json, '$.prNumber') AS INTEGER) AS pr_number,
            json_extract_string(sle.meta_payload_json, '$.prUrl') AS pr_url,
            json_extract_string(sle.meta_payload_json, '$.prRepository') AS pr_repository
        FROM stg_log_entries sle
        WHERE sle.type = 'pr-link'
        """
    )
    lineage_upsert(
        conn, run=run,
        table="fact_pr_links",
        inbound_table="_inbound_pr",
        natural_key="entry_id",
        payload_cols=_PR_PAYLOAD_COLS,
        hash_cols=_PR_HASH_COLS,
    )
