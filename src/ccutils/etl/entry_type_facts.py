"""Populate the entry-type facts: entry events, progress events, system events.

Each populator builds an inbound temp table from staging, then delegates to
lineage_upsert() for the UPDATE/INSERT/soft-delete choreography.
"""

from __future__ import annotations

from ccutils.etl.lineage import EtlRun
from ccutils.etl.upsert import lineage_upsert


# --------------------------------------------------------------------------
# fact_entry_events
# --------------------------------------------------------------------------

#: The entry types that land in fact_entry_events, and nothing else does.
#: A list, not "every type that is not a message": taking a new type in is a
#: decision about what the warehouse holds, made per type. It was five
#: tables until 1.0.0, one per group below.
ENTRY_EVENT_TYPES = (
    "attachment",
    "permission-mode", "custom-title", "agent-name", "last-prompt",
    "file-history-snapshot",
    "queue-operation",
    "pr-link",
)

_ENTRY_PAYLOAD_COLS = [
    "sequence_num", "timestamp", "derived_timestamp",
    "entry_type", "subtype", "value_text", "payload_json",
]
_ENTRY_HASH_COLS = _ENTRY_PAYLOAD_COLS

# What each entry type contributes. `subtype` is the sub-kind the entry
# states, where it states one. `value_text` is its one scalar, where it has
# one. `payload_json` is the staged payload whole, so nothing a narrower
# table used to hold as a column is lost: a pr-link's number and repository,
# and a snapshot's isSnapshotUpdate flag and backup map, are read from it.
#
# Time. The meta entries (permission-mode, custom-title, agent-name,
# last-prompt) state no timestamp at all, so `timestamp` is NULL on them and
# the only order they have is their position in the file, `sequence_num`.
# `derived_timestamp` places such an entry by its neighbours: the stated
# time of the nearest entry before it in the same file, or after it when
# nothing timestamped precedes. It is NULL where the entry states its own,
# so the stated and the derived value never share a column. `key_timestamp`
# is whichever exists and feeds only date_key / time_key.
_PROJECT_ENTRY_EVENTS_SQL = """
CREATE TEMP TABLE _inbound_entry_events AS
WITH stated AS (
    -- EVERY staged entry, not only the types kept below: an entry is placed
    -- by whatever sits around it in the file.
    SELECT
        sle.*,
        COALESCE(
            TRY_CAST(sle.timestamp AS TIMESTAMP),
            -- A file-history-snapshot entry carries no timestamp of its
            -- own; the snapshot inside it states one.
            CASE WHEN sle.type = 'file-history-snapshot' THEN TRY_CAST(
                json_extract_string(sle.meta_payload_json, '$.snapshot.timestamp')
                AS TIMESTAMP) END
        ) AS stated_ts
    FROM etl.log_entries sle
),
placed AS (
    SELECT
        s.*,
        LAST_VALUE(s.stated_ts IGNORE NULLS) OVER (
            PARTITION BY s.source_path ORDER BY s.sequence_num
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS ts_before,
        FIRST_VALUE(s.stated_ts IGNORE NULLS) OVER (
            PARTITION BY s.source_path ORDER BY s.sequence_num
            ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS ts_after
    FROM stated s
)
SELECT
    sle.entry_id,
    sle.session_id,
    sle.sequence_num,
    sle.stated_ts AS timestamp,
    CASE WHEN sle.stated_ts IS NULL
         THEN COALESCE(sle.ts_before, sle.ts_after) END AS derived_timestamp,
    COALESCE(sle.stated_ts, sle.ts_before, sle.ts_after) AS key_timestamp,
    sle.type AS entry_type,
    CASE sle.type
        WHEN 'attachment' THEN json_extract_string(sle.attachment_json, '$.type')
        WHEN 'queue-operation' THEN json_extract_string(sle.meta_payload_json, '$.operation')
    END AS subtype,
    CASE sle.type
        WHEN 'permission-mode' THEN json_extract_string(sle.meta_payload_json, '$.permission_mode')
        WHEN 'custom-title' THEN json_extract_string(sle.meta_payload_json, '$.customTitle')
        WHEN 'agent-name' THEN json_extract_string(sle.meta_payload_json, '$.agentName')
        WHEN 'last-prompt' THEN json_extract_string(sle.meta_payload_json, '$.lastPrompt')
        WHEN 'queue-operation' THEN json_extract_string(sle.meta_payload_json, '$.content')
        WHEN 'pr-link' THEN json_extract_string(sle.meta_payload_json, '$.prUrl')
        -- The message the snapshot belongs to.
        WHEN 'file-history-snapshot' THEN json_extract_string(sle.meta_payload_json, '$.messageId')
    END AS value_text,
    CAST(CASE WHEN sle.type = 'attachment' THEN sle.attachment_json
              ELSE sle.meta_payload_json END AS VARCHAR) AS payload_json
FROM placed sle
WHERE sle.type IN ({types})
"""


def populate_fact_entry_events(conn, *, run: EtlRun) -> None:
    """One row per entry of a type in ENTRY_EVENT_TYPES.

    One row per ENTRY, which for the meta types is not one row per change:
    Claude Code restates the current permission mode far more often than it
    changes it. Changes are found by comparing a row with the one before it
    in `sequence_num` order, which is what the views do.
    """
    conn.execute("DROP TABLE IF EXISTS _inbound_entry_events")
    conn.execute(_PROJECT_ENTRY_EVENTS_SQL.format(
        types=", ".join(f"'{t}'" for t in ENTRY_EVENT_TYPES)))
    lineage_upsert(
        conn, run=run,
        table="fact_entry_events",
        inbound_table="_inbound_entry_events",
        natural_key="entry_id",
        payload_cols=_ENTRY_PAYLOAD_COLS,
        hash_cols=_ENTRY_HASH_COLS,
        timestamp_col="key_timestamp",
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
        FROM etl.log_entries sle
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
    "hook_count", "prevented_continuation", "has_output",
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
        FROM etl.log_entries sle
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
