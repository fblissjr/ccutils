"""Populate bridge_session_file from fact_file_operations.

Grain: one row per (session, file) pair. Cheap aggregate -- counts of
read/write/edit ops per pair plus the timestamp window.

The natural key is session_file_key = md5(session_id || '|' || file_key).
Rebuilds drop-and-reload by deleting prior rows for the sessions
currently in fact_file_operations and re-inserting; hash_diff guards
ensure re-running on unchanged source is a no-op via lineage_upsert.

Run AFTER populate_fact_file_operations.
"""

from __future__ import annotations

from ccutils.etl.lineage import EtlRun
from ccutils.etl.upsert import lineage_upsert


# session_key is derived by lineage_upsert from session_id, so we don't
# list it here. file_key is part of the natural key but stays in payload
# so updates flow through.
_PAYLOAD_COLS = [
    "file_key",
    "first_operation_timestamp", "last_operation_timestamp",
    "operation_count", "read_count", "write_count", "edit_count",
    "total_chars_written",
]
_HASH_COLS = [
    "file_key",
    "first_operation_timestamp", "last_operation_timestamp",
    "operation_count", "read_count", "write_count", "edit_count",
    "total_chars_written",
]


def populate_bridge_session_file(conn, *, run: EtlRun) -> None:
    """Aggregate fact_file_operations by (session, file) into bridge_session_file."""
    conn.execute("DROP TABLE IF EXISTS _inbound_bridge_session_file")
    conn.execute(
        """
        CREATE TEMP TABLE _inbound_bridge_session_file AS
        SELECT
            md5(ffo.session_id || '|' || ffo.file_key) AS session_file_key,
            ffo.session_id,
            ffo.session_key,
            ffo.file_key,
            -- Pick a representative timestamp for the lineage helper to
            -- derive date_key / time_key from. first_operation_timestamp
            -- is the natural choice.
            MIN(ffo.timestamp) AS timestamp,
            MIN(ffo.timestamp) AS first_operation_timestamp,
            MAX(ffo.timestamp) AS last_operation_timestamp,
            COUNT(*) AS operation_count,
            SUM(CASE WHEN ffo.operation_type = 'read' THEN 1 ELSE 0 END)
                AS read_count,
            SUM(CASE WHEN ffo.operation_type = 'write' THEN 1 ELSE 0 END)
                AS write_count,
            SUM(CASE WHEN ffo.operation_type = 'edit' THEN 1 ELSE 0 END)
                AS edit_count,
            SUM(CASE WHEN ffo.operation_type IN ('write', 'edit')
                     THEN COALESCE(ffo.file_size_chars, 0) ELSE 0 END)
                AS total_chars_written
        FROM fact_file_operations ffo
        WHERE ffo.is_deleted = FALSE
          AND ffo.session_id IS NOT NULL
          AND ffo.file_key IS NOT NULL
          -- Scope to the session currently being ETL'd. Aggregates for
          -- other sessions don't change unless their underlying file
          -- ops do, so re-aggregating them would just churn through the
          -- hash_diff no-op path.
          AND ffo.session_id IN (
              SELECT DISTINCT session_id FROM stg_log_entries
              WHERE session_id IS NOT NULL
          )
        GROUP BY ffo.session_id, ffo.session_key, ffo.file_key
        """
    )
    lineage_upsert(
        conn, run=run,
        table="bridge_session_file",
        inbound_table="_inbound_bridge_session_file",
        natural_key="session_file_key",
        payload_cols=_PAYLOAD_COLS,
        hash_cols=_HASH_COLS,
        timestamp_col="first_operation_timestamp",
    )
