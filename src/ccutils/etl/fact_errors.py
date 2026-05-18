"""Populate fact_errors from fact_tool_results where is_error = TRUE.

One row per failed tool call. error_type is classified by the same
zero-dep regex rules as ccutils.etl.heuristics.classify_error_type, but
applied inline via DuckDB regexp_matches so the populator stays
SQL-only. The Python classifier and the SQL CASE here MUST stay in
sync; the test suite exercises both paths.

Run AFTER populate_fact_tool_uses + populate_fact_tool_results.
"""

from __future__ import annotations

from ccutils.etl.lineage import EtlRun
from ccutils.etl.upsert import lineage_upsert


_PAYLOAD_COLS = [
    "tool_use_id", "tool_key", "timestamp",
    "error_type", "error_message",
]
_HASH_COLS = [
    "tool_use_id", "tool_key",
    "error_type", "error_message",
]

# Inline SQL CASE that mirrors heuristics.classify_error_type. Patterns
# below use DuckDB's (?i) case-insensitive flag; first match wins so the
# CASE order matters and must match the Python _ERROR_RULES list.
_ERROR_TYPE_CASE = """
CASE
    WHEN regexp_matches(error_message, '(?i)permission denied|EACCES')
        THEN 'permission_denied'
    WHEN regexp_matches(error_message, '(?i)not found|ENOENT|no such file')
        THEN 'file_not_found'
    WHEN regexp_matches(error_message, '(?i)syntax error|SyntaxError')
        THEN 'syntax_error'
    WHEN regexp_matches(error_message, '(?i)timeout|ETIMEDOUT')
        THEN 'timeout'
    WHEN regexp_matches(error_message, '(?i)ImportError|ModuleNotFoundError')
        THEN 'import_error'
    ELSE 'tool_error'
END
"""


def populate_fact_errors(conn, *, run: EtlRun) -> None:
    """Derive one fact_errors row per fact_tool_results.is_error = TRUE."""
    conn.execute("DROP TABLE IF EXISTS _inbound_errors")
    conn.execute(
        f"""
        CREATE TEMP TABLE _inbound_errors AS
        WITH base AS (
            SELECT
                md5(ftr.session_id || '|' || ftr.tool_use_id) AS error_id,
                ftr.tool_use_id,
                ftr.session_id,
                ftr.tool_key,
                ftr.timestamp,
                COALESCE(ftr.result_content_text, '') AS error_message
            FROM fact_tool_results ftr
            WHERE ftr.is_deleted = FALSE
              AND ftr.is_error = TRUE
              AND ftr.session_id IN (
                  SELECT DISTINCT session_id FROM stg_log_entries
                  WHERE session_id IS NOT NULL
              )
        )
        SELECT
            error_id,
            tool_use_id,
            session_id,
            tool_key,
            timestamp,
            {_ERROR_TYPE_CASE} AS error_type,
            error_message
        FROM base
        """
    )
    lineage_upsert(
        conn, run=run,
        table="fact_errors",
        inbound_table="_inbound_errors",
        natural_key="error_id",
        payload_cols=_PAYLOAD_COLS,
        hash_cols=_HASH_COLS,
    )
