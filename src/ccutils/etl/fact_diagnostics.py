"""Populate fact_diagnostics from fact_attachments where type='diagnostics'.

A single diagnostics attachment carries a nested `files` list, each
file carries a list of `diagnostics`. We flatten to one fact row per
individual diagnostic with the file_uri promoted to a column and
file_key looked up against dim_file (NULL if the file isn't tracked).

Run AFTER populate_fact_attachments (and after populate_dim_file if
you want file_key FKs populated).
"""

from __future__ import annotations

from ccutils.etl.lineage import EtlRun
from ccutils.etl.upsert import lineage_upsert


_PAYLOAD_COLS = [
    "entry_id", "timestamp", "file_path", "file_key",
    "severity", "source", "code", "message",
    "range_start_line", "range_start_col",
    "range_end_line", "range_end_col",
]
_HASH_COLS = [
    "timestamp", "file_path", "severity", "source", "code", "message",
    "range_start_line", "range_start_col",
    "range_end_line", "range_end_col",
]


def populate_fact_diagnostics(conn, *, run: EtlRun) -> None:
    """Flatten diagnostics attachments into one row per diagnostic."""
    conn.execute("DROP TABLE IF EXISTS _inbound_diagnostics")
    conn.execute(
        """
        CREATE TEMP TABLE _inbound_diagnostics AS
        WITH per_file AS (
            -- One row per (attachment, file_entry). Each file_entry has a
            -- `uri` and a `diagnostics` list.
            SELECT
                fa.entry_id,
                fa.session_id,
                fa.timestamp,
                json_extract_string(file_entry, '$.uri') AS file_uri,
                json_extract(file_entry, '$.diagnostics') AS diag_list_json
            FROM fact_attachments fa,
            LATERAL (
                SELECT unnest(json_extract(fa.attachment_json, '$.files')::JSON[])
                    AS file_entry
            )
            WHERE fa.is_deleted = FALSE
              AND fa.attachment_type = 'diagnostics'
              AND json_type(fa.attachment_json, '$.files') = 'ARRAY'
              -- Scope to current session: prior sessions' diagnostics
              -- are already in target and would no-op through hash_diff.
              AND fa.session_id IN (
                  SELECT DISTINCT session_id FROM stg_log_entries
                  WHERE session_id IS NOT NULL
              )
        ),
        per_diag AS (
            SELECT
                per_file.entry_id,
                per_file.session_id,
                per_file.timestamp,
                per_file.file_uri AS file_path,
                d AS diag,
                idx
            FROM per_file,
            LATERAL (
                SELECT
                    unnest(per_file.diag_list_json::JSON[]) AS d,
                    generate_subscripts(per_file.diag_list_json::JSON[], 1)
                        AS idx
            )
        )
        SELECT
            -- diagnostic_id is stable per (entry, file, line, col, code)
            md5(
                entry_id || '|' || COALESCE(file_path, '') || '|'
                || CAST(idx AS VARCHAR)
            ) AS diagnostic_id,
            entry_id,
            session_id,
            timestamp,
            file_path,
            md5(file_path) AS file_key,
            json_extract_string(diag, '$.severity') AS severity,
            json_extract_string(diag, '$.source') AS source,
            json_extract_string(diag, '$.code') AS code,
            json_extract_string(diag, '$.message') AS message,
            CAST(json_extract(diag, '$.range.start.line') AS INTEGER)
                AS range_start_line,
            CAST(json_extract(diag, '$.range.start.character') AS INTEGER)
                AS range_start_col,
            CAST(json_extract(diag, '$.range.end.line') AS INTEGER)
                AS range_end_line,
            CAST(json_extract(diag, '$.range.end.character') AS INTEGER)
                AS range_end_col
        FROM per_diag
        """
    )
    lineage_upsert(
        conn, run=run,
        table="fact_diagnostics",
        inbound_table="_inbound_diagnostics",
        natural_key="diagnostic_id",
        payload_cols=_PAYLOAD_COLS,
        hash_cols=_HASH_COLS,
    )
