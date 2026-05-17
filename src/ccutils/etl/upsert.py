"""Shared lineage-upsert helper used by every v0.15 fact populator.

The pattern: build a temp table of inbound rows (one row per natural key),
compute hash_diff in SQL, then:
  1. UPDATE existing rows whose hash_diff changed.
  2. INSERT new rows.
  3. Soft-delete rows for the loaded sessions whose natural_key is no
     longer in the inbound batch.

Centralizing here means every v0.15 fact follows the same lineage convention
without copy-paste drift. Adding a new lineage column happens in one place.

Note on identifier interpolation: `table`, `inbound_table`, and column names
are f-string-interpolated into SQL because DuckDB doesn't accept placeholders
for identifiers. All values for these come from this module or other
ccutils code -- never from user input. `version_key` and `etl_run_id` ARE
parameterized.
"""

from __future__ import annotations

import re

from ccutils.etl.lineage import EtlRun


# Identifier names that pass into SQL identifier positions are validated
# against this pattern to prevent any future use from accepting unsafe input.
_IDENT_RE = re.compile(r"^[a-zA-Z_][\w]{0,127}$")


def _validate_ident(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


def hash_diff_sql(cols: list[str]) -> str:
    """SQL expression that computes an MD5 hash_diff over the given columns.

    Each column is COALESCEd to '' (so NULL doesn't propagate) and CAST to
    VARCHAR. Pipe-separated, then md5'd. Equivalent to the Python
    `hash_diff(**attrs)` helper in lineage.py.
    """
    for c in cols:
        _validate_ident(c)
    return "md5(" + " || '|' || ".join(
        f"COALESCE(CAST({c} AS VARCHAR), '')" for c in cols
    ) + ")"


def lineage_upsert(
    conn,
    *,
    run: EtlRun,
    table: str,
    inbound_table: str,
    natural_key: str,
    payload_cols: list[str],
    hash_cols: list[str],
    derive_session_keys: bool = True,
    record_source: str = "claude_code_jsonl",
) -> None:
    """Generic UPDATE/INSERT/soft-delete pattern shared by every populator.

    Args:
        conn: DuckDB connection.
        run: ETL run handle. version_key and etl_run_id are parameterized.
        table: target fact table.
        inbound_table: name of the populated temp table. MUST already contain
            entry_id (or whatever natural_key is set to), session_id, and all
            payload_cols. If derive_session_keys is True, ALSO must contain
            `timestamp` -- session_key/date_key/time_key/hash_diff are added.
        natural_key: column name used to match rows (e.g. 'entry_id' for
            most entry-type facts; 'tool_use_id' for fact_tool_uses/results).
        payload_cols: columns copied from inbound -> target on INSERT/UPDATE.
            EXCLUDES lineage columns and natural_key + session_id (handled
            separately). Their VALUES are interpolated into SQL but the
            VALUE itself is never user-controlled.
        hash_cols: columns used in the hash_diff computation.
        derive_session_keys: when True (default), adds session_key, date_key,
            time_key, hash_diff to the inbound table. Set False when the
            caller has already done that work.
        record_source: provenance label stamped on inserted rows.
    """
    _validate_ident(table)
    _validate_ident(inbound_table)
    _validate_ident(natural_key)
    for c in payload_cols:
        _validate_ident(c)
    for c in hash_cols:
        _validate_ident(c)

    if derive_session_keys:
        for ddl in (
            f"ALTER TABLE {inbound_table} ADD COLUMN IF NOT EXISTS session_key VARCHAR",
            f"ALTER TABLE {inbound_table} ADD COLUMN IF NOT EXISTS date_key INTEGER",
            f"ALTER TABLE {inbound_table} ADD COLUMN IF NOT EXISTS time_key INTEGER",
            f"ALTER TABLE {inbound_table} ADD COLUMN IF NOT EXISTS hash_diff VARCHAR",
        ):
            conn.execute(ddl)
        conn.execute(f"UPDATE {inbound_table} SET session_key = md5(session_id)")
        conn.execute(
            f"UPDATE {inbound_table} "
            f"SET date_key = CAST(strftime(timestamp, '%Y%m%d') AS INTEGER), "
            f"    time_key = CAST(strftime(timestamp, '%H%M') AS INTEGER) "
            f"WHERE timestamp IS NOT NULL"
        )
        conn.execute(
            f"UPDATE {inbound_table} SET hash_diff = {hash_diff_sql(hash_cols)}"
        )

    set_clause = ",\n            ".join(
        f"{c} = im.{c}" for c in (
            *payload_cols, "session_key", "date_key", "time_key", "session_id"
        )
    )
    conn.execute(
        f"""
        UPDATE {table} tgt
        SET
            last_updated_at = current_timestamp,
            last_updated_by_version_key = ?,
            etl_run_id = ?,
            hash_diff = im.hash_diff,
            {set_clause},
            is_deleted = FALSE,
            deleted_at = NULL
        FROM {inbound_table} im
        WHERE tgt.{natural_key} = im.{natural_key}
          AND tgt.hash_diff IS DISTINCT FROM im.hash_diff
        """,
        [run.version_key, run.etl_run_id],
    )

    all_cols = [natural_key, "session_id", "session_key", "date_key", "time_key", *payload_cols]
    insert_col_list = ", ".join(all_cols)
    select_col_list = ", ".join(f"im.{c}" for c in all_cols)
    conn.execute(
        f"""
        INSERT INTO {table} (
            created_by_version_key, last_updated_by_version_key,
            etl_run_id, record_source, hash_diff,
            {insert_col_list}
        )
        SELECT
            ?, ?, ?, ?, im.hash_diff,
            {select_col_list}
        FROM {inbound_table} im
        WHERE NOT EXISTS (
            SELECT 1 FROM {table} tgt WHERE tgt.{natural_key} = im.{natural_key}
        )
        """,
        [run.version_key, run.version_key, run.etl_run_id, record_source],
    )

    conn.execute(
        f"""
        UPDATE {table} tgt
        SET is_deleted = TRUE,
            deleted_at = current_timestamp,
            last_updated_at = current_timestamp,
            last_updated_by_version_key = ?,
            etl_run_id = ?
        WHERE tgt.is_deleted = FALSE
          AND tgt.session_id IN (SELECT DISTINCT session_id FROM {inbound_table})
          AND tgt.{natural_key} NOT IN (SELECT {natural_key} FROM {inbound_table})
        """,
        [run.version_key, run.etl_run_id],
    )

    conn.execute(f"DROP TABLE {inbound_table}")
