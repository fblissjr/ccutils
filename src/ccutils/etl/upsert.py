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
from ccutils.etl.utils import fetch_scalar


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
    timestamp_col: str = "timestamp",
    record_source: str = "claude_code_jsonl",
    soft_delete_scope_sql: str | None = None,
) -> None:
    """Generic UPDATE/INSERT/soft-delete pattern shared by every populator.

    Args:
        conn: DuckDB connection.
        run: ETL run handle. version_key and etl_run_id are parameterized.
        table: target fact table.
        inbound_table: name of the populated temp table. MUST already contain
            entry_id (or whatever natural_key is set to), session_id, and all
            payload_cols. If derive_session_keys is True, ALSO must contain
            the `timestamp_col` -- session_key/date_key/time_key/hash_diff
            are derived from it.
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
        timestamp_col: column on the inbound table from which to derive
            date_key and time_key. Defaults to "timestamp"; aggregate
            populators (fact_session_summary uses "first_timestamp")
            override.
        record_source: provenance label stamped on inserted rows.
        soft_delete_scope_sql: optional extra WHERE clause for the
            soft-delete step. Use when multiple populators write to the
            same table and the helper must not soft-delete rows belonging
            to other populators (e.g. fact_session_facets is written by
            both the Tier 1 and Tier 2 populators; each needs to scope
            its soft-delete by `facet_type_key IN (... WHERE tier=N)`).
            Default None preserves the original "this populator owns the
            session" semantics.
    """
    _validate_ident(table)
    _validate_ident(inbound_table)
    _validate_ident(natural_key)
    _validate_ident(timestamp_col)
    for c in payload_cols:
        _validate_ident(c)
    for c in hash_cols:
        _validate_ident(c)

    with run.step(f"upsert:{table}", kind="upsert") as st:
        st.rows_read = fetch_scalar(
            conn, f"SELECT COUNT(*) FROM {inbound_table}"
        )

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
                f"SET date_key = CAST(strftime({timestamp_col}, '%Y%m%d') AS INTEGER), "
                f"    time_key = CAST(strftime({timestamp_col}, '%H%M') AS INTEGER) "
                f"WHERE {timestamp_col} IS NOT NULL"
            )
            conn.execute(
                f"UPDATE {inbound_table} SET hash_diff = {hash_diff_sql(hash_cols)}"
            )

        # ASSERT the inbound batch already holds one row per natural key.
        #
        # This does NOT collapse duplicates. It used to, and that was the
        # wrong layer: a duplicate key means the populator's projection does
        # not produce the grain it declares, and only that populator knows
        # whether collapsing is safe or what the right survivor is. Resolving
        # it here applied one fact's judgment to all 13 and did so silently --
        # the row counts could not even distinguish a collapse from an
        # unchanged row.
        #
        # The INSERT below guards with NOT EXISTS against the TARGET only, so
        # two inbound rows sharing a key would both pass and both insert,
        # breaking the uniqueness every consumer assumes from `natural_key`.
        # Measured on a real 2,344-session corpus before the projections were
        # fixed: 6 of 13 facts violated their own declared key
        # (fact_tool_results 29 keys, fact_file_operations 8, fact_tool_uses 7,
        # fact_tool_chain_steps 7, fact_agent_delegations 3, fact_errors 1).
        #
        # Failing loud also removes a hazard the collapse carried: dropping a
        # row could drop its session_id from the soft-delete scope below
        # (`session_id IN (stg UNION inbound)`), leaving stale rows for that
        # session soft-deleted never -- silently.
        dupes = fetch_scalar(
            conn,
            f"""
            SELECT COUNT(*) FROM (
                SELECT {natural_key} FROM {inbound_table}
                WHERE {natural_key} IS NOT NULL
                GROUP BY 1 HAVING COUNT(*) > 1
            )
            """,
        )
        if dupes:
            raise ValueError(
                f"{inbound_table} has {dupes} duplicate value(s) of "
                f"natural_key '{natural_key}' -- {table} declares that key as "
                "unique, so its projection must emit one row per key. Fix the "
                "projection; do not collapse here (only the populator knows "
                "whether collapsing is safe and which row should survive)."
            )

        # When natural_key IS session_id, don't list it twice.
        extra_keys = ["session_key", "date_key", "time_key"]
        if natural_key != "session_id":
            extra_keys.append("session_id")
        set_clause = ",\n            ".join(
            f"{c} = im.{c}" for c in (*payload_cols, *extra_keys)
        )
        # The UPDATE must address exactly ONE physical row per natural key:
        # the live row, else the lowest-rowid soft-deleted one (that keeps
        # revival working -- the INSERT's NOT EXISTS matches deleted rows, so
        # this UPDATE is the only path back). Matching on the key alone
        # touches every physical twin sharing it, and the SET's
        # `is_deleted = FALSE` then RESURRECTS duplicates that
        # `_repair_duplicate_natural_keys` soft-deleted on open. Observed on
        # a real pre-fix warehouse: a new hash column changed every row's
        # hash, one batch run revived all 29 repaired tool_use_id twins, and
        # the delegation-completion pass died on them after every session
        # had already been processed.
        st.rows_updated = fetch_scalar(
            conn,
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
            JOIN (
                SELECT {natural_key} AS canon_key,
                       first(rowid ORDER BY is_deleted, rowid) AS canon_rowid
                FROM {table}
                GROUP BY 1
            ) canon ON canon.canon_key = im.{natural_key}
            WHERE tgt.{natural_key} = im.{natural_key}
              AND tgt.rowid = canon.canon_rowid
              AND tgt.hash_diff IS DISTINCT FROM im.hash_diff
            """,
            [run.version_key, run.etl_run_id],
        )

        all_cols = [natural_key]
        if natural_key != "session_id":
            all_cols.append("session_id")
        all_cols.extend(["session_key", "date_key", "time_key", *payload_cols])
        insert_col_list = ", ".join(all_cols)
        select_col_list = ", ".join(f"im.{c}" for c in all_cols)
        st.rows_inserted = fetch_scalar(
            conn,
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

        extra_scope = (
            f" AND ({soft_delete_scope_sql})" if soft_delete_scope_sql else ""
        )
        st.rows_soft_deleted = fetch_scalar(
            conn,
            f"""
            UPDATE {table} tgt
            SET is_deleted = TRUE,
                deleted_at = current_timestamp,
                last_updated_at = current_timestamp,
                last_updated_by_version_key = ?,
                etl_run_id = ?
            WHERE tgt.is_deleted = FALSE
              AND tgt.session_id IN (
                  SELECT DISTINCT session_id FROM stg_log_entries WHERE session_id IS NOT NULL
                  UNION
                  SELECT DISTINCT session_id FROM {inbound_table} WHERE session_id IS NOT NULL
              )
              AND tgt.{natural_key} NOT IN (
                  SELECT {natural_key} FROM {inbound_table} WHERE {natural_key} IS NOT NULL
              )
              {extra_scope}
            """,
            [run.version_key, run.etl_run_id],
        )

        conn.execute(f"DROP TABLE {inbound_table}")
