"""Load Tier 1 Parquet into the Tier 2 staging table.

`stg_log_entries` is one row per JSONL line, with the envelope as typed
columns and polymorphic payloads as JSON strings. Fact-table populators
(Phase C2 onward) read from staging and project into their grain.

Trunc-and-reload semantics: reloading a session by source_path replaces
its existing staging rows. No append-with-duplicates.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from ccutils.etl.lineage import EtlRun
from ccutils.parsers.parquet_writer import _LOG_ENTRY_SCHEMA


# Public re-export: callers can import the Pyarrow schema from this module
# rather than from the writer (the staging module is the contract surface).
STG_LOG_ENTRIES_SCHEMA: pa.Schema = _LOG_ENTRY_SCHEMA


def load_session_to_staging(
    conn,
    log_entries_parquet: str | Path,
    *,
    run: EtlRun | None = None,
) -> int:
    """Load one session's log_entries.parquet into stg_log_entries.

    Returns the row count loaded. Idempotent by source_path: existing rows
    for the same source are DELETEd before INSERT, so re-running the same
    session against staging never doubles up.

    `run` is unused today -- the etl_run_id stamped on rows comes from the
    Parquet file (written by parquet_writer). Accepted for symmetry with
    other loaders that may need to associate row-level work to a run.
    """
    log_entries_parquet = Path(log_entries_parquet)
    parquet_path_str = str(log_entries_parquet)

    # Read just enough to find the source_path values present in this file.
    # In practice every row of a single per-session file shares one source.
    source_paths = [
        r[0]
        for r in conn.execute(
            f"SELECT DISTINCT source_path FROM read_parquet('{parquet_path_str}')"
        ).fetchall()
    ]
    if source_paths:
        placeholders = ",".join("?" for _ in source_paths)
        conn.execute(
            f"DELETE FROM stg_log_entries WHERE source_path IN ({placeholders})",
            source_paths,
        )

    conn.execute(
        f"INSERT INTO stg_log_entries SELECT * FROM read_parquet('{parquet_path_str}')"
    )

    return conn.execute(
        "SELECT COUNT(*) FROM stg_log_entries WHERE source_path = ANY (?)",
        [source_paths],
    ).fetchone()[0] if source_paths else 0


def load_archive_to_staging(
    conn,
    lake_root: str | Path,
    *,
    run: EtlRun | None = None,
) -> int:
    """Load every per-session log_entries.parquet under a Parquet-lake root.

    Returns the count of sessions loaded.
    """
    lake_root = Path(lake_root)
    paths = sorted(lake_root.rglob("log_entries.parquet"))
    for p in paths:
        load_session_to_staging(conn, p, run=run)
    return len(paths)
