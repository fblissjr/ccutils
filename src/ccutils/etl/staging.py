"""Load Tier 1 Parquet into the Tier 2 staging table.

`stg_log_entries` is one row per JSONL line, with the envelope as typed
columns and polymorphic payloads as JSON strings. Fact-table populators
(Phase C2 onward) read from staging and project into their grain.

Trunc-and-reload semantics: reloading a session by source_path replaces
its existing staging rows. No append-with-duplicates.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pyarrow as pa

from ccutils.parsers.parquet_writer import _LOG_ENTRY_SCHEMA


# Public re-export: callers can import the Pyarrow schema from this module
# rather than from the writer (the staging module is the contract surface).
STG_LOG_ENTRIES_SCHEMA: pa.Schema = _LOG_ENTRY_SCHEMA


class StagingLoad(NamedTuple):
    """What one staging load did: row count + the session's CDC window
    (min/max parseable entry timestamp), all from a single scan."""

    rows: int
    data_start_ts: object  # datetime | None
    data_end_ts: object    # datetime | None


def load_session_to_staging(
    conn,
    log_entries_parquet: str | Path,
) -> StagingLoad:
    """Load one session's log_entries.parquet into stg_log_entries.

    Returns a StagingLoad (row count + CDC data window). Idempotent by
    source_path: existing rows for the same source are DELETEd before
    INSERT, so re-running the same session against staging never doubles
    up.
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

    # One pass fixes session_id for two cases, in precedence order:
    # 1. Subagent identity override (REAL Claude Code contract): agent
    #    transcript entries carry the PARENT's sessionId on every line.
    #    The transcript's true identity is the file itself, so agent files
    #    get session_id from the filename stem ('agent-<id>'). Without
    #    this, all of a parent's agents collapse into the parent's
    #    dim_session row, subagent enrichment mislabels the parent
    #    is_agent with a SELF-referencing parent_session_key, and
    #    depth_level flattens to 0. (The parquet writer now stamps agent
    #    rows correctly at Tier 1; this override also repairs lakes
    #    written before that.)
    # 2. NULL backfill for entry types that don't carry sessionId at the
    #    JSONL top level (file-history-snapshot, queue-operation, summary,
    #    ai-title, ...): the filename stem.
    conn.execute(
        """
        UPDATE stg_log_entries
        SET session_id = CASE
            WHEN regexp_matches(source_path, '/subagents/agent-[^/]+\\.jsonl$')
                THEN regexp_extract(source_path, '/subagents/(agent-[^/]+)\\.jsonl$', 1)
            ELSE regexp_extract(source_path, '([^/]+)\\.jsonl$', 1)
        END
        WHERE session_id IS NULL
           OR regexp_matches(source_path, '/subagents/agent-[^/]+\\.jsonl$')
        """
    )

    if not source_paths:
        return StagingLoad(0, None, None)
    row = conn.execute(
        "SELECT COUNT(*), MIN(TRY_CAST(timestamp AS TIMESTAMP)), "
        "       MAX(TRY_CAST(timestamp AS TIMESTAMP)) "
        "FROM stg_log_entries WHERE source_path = ANY (?)",
        [source_paths],
    ).fetchone()
    return StagingLoad(*row)


def load_archive_to_staging(
    conn,
    lake_root: str | Path,
) -> int:
    """Load every per-session log_entries.parquet under a Parquet-lake root.

    Returns the count of sessions loaded.
    """
    lake_root = Path(lake_root)
    paths = sorted(lake_root.rglob("log_entries.parquet"))
    for p in paths:
        load_session_to_staging(conn, p)
    return len(paths)
