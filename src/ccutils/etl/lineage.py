"""Provenance + version + change-detection helpers shared by every Tier 3 writer.

Three responsibilities:

1. **ETL run lifecycle.** `EtlRun.start(conn, source_path)` inserts a `running`
   row into `fact_etl_runs`, allocates a UUID4 hex `etl_run_id`, and resolves
   (creating if needed) the `dim_etl_version` row. Returns a handle whose
   `.complete(...)` / `.fail(...)` methods close out the batch.

2. **Hash-diff change detection.** `hash_diff(**attrs)` returns a stable
   MD5 over the supplied attributes. Populators write it to every fact row
   and only UPDATE when the hash actually changed -- so `last_updated_at`
   is a precise temporal signal, not a "last ETL touch" timestamp.

3. **Record-source allow-list.** `record_source_label(name)` validates that
   the caller's `record_source` is one of the recognized values. Catches
   typos before they land in 100k rows.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any

from ccutils.schemas.star.utils import generate_dimension_key


PARSER_VERSION = "0.15.0-dev"
DEFAULT_BUSINESS_RULES_VERSION = "1"


# Provenance label allow-list. Add new values here when a new source goes live.
_RECORD_SOURCES: frozenset[str] = frozenset({
    "claude_code_jsonl",   # Tier 0 Claude Code project session JSONL
    "history_jsonl",       # Claude Code prompt-history JSONL
    "claude_ai_export",    # Claude.ai account export
    "derived_post_etl",    # DAG-invariant facts derived from other facts
})


def record_source_label(name: str) -> str:
    if name not in _RECORD_SOURCES:
        raise ValueError(
            f"Unknown record_source {name!r}. Add it to _RECORD_SOURCES in lineage.py "
            f"if it's a new legitimate source."
        )
    return name


def hash_diff(**attrs: Any) -> str:
    """Stable MD5 over mutable-content attributes. None values are skipped.

    Skipping None means adding a new optional column doesn't invalidate every
    existing row's hash and trigger spurious UPDATEs on re-ETL.

    Key order does not matter (sorted before hashing). Values are stringified
    via repr() to distinguish `1` from `"1"` from `True`.
    """
    parts = [f"{k}={v!r}" for k, v in sorted(attrs.items()) if v is not None]
    combined = "|".join(parts)
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


def _resolve_version_key(
    conn,
    ccutils_version: str,
    business_rules_version: str,
    description: str | None,
) -> str:
    """Return the dim_etl_version surrogate key, inserting a row if first seen."""
    version_key = generate_dimension_key(ccutils_version, business_rules_version)
    exists = conn.execute(
        "SELECT 1 FROM dim_etl_version WHERE version_key = ?", [version_key]
    ).fetchone()
    if exists is None:
        conn.execute(
            """
            INSERT INTO dim_etl_version
                (version_key, ccutils_version, business_rules_version, description)
            VALUES (?, ?, ?, ?)
            """,
            [version_key, ccutils_version, business_rules_version, description],
        )
    return version_key


@dataclass
class EtlRun:
    """Handle for the lifecycle of one ETL batch.

    Use as:
        run = EtlRun.start(conn, source_path="/path/to/archive")
        try:
            ... do work, stamping facts with run.etl_run_id / run.version_key ...
            run.complete(sessions_inserted=N, facts_inserted=M)
        except Exception as e:
            run.fail(str(e))
            raise
    """

    conn: Any
    etl_run_id: str
    version_key: str

    @classmethod
    def start(
        cls,
        conn,
        *,
        source_path: str,
        ccutils_version: str = PARSER_VERSION,
        business_rules_version: str = DEFAULT_BUSINESS_RULES_VERSION,
        description: str | None = None,
    ) -> "EtlRun":
        version_key = _resolve_version_key(
            conn, ccutils_version, business_rules_version, description
        )
        etl_run_id = uuid.uuid4().hex
        conn.execute(
            """
            INSERT INTO fact_etl_runs (etl_run_id, version_key, source_path, status)
            VALUES (?, ?, ?, 'running')
            """,
            [etl_run_id, version_key, source_path],
        )
        return cls(conn=conn, etl_run_id=etl_run_id, version_key=version_key)

    def complete(
        self,
        *,
        sessions_seen: int = 0,
        sessions_inserted: int = 0,
        sessions_updated: int = 0,
        sessions_unchanged: int = 0,
        sessions_soft_deleted: int = 0,
        facts_inserted: int = 0,
        facts_updated: int = 0,
    ) -> None:
        self.conn.execute(
            """
            UPDATE fact_etl_runs
            SET status = 'success',
                completed_at = current_timestamp,
                sessions_seen = ?,
                sessions_inserted = ?,
                sessions_updated = ?,
                sessions_unchanged = ?,
                sessions_soft_deleted = ?,
                facts_inserted = ?,
                facts_updated = ?
            WHERE etl_run_id = ?
            """,
            [
                sessions_seen, sessions_inserted, sessions_updated,
                sessions_unchanged, sessions_soft_deleted,
                facts_inserted, facts_updated,
                self.etl_run_id,
            ],
        )

    def fail(self, error_message: str) -> None:
        self.conn.execute(
            """
            UPDATE fact_etl_runs
            SET status = 'failed',
                completed_at = current_timestamp,
                error_message = ?
            WHERE etl_run_id = ?
            """,
            [error_message, self.etl_run_id],
        )
