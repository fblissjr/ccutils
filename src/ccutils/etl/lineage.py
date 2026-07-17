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
from contextlib import contextmanager
from dataclasses import dataclass, field
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
class StepCounts:
    """Mutable row-count slots a step body fills in before the context
    manager closes the step out. None means "not applicable/unknown"."""

    rows_read: int | None = None
    rows_inserted: int | None = None
    rows_updated: int | None = None
    rows_soft_deleted: int | None = None


@dataclass
class EtlRun:
    """Handle for the lifecycle of one per-session ETL run.

    Use as:
        run = EtlRun.start(conn, source_path="/path/to/session.jsonl")
        try:
            with run.step("load_staging") as st:
                ... do work, set st.rows_read etc ...
            run.complete(sessions_inserted=1)
        except Exception as e:
            run.fail(str(e))
            raise

    ``complete`` derives facts_inserted / facts_updated from this run's
    fact_etl_steps rows -- populators report real affected-row counts via
    ``step``, so the run row can't drift from what actually happened.
    """

    conn: Any
    etl_run_id: str
    version_key: str
    batch_run_id: str | None = None
    _step_seq: int = field(default=0, repr=False)

    @classmethod
    def start(
        cls,
        conn,
        *,
        source_path: str,
        batch_run_id: str | None = None,
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
            INSERT INTO fact_etl_runs
                (etl_run_id, version_key, source_path, status, batch_run_id)
            VALUES (?, ?, ?, 'running', ?)
            """,
            [etl_run_id, version_key, source_path, batch_run_id],
        )
        return cls(
            conn=conn, etl_run_id=etl_run_id, version_key=version_key,
            batch_run_id=batch_run_id,
        )

    @contextmanager
    def step(self, step_name: str):
        """Record one DAG node in fact_etl_steps around the wrapped body.

        Yields a StepCounts whose slots the body may fill (lineage_upsert
        fills all four; stage wrappers set what they cheaply know). On an
        exception the step row is marked failed with the error, and the
        exception propagates.
        """
        self._step_seq += 1
        step_id = uuid.uuid4().hex
        self.conn.execute(
            """
            INSERT INTO fact_etl_steps
                (step_id, etl_run_id, batch_run_id, step_name, step_order, status)
            VALUES (?, ?, ?, ?, ?, 'running')
            """,
            [step_id, self.etl_run_id, self.batch_run_id, step_name,
             self._step_seq],
        )
        counts = StepCounts()
        try:
            yield counts
        except Exception as e:
            self.conn.execute(
                """
                UPDATE fact_etl_steps
                SET status = 'failed',
                    completed_at = current_timestamp,
                    error_message = ?
                WHERE step_id = ?
                """,
                [str(e), step_id],
            )
            raise
        self.conn.execute(
            """
            UPDATE fact_etl_steps
            SET status = 'success',
                completed_at = current_timestamp,
                rows_read = ?,
                rows_inserted = ?,
                rows_updated = ?,
                rows_soft_deleted = ?
            WHERE step_id = ?
            """,
            [counts.rows_read, counts.rows_inserted, counts.rows_updated,
             counts.rows_soft_deleted, step_id],
        )

    def complete(
        self,
        *,
        sessions_seen: int = 0,
        sessions_inserted: int = 0,
        sessions_updated: int = 0,
        sessions_unchanged: int = 0,
        sessions_soft_deleted: int = 0,
        data_start_ts=None,
        data_end_ts=None,
    ) -> None:
        """Close the run out as success.

        facts_inserted / facts_updated are derived from this run's
        `upsert:%` steps only -- stage steps (load_staging etc.) record
        real row counts at step grain but are NOT facts, so they are
        excluded from the run-level fact totals.

        data_start_ts / data_end_ts must be supplied by the caller (the
        orchestrator reads them from staging before staging_scope clears
        it); omitting them stores a NULL CDC window.
        """
        facts_inserted, facts_updated = self.conn.execute(
            """
            SELECT COALESCE(SUM(rows_inserted), 0),
                   COALESCE(SUM(rows_updated), 0)
            FROM fact_etl_steps
            WHERE etl_run_id = ? AND step_name LIKE 'upsert:%'
            """,
            [self.etl_run_id],
        ).fetchone()
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
                data_start_ts = ?,
                data_end_ts = ?,
                facts_inserted = ?,
                facts_updated = ?
            WHERE etl_run_id = ?
            """,
            [
                sessions_seen, sessions_inserted, sessions_updated,
                sessions_unchanged, sessions_soft_deleted,
                data_start_ts, data_end_ts,
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


@dataclass
class BatchRun:
    """Handle for one CLI orchestration over many sessions.

    Start before the per-session loop, pass ``batch_run_id`` into every
    ``run_v15_etl`` call, then ``complete()`` -- which derives every count
    (sessions seen/succeeded/failed, row totals, CDC data window) from the
    child fact_etl_runs / fact_etl_steps rows rather than trusting the
    caller to tally. Status lands 'success' when no child failed, else
    'partial'; ``fail()`` is for the orchestration itself dying.
    """

    conn: Any
    batch_run_id: str
    version_key: str

    @classmethod
    def start(
        cls,
        conn,
        *,
        source_root: str,
        output_format: str | None = None,
        ccutils_version: str = PARSER_VERSION,
        business_rules_version: str = DEFAULT_BUSINESS_RULES_VERSION,
        description: str | None = None,
    ) -> "BatchRun":
        version_key = _resolve_version_key(
            conn, ccutils_version, business_rules_version, description
        )
        batch_run_id = uuid.uuid4().hex
        conn.execute(
            """
            INSERT INTO fact_etl_batch_runs
                (batch_run_id, version_key, source_root, output_format, status)
            VALUES (?, ?, ?, ?, 'running')
            """,
            [batch_run_id, version_key, source_root, output_format],
        )
        return cls(conn=conn, batch_run_id=batch_run_id, version_key=version_key)

    def complete(self, *, expected_sessions: int | None = None) -> None:
        """Roll children up onto the batch row and close it out.

        expected_sessions is the number of sessions the caller ATTEMPTED.
        A session that died before EtlRun.start wrote its child row, or a
        child left 'running' by a hard crash, is invisible to the child
        rollup -- sessions_seen = max(child count, expected_sessions)
        counts every non-success as failed, so the batch cannot report a
        clean 'success' while the CLI printed failures.
        """
        child_count, succeeded, data_start_ts, data_end_ts = self.conn.execute(
            """
            SELECT
                COUNT(*),
                COUNT(*) FILTER (WHERE status = 'success'),
                MIN(data_start_ts),
                MAX(data_end_ts)
            FROM fact_etl_runs
            WHERE batch_run_id = ?
            """,
            [self.batch_run_id],
        ).fetchone()
        rows_read, rows_inserted, rows_updated, rows_soft_deleted = (
            self.conn.execute(
                """
                SELECT
                    COALESCE(SUM(rows_read), 0),
                    COALESCE(SUM(rows_inserted), 0),
                    COALESCE(SUM(rows_updated), 0),
                    COALESCE(SUM(rows_soft_deleted), 0)
                FROM fact_etl_steps
                WHERE batch_run_id = ? AND step_name LIKE 'upsert:%'
                """,
                [self.batch_run_id],
            ).fetchone()
        )
        seen = max(child_count, expected_sessions or 0)
        failed = seen - succeeded
        status = "success" if failed == 0 else "partial"
        self.conn.execute(
            """
            UPDATE fact_etl_batch_runs
            SET completed_at = current_timestamp,
                sessions_seen = ?,
                sessions_succeeded = ?,
                sessions_failed = ?,
                status = ?,
                data_start_ts = ?,
                data_end_ts = ?,
                rows_read = ?,
                rows_inserted = ?,
                rows_updated = ?,
                rows_soft_deleted = ?
            WHERE batch_run_id = ?
            """,
            [
                seen, succeeded, failed, status,
                data_start_ts, data_end_ts,
                rows_read, rows_inserted, rows_updated, rows_soft_deleted,
                self.batch_run_id,
            ],
        )

    def fail(self, error_message: str) -> None:
        self.conn.execute(
            """
            UPDATE fact_etl_batch_runs
            SET status = 'failed',
                completed_at = current_timestamp,
                error_message = ?
            WHERE batch_run_id = ?
            """,
            [error_message, self.batch_run_id],
        )
