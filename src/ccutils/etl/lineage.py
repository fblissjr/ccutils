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


from ccutils._version import PARSER_VERSION
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


def _mark_failed(conn, *, table: str, id_col: str, id_val: str, error: str) -> None:
    """Shared failure-marking for the three audit tables (steps/runs/batches):
    status='failed' + completed_at + error_message, matched on the id column."""
    conn.execute(
        f"""
        UPDATE {table}
        SET status = 'failed',
            completed_at = current_timestamp,
            error_message = ?
        WHERE {id_col} = ?
        """,
        [error, id_val],
    )


def _sum_upsert_steps(conn, *, scope_col: str, scope_id: str):
    """Rollup of fact-populating steps (step_kind='upsert') for one run or
    batch: (rows_read, rows_inserted, rows_updated, rows_soft_deleted).
    Stage steps (load_staging etc.) report real counts at step grain but
    are not facts, so they are excluded here -- the single definition all
    three consumers (EtlRun.complete, BatchRun.complete, and mirrored in
    the semantic_etl_runs view) agree on."""
    return conn.execute(
        f"""
        SELECT
            COALESCE(SUM(rows_read), 0),
            COALESCE(SUM(rows_inserted), 0),
            COALESCE(SUM(rows_updated), 0),
            COALESCE(SUM(rows_soft_deleted), 0)
        FROM fact_etl_steps
        WHERE {scope_col} = ? AND step_kind = 'upsert'
        """,
        [scope_id],
    ).fetchone()


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
        run_kind: str = "session",
    ) -> "EtlRun":
        version_key = _resolve_version_key(
            conn, ccutils_version, business_rules_version, description
        )
        etl_run_id = uuid.uuid4().hex
        # run_kind is stamped at INSERT, not at complete(), so a run that
        # crashes before completing is still classified. BatchRun.complete
        # relies on that to keep counting crashed sessions as failed ones.
        conn.execute(
            """
            INSERT INTO fact_etl_runs
                (etl_run_id, version_key, source_path, status, batch_run_id,
                 run_kind)
            VALUES (?, ?, ?, 'running', ?, ?)
            """,
            [etl_run_id, version_key, source_path, batch_run_id, run_kind],
        )
        return cls(
            conn=conn, etl_run_id=etl_run_id, version_key=version_key,
            batch_run_id=batch_run_id,
        )

    @contextmanager
    def step(self, step_name: str, *, kind: str = "stage"):
        """Record one DAG node in fact_etl_steps around the wrapped body.

        Yields a StepCounts whose slots the body may fill (lineage_upsert
        fills all four; stage wrappers set what they cheaply know). On an
        exception the step row is marked failed with the error, and the
        exception propagates.

        kind is the scoping key for fact rollups: 'upsert' steps (fact
        populators via lineage_upsert) count toward facts_inserted /
        rows_* totals; 'stage' steps (default) are recorded but excluded.
        """
        if kind not in ("stage", "upsert"):
            raise ValueError(
                f"step kind must be 'stage' or 'upsert', got {kind!r} -- "
                "a typo here silently zeroes every fact rollup"
            )
        self._step_seq += 1
        step_id = uuid.uuid4().hex
        self.conn.execute(
            """
            INSERT INTO fact_etl_steps
                (step_id, etl_run_id, batch_run_id, step_name, step_kind,
                 step_order, status)
            VALUES (?, ?, ?, ?, ?, ?, 'running')
            """,
            [step_id, self.etl_run_id, self.batch_run_id, step_name, kind,
             self._step_seq],
        )
        counts = StepCounts()
        try:
            yield counts
        except BaseException as e:
            # BaseException so a KeyboardInterrupt mid-step doesn't leave
            # the step row stuck 'running' (mirrors BatchRun.__exit__).
            _mark_failed(
                self.conn, table="fact_etl_steps", id_col="step_id",
                id_val=step_id, error=str(e) or type(e).__name__,
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
        step_kind='upsert' steps only -- stage steps (load_staging etc.)
        record real row counts at step grain but are NOT facts, so they
        are excluded from the run-level fact totals.

        data_start_ts / data_end_ts must be supplied by the caller (the
        orchestrator gets them from the staging load); omitting them
        stores a NULL CDC window.
        """
        _, facts_inserted, facts_updated, _ = _sum_upsert_steps(
            self.conn, scope_col="etl_run_id", scope_id=self.etl_run_id
        )
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
        _mark_failed(
            self.conn, table="fact_etl_runs", id_col="etl_run_id",
            id_val=self.etl_run_id, error=error_message,
        )


@dataclass
class BatchRun:
    """Handle for one CLI orchestration over many sessions.

    Use as a context manager:

        with BatchRun.start(conn, source_root=..., output_format=...) as batch:
            ... per-session loop passing batch.batch_run_id ...
            batch.complete(expected_sessions=N)

    ``__exit__`` marks the batch row failed on ANY escaping exception
    (including KeyboardInterrupt) so it can never stick at 'running';
    ``complete()`` derives every count (sessions seen/succeeded/failed,
    row totals, CDC data window) from the child fact_etl_runs /
    fact_etl_steps rows rather than trusting the caller to tally. Status
    lands 'success' when no child failed, else 'partial'.
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
              -- Sessions only. Cross-session passes (the delegation
              -- reconciliation) are child runs of this batch but are not
              -- sessions; counting them reported one more session than the
              -- CLI actually processed.
              AND run_kind = 'session'
            """,
            [self.batch_run_id],
        ).fetchone()
        rows_read, rows_inserted, rows_updated, rows_soft_deleted = (
            _sum_upsert_steps(
                self.conn, scope_col="batch_run_id", scope_id=self.batch_run_id
            )
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
        _mark_failed(
            self.conn, table="fact_etl_batch_runs", id_col="batch_run_id",
            id_val=self.batch_run_id, error=error_message,
        )

    def __enter__(self) -> "BatchRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Any escaping exception -- including complete() itself failing or
        # a KeyboardInterrupt mid-loop -- marks the batch failed instead of
        # leaving the row stuck at 'running'. Guarded on the row still
        # being 'running' so an exception AFTER a successful complete()
        # cannot clobber a truthful success/partial status. Never
        # suppresses.
        if exc is not None:
            status = self.conn.execute(
                "SELECT status FROM fact_etl_batch_runs WHERE batch_run_id = ?",
                [self.batch_run_id],
            ).fetchone()
            if status is not None and status[0] == "running":
                self.fail(str(exc) or exc_type.__name__)
        return False
