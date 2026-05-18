"""End-to-end v0.15 ETL orchestrator (Phase C6).

Single entry point: parses one JSONL session, writes per-session Parquet,
loads staging, then runs every v0.15 fact populator in dependency order.

Pipeline:
    JSONL  ->  Pydantic parse + write Parquet (Tier 1)
            ->  load_session_to_staging (Tier 2)
            ->  upsert dim_session / dim_project / dim_model / dim_tool
                (minimal stub rows so fact FKs join)
            ->  populate_fact_messages
            ->  populate_fact_tool_uses
            ->  populate_fact_tool_results
            ->  populate_fact_token_usage
            ->  populate_fact_attachments
            ->  populate_fact_progress_events
            ->  populate_fact_system_events
            ->  populate_fact_meta_events
            ->  populate_fact_file_history_snapshots
            ->  populate_fact_queue_operations
            ->  populate_fact_pr_links
            ->  populate_fact_session_summary  (must run last; aggregates over all the above)

The EtlRun lifecycle ensures any exception marks the run failed in
fact_etl_runs instead of leaving 'running' rows around.

Dimensions are populated as STUB rows (just the surrogate key + natural
key + minimal envelope fields). Full dimension enrichment (intent,
complexity, heuristics, etc.) is left to a separate dim-population
pass not yet rewritten for v0.15.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ccutils.etl.entry_type_facts import (
    populate_fact_attachments,
    populate_fact_file_history_snapshots,
    populate_fact_meta_events,
    populate_fact_pr_links,
    populate_fact_progress_events,
    populate_fact_queue_operations,
    populate_fact_system_events,
)
from ccutils.etl.bridge_session_file import populate_bridge_session_file
from ccutils.etl.fact_diagnostics import populate_fact_diagnostics
from ccutils.etl.fact_file_operations import (
    populate_dim_file,
    populate_fact_file_operations,
)
from ccutils.etl.fact_plan_revisions import populate_fact_plan_revisions
from ccutils.etl.fact_messages import populate_fact_messages
from ccutils.etl.fact_session_summary import populate_fact_session_summary
from ccutils.etl.fact_token_usage import populate_fact_token_usage
from ccutils.etl.fact_tool_calls import (
    populate_fact_tool_results,
    populate_fact_tool_uses,
)
from ccutils.etl.lineage import EtlRun
from ccutils.etl.staging import load_session_to_staging
from ccutils.parsers.parquet_writer import write_session_to_parquet


def _upsert_minimal_dimensions(conn) -> None:
    """Insert stub rows into dim_session / dim_project / dim_model / dim_tool
    for any surrogate key referenced by the staging table but not yet
    present in the dim. Skill anti-pattern note: dimensions are intended
    to be wider than this (with heuristic enrichment, slug, depth_level,
    etc.) -- but for query consumers that only need FK existence, the
    surrogate + natural key + minimal envelope is enough. Full enrichment
    is a follow-up.
    """
    # dim_session: surrogate from staging.session_id.
    #
    # We derive project_key from source_path's parent dir here so the FK
    # is set on insert -- semantic_project_context / semantic_sessions
    # join dim_session -> dim_project, and they go empty without it.
    # first_timestamp / last_timestamp come straight from the staging
    # min/max so date-range filtering on dim_session works immediately
    # (without waiting on Phase D enrichment).
    conn.execute(
        """
        INSERT INTO dim_session (
            session_key, session_id, project_key,
            cwd, git_branch, version, slug, entrypoint,
            first_timestamp, last_timestamp
        )
        SELECT
            md5(sle.session_id) AS session_key,
            sle.session_id,
            md5(regexp_replace(ANY_VALUE(sle.source_path), '/[^/]+$', '')) AS project_key,
            ANY_VALUE(sle.cwd) AS cwd,
            ANY_VALUE(sle.git_branch) AS git_branch,
            ANY_VALUE(sle.version) AS version,
            ANY_VALUE(sle.slug) AS slug,
            ANY_VALUE(sle.entrypoint) AS entrypoint,
            MIN(TRY_CAST(sle.timestamp AS TIMESTAMP)) AS first_timestamp,
            MAX(TRY_CAST(sle.timestamp AS TIMESTAMP)) AS last_timestamp
        FROM stg_log_entries sle
        WHERE sle.session_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM dim_session ds
              WHERE ds.session_key = md5(sle.session_id)
          )
        GROUP BY sle.session_id
        """
    )

    # Idempotent backfill: if a session already exists from a prior ETL run
    # but lacks project_key / timestamps (legacy minimal-dim path), set them.
    conn.execute(
        """
        UPDATE dim_session ds
        SET project_key = COALESCE(ds.project_key, sub.project_key),
            first_timestamp = COALESCE(ds.first_timestamp, sub.first_timestamp),
            last_timestamp = COALESCE(ds.last_timestamp, sub.last_timestamp)
        FROM (
            SELECT
                md5(sle.session_id) AS session_key,
                md5(regexp_replace(ANY_VALUE(sle.source_path), '/[^/]+$', '')) AS project_key,
                MIN(TRY_CAST(sle.timestamp AS TIMESTAMP)) AS first_timestamp,
                MAX(TRY_CAST(sle.timestamp AS TIMESTAMP)) AS last_timestamp
            FROM stg_log_entries sle
            WHERE sle.session_id IS NOT NULL
            GROUP BY sle.session_id
        ) sub
        WHERE ds.session_key = sub.session_key
          AND (ds.project_key IS NULL
               OR ds.first_timestamp IS NULL
               OR ds.last_timestamp IS NULL)
        """
    )

    # dim_project: surrogate from staging.source_path's parent dir
    conn.execute(
        """
        INSERT INTO dim_project (project_key, project_path, project_name)
        SELECT
            md5(regexp_replace(sle.source_path, '/[^/]+$', '')) AS project_key,
            regexp_replace(sle.source_path, '/[^/]+$', '') AS project_path,
            -- project_name is the last path segment of project_path
            regexp_extract(regexp_replace(sle.source_path, '/[^/]+$', ''),
                           '([^/]+)$', 1) AS project_name
        FROM (SELECT DISTINCT source_path FROM stg_log_entries) sle
        WHERE NOT EXISTS (
            SELECT 1 FROM dim_project dp
            WHERE dp.project_key = md5(regexp_replace(sle.source_path, '/[^/]+$', ''))
        )
        """
    )

    # dim_model: from assistant message.model values
    conn.execute(
        """
        INSERT INTO dim_model (model_key, model_name, model_family)
        SELECT DISTINCT
            md5(json_extract_string(sle.message_json, '$.model')) AS model_key,
            json_extract_string(sle.message_json, '$.model') AS model_name,
            CASE
                WHEN json_extract_string(sle.message_json, '$.model') LIKE '%opus%' THEN 'opus'
                WHEN json_extract_string(sle.message_json, '$.model') LIKE '%sonnet%' THEN 'sonnet'
                WHEN json_extract_string(sle.message_json, '$.model') LIKE '%haiku%' THEN 'haiku'
                ELSE 'unknown'
            END AS model_family
        FROM stg_log_entries sle
        WHERE sle.type = 'assistant'
          AND json_extract_string(sle.message_json, '$.model') IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM dim_model dm
              WHERE dm.model_key = md5(json_extract_string(sle.message_json, '$.model'))
          )
        """
    )

    # dim_tool: from every distinct tool_use.name across all assistant content
    conn.execute(
        """
        INSERT INTO dim_tool (tool_key, tool_name, tool_category)
        SELECT DISTINCT
            md5(tool_name) AS tool_key,
            tool_name,
            'unknown' AS tool_category  -- categorization left to a heuristic pass
        FROM (
            SELECT DISTINCT
                json_extract_string(b.block, '$.name') AS tool_name
            FROM stg_log_entries sle,
            LATERAL (
                SELECT unnest(json_extract(sle.message_json, '$.content')::JSON[]) AS block
            ) b
            WHERE sle.type = 'assistant'
              AND json_type(sle.message_json, '$.content') = 'ARRAY'
              AND json_extract_string(b.block, '$.type') = 'tool_use'
        )
        WHERE tool_name IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM dim_tool dt WHERE dt.tool_key = md5(tool_name)
          )
        """
    )


def run_v15_etl(
    conn,
    session_path: str | Path,
    *,
    project_name: str = "unknown",
    parquet_lake_root: str | Path | None = None,
) -> dict[str, Any]:
    """End-to-end ETL for one session JSONL.

    Args:
        conn: DuckDB connection.
        session_path: JSONL file path.
        project_name: ignored at fact level (kept for caller compat); the
            actual project_name in dim_project is derived from source_path.
        parquet_lake_root: where to write the per-session Parquet files.
            Defaults to a sibling 'parquet_lake' dir next to session_path.

    Returns:
        dict with 'etl_run_id' and 'sessions_inserted' for caller use.
    """
    session_path = Path(session_path)
    if parquet_lake_root is None:
        parquet_lake_root = session_path.parent / "parquet_lake"
    parquet_lake_root = Path(parquet_lake_root)

    run = EtlRun.start(conn, source_path=str(session_path))
    try:
        # Tier 1: parse + write Parquet
        log_path, _ = write_session_to_parquet(
            session_path,
            parquet_lake_root,
            etl_run_id=run.etl_run_id,
            project_slug=project_name,
        )

        # Tier 2: load staging
        load_session_to_staging(conn, log_path)

        # Stub dimensions so fact FKs resolve
        _upsert_minimal_dimensions(conn)

        # Populate every v0.15 fact in order. fact_session_summary MUST be
        # last -- it aggregates over the others.
        populate_fact_messages(conn, run=run)
        populate_fact_tool_uses(conn, run=run)
        populate_fact_tool_results(conn, run=run)
        populate_fact_token_usage(conn, run=run)
        populate_fact_attachments(conn, run=run)
        populate_fact_progress_events(conn, run=run)
        populate_fact_system_events(conn, run=run)
        populate_fact_meta_events(conn, run=run)
        populate_fact_file_history_snapshots(conn, run=run)
        populate_fact_queue_operations(conn, run=run)
        populate_fact_pr_links(conn, run=run)
        # dim_file + fact_file_operations depend on fact_tool_uses + fact_tool_results
        populate_dim_file(conn, run=run)
        populate_fact_file_operations(conn, run=run)
        # bridge_session_file aggregates fact_file_operations
        populate_bridge_session_file(conn, run=run)
        # fact_diagnostics flattens fact_attachments where type='diagnostics'
        populate_fact_diagnostics(conn, run=run)
        # fact_plan_revisions classifies ExitPlanMode outcomes from
        # fact_tool_results.is_error (R16 tri-state)
        populate_fact_plan_revisions(conn, run=run)
        populate_fact_session_summary(conn, run=run)

        run.complete(sessions_inserted=1)
    except Exception as e:
        run.fail(str(e))
        raise

    return {"etl_run_id": run.etl_run_id, "sessions_inserted": 1}
