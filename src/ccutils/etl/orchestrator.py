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

from contextlib import contextmanager
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
from ccutils.etl.facets import FacetExtractor
from ccutils.etl.facets.populator import populate_tier2_facets
from ccutils.etl.bridge_session_file import populate_bridge_session_file
from ccutils.etl.dim_session_chain import populate_dim_session_chain
from ccutils.etl.dim_session_heuristics import populate_dim_session_heuristics
from ccutils.etl.fact_agent_delegations import (
    populate_delegation_completion,
    populate_fact_agent_delegations,
)
from ccutils.etl.fact_diagnostics import populate_fact_diagnostics
from ccutils.etl.fact_errors import populate_fact_errors
from ccutils.etl.fact_file_operations import (
    populate_dim_file,
    populate_fact_file_operations,
)
from ccutils.etl.fact_plan_revisions import populate_fact_plan_revisions
from ccutils.etl.fact_tool_chain_steps import populate_fact_tool_chain_steps
from ccutils.etl.subagent_enrichment import populate_subagent_dim_session
from ccutils.etl.fact_messages import populate_fact_messages
from ccutils.etl.fact_session_facets import populate_tier1_facets
from ccutils.etl.fact_session_summary import populate_fact_session_summary
from ccutils.etl.fact_token_usage import populate_fact_token_usage
from ccutils.etl.fact_tool_calls import (
    populate_fact_tool_results,
    populate_fact_tool_uses,
)
from ccutils.etl.lineage import EtlRun
from ccutils.etl.staging import load_session_to_staging
from ccutils.etl.utils import (
    MODEL_BASE_SQL,
    MODEL_FAMILY_SQL,
    TOOL_CATEGORY_SQL,
    insert_missing_dim_dates,
    project_dir_sql,
    project_key_from_dir_sql,
    project_key_sql,
)
from ccutils.parsers.parquet_writer import write_session_to_parquet


def _upsert_minimal_dimensions(conn) -> int:
    """Insert stub rows into dim_session / dim_project / dim_model / dim_tool
    for any surrogate key referenced by the staging table but not yet
    present in the dim. Skill anti-pattern note: dimensions are intended
    to be wider than this (with heuristic enrichment, slug, depth_level,
    etc.) -- but for query consumers that only need FK existence, the
    surrogate + natural key + minimal envelope is enough. Full enrichment
    is a follow-up.

    Returns the number of NEW dim_session rows inserted (0 when every
    staged session already existed) so the caller can report
    sessions_inserted vs sessions_updated honestly.
    """
    # dim_session: surrogate from staging.session_id.
    #
    # We derive project_key from source_path's parent dir here so the FK
    # is set on insert -- semantic_project_context / semantic_sessions
    # join dim_session -> dim_project, and they go empty without it.
    # first_timestamp / last_timestamp come straight from the staging
    # min/max so date-range filtering on dim_session works immediately
    # (without waiting on Phase D enrichment).
    sessions_inserted = conn.execute(
        f"""
        INSERT INTO dim_session (
            session_key, session_id, project_key,
            cwd, git_branch, version, slug, entrypoint,
            first_timestamp, last_timestamp
        )
        SELECT
            md5(sle.session_id) AS session_key,
            sle.session_id,
            {project_key_sql("ANY_VALUE(sle.source_path)")} AS project_key,
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
    ).fetchone()[0]

    # Idempotent backfill: if a session already exists from a prior ETL run
    # but lacks project_key / timestamps (legacy minimal-dim path), set them.
    conn.execute(
        f"""
        UPDATE dim_session ds
        SET project_key = COALESCE(ds.project_key, sub.project_key),
            first_timestamp = COALESCE(ds.first_timestamp, sub.first_timestamp),
            last_timestamp = COALESCE(ds.last_timestamp, sub.last_timestamp)
        FROM (
            SELECT
                md5(sle.session_id) AS session_key,
                {project_key_sql("ANY_VALUE(sle.source_path)")} AS project_key,
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

    # dim_project: surrogate from the session's project directory (walks up
    # past <uuid>/subagents layers -- see project_dir_sql). The dir is
    # computed once in the inner SELECT and reused everywhere.
    conn.execute(
        f"""
        INSERT INTO dim_project (project_key, project_path, project_name)
        SELECT
            {project_key_from_dir_sql("sle.project_dir")} AS project_key,
            sle.project_dir AS project_path,
            -- project_name is the last path segment of project_path
            regexp_extract(sle.project_dir, '([^/]+)$', 1) AS project_name
        FROM (
            SELECT DISTINCT {project_dir_sql("source_path")} AS project_dir
            FROM stg_log_entries
        ) sle
        WHERE NOT EXISTS (
            SELECT 1 FROM dim_project dp
            WHERE dp.project_key = {project_key_from_dir_sql("sle.project_dir")}
        )
        """
    )

    # dim_model: from assistant message.model values
    conn.execute(
        """
        INSERT INTO dim_model (model_key, model_name, model_base, model_family)
        SELECT DISTINCT
            md5(json_extract_string(sle.message_json, '$.model')) AS model_key,
            json_extract_string(sle.message_json, '$.model') AS model_name,
            """
        + MODEL_BASE_SQL.format(col="json_extract_string(sle.message_json, '$.model')")
        + """ AS model_base,
            """
        + MODEL_FAMILY_SQL.format(col="json_extract_string(sle.message_json, '$.model')")
        + """ AS model_family
        FROM stg_log_entries sle
        WHERE sle.type = 'assistant'
          AND json_extract_string(sle.message_json, '$.model') IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM dim_model dm
              WHERE dm.model_key = md5(json_extract_string(sle.message_json, '$.model'))
          )
        """
    )

    # dim_tool: from every distinct tool_use.name across all assistant content.
    # `cand` (not `tool_name`) is referenced inside the correlated NOT EXISTS
    # subquery deliberately: dim_tool has its own `tool_name` column, so an
    # unqualified `tool_name` there binds to dim_tool.tool_name instead of the
    # candidate row -- once dim_tool has any row that self-reference is
    # trivially true (every existing row satisfies tool_key = md5(tool_name)
    # by construction), so NOT EXISTS goes permanently false and every tool
    # introduced after the first session silently fails to insert.
    conn.execute(
        """
        INSERT INTO dim_tool (tool_key, tool_name, tool_category)
        SELECT DISTINCT
            md5(cand.tool_name) AS tool_key,
            cand.tool_name,
            """
        + TOOL_CATEGORY_SQL.format(col="cand.tool_name")
        + """ AS tool_category
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
        ) cand
        WHERE cand.tool_name IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM dim_tool dt WHERE dt.tool_key = md5(cand.tool_name)
          )
        """
    )

    # Backfill model_base / model_family on rows written before the
    # structural rule (they hold NULL / 'unknown'). Same reasoning as the
    # tool_category backfill below.
    conn.execute(
        """
        UPDATE dim_model SET
            model_base = """
        + MODEL_BASE_SQL.format(col="model_name")
        + """,
            model_family = """
        + MODEL_FAMILY_SQL.format(col="model_name")
        + """
        WHERE model_base IS NULL OR model_family IS NULL
           OR model_family = 'unknown'
        """
    )

    # Backfill categories on rows that predate the categorization pass (every
    # warehouse built before it holds 'unknown'). Scoped to 'unknown' so a
    # future manual override of a category is not stomped on every run.
    conn.execute(
        """
        UPDATE dim_tool SET tool_category = """
        + TOOL_CATEGORY_SQL.format(col="tool_name")
        + """
        WHERE tool_category IS NULL OR tool_category = 'unknown'
        """
    )

    # dim_date: one row per calendar date seen in staging. Without these
    # rows all nine dim_date-joining semantic views return NULL dates.
    insert_missing_dim_dates(conn, "stg_log_entries", "timestamp")

    return sessions_inserted


@contextmanager
def staging_scope(conn):
    """Clear the staging table at exit so one session's rows never bleed
    into the next (the warehouse is persistent; staging is per-session)."""
    try:
        yield
    finally:
        conn.execute("DELETE FROM stg_log_entries")


def run_v15_etl(
    conn,
    session_path: str | Path,
    *,
    project_name: str = "unknown",
    parquet_lake_root: str | Path | None = None,
    facet_extractor: FacetExtractor | None = None,
    include_thinking: bool = True,
    batch_run_id: str | None = None,
) -> dict[str, Any]:
    """End-to-end ETL for one session JSONL.

    Args:
        conn: DuckDB connection.
        session_path: JSONL file path.
        project_name: ignored at fact level (kept for caller compat); the
            actual project_name in dim_project is derived from source_path.
        parquet_lake_root: where to write the per-session Parquet files.
            Defaults to a sibling 'parquet_lake' dir next to session_path.
        facet_extractor: optional Tier 2 LLM-facet extractor. None (the
            default) disables Tier 2 entirely. When supplied, the Tier 2
            populator runs after Tier 1 and before fact_session_summary
            so the summary roll-up still sees every facet.
        include_thinking: when False, `stg_log_entries` is cleared at the
            end of this call. `fact_messages.content_text` already excludes
            thinking by SQL projection (the populator picks only
            `type='text'` blocks); the per-call truncate removes the raw
            staging JSON so no thinking text survives in the warehouse.
            The Parquet lake is unaffected -- it's the re-derivable cache
            and intentionally captures everything.
        batch_run_id: optional fact_etl_batch_runs id linking this run to
            the orchestration that spawned it (stamped on fact_etl_runs and
            every fact_etl_steps row). None for standalone runs.

    Returns:
        dict with 'etl_run_id' and 'sessions_inserted' for caller use.
    """
    # resolve() so the subagent-layout matchers (Tier-1 stamp, staging
    # override, enrichment) see the full path even when the CLI is invoked
    # with a relative path from inside the project/subagents directory --
    # otherwise the identity override silently skips and the agent
    # collapses into its parent.
    session_path = Path(session_path).resolve()
    if parquet_lake_root is None:
        parquet_lake_root = session_path.parent / "parquet_lake"
    parquet_lake_root = Path(parquet_lake_root)

    run = EtlRun.start(
        conn, source_path=str(session_path), batch_run_id=batch_run_id
    )
    with staging_scope(conn):
        try:
            # Tier 1: parse + write Parquet
            with run.step("write_parquet"):
                log_path, _ = write_session_to_parquet(
                    session_path,
                    parquet_lake_root,
                    etl_run_id=run.etl_run_id,
                    project_slug=project_name,
                )

            # Tier 2: load staging. rows_* here are STAGING rows -- real at
            # step grain, excluded from run-level fact totals (complete()
            # only sums step_kind='upsert' steps). The load also returns
            # the session's CDC window so nothing rescans staging for it.
            with run.step("load_staging") as st:
                staged = load_session_to_staging(conn, log_path)
                st.rows_read = st.rows_inserted = staged.rows

            # Stub dimensions so fact FKs resolve
            with run.step("upsert_dimensions") as st:
                new_sessions = _upsert_minimal_dimensions(conn)
                st.rows_inserted = new_sessions
            # Subagent enrichment looks at the JSONL source_path + sidecar
            # .meta.json to set is_agent / agent_id / parent_session_key /
            # agent_type / agent_description on dim_session.
            with run.step("subagent_enrichment"):
                populate_subagent_dim_session(conn, run=run)

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
            # fact_agent_delegations captures Task tool spawns + agent rollup
            populate_fact_agent_delegations(conn, run=run)
            # fact_errors flattens fact_tool_results where is_error=TRUE; must
            # run before dim_session_heuristics so error_count is accurate.
            populate_fact_errors(conn, run=run)
            # fact_tool_chain_steps captures tool sequences per assistant turn
            populate_fact_tool_chain_steps(conn, run=run)
            # dim_session enrichment runs after all facts so the classifiers
            # see complete metrics + file-extension data
            with run.step("dim_session_heuristics"):
                populate_dim_session_heuristics(
                    conn, run=run, include_thinking=include_thinking,
                )
            # dim_session_chain groups sessions sharing a slug; rebuilt fresh
            # each run since adding a new session can re-aggregate the chain
            with run.step("dim_session_chain"):
                populate_dim_session_chain(conn, run=run)
            # Tier 1 facets: 19 SQL-computed facets per session (F01..F19) into
            # fact_session_facets. Runs after every source fact / dim is in
            # place; runs before fact_session_summary so summary stays the
            # final aggregate roll-up.
            populate_tier1_facets(conn, run=run)
            # Tier 2 facets (LLM-extracted) only run when a FacetExtractor is
            # injected. Default None disables Tier 2 entirely.
            if facet_extractor is not None:
                populate_tier2_facets(
                    conn, run=run, extractor=facet_extractor,
                    include_thinking=include_thinking,
                )
            populate_fact_session_summary(conn, run=run)

            # No per-thinking staging cleanup here: `staging_scope` clears
            # stg_log_entries unconditionally at exit, so the raw message_json
            # (which carries thinking blocks) never survives the run regardless
            # of include_thinking. include_thinking still governs what lands in
            # the FACTS (dim_session.last_assistant_message, Tier 2 inputs,
            # fact_messages.content_text projection).

            # complete() derives facts_inserted/updated from steps; the CDC
            # window came back from the staging load.
            run.complete(
                sessions_seen=1,
                sessions_inserted=new_sessions,
                sessions_updated=1 - new_sessions,
                data_start_ts=staged.data_start_ts,
                data_end_ts=staged.data_end_ts,
            )
        except BaseException as e:
            # BaseException so a KeyboardInterrupt doesn't leave the run
            # row stuck 'running' (batch + step grains already do this).
            run.fail(str(e) or type(e).__name__)
            raise

    return {"etl_run_id": run.etl_run_id, "sessions_inserted": new_sessions}


def run_post_session_reconciliation(conn, *, batch_run_id: str | None = None):
    """Cross-session passes that cannot run inside the per-session ETL.

    ``run_v15_etl`` handles one session at a time, in arbitrary order and
    potentially in parallel, so anything needing rows from a DIFFERENT
    session must run after the whole loop. Today that is the agent-delegation
    completion pass: a parent is normally ETL'd before the agent it spawned,
    so at per-session time the agent's tokens, duration and stop_reason do
    not exist yet.

    BOTH batch entry points must call this. ``local`` and ``all`` diverging
    on post-ETL steps is a bug this project already shipped once -- `all`
    ran the post-ETL functions and `local` did not, silently leaving derived
    tables empty for anyone who used `local`. Adding a cross-session pass to
    only one path recreates exactly that.

    Idempotent: re-running recomputes from current warehouse state, and
    ``lineage_upsert``'s hash_diff means unchanged rows are not rewritten.
    """
    run = EtlRun.start(
        conn,
        source_path="<post-session-reconciliation>",
        batch_run_id=batch_run_id,
        description="cross-session reconciliation",
        run_kind="reconciliation",
    )
    try:
        populate_delegation_completion(conn, run=run)
        run.complete(sessions_seen=0, sessions_inserted=0, sessions_updated=0)
    except BaseException as e:
        run.fail(str(e) or type(e).__name__)
        raise
    return {"etl_run_id": run.etl_run_id}
