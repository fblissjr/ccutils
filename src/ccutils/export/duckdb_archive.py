"""DuckDB / JSON archive generation.

Star schema only. Drives the v0.15 four-tier per-session ETL
(``ccutils.etl.orchestrator.run_v15_etl``) across a project tree and
writes either a DuckDB database or its JSON export.

The legacy `schema_type` parameter is gone; star is the only schema.
The legacy `simple` 4-table schema was removed when v0.15 stabilized.
"""

import os
import time
from pathlib import Path

import duckdb

from ..etl.global_sources import run_global_sources
from ..parsers import find_all_sessions
from ..schemas import (
    create_star_schema,
    export_star_schema_to_json,
)
from ..etl.lineage import BatchRun
from ..etl.orchestrator import run_v15_etl


# Tables `_count_rows` polls for the progress display. Every fact table
# `run_v15_etl` populates -- omitting one undercounts row totals by
# multiples on real corpora (fact_attachments and fact_file_operations
# in particular dwarf fact_messages). Order doesn't matter (SUM).
_PROGRESS_TABLES = (
    "fact_messages",
    "fact_tool_uses",
    "fact_tool_results",
    "fact_token_usage",
    "fact_session_summary",
    "fact_attachments",
    "fact_progress_events",
    "fact_system_events",
    "fact_meta_events",
    "fact_file_history_snapshots",
    "fact_queue_operations",
    "fact_pr_links",
    "fact_file_operations",
    "fact_diagnostics",
    "fact_plan_revisions",
    "fact_agent_delegations",
    "fact_errors",
    "fact_tool_chain_steps",
    "fact_session_facets",
)


def generate_duckdb_archive(
    source_folder,
    output_dir,
    include_agents=False,
    include_thinking=True,
    truncate_output=2000,
    progress_callback=None,
    max_workers=1,
    batch_size=10,
    private=False,
    facet_extractor=None,
    output_format="duckdb",
    projects=None,
    scope_history=False,
):
    """Generate a DuckDB archive for all sessions under ``source_folder``.

    Writes ``archive.duckdb`` plus a sibling ``parquet_lake/`` directory.
    Stage-and-load: parses sessions (parallelizable with ``max_workers``)
    then bulk-inserts in batches of ``batch_size``.

    Args:
        source_folder: Path to Claude Code projects folder.
        output_dir: Path for output.
        include_agents: Whether to include agent sessions.
        include_thinking: When False, `stg_log_entries` is cleared after each
            session ETL so the raw message_json (which contains thinking
            blocks) doesn't survive in the warehouse.
            `fact_messages.content_text` already excludes thinking
            unconditionally (SQL projection). The Parquet lake is the
            re-derivable cache and intentionally captures everything --
            delete it post-run if you don't want thinking in any cache.
        truncate_output: ACCEPTED FOR BACK-COMPAT; v0.15 stores full payloads.
        progress_callback: callback(project_name, session_name, current,
            total, stats) where stats has 'rows_inserted', 'db_size_mb',
            'rate'.
        max_workers: Reserved for future parallelism; the current implementation
            is fully sequential (DuckDB connections aren't write-safe across
            threads). Pass anything; only `batch_size`'s progress-reporting
            cadence is affected.
        batch_size: Sessions per progress-report batch (default: 10).
        private: ACCEPTED FOR BACK-COMPAT but NOT honored -- v0.15 has no
            PathSanitizer wiring. The CLI rejects `--private` for duckdb/json
            upstream so library callers won't normally hit this. If you call
            this function programmatically with private=True, you will NOT
            get sanitized paths.
        facet_extractor: Optional Tier 2 facet extractor; None disables.
        output_format: label recorded on the fact_etl_batch_runs row --
            "duckdb" for a direct archive, "json" when driven by
            generate_json_archive.
        scope_history: keep dim_prompt to the projects this archive covers.
            The CALLER decides, because it follows intent and `projects`
            cannot signal it -- the CLI passes a pre-scanned list whether or
            not `-p` was given. See run_global_sources for why scoping a
            full-corpus build would lose rows rather than being a no-op.
        projects: optional pre-scanned project list (find_all_sessions
            output). Passing it avoids a second full tree walk AND is how
            the CLI's -p project filter reaches this path; None rescans
            unfiltered with complete (unsummarized-inclusive) coverage.

    Returns:
        dict with statistics including row counts.
    """
    source_folder = Path(source_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "archive.duckdb"
    parquet_lake = output_dir / "parquet_lake"

    conn = create_star_schema(db_path)

    processed_count = 0
    successful_sessions = 0
    failed_sessions = []
    start_time = time.time()

    # One orchestration row per CLI invocation, started BEFORE session
    # discovery so even a crash during the scan is captured as a failed
    # batch (the invariant the README advertises). BatchRun's __exit__
    # marks the row failed on any escaping exception (including
    # KeyboardInterrupt / complete() itself erroring), so the batch can
    # never stick at 'running'.
    with BatchRun.start(
        conn, source_root=str(source_folder), output_format=output_format
    ) as batch:
        # Warehouse runs want complete coverage: no summary-based curation.
        if projects is None:
            projects = find_all_sessions(
                source_folder, include_agents=include_agents,
                include_unsummarized=True,
            )
        total_session_count = sum(len(p["sessions"]) for p in projects)
        # Per-session ETL closure. include_thinking forwards through; the
        # other legacy back-compat kwargs (truncate_output, private) are
        # explicitly named at the closure boundary rather than absorbed by
        # **kwargs (CLAUDE.md closure rule -- the **_legacy_kwargs shim once
        # hid a --private silent drop). Naming what we discard makes
        # signature drift fail loud.
        def _etl(
            conn, session_path, project_name,
            include_thinking=True,
            truncate_output=None, private=None,
        ):
            _ = truncate_output, private  # explicitly discarded; v0.15 doesn't use
            return run_v15_etl(
                conn,
                session_path,
                project_name=project_name,
                parquet_lake_root=parquet_lake,
                facet_extractor=facet_extractor,
                include_thinking=include_thinking,
                batch_run_id=batch.batch_run_id,
            )

        session_tasks = []
        for project in projects:
            project_name = project["name"]
            for session in project["sessions"]:
                session_tasks.append((project_name, session["path"]))

        if max_workers > 1 and len(session_tasks) > 1:
            _process_parallel(
                conn,
                session_tasks,
                _etl,
                include_thinking,
                truncate_output,
                batch_size,
                progress_callback,
                db_path,
                start_time,
                failed_sessions,
                private,
            )
            successful_sessions = len(session_tasks) - len(failed_sessions)
        else:
            for project_name, session_path in session_tasks:
                try:
                    _etl(
                        conn,
                        session_path,
                        project_name,
                        include_thinking=include_thinking,
                        truncate_output=truncate_output,
                        private=private,
                    )
                    successful_sessions += 1
                except Exception as e:
                    failed_sessions.append(
                        {
                            "project": project_name,
                            "session": session_path.stem,
                            "error": str(e),
                        }
                    )

                processed_count += 1
                if progress_callback:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    db_size = _get_db_size_mb(db_path)
                    stats = {
                        "rows_inserted": _count_rows(conn),
                        "db_size_mb": db_size,
                        "rate": rate,
                    }
                    progress_callback(
                        project_name,
                        session_path.stem,
                        processed_count,
                        total_session_count,
                        stats,
                    )

        # Every source that is global rather than per-session, in one
        # call that both entry points make. See run_global_sources.
        # Scope history iff the user asked for a subset. An unfiltered
        # full-corpus run asked for everything, and scoping it would
        # drop the ~11% of prompts whose project directory no longer
        # exists or never had a session.
        run_global_sources(conn, batch_run_id=batch.batch_run_id,
                           scope_to_covered_projects=scope_history)

        # Per-session failures were isolated above (they land as failed
        # fact_etl_runs children and make the batch 'partial').
        batch.complete(expected_sessions=total_session_count)

    final_row_count = _count_rows(conn)
    final_db_size = _get_db_size_mb(db_path)

    conn.close()

    return {
        "total_projects": len(projects),
        "total_sessions": successful_sessions,
        "failed_sessions": failed_sessions,
        "output_dir": output_dir,
        "db_path": db_path,
        "rows_inserted": final_row_count,
        "db_size_mb": final_db_size,
    }


def _process_parallel(
    conn,
    session_tasks,
    etl_func,
    include_thinking,
    truncate_output,
    batch_size,
    progress_callback,
    db_path,
    start_time,
    failed_sessions,
    private=False,
):
    """Process sessions in batches with progress reporting.

    Note: DuckDB connections are not thread-safe for writes, so the
    actual DB writes are serialized; batching only affects how often we
    report progress.
    """
    total = len(session_tasks)
    processed = 0
    rows_total = 0

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = session_tasks[batch_start:batch_end]

        for project_name, session_path in batch:
            try:
                etl_func(
                    conn,
                    session_path,
                    project_name,
                    include_thinking=include_thinking,
                    truncate_output=truncate_output,
                    private=private,
                )
            except Exception as e:
                failed_sessions.append(
                    {
                        "project": project_name,
                        "session": session_path.stem,
                        "error": str(e),
                    }
                )

            processed += 1
            if progress_callback:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                db_size = _get_db_size_mb(db_path)
                if processed % 5 == 0:
                    rows_total = _count_rows(conn)
                stats = {
                    "rows_inserted": rows_total,
                    "db_size_mb": db_size,
                    "rate": rate,
                }
                progress_callback(
                    project_name,
                    session_path.stem,
                    processed,
                    total,
                    stats,
                )


def _count_rows(conn):
    """Sum row counts across the v0.15 fact tables that the progress
    display surfaces. Missing tables are silently skipped (a fresh DDL
    may not have populated them yet)."""
    total = 0
    for table in _PROGRESS_TABLES:
        try:
            result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            total += result[0] if result else 0
        except Exception:
            pass
    return total


def _get_db_size_mb(db_path):
    """Database file size in MB."""
    try:
        size_bytes = os.path.getsize(db_path)
        return round(size_bytes / (1024 * 1024), 2)
    except Exception:
        return 0.0


def generate_json_archive(
    source_folder,
    output_dir,
    include_agents=False,
    include_thinking=True,
    truncate_output=2000,
    progress_callback=None,
    max_workers=1,
    batch_size=10,
    private=False,
    facet_extractor=None,
    projects=None,
    scope_history=False,
):
    """Generate a JSON archive for all sessions under ``source_folder``.

    Builds the v0.15 star DuckDB in a temp dir, then exports it as a
    JSON directory tree (meta.json + dimensions/ + facts/). The DuckDB
    is discarded after export.

    Footgun: pairing this with `facet_extractor` pays the full LLM API
    cost during the temp-DB build, then throws the DB away. Use
    ``generate_duckdb_archive`` if you want the queryable DuckDB.
    """
    import tempfile

    source_folder = Path(source_folder)
    output_dir = Path(output_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        stats = generate_duckdb_archive(
            source_folder,
            tmp_path,
            include_agents=include_agents,
            include_thinking=include_thinking,
            truncate_output=truncate_output,
            progress_callback=progress_callback,
            max_workers=max_workers,
            batch_size=batch_size,
            private=private,
            facet_extractor=facet_extractor,
            output_format="json",
            projects=projects,
            # Forward the scope: JSON is the most shareable output we
            # produce, so a filtered build leaking machine-wide prompts
            # here is the worst place for it.
            scope_history=scope_history,
        )

        db_path = tmp_path / "archive.duckdb"
        conn = duckdb.connect(str(db_path))
        export_star_schema_to_json(conn, output_dir)
        conn.close()

    stats["output_dir"] = output_dir
    stats["db_path"] = None  # JSON output discards the DuckDB
    # db_size_mb on the returned dict still reflected the now-deleted
    # tempfile; clear it so the CLI's "Size: X MB" line doesn't print a
    # number referring to a path that no longer exists.
    stats["db_size_mb"] = None
    return stats
