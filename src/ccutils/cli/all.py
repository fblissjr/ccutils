"""Batch conversion command for all sessions."""

from datetime import datetime
from pathlib import Path

import click
from click_option_group import optgroup

from ..parsers import find_all_sessions
from ..schemas import resolve_schema_format
from ..export import (
    generate_batch_html,
    generate_duckdb_archive,
    generate_star_json_archive,
)
from .utils import maybe_open_browser, run_embedding_pipeline


@click.command("all")
@optgroup.group("Output")
@optgroup.option(
    "-s",
    "--source",
    type=click.Path(exists=True),
    help="Source directory (default: ~/.claude/projects).",
)
@optgroup.option(
    "-o",
    "--output",
    type=click.Path(),
    default="./claude-archive",
    help="Output directory (default: ./claude-archive).",
)
@optgroup.option(
    "--format",
    "output_format",
    type=click.Choice(["html", "duckdb", "duckdb-star", "json", "json-star", "both"]),
    default="html",
    help="Output format: html (default), duckdb[-star], json[-star], or both.",
)
@optgroup.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open result in browser.",
)
@optgroup.group("Selection")
@optgroup.option(
    "-p",
    "--project",
    "project_filter",
    help="Filter by project name (partial match).",
)
@optgroup.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be converted without creating files.",
)
@optgroup.group("Content")
@optgroup.option(
    "--no-thinking",
    is_flag=True,
    help="Exclude thinking blocks from export.",
)
@optgroup.option(
    "--no-agents",
    is_flag=True,
    help="Exclude agent-* session files.",
)
@optgroup.option(
    "--private",
    is_flag=True,
    help="Sanitize file paths for sharing.",
)
@optgroup.group("Processing")
@optgroup.option(
    "-j",
    "--jobs",
    default=1,
    type=int,
    help="Parallel workers (default: 1).",
)
@optgroup.option(
    "--batch-size",
    default=10,
    type=int,
    help="Sessions per batch (default: 10).",
)
@optgroup.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Suppress all output except errors.",
)
@optgroup.option(
    "--no-search-index",
    is_flag=True,
    help="Skip search index generation.",
)
@optgroup.group("Embeddings")
@optgroup.option(
    "--embed",
    default=None,
    is_flag=False,
    flag_value="default",
    help="Run ColBERT embeddings (optionally specify model name).",
)
def all_cmd(
    source,
    output,
    output_format,
    open_browser,
    project_filter,
    dry_run,
    no_thinking,
    no_agents,
    private,
    jobs,
    batch_size,
    quiet,
    no_search_index,
    embed,
):
    """Convert all local Claude Code sessions to HTML, DuckDB, or JSON archives.

    Use --format to choose output:

    \b
    - html: Browsable HTML archive with master index and per-project pages
    - duckdb: DuckDB database with simple schema (4 tables)
    - duckdb-star: DuckDB database with star schema (22 tables + 10 views)
    - json: JSON files with simple schema
    - json-star: JSON directory with star schema (dimensions/ + facts/)
    - both: Generate both HTML archive and simple DuckDB database

    Thinking blocks and agent sessions are included by default. Use --no-thinking
    or --no-agents to exclude them.
    """
    include_thinking = not no_thinking
    include_agents = not no_agents

    # Resolve embed model
    embed_model = None
    if embed and embed != "default":
        embed_model = embed

    # Default source folder
    if source is None:
        source = Path.home() / ".claude" / "projects"
    else:
        source = Path(source)

    if not source.exists():
        raise click.ClickException(f"Source directory not found: {source}")

    output = Path(output)

    if not quiet:
        click.echo(f"Scanning {source}...")

    projects = find_all_sessions(
        source, include_agents=include_agents, project_filter=project_filter
    )

    if not projects:
        if not quiet:
            click.echo("No sessions found.")
        return

    # Calculate totals
    total_sessions = sum(len(p["sessions"]) for p in projects)

    if not quiet:
        click.echo(f"Found {len(projects)} projects with {total_sessions} sessions")

    if dry_run:
        # Dry-run always outputs (it's the point of dry-run), but respects --quiet
        if not quiet:
            click.echo("\nDry run - would convert:")
            for project in projects:
                click.echo(
                    f"\n  {project['name']} ({len(project['sessions'])} sessions)"
                )
                for session in project["sessions"][:3]:  # Show first 3
                    mod_time = datetime.fromtimestamp(session["mtime"])
                    click.echo(
                        f"    - {session['path'].stem} ({mod_time.strftime('%Y-%m-%d')})"
                    )
                if len(project["sessions"]) > 3:
                    click.echo(f"    ... and {len(project['sessions']) - 3} more")
        return

    if not quiet:
        click.echo(f"\nGenerating archive in {output}...")

    # Resolve schema type from format
    resolved_schema, resolved_format = resolve_schema_format(output_format)

    # Progress callback for non-quiet mode with enhanced stats
    def on_progress(project_name, session_name, current, total, stats=None):
        if quiet:
            return
        if stats and current % 5 == 0:
            # Enhanced progress with stats
            rate = stats.get("rate", 0)
            db_size = stats.get("db_size_mb", 0)
            rows = stats.get("rows_inserted", 0)
            click.echo(
                f"  [{current}/{total}] {project_name}/{session_name[:8]}... "
                f"({rows} rows, {db_size:.1f} MB, {rate:.1f} sess/sec)"
            )
        elif current % 10 == 0:
            click.echo(f"  Processed {current}/{total} sessions...")

    stats = None
    duckdb_stats = None

    # Generate HTML if requested
    if output_format in ("html", "both"):
        # HTML progress callback has different signature (no stats)
        def html_progress(proj, sess, cur, tot):
            on_progress(proj, sess, cur, tot, None)

        stats = generate_batch_html(
            source,
            output,
            include_agents=include_agents,
            progress_callback=html_progress,
            no_search_index=no_search_index,
            private=private,
        )

    # Generate DuckDB if requested (simple or star schema)
    if output_format in ("duckdb", "duckdb-star", "both"):
        if not quiet:
            if output_format == "both":
                click.echo("\nGenerating DuckDB archive...")
            elif output_format == "duckdb-star":
                click.echo(f"Using star schema ({resolved_schema})")

        duckdb_stats = generate_duckdb_archive(
            source,
            output,
            schema_type=resolved_schema,
            include_agents=include_agents,
            include_thinking=include_thinking,
            progress_callback=on_progress if output_format != "both" else None,
            max_workers=jobs,
            batch_size=batch_size,
            private=private,
        )
        if stats is None:
            stats = duckdb_stats

    # Run embedding pipeline if requested (star schema only)
    if embed and duckdb_stats and duckdb_stats.get("db_path"):
        import duckdb as _duckdb

        emb_conn = _duckdb.connect(str(duckdb_stats["db_path"]))
        run_embedding_pipeline(emb_conn, embed_model, quiet=quiet)
        emb_conn.close()

    # Generate JSON star schema if requested
    if output_format == "json-star":
        if not quiet:
            click.echo("Generating JSON star schema archive...")
        duckdb_stats = generate_star_json_archive(
            source,
            output,
            include_agents=include_agents,
            include_thinking=include_thinking,
            progress_callback=on_progress,
            max_workers=jobs,
            batch_size=batch_size,
            private=private,
        )
        if stats is None:
            stats = duckdb_stats

    # Generate simple JSON if requested
    if output_format == "json":
        if not quiet:
            click.echo("Generating JSON archive...")
        from ..schemas import export_sessions_to_json

        # Collect all session paths
        session_paths = []
        for project in projects:
            for session in project["sessions"]:
                session_paths.append(session["path"])

        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "sessions.json"
        export_sessions_to_json(
            session_paths,
            json_path,
            include_thinking=include_thinking,
            private=private,
        )
        stats = {
            "total_projects": len(projects),
            "total_sessions": total_sessions,
            "failed_sessions": [],
            "output_dir": output,
            "db_path": None,
        }

    # Report any failures
    if stats and stats.get("failed_sessions"):
        click.echo(f"\nWarning: {len(stats['failed_sessions'])} session(s) failed:")
        for failure in stats["failed_sessions"]:
            click.echo(
                f"  {failure['project']}/{failure['session']}: {failure['error']}"
            )

    if not quiet and stats:
        click.echo(
            f"\nGenerated archive with {stats['total_projects']} projects, "
            f"{stats['total_sessions']} sessions"
        )
        click.echo(f"Output: {output.resolve()}")
        if duckdb_stats:
            if duckdb_stats.get("db_path"):
                click.echo(f"DuckDB: {duckdb_stats['db_path']}")
            if duckdb_stats.get("rows_inserted"):
                click.echo(f"Rows: {duckdb_stats['rows_inserted']}")
            if duckdb_stats.get("db_size_mb"):
                click.echo(f"Size: {duckdb_stats['db_size_mb']:.2f} MB")

    if open_browser and output_format in ("html", "both"):
        maybe_open_browser(output)
