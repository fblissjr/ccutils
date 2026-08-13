"""Batch conversion command for all sessions."""

from datetime import datetime
from pathlib import Path

import click
from click_option_group import optgroup

from ..parsers import find_all_sessions
from ..parsers.discovery import curate_projects
from ..export import (
    generate_batch_html,
    generate_batch_markdown,
    generate_duckdb_archive,
    generate_json_archive,
)
from .utils import (
    build_facet_extractor_or_exit,
    default_archive_output,
    maybe_open_browser,
    run_embedding_pipeline,
    warn_private_best_effort,
)


@click.command("all")
@optgroup.group("Output")
@optgroup.option(
    "-s",
    "--source",
    type=click.Path(exists=True),
    help="Source directory (default: ~/.claude/projects).",  # path-privacy: ignore
)
@optgroup.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output directory (default: ~/.ccutils/claude-archive).",  # path-privacy: ignore
)
@optgroup.option(
    "--format",
    "output_format",
    type=click.Choice(["html", "markdown", "duckdb", "json", "both"]),
    default="html",
    help="Output format: html (default), markdown, duckdb, json, or both (html+duckdb). duckdb/json write the v0.15 star schema.",
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
    "--include-temp-sessions",
    is_flag=True,
    help=(
        "Include sessions whose cwd is under the OS temp directory "
        "(excluded by default -- typically sandboxed/ephemeral tooling "
        "like eval harnesses, not real projects)."
    ),
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
@optgroup.group("Enrichment")
@optgroup.option(
    "--batch-llm-facets",
    is_flag=True,
    default=False,
    help=(
        "Extract Tier 2 LLM facets (F20 task_description via Haiku) into "
        "fact_session_facets across the whole batch. Requires "
        "ANTHROPIC_API_KEY or a ccutils-anthropic keychain entry. Star "
        "schema only."
    ),
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
    include_temp_sessions,
    private,
    jobs,
    batch_size,
    quiet,
    no_search_index,
    embed,
    batch_llm_facets,
):
    """Convert all local Claude Code sessions to HTML, DuckDB, or JSON archives.

    Use --format to choose output:

    \b
    - html: Browsable HTML archive with master index and per-project pages
    - markdown: One .md transcript per session in per-project directories
    - duckdb: DuckDB database with the v0.15 star schema
    - json: JSON directory with the v0.15 star schema (dimensions/ + facts/)
    - both: Generate both HTML archive and DuckDB database

    Thinking blocks and agent sessions are included by default. Use --no-thinking
    or --no-agents to exclude them.
    """
    include_thinking = not no_thinking
    include_agents = not no_agents

    # Honesty guards (mirror cli/local.py): v0.15 has no PathSanitizer wiring
    # yet, so --private would silently produce a non-sanitized database on
    # duckdb/json. Fail loud rather than ship the regression. Render-only
    # formats (html, markdown) sanitize on the render path and are exempt.
    # --no-thinking IS wired (truncates stg_log_entries;
    # fact_messages.content_text already excludes thinking by SQL projection).
    # --embed against --format json discards the embeddings (DB is built in
    # a tempdir and thrown away after export).
    if output_format in ("duckdb", "json", "both") and private:
        raise click.UsageError(
            "--private is not yet wired through the v0.15 ETL; it only "
            "affects the render formats (html, markdown). Either drop "
            "--private or use --format html / --format markdown."
        )
    if private:
        warn_private_best_effort()
    if embed and output_format == "json":
        raise click.UsageError(
            "--embed cannot combine with --format json: the JSON archive is "
            "built in a temporary DuckDB that's discarded after export, so "
            "the embeddings would be lost. Use --format duckdb if you need "
            "embeddings."
        )

    # Build the Tier 2 facet extractor at the CLI boundary so credential
    # failures exit cleanly here rather than as a stack trace deep in the
    # batch.
    facet_extractor = build_facet_extractor_or_exit(batch_llm_facets)

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

    # No -o: land outside any worktree (see default_archive_output --
    # the archive holds unredacted transcripts for every project on the
    # machine and must never default into a checkout).
    output = Path(output) if output else default_archive_output()

    if not quiet:
        click.echo(f"Scanning {source}...")

    # One scan, passed through to every exporter below (also the only way
    # -p reaches them). Warehouse formats ingest everything; html/markdown
    # apply the curated skip of warmup / no-summary sessions themselves
    # (curate_projects inside the exporters), so for --format both the
    # two halves intentionally cover different session sets -- the count
    # line reports both numbers.
    projects = find_all_sessions(
        source,
        include_agents=include_agents,
        project_filter=project_filter,
        include_unsummarized=output_format in ("duckdb", "json", "both"),
        include_temp_sessions=include_temp_sessions,
    )

    if not projects:
        if not quiet:
            click.echo("No sessions found.")
        return

    # Calculate totals
    total_sessions = sum(len(p["sessions"]) for p in projects)

    if not quiet:
        if output_format == "both":
            curated_sessions = sum(
                len(p["sessions"]) for p in curate_projects(projects)
            )
            click.echo(
                f"Found {len(projects)} projects with {total_sessions} sessions "
                f"({curated_sessions} eligible for the HTML half; warehouse "
                f"ingests all {total_sessions})"
            )
        else:
            click.echo(
                f"Found {len(projects)} projects with {total_sessions} sessions"
            )

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
            include_thinking=include_thinking,
            no_search_index=no_search_index,
            private=private,
            projects=projects,
        )

    # Generate markdown if requested (render-only: one .md per session in
    # per-project directories, no index pages, no ETL)
    if output_format == "markdown":

        def markdown_progress(proj, sess, cur, tot):
            on_progress(proj, sess, cur, tot, None)

        stats = generate_batch_markdown(
            source,
            output,
            include_agents=include_agents,
            include_thinking=include_thinking,
            private=private,
            progress_callback=markdown_progress,
            projects=projects,
        )

    # Generate DuckDB if requested (always v0.15 star schema)
    if output_format in ("duckdb", "both"):
        if not quiet and output_format == "both":
            click.echo("\nGenerating DuckDB archive...")

        duckdb_stats = generate_duckdb_archive(
            source,
            output,
            include_agents=include_agents,
            include_thinking=include_thinking,
            progress_callback=on_progress if output_format != "both" else None,
            max_workers=jobs,
            batch_size=batch_size,
            private=private,
            facet_extractor=facet_extractor,
            projects=projects,
        )
        if stats is None:
            stats = duckdb_stats

    # Run embedding pipeline if requested
    if embed and duckdb_stats and duckdb_stats.get("db_path"):
        import duckdb as _duckdb

        emb_conn = _duckdb.connect(str(duckdb_stats["db_path"]))
        run_embedding_pipeline(emb_conn, embed_model, quiet=quiet)
        emb_conn.close()

    # Generate JSON archive if requested (v0.15 star schema as JSON dir)
    if output_format == "json":
        if not quiet:
            click.echo("Generating JSON archive...")
        duckdb_stats = generate_json_archive(
            source,
            output,
            include_agents=include_agents,
            include_thinking=include_thinking,
            progress_callback=on_progress,
            max_workers=jobs,
            batch_size=batch_size,
            private=private,
            facet_extractor=facet_extractor,
            projects=projects,
        )
        if stats is None:
            stats = duckdb_stats

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
