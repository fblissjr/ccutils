"""Session selection and conversion command."""

import os
import tempfile
from pathlib import Path

import click
import questionary
from click_option_group import optgroup
from rich.console import Console

from ..parsers import (
    find_agent_sessions,
    flatten_selected_sessions,
)
from ..parsers.discovery import (
    find_local_sessions_rich,
    group_by_project,
)
from ..tui import (
    build_flat_choices,
    build_project_choices,
    build_session_choices,
    questionary_style,
    render_project_table,
    render_session_table,
)
from ..schemas.star import (
    create_star_schema,
    export_star_schema_to_json,
)
from ..etl.lineage import BatchRun
from ..etl.orchestrator import run_v15_etl
from ..export import (
    generate_html,
    generate_markdown,
    generate_multi_session_index,
)
from .utils import (
    build_facet_extractor_or_exit,
    maybe_open_browser,
    run_embedding_pipeline,
    warn_private_best_effort,
)


@click.command("local")
@click.argument("input_file", required=False, default=None, type=click.Path())
@optgroup.group("Output")
@optgroup.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output directory (default: ./claude-archive or temp dir for single file).",
)
@optgroup.option(
    "--format",
    "output_format",
    type=click.Choice(["html", "markdown", "duckdb", "json"]),
    default="html",
    help="Output format: html (default), markdown, duckdb, or json. Both duckdb and json write the v0.15 star schema.",
)
@optgroup.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open result in browser.",
)
@optgroup.group("Selection")
@optgroup.option(
    "--flat",
    is_flag=True,
    help="Flat single-list mode (skip project grouping).",
)
@optgroup.option(
    "--expand-chains",
    is_flag=True,
    help="Show individual sessions in resumed chains.",
)
@optgroup.option(
    "-p",
    "--project",
    "project_filter",
    help="Filter by project name (partial match).",
)
@optgroup.group("Content")
@optgroup.option(
    "--no-thinking",
    is_flag=True,
    help="Exclude thinking blocks from export.",
)
@optgroup.option(
    "--no-subagents",
    is_flag=True,
    help="Exclude related agent sessions.",
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
    "--with-llm-facets",
    is_flag=True,
    default=False,
    help=(
        "Extract Tier 2 LLM facets (F20 task_description via Haiku) into "
        "fact_session_facets. Requires ANTHROPIC_API_KEY or a "
        "ccutils-anthropic keychain entry. Star schema only."
    ),
)
def local_cmd(
    input_file,
    output,
    output_format,
    open_browser,
    flat,
    expand_chains,
    project_filter,
    no_thinking,
    no_subagents,
    include_temp_sessions,
    private,
    embed,
    with_llm_facets,
):
    """Convert Claude Code sessions to HTML, Markdown, DuckDB, or JSON.

    With no arguments, launches an interactive picker for local sessions.
    Pass a session file to convert it directly.

    \b
    Examples:
      ccutils                                    # interactive picker
      ccutils session.jsonl                      # convert file, open in browser
      ccutils session.jsonl --format duckdb -o ./out  # star schema DuckDB
      ccutils --format duckdb -o ./archive  # pick sessions, star schema
      ccutils session.jsonl --format duckdb -o ./out --with-llm-facets
    """
    include_thinking = not no_thinking

    # Honesty guard: v0.15 has no PathSanitizer wiring yet, so --private
    # would silently produce a non-sanitized database on the duckdb/json
    # paths. Fail loud rather than ship the regression. Render-only formats
    # (html, markdown) sanitize on the render path and are exempt.
    # --no-thinking IS wired (truncates stg_log_entries;
    # fact_messages.content_text already excludes thinking by SQL projection).
    if output_format in ("duckdb", "json") and private:
        raise click.UsageError(
            "--private is not yet wired through the v0.15 ETL; it only "
            "affects the render formats (html, markdown). Either drop "
            "--private or use --format html / --format markdown."
        )
    if private:
        warn_private_best_effort()

    # Build the Tier 2 facet extractor at the CLI boundary -- credential
    # failures exit cleanly here rather than as a stack trace from inside
    # _run_export_pipeline. Returns None when --with-llm-facets is off.
    facet_extractor = build_facet_extractor_or_exit(with_llm_facets)

    if input_file:
        # Direct file conversion mode
        _convert_file(
            input_file, output, output_format, open_browser,
            include_thinking, private, facet_extractor,
        )
    else:
        # Interactive picker mode
        _interactive_mode(
            output, output_format, open_browser, flat, expand_chains,
            project_filter, include_thinking, not no_subagents, private,
            embed, facet_extractor, include_temp_sessions,
        )


def _convert_file(input_file, output, output_format, open_browser,
                   include_thinking, private, facet_extractor=None):
    """Convert a single session file to the requested format."""
    json_file_path = Path(input_file)
    if not json_file_path.exists():
        raise click.ClickException(f"File not found: {input_file}")

    # Single file with no -o: use temp dir and auto-open browser
    if output is None:
        open_browser = True
        output = Path(tempfile.gettempdir()) / f"claude-session-{json_file_path.stem}"
    output = Path(output)

    _run_export_pipeline(
        session_files=[json_file_path],
        output=output,
        output_format=output_format,
        include_thinking=include_thinking,
        private=private,
        project_name=json_file_path.parent.name or "unknown",
        open_browser=open_browser,
        facet_extractor=facet_extractor,
    )


def _interactive_mode(output, output_format, open_browser, flat, expand_chains,
                      project_filter, include_thinking, include_subagents,
                      private, embed, facet_extractor=None,
                      include_temp_sessions=False):
    """Interactive session picker followed by export."""
    projects_folder = Path.home() / ".claude" / "projects"

    if not projects_folder.exists():
        click.echo(f"Projects folder not found: {projects_folder}")
        click.echo("No local Claude Code sessions available.")
        return

    console = Console()
    style = questionary_style()

    if flat:
        selected = _flat_mode_selection(
            projects_folder, 100, project_filter, expand_chains, style,
            include_temp_sessions,
        )
    else:
        selected = _two_phase_selection(
            projects_folder, 100, project_filter, expand_chains, console, style,
            include_temp_sessions,
        )

    if not selected:
        click.echo("No sessions selected.")
        return

    selected = flatten_selected_sessions(selected)
    click.echo(f"Selected {len(selected)} session(s)")

    # Auto-include subagents (default behavior, opt out with --no-subagents)
    agent_map = {}
    if include_subagents:
        agent_map = find_agent_sessions(selected, recursive=True)
        agent_count = sum(len(agents) for agents in agent_map.values())
        if agent_count > 0:
            click.echo(f"Including {agent_count} related agent session(s)")
            for parent, agents in agent_map.items():
                for agent_path in agents:
                    if agent_path not in selected:
                        selected.append(agent_path)

    # Picker mode: default to ./claude-archive
    if output is None:
        output = Path("./claude-archive")
    output = Path(output)

    _run_export_pipeline(
        session_files=selected,
        output=output,
        output_format=output_format,
        include_thinking=include_thinking,
        private=private,
        open_browser=open_browser,
        agent_map=agent_map,
        embed=embed,
        facet_extractor=facet_extractor,
    )


def _etl_session_files(
    conn,
    session_files,
    *,
    project_name,
    parquet_lake,
    facet_extractor,
    include_thinking,
    output_format=None,
):
    """Run run_v15_etl over each file, isolating per-file failures.

    One empty/unparseable session (e.g. write_session_to_parquet raising
    "No valid JSON log entries found") must not abort the whole export --
    it is reported and skipped, mirroring the per-session isolation in the
    batch path (export/duckdb_archive.py). Returns the list of
    (session_file, exception) failures.

    Records one fact_etl_batch_runs row for the invocation; each session's
    EtlRun links back via batch_run_id and complete() rolls the counts up.
    Anything that escapes the per-file isolation (KeyboardInterrupt, a
    failure inside complete() itself) marks the batch row failed instead
    of leaving it stuck 'running'.
    """
    source_root = (
        os.path.commonpath([str(f.parent) for f in session_files])
        if session_files else ""
    )
    failures = []
    # BatchRun's __exit__ marks the row failed on ANY escaping exception
    # (including KeyboardInterrupt / complete() itself erroring), so the
    # batch can never stick at 'running'.
    with BatchRun.start(
        conn, source_root=source_root, output_format=output_format,
    ) as batch:
        for idx, session_file in enumerate(session_files, 1):
            click.echo(f"[{idx}/{len(session_files)}] {session_file.name}")
            try:
                run_v15_etl(
                    conn,
                    session_file,
                    project_name=project_name or session_file.parent.name,
                    parquet_lake_root=parquet_lake,
                    facet_extractor=facet_extractor,
                    include_thinking=include_thinking,
                    batch_run_id=batch.batch_run_id,
                )
            except Exception as exc:  # noqa: BLE001 -- isolate one bad file
                failures.append((session_file, exc))
                click.echo(f"  skipped {session_file.name}: {exc}", err=True)
        batch.complete(expected_sessions=len(session_files))
    if failures:
        click.echo(
            f"{len(failures)} of {len(session_files)} session(s) failed "
            "and were skipped.",
            err=True,
        )
    return failures


def _run_export_pipeline(
    session_files,
    output,
    output_format,
    include_thinking,
    private,
    open_browser=False,
    project_name=None,
    agent_map=None,
    embed=None,
    facet_extractor=None,
):
    """Shared export pipeline for all output formats.

    Args:
        session_files: List of Path objects to session JSONL files.
        output: Output path (directory for HTML/JSON, file for DuckDB).
        output_format: One of html, markdown, duckdb, json. The simple
            4-table schema is gone; duckdb/json now always write the v0.15
            star schema (DDL in schemas/star, ETL in etl/orchestrator).
            markdown is render-only like html (no ETL, no warehouse).
        include_thinking: Whether to include thinking blocks.
        private: Whether to sanitize file paths.
        open_browser: Open result in browser after HTML export.
        project_name: Override project name (for single-file mode).
        agent_map: Agent session relationships (from interactive picker).
        embed: Embedding model name, "default", or None.
        facet_extractor: Pre-built FacetExtractor instance (or None).
            Construction happens at the CLI boundary in local_cmd so
            credential errors exit cleanly before any work starts.
    """
    if agent_map is None:
        agent_map = {}

    # Resolve embed model
    embed_model = None
    if embed and embed != "default":
        embed_model = embed

    # github_repo is auto-detected by generate_html() from git push output
    # in the JSONL session data, or from the current working directory's git remote.
    if output_format == "html":
        if len(session_files) == 1 and not agent_map:
            generate_html(session_files[0], output, private=private)
        else:
            output.mkdir(parents=True, exist_ok=True)
            for idx, session_file in enumerate(session_files, 1):
                session_output = output / session_file.stem
                click.echo(f"[{idx}/{len(session_files)}] {session_file.name}")
                generate_html(session_file, session_output, private=private)
            generate_multi_session_index(output, session_files, agent_map=agent_map)
            click.echo(f"Generated {len(session_files)} session(s) with master index")

        click.echo(f"Output: {output.resolve()}")
        if open_browser:
            maybe_open_browser(output)

    elif output_format == "markdown":
        # Render-only format like html: no ETL, no warehouse. Single file
        # with an explicit .md output path writes that file; everything
        # else writes one <session-stem>.md per session into the output
        # directory (no index pages).
        if len(session_files) == 1 and output.suffix == ".md":
            md_path = generate_markdown(
                session_files[0], output,
                include_thinking=include_thinking, private=private,
            )
            click.echo(f"Output: {md_path.resolve()}")
        else:
            output.mkdir(parents=True, exist_ok=True)
            for idx, session_file in enumerate(session_files, 1):
                click.echo(f"[{idx}/{len(session_files)}] {session_file.name}")
                generate_markdown(
                    session_file, output,
                    include_thinking=include_thinking, private=private,
                )
            click.echo(f"Output: {output.resolve()}")

    elif output_format == "duckdb":
        db_path = (
            output.with_suffix(".duckdb") if output.suffix != ".duckdb" else output
        )
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = create_star_schema(db_path)
        parquet_lake = output.parent / "parquet_lake"
        _etl_session_files(
            conn, session_files,
            project_name=project_name,
            parquet_lake=parquet_lake,
            facet_extractor=facet_extractor,
            include_thinking=include_thinking,
            output_format="duckdb",
        )
        if embed:
            run_embedding_pipeline(conn, embed_model)
        conn.close()

        click.echo(f"Exported to {db_path}")

    elif output_format == "json":
        # JSON output is a directory containing meta.json + dimensions/ + facts/.
        # If user passes `-o ./out.json`, we treat that literally as the
        # directory name (was: silently strip the `.json` suffix to `./out`).
        # The legacy simple-JSON path wrote a single file at that exact path;
        # v0.15's directory shape is intentionally different, but the rename
        # belongs to the user, not us.
        output_dir = output
        output_dir.mkdir(parents=True, exist_ok=True)
        click.echo(f"Exporting {len(session_files)} session(s) to JSON...")

        conn = create_star_schema(":memory:")
        parquet_lake = output_dir / "parquet_lake"
        _etl_session_files(
            conn, session_files,
            project_name=project_name,
            parquet_lake=parquet_lake,
            facet_extractor=facet_extractor,
            include_thinking=include_thinking,
            output_format="json",
        )
        if embed:
            run_embedding_pipeline(conn, embed_model)

        export_star_schema_to_json(conn, output_dir)
        conn.close()
        click.echo(f"Exported to {output_dir}/")


def _flat_mode_selection(projects_folder, limit, project_filter, expand_chains, style,
                          include_temp_sessions=False):
    """Flat mode: single list of all sessions sorted by date with rich metadata."""
    click.echo("Scanning sessions...")
    sessions = find_local_sessions_rich(
        projects_folder, limit=limit, project_filter=project_filter,
        include_temp_sessions=include_temp_sessions,
    )

    if not sessions:
        return None

    grouped = group_by_project(sessions)

    choices = build_flat_choices(grouped, expand_chains=expand_chains)

    selected = questionary.checkbox(
        "Select sessions to convert (SPACE to select, ENTER to confirm):",
        choices=choices,
        style=style,
    ).ask()

    return selected


def _two_phase_selection(
    projects_folder,
    limit,
    project_filter,
    expand_chains,
    console,
    style,
    include_temp_sessions=False,
):
    """Two-phase selection: pick projects, then pick sessions."""
    click.echo("Scanning sessions...")
    sessions = find_local_sessions_rich(
        projects_folder, limit=limit, project_filter=project_filter,
        include_temp_sessions=include_temp_sessions,
    )

    if not sessions:
        return None

    grouped = group_by_project(sessions)

    # If only one project found (or -p narrowed to one), skip phase 1
    if len(grouped) == 1:
        selected_projects = list(grouped.keys())
    else:
        # Phase 1: Project selection
        render_project_table(grouped, console=console)

        project_choices = build_project_choices(grouped)
        selected_projects = questionary.checkbox(
            "Select projects (SPACE to select, ENTER to confirm):",
            choices=project_choices,
            style=style,
        ).ask()

        if not selected_projects:
            return None

    # Phase 2: Session selection within chosen projects
    # Print session tables for selected projects
    for project_path in selected_projects:
        if project_path in grouped:
            project_sessions = grouped[project_path]
            project_name = project_sessions[0].project_name
            render_session_table(project_name, project_sessions, console=console)

    session_choices = build_session_choices(
        sessions, selected_projects, expand_chains=expand_chains
    )

    selected = questionary.checkbox(
        "Select sessions to convert (SPACE to select, ENTER to confirm):",
        choices=session_choices,
        style=style,
    ).ask()

    return selected
