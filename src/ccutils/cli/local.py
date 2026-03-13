"""Session selection and conversion command."""

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
from ..schemas import (
    create_duckdb_schema,
    export_session_to_duckdb,
    export_sessions_to_json,
    resolve_schema_format,
)
from ..schemas.star import (
    create_semantic_model,
    create_star_schema,
    export_star_schema_to_json,
    run_star_schema_etl,
)
from ..export import (
    finalize_star_schema,
    generate_html,
    generate_multi_session_index,
)
from .utils import maybe_open_browser


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
    type=click.Choice(["html", "duckdb", "duckdb-star", "json", "json-star"]),
    default="html",
    help="Output format: html (default), duckdb[-star], or json[-star].",
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
    private,
    embed,
):
    """Convert Claude Code sessions to HTML, DuckDB, or JSON.

    With no arguments, launches an interactive picker for local sessions.
    Pass a session file to convert it directly.

    \b
    Examples:
      ccutils                                    # interactive picker
      ccutils session.jsonl                      # convert file, open in browser
      ccutils session.jsonl --format duckdb-star -o ./out  # star schema
      ccutils --format duckdb-star -o ./archive  # pick sessions, star schema
    """
    include_thinking = not no_thinking

    if input_file:
        # Direct file conversion mode
        _convert_file(
            input_file, output, output_format, open_browser,
            include_thinking, private,
        )
    else:
        # Interactive picker mode
        _interactive_mode(
            output, output_format, open_browser, flat, expand_chains,
            project_filter, include_thinking, not no_subagents, private, embed,
        )


def _convert_file(input_file, output, output_format, open_browser,
                   include_thinking, private):
    """Convert a single session file to the requested format."""
    json_file_path = Path(input_file)
    if not json_file_path.exists():
        raise click.ClickException(f"File not found: {input_file}")

    # Single file: default to temp dir + auto-open browser
    auto_open = output is None
    if output is None:
        output = Path(tempfile.gettempdir()) / f"claude-session-{json_file_path.stem}"
    output = Path(output)

    project_name = json_file_path.parent.name or "unknown"

    _run_export_pipeline(
        session_files=[json_file_path],
        output=output,
        output_format=output_format,
        include_thinking=include_thinking,
        private=private,
        project_name=project_name,
        open_browser=open_browser,
        auto_open=auto_open,
    )


def _interactive_mode(output, output_format, open_browser, flat, expand_chains,
                      project_filter, include_thinking, include_subagents,
                      private, embed):
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
        )
    else:
        selected = _two_phase_selection(
            projects_folder, 100, project_filter, expand_chains, console, style,
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
        auto_open=False,
        agent_map=agent_map,
        embed=embed,
    )


def _run_export_pipeline(
    session_files,
    output,
    output_format,
    include_thinking,
    private,
    open_browser=False,
    auto_open=False,
    project_name=None,
    agent_map=None,
    embed=None,
):
    """Shared export pipeline for all output formats.

    Args:
        session_files: List of Path objects to session JSONL files.
        output: Output path (directory for HTML/JSON, file for DuckDB).
        output_format: One of html, duckdb, duckdb-star, json, json-star.
        include_thinking: Whether to include thinking blocks.
        private: Whether to sanitize file paths.
        open_browser: Explicit --open flag.
        auto_open: Auto-open browser (single file with no -o).
        project_name: Override project name (for single-file mode).
        agent_map: Agent session relationships (from interactive picker).
        embed: Embedding model name, "default", or None.
    """
    if agent_map is None:
        agent_map = {}

    schema, fmt = resolve_schema_format(output_format)

    # Resolve embed model
    embed_model = None
    if embed and embed != "default":
        embed_model = embed

    # github_repo is auto-detected by generate_html() from git push output
    # in the JSONL session data, or from the current working directory's git remote.
    if fmt == "html":
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
        if open_browser or auto_open:
            maybe_open_browser(output)

    elif fmt == "duckdb":
        db_path = (
            output.with_suffix(".duckdb") if output.suffix != ".duckdb" else output
        )
        db_path.parent.mkdir(parents=True, exist_ok=True)

        if schema == "simple":
            conn = create_duckdb_schema(db_path)
            for idx, session_file in enumerate(session_files, 1):
                click.echo(f"[{idx}/{len(session_files)}] {session_file.name}")
                export_session_to_duckdb(
                    conn,
                    session_file,
                    project_name or session_file.parent.name,
                    include_thinking=include_thinking,
                    private=private,
                )
            conn.close()
        else:  # star schema
            conn = create_star_schema(db_path)
            for idx, session_file in enumerate(session_files, 1):
                click.echo(f"[{idx}/{len(session_files)}] {session_file.name}")
                run_star_schema_etl(
                    conn,
                    session_file,
                    project_name or session_file.parent.name,
                    include_thinking=include_thinking,
                    private=private,
                )
            finalize_star_schema(conn)
            create_semantic_model(conn)

            if embed:
                _run_embedding_pipeline(conn, embed_model)

            conn.close()

        click.echo(f"Exported to {db_path}")

    elif fmt == "json":
        if schema == "simple":
            if output.is_dir() or output.name == "" or output.suffix == "":
                json_path = output / "sessions.json"
            elif output.suffix != ".json":
                json_path = output.with_suffix(".json")
            else:
                json_path = output
            json_path.parent.mkdir(parents=True, exist_ok=True)
            click.echo(f"Exporting {len(session_files)} session(s) to JSON...")
            export_sessions_to_json(
                session_files, json_path,
                include_thinking=include_thinking, private=private,
            )
            click.echo(f"Exported to {json_path}")
        else:  # star schema
            if output.is_dir() or output.name in ("", "."):
                output_dir = output / "star_schema"
            elif output.suffix != "":
                output_dir = output.with_suffix("")
            else:
                output_dir = output
            output_dir.mkdir(parents=True, exist_ok=True)
            click.echo(f"Exporting {len(session_files)} session(s) to star schema JSON...")

            conn = create_star_schema(":memory:")
            for idx, session_file in enumerate(session_files, 1):
                click.echo(f"[{idx}/{len(session_files)}] {session_file.name}")
                run_star_schema_etl(
                    conn,
                    session_file,
                    project_name or session_file.parent.name,
                    include_thinking=include_thinking,
                    private=private,
                )
            finalize_star_schema(conn)
            create_semantic_model(conn)

            if embed:
                _run_embedding_pipeline(conn, embed_model)

            export_star_schema_to_json(conn, output_dir)
            conn.close()
            click.echo(f"Exported to {output_dir}/")


def _run_embedding_pipeline(conn, embed_model=None):
    """Run ColBERT embedding pipeline on a star schema connection."""
    try:
        from ..schemas.star.embeddings import EmbeddingPipeline

        click.echo("Running ColBERT embedding pipeline...")
        pipeline = EmbeddingPipeline(model_name=embed_model)
        result = pipeline.embed_sessions(conn)
        click.echo(f"  Embedded {result['sessions_embedded']} sessions")
        match_result = pipeline.match_delegations(conn)
        if match_result["delegations_rescored"] > 0:
            click.echo(
                f"  Re-scored {match_result['delegations_rescored']} delegations"
            )
    except ImportError:
        click.echo(
            "Warning: pylate not installed. " "Install with: uv add ccutils[colbert]"
        )


def _flat_mode_selection(projects_folder, limit, project_filter, expand_chains, style):
    """Flat mode: single list of all sessions sorted by date with rich metadata."""
    click.echo("Scanning sessions...")
    sessions = find_local_sessions_rich(
        projects_folder, limit=limit, project_filter=project_filter
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
):
    """Two-phase selection: pick projects, then pick sessions."""
    click.echo("Scanning sessions...")
    sessions = find_local_sessions_rich(
        projects_folder, limit=limit, project_filter=project_filter
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
