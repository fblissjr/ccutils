"""Local session selection and conversion command."""

import shutil
from datetime import datetime
from pathlib import Path

import click
import questionary
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
    generate_html,
    generate_multi_session_index,
)
from .utils import maybe_open_browser


@click.command("local")
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output directory. If not specified, writes to temp dir and opens in browser.",
)
@click.option(
    "-a",
    "--output-auto",
    is_flag=True,
    help="Auto-name output subdirectory based on session filename (uses -o as parent, or current dir).",
)
@click.option(
    "--repo",
    help="GitHub repo (owner/name) for commit links. Auto-detected from git push output if not specified.",
)
@click.option(
    "--json",
    "include_json",
    is_flag=True,
    help="Include the original JSONL session file in the output directory.",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open the generated index.html in your default browser (default if no -o specified).",
)
@click.option(
    "--limit",
    default=100,
    help="Maximum number of sessions to show (default: 100).",
)
@click.option(
    "-p",
    "--project",
    "project_filter",
    help="Filter by project name (partial match, case-insensitive).",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["html", "duckdb", "duckdb-star", "json", "json-star"]),
    default="html",
    help="Output format: html (default), duckdb[-star], or json[-star].",
)
@click.option(
    "--schema",
    "schema_type",
    type=click.Choice(["simple", "star"]),
    default=None,
    help="Data schema: simple (4 tables) or star (dimensional). Auto-inferred from format.",
)
@click.option(
    "--include-subagents",
    is_flag=True,
    help="Auto-include related agent sessions (recursive by default).",
)
@click.option(
    "--include-thinking",
    is_flag=True,
    help="Include thinking blocks in DuckDB/JSON export (can be large).",
)
@click.option(
    "--expand-chains",
    is_flag=True,
    help="Show individual sessions in chains instead of collapsed view.",
)
@click.option(
    "--flat",
    is_flag=True,
    help="Show all sessions in a flat list sorted by date (disables project grouping).",
)
@click.option(
    "--private",
    is_flag=True,
    help="Sanitize file paths in output to remove home directory and absolute paths.",
)
def local_cmd(
    output,
    output_auto,
    repo,
    include_json,
    open_browser,
    limit,
    project_filter,
    output_format,
    schema_type,
    include_subagents,
    include_thinking,
    expand_chains,
    flat,
    private,
):
    """Select and convert local Claude Code sessions to HTML or DuckDB.

    Two-phase selection: first pick project(s), then pick session(s) within them.
    Use --flat to skip project selection and show all sessions in a single list.
    Supports multi-select: use SPACE to select multiple, ENTER to confirm.
    """
    projects_folder = Path.home() / ".claude" / "projects"

    if not projects_folder.exists():
        click.echo(f"Projects folder not found: {projects_folder}")
        click.echo("No local Claude Code sessions available.")
        return

    console = Console()
    style = questionary_style()

    if flat:
        # --flat mode: use old behavior with improved formatting
        selected = _flat_mode_selection(
            projects_folder,
            limit,
            project_filter,
            expand_chains,
            style,
        )
    else:
        # Two-phase mode: project selection then session selection
        selected = _two_phase_selection(
            projects_folder,
            limit,
            project_filter,
            include_subagents,
            expand_chains,
            console,
            style,
        )

    if not selected:
        click.echo("No sessions selected.")
        return

    # Flatten selection: chains return lists of paths, standalone return single paths
    selected = flatten_selected_sessions(selected)
    click.echo(f"Selected {len(selected)} session(s)")

    # Auto-include subagents if requested
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

    # Determine output path - default to ./claude-archive (not temp dir)
    if output_auto:
        parent_dir = Path(output) if output else Path(".")
        if len(selected) == 1:
            output = parent_dir / selected[0].stem
        else:
            output = (
                parent_dir / f"multi-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
    elif output is None:
        # Default to local ./claude-archive directory
        output = Path("./claude-archive")

    output = Path(output)

    # Resolve schema and format from potentially compound format names
    schema, fmt = resolve_schema_format(schema_type, output_format)

    # Execute based on format
    if fmt == "html":
        if len(selected) == 1 and not agent_map:
            # Single session, no agents - use existing simple path
            generate_html(selected[0], output, github_repo=repo, private=private)
        else:
            # Multiple sessions or has agents - use batch structure with master index
            output.mkdir(parents=True, exist_ok=True)
            for idx, session_file in enumerate(selected, 1):
                session_output = output / session_file.stem
                click.echo(f"[{idx}/{len(selected)}] {session_file.name}")
                generate_html(
                    session_file, session_output, github_repo=repo, private=private
                )
            # Generate master index with agent relationships
            generate_multi_session_index(output, selected, agent_map=agent_map)
            click.echo(f"Generated {len(selected)} session(s) with master index")

    elif fmt == "duckdb":
        db_path = (
            output.with_suffix(".duckdb") if output.suffix != ".duckdb" else output
        )
        db_path.parent.mkdir(parents=True, exist_ok=True)

        if schema == "simple":
            conn = create_duckdb_schema(db_path)
            for idx, session_file in enumerate(selected, 1):
                click.echo(f"[{idx}/{len(selected)}] {session_file.name}")
                export_session_to_duckdb(
                    conn,
                    session_file,
                    session_file.parent.name,
                    include_thinking=include_thinking,
                    private=private,
                )
            conn.close()
        else:  # star schema
            conn = create_star_schema(db_path)
            for idx, session_file in enumerate(selected, 1):
                click.echo(f"[{idx}/{len(selected)}] {session_file.name}")
                run_star_schema_etl(
                    conn,
                    session_file,
                    session_file.parent.name,
                    include_thinking=include_thinking,
                    private=private,
                )
            # Generate semantic model metadata after all ETL is complete
            create_semantic_model(conn)
            conn.close()

        click.echo(f"Exported to {db_path}")
        return  # Skip browser open for DuckDB

    elif fmt == "json":
        if schema == "simple":
            # Handle directory paths (like ".") - generate default filename
            if output.is_dir() or output.name == "" or output.suffix == "":
                json_path = output / "sessions.json"
            elif output.suffix != ".json":
                json_path = output.with_suffix(".json")
            else:
                json_path = output
            json_path.parent.mkdir(parents=True, exist_ok=True)
            click.echo(f"Exporting {len(selected)} session(s) to JSON...")
            export_sessions_to_json(
                selected, json_path, include_thinking=include_thinking, private=private
            )
            click.echo(f"Exported to {json_path}")
        else:  # star schema
            # Star schema JSON exports to a directory
            # Handle paths like "." or paths with extensions
            if output.is_dir() or output.name in ("", "."):
                output_dir = output / "star_schema"
            elif output.suffix != "":
                output_dir = output.with_suffix("")
            else:
                output_dir = output
            output_dir.mkdir(parents=True, exist_ok=True)
            click.echo(f"Exporting {len(selected)} session(s) to star schema JSON...")
            # First create DuckDB in memory, then export to JSON
            conn = create_star_schema(":memory:")
            for idx, session_file in enumerate(selected, 1):
                click.echo(f"[{idx}/{len(selected)}] {session_file.name}")
                run_star_schema_etl(
                    conn,
                    session_file,
                    session_file.parent.name,
                    include_thinking=include_thinking,
                    private=private,
                )
            create_semantic_model(conn)
            export_star_schema_to_json(conn, output_dir)
            conn.close()
            click.echo(f"Exported to {output_dir}/")
        return  # Skip browser open for JSON

    # Show output directory
    click.echo(f"Output: {output.resolve()}")

    # Copy JSONL file to output directory if requested
    if include_json and fmt == "html":
        output.mkdir(exist_ok=True)
        for session_file in selected:
            json_dest = output / session_file.name
            shutil.copy(session_file, json_dest)
        click.echo(f"Copied {len(selected)} JSONL file(s)")

    if open_browser and fmt == "html":
        maybe_open_browser(output)


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
    include_subagents,
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
