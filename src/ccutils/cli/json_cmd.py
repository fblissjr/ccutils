"""Single-file conversion command (convert JSON/JSONL to HTML, DuckDB, or JSON)."""

import shutil
import tempfile
from pathlib import Path

import click

from ..export import generate_html
from ..export import finalize_star_schema
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
from .utils import is_url, fetch_url_to_tempfile, maybe_open_browser


@click.command("convert")
@click.argument("json_file", type=click.Path())
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output path. If not specified, writes to temp dir and opens in browser.",
)
@click.option(
    "-a",
    "--output-auto",
    is_flag=True,
    help="Auto-name output subdirectory based on filename (uses -o as parent, or current dir).",
)
@click.option(
    "--repo",
    help="GitHub repo (owner/name) for commit links. Auto-detected from git push output if not specified.",
)
@click.option(
    "--json",
    "include_json",
    is_flag=True,
    help="Include the original JSON session file in the output directory (HTML mode only).",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open the generated index.html in your default browser (default if no -o specified).",
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
    "--include-thinking",
    is_flag=True,
    help="Include thinking blocks in DuckDB/JSON export (can be large).",
)
@click.option(
    "--private",
    is_flag=True,
    help="Sanitize file paths in output to remove home directory and absolute paths.",
)
def convert_cmd(
    json_file,
    output,
    output_auto,
    repo,
    include_json,
    open_browser,
    output_format,
    schema_type,
    include_thinking,
    private,
):
    """Convert a single Claude Code session JSON/JSONL file or URL.

    Supports all output formats: HTML (default), DuckDB, or JSON export
    with simple or star schema.
    """
    # Handle URL input
    if is_url(json_file):
        click.echo(f"Fetching {json_file}...")
        temp_file = fetch_url_to_tempfile(json_file)
        json_file_path = temp_file
        # Use URL path for naming
        url_name = Path(json_file.split("?")[0]).stem or "session"
    else:
        # Validate that local file exists
        json_file_path = Path(json_file)
        if not json_file_path.exists():
            raise click.ClickException(f"File not found: {json_file}")
        url_name = None

    # Resolve schema and format
    schema, fmt = resolve_schema_format(schema_type, output_format)

    # Determine output path
    auto_open = output is None and not output_auto
    if output_auto:
        parent_dir = Path(output) if output else Path(".")
        output = parent_dir / (url_name or json_file_path.stem)
    elif output is None:
        output = (
            Path(tempfile.gettempdir())
            / f"claude-session-{url_name or json_file_path.stem}"
        )

    output = Path(output)
    project_name = url_name or json_file_path.parent.name or "unknown"

    if fmt == "html":
        generate_html(json_file_path, output, github_repo=repo, private=private)
        click.echo(f"Output: {output.resolve()}")

        # Copy JSON file to output directory if requested
        if include_json:
            output.mkdir(exist_ok=True)
            json_dest = output / json_file_path.name
            shutil.copy(json_file_path, json_dest)
            json_size_kb = json_dest.stat().st_size / 1024
            click.echo(f"JSON: {json_dest} ({json_size_kb:.1f} KB)")

        if open_browser or auto_open:
            maybe_open_browser(output)

    elif fmt == "duckdb":
        db_path = (
            output.with_suffix(".duckdb") if output.suffix != ".duckdb" else output
        )
        db_path.parent.mkdir(parents=True, exist_ok=True)

        if schema == "simple":
            conn = create_duckdb_schema(db_path)
            export_session_to_duckdb(
                conn,
                json_file_path,
                project_name,
                include_thinking=include_thinking,
                private=private,
            )
            conn.close()
        else:  # star
            conn = create_star_schema(db_path)
            run_star_schema_etl(
                conn,
                json_file_path,
                project_name,
                include_thinking=include_thinking,
                private=private,
            )
            finalize_star_schema(conn)
            create_semantic_model(conn)
            conn.close()

        click.echo(f"Exported to {db_path}")

    elif fmt == "json":
        if schema == "simple":
            if output.is_dir() or output.suffix == "":
                json_path = output / "sessions.json"
            elif output.suffix != ".json":
                json_path = output.with_suffix(".json")
            else:
                json_path = output
            json_path.parent.mkdir(parents=True, exist_ok=True)
            export_sessions_to_json(
                [json_file_path],
                json_path,
                include_thinking=include_thinking,
                private=private,
            )
            click.echo(f"Exported to {json_path}")
        else:  # star
            if output.is_dir() or output.name in ("", "."):
                output_dir = output / "star_schema"
            elif output.suffix != "":
                output_dir = output.with_suffix("")
            else:
                output_dir = output
            output_dir.mkdir(parents=True, exist_ok=True)

            conn = create_star_schema(":memory:")
            run_star_schema_etl(
                conn,
                json_file_path,
                project_name,
                include_thinking=include_thinking,
                private=private,
            )
            finalize_star_schema(conn)
            create_semantic_model(conn)
            export_star_schema_to_json(conn, output_dir)
            conn.close()
            click.echo(f"Exported to {output_dir}/")
