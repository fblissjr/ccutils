"""Import command for Claude.ai account exports."""

import html
import tempfile
from pathlib import Path

import click

from ..parsers.claude_ai import parse_claude_ai_export, load_export_files
from ..export import generate_html
from ..schemas.simple import create_duckdb_schema, export_session_to_duckdb
from .utils import maybe_open_browser


@click.command("import")
@click.argument("export_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output path. For HTML: directory. For DuckDB: .duckdb file.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["html", "duckdb"]),
    default="html",
    help="Output format: html (default) or duckdb.",
)
@click.option(
    "--conversation-id",
    "-c",
    "conversation_ids",
    multiple=True,
    help="Filter by conversation UUID (can specify multiple).",
)
@click.option(
    "--include-thinking/--no-thinking",
    default=True,
    help="Include thinking blocks (default: yes).",
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Interactively select conversations to export.",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open the result in browser after export (HTML only).",
)
@click.option(
    "--list",
    "list_only",
    is_flag=True,
    help="List conversations in the export without converting.",
)
def import_cmd(
    export_path,
    output,
    output_format,
    conversation_ids,
    include_thinking,
    interactive,
    open_browser,
    list_only,
):
    """Import a Claude.ai account export (from Settings > Privacy).

    EXPORT_PATH should be the directory containing conversations.json,
    projects.json, etc.

    Examples:

    \b
      # Export to HTML (opens in browser)
      ccutils import ./my-claude-export --open

    \b
      # Export specific conversations to DuckDB
      ccutils import ./export -c abc123 -c def456 --format duckdb -o data.duckdb

    \b
      # Interactive selection
      ccutils import ./export --interactive
    """
    export_path = Path(export_path)

    # Validate export directory
    if not (export_path / "conversations.json").exists():
        raise click.ClickException(
            f"conversations.json not found in {export_path}. "
            "This doesn't appear to be a valid Claude.ai export."
        )

    # Load export data for listing/interactive modes
    if list_only or interactive:
        data = load_export_files(export_path)
        conversations = data["conversations"]

        if not conversations:
            click.echo("No conversations found in export.")
            return

        if list_only:
            _list_conversations(conversations)
            return

        if interactive:
            conversation_ids = _interactive_select(conversations)
            if not conversation_ids:
                click.echo("No conversations selected.")
                return

    # Convert conversation_ids tuple to list (or None)
    conv_filter = list(conversation_ids) if conversation_ids else None

    # Parse the export
    click.echo(f"Parsing Claude.ai export from {export_path}...")
    parsed = parse_claude_ai_export(
        export_path,
        conversation_ids=conv_filter,
        include_thinking=include_thinking,
    )

    loglines = parsed["loglines"]

    if not loglines:
        click.echo("No messages found to export.")
        return

    # Count conversations
    session_ids = set(ll.get("sessionId") for ll in loglines)
    click.echo(
        f"Found {len(loglines)} messages across {len(session_ids)} conversations"
    )

    if output_format == "html":
        _export_to_html(parsed, output, open_browser)
    elif output_format == "duckdb":
        _export_to_duckdb(parsed, output, include_thinking)


def _list_conversations(conversations):
    """List all conversations in the export."""
    click.echo(f"\nFound {len(conversations)} conversations:\n")
    for conv in sorted(
        conversations, key=lambda c: c.get("updated_at", ""), reverse=True
    ):
        name = conv.get("name", "(untitled)")
        uuid = conv.get("uuid", "")
        msg_count = len(conv.get("chat_messages", []))
        updated = conv.get("updated_at", "")[:10]  # Just the date
        click.echo(f"  {uuid[:8]}  {updated}  ({msg_count:3d} msgs)  {name[:60]}")


def _interactive_select(conversations):
    """Interactively select conversations using questionary."""
    try:
        import questionary
    except ImportError:
        raise click.ClickException(
            "Interactive mode requires questionary. Install with: uv add questionary"
        )

    # Build styled choices
    from ..tui.selection import build_import_choices
    from ..tui.theme import questionary_style

    choices = build_import_choices(conversations)

    # Multi-select
    selected = questionary.checkbox(
        "Select conversations to export:",
        choices=choices,
        style=questionary_style(),
    ).ask()

    return selected if selected else []


def _export_to_html(parsed, output, open_browser):
    """Export parsed data to HTML."""
    loglines = parsed["loglines"]
    sessions = _group_loglines_by_session(loglines)
    auto_open = output is None

    output = (
        Path(output) if output else Path(tempfile.gettempdir()) / "claude-ai-export"
    )
    output.mkdir(parents=True, exist_ok=True)

    for session_id, session_loglines in sessions.items():
        session_name = session_id[:8]
        session_output = output / session_name
        generate_html(loglines=session_loglines, output_dir=session_output)
        click.echo(f"  Generated: {session_output}")

    # Create index if multiple sessions
    if len(sessions) > 1:
        _create_multi_session_index(output, sessions, parsed["_metadata"])

    click.echo(f"\nOutput: {output.resolve()}")

    if open_browser or auto_open:
        if len(sessions) == 1:
            session_name = list(sessions.keys())[0][:8]
            maybe_open_browser(output / session_name)
        else:
            maybe_open_browser(output)


def _create_multi_session_index(output_dir, sessions, metadata):
    """Create an index.html linking to all session directories."""
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Claude.ai Export</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
        h1 { color: #1a1a2e; }
        .session { padding: 12px; margin: 8px 0; background: #f5f5f5; border-radius: 8px; }
        .session a { color: #4a4a6a; text-decoration: none; font-weight: 500; }
        .session a:hover { color: #1a1a2e; }
        .meta { color: #666; font-size: 0.9em; margin-top: 4px; }
        .stats { color: #888; font-size: 0.85em; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>Claude.ai Export</h1>
    <p class="stats">Source: Claude.ai account export | Conversations: {conv_count}</p>
    <div class="sessions">
""".format(
        conv_count=len(sessions)
    )

    for session_id, loglines in sessions.items():
        session_name = html.escape(session_id[:8])
        msg_count = len(loglines)
        # Try to get conversation name from metadata or first message
        conv_name = html.escape(session_id)  # fallback
        html_content += f"""        <div class="session">
            <a href="{session_name}/index.html">{conv_name}</a>
            <div class="meta">{msg_count} messages</div>
        </div>
"""

    html_content += """    </div>
</body>
</html>"""

    (output_dir / "index.html").write_text(html_content)


def _group_loglines_by_session(loglines):
    """Group loglines by sessionId."""
    sessions = {}
    for ll in loglines:
        sid = ll.get("sessionId", "unknown")
        if sid not in sessions:
            sessions[sid] = []
        sessions[sid].append(ll)
    return sessions


def _resolve_db_path(output):
    """Resolve DuckDB output path from user-provided output argument."""
    if output is None:
        return Path("claude-ai-export.duckdb")
    p = Path(output)
    if not p.suffix:
        return p.with_suffix(".duckdb")
    return p


def _export_to_duckdb(parsed, output, include_thinking):
    """Export parsed data to DuckDB."""
    loglines = parsed["loglines"]
    sessions = _group_loglines_by_session(loglines)

    db_path = _resolve_db_path(output)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"Creating DuckDB database: {db_path}")
    conn = create_duckdb_schema(db_path)

    for session_id, session_loglines in sessions.items():
        export_session_to_duckdb(
            conn,
            session_path=None,
            project_name="Claude.ai Import",
            include_thinking=include_thinking,
            loglines=session_loglines,
            session_id_override=session_id,
        )

    conn.close()
    click.echo(f"Exported {len(sessions)} conversations to {db_path}")
