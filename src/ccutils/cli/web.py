"""Web session import command."""

import tempfile
from pathlib import Path

import click
import httpx
import questionary

from ..api import (
    enrich_sessions_with_repos,
    fetch_session,
    fetch_sessions,
    filter_sessions_by_repo,
)
from ..export import generate_html
from .utils import resolve_credentials, maybe_open_browser


@click.command("web")
@click.argument("session_id", required=False)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output directory. If not specified, writes to temp dir and opens in browser.",
)
@click.option("--token", help="API access token (auto-detected from keychain on macOS)")
@click.option(
    "--org-uuid", help="Organization UUID (auto-detected from ~/.claude.json)"
)
@click.option(
    "--repo",
    help="GitHub repo (owner/name). Filters session list.",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open the generated index.html in your default browser (default if no -o specified).",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Show raw API response structure for debugging pagination.",
)
@click.option(
    "--limit",
    type=int,
    help="Request specific number of sessions per page (for debugging API).",
)
@click.option(
    "--private",
    is_flag=True,
    help="Sanitize file paths in output to remove home directory and absolute paths.",
)
def web_cmd(
    session_id,
    output,
    token,
    org_uuid,
    repo,
    open_browser,
    debug,
    limit,
    private,
):
    """Select and convert a web session from the Claude API to HTML.

    If SESSION_ID is not provided, displays an interactive picker to select a session.
    """
    try:
        token, org_uuid = resolve_credentials(token, org_uuid)
    except click.ClickException:
        raise

    # If no session ID provided, show interactive picker
    if session_id is None:
        try:
            sessions_data = fetch_sessions(token, org_uuid, debug=debug, limit=limit)
        except httpx.HTTPStatusError as e:
            raise click.ClickException(
                f"API request failed: {e.response.status_code} {e.response.text}"
            )
        except httpx.RequestError as e:
            raise click.ClickException(f"Network error: {e}")

        if debug:
            click.echo("\n=== DEBUG: API Response Structure ===")
            click.echo(f"Top-level keys: {list(sessions_data.keys())}")
            pagination_keys = ["has_more", "first_id", "last_id"]
            found = {
                k: sessions_data.get(k) for k in pagination_keys if k in sessions_data
            }
            click.echo(f"Pagination fields: {found}")
            click.echo(f"Session count: {len(sessions_data.get('data', []))}")
            if sessions_data.get("data"):
                first = sessions_data["data"][0]
                last = sessions_data["data"][-1]
                click.echo(
                    f"First session: {first.get('created_at', 'N/A')} - {first.get('id', 'N/A')[:8]}..."
                )
                click.echo(
                    f"Last session: {last.get('created_at', 'N/A')} - {last.get('id', 'N/A')[:8]}..."
                )
            click.echo("=====================================\n")

        sessions = sessions_data.get("data", [])
        if not sessions:
            raise click.ClickException("No sessions found.")

        # Enrich sessions with repo information from session metadata
        sessions = enrich_sessions_with_repos(sessions)

        # Filter by repo if specified
        if repo:
            sessions = filter_sessions_by_repo(sessions, repo)
            if not sessions:
                raise click.ClickException(f"No sessions found for repo: {repo}")

        # Build styled choices for questionary
        from ..tui.selection import build_web_session_choices
        from ..tui.theme import questionary_style

        choices = build_web_session_choices(sessions)

        selected = questionary.select(
            "Select a session to import:",
            choices=choices,
            style=questionary_style(),
        ).ask()

        if selected is None:
            # User cancelled
            raise click.ClickException("No session selected.")

        session_id = selected

    # Fetch the session
    click.echo(f"Fetching session {session_id}...")
    try:
        session_data = fetch_session(token, org_uuid, session_id)
    except httpx.HTTPStatusError as e:
        raise click.ClickException(
            f"API request failed: {e.response.status_code} {e.response.text}"
        )
    except httpx.RequestError as e:
        raise click.ClickException(f"Network error: {e}")

    # Determine output directory and whether to open browser
    auto_open = output is None
    if output is None:
        output = Path(tempfile.gettempdir()) / f"claude-session-{session_id}"

    output = Path(output)
    click.echo(f"Generating HTML in {output}/...")
    generate_html(
        output_dir=output,
        github_repo=repo,
        loglines=session_data.get("loglines", []),
        private=private,
    )

    # Show output directory
    click.echo(f"Output: {output.resolve()}")

    if open_browser or auto_open:
        maybe_open_browser(output)
