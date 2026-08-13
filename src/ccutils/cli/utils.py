"""Utility functions for CLI commands."""

import platform
import sys
import tempfile
import webbrowser
from pathlib import Path

import click
import httpx

from ..api import (
    CredentialsError,
    get_access_token_from_keychain,
    get_org_uuid_from_config,
    resolve_anthropic_key,
)
from ..etl.facets import AnthropicFacetExtractor


def default_archive_output():
    """Where an archive lands when the user passes no ``-o/--output``.

    A generated archive contains unredacted Claude Code transcripts for
    EVERY project on the machine, so the default location is a privacy
    decision. It must never resolve inside a git worktree: a cwd-relative
    default writes machine-wide transcript data into whatever checkout the
    command was run from, one ``git add -A`` from being published, with a
    single .gitignore line as the only guard. Home-anchored and absolute,
    so it is the same directory wherever the tool is invoked from.

    Resolved per call rather than at import so a changed home directory
    (and test sandboxes) is honored. ``-o/--output`` overrides it as before.
    """
    return Path.home() / ".ccutils" / "claude-archive"


def build_facet_extractor_or_exit(with_llm_facets: bool):
    """Resolve Anthropic credentials and construct an AnthropicFacetExtractor
    at the CLI boundary. CredentialsError surfaces as a helpful message +
    non-zero exit code rather than a stack trace deep in the ETL.

    Returns None when the flag is off (default), keeping the basic
    pipeline credential-free.

    Shared by `local_cmd` and `all_cmd` so credential resolution and the
    error message stay in lockstep. Any change to the error wording or
    keychain service name happens here once.
    """
    if not with_llm_facets:
        return None
    try:
        api_key = resolve_anthropic_key()
    except CredentialsError as e:
        click.echo(str(e), err=True)
        sys.exit(2)
    return AnthropicFacetExtractor(api_key=api_key)


def warn_private_best_effort():
    """One-time notice that --private sanitization is best-effort.

    PathSanitizer only rewrites cwd/home-prefixed paths in a subset of
    channels (tool_use inputs + string tool_results); message text,
    thinking blocks, non-message entries, the batch search index, and
    foreign/pasted paths are NOT sanitized. Callers should review output
    before sharing. See the --private known-limitations note in README.
    """
    click.echo(
        "Note: --private is best-effort -- it masks cwd/home paths in a "
        "subset of fields, not message text, thinking, or the batch search "
        "index. Review the output before sharing.",
        err=True,
    )


def is_url(path):
    """Check if a path is a URL (starts with http:// or https://)."""
    return path.startswith("http://") or path.startswith("https://")


def fetch_url_to_tempfile(url):
    """Fetch a URL and save to a temporary file.

    Returns the Path to the temporary file.
    Raises click.ClickException on network errors.
    """
    try:
        response = httpx.get(url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
    except httpx.RequestError as e:
        raise click.ClickException(f"Failed to fetch URL: {e}")
    except httpx.HTTPStatusError as e:
        raise click.ClickException(
            f"Failed to fetch URL: {e.response.status_code} {e.response.reason_phrase}"
        )

    # Determine file extension from URL
    url_path = url.split("?")[0]  # Remove query params
    if url_path.endswith(".jsonl"):
        suffix = ".jsonl"
    elif url_path.endswith(".json"):
        suffix = ".json"
    else:
        suffix = ".jsonl"  # Default to JSONL

    # Extract a name from the URL for the temp file
    url_name = Path(url_path).stem or "session"

    temp_dir = Path(tempfile.gettempdir())
    temp_file = temp_dir / f"claude-url-{url_name}{suffix}"
    temp_file.write_text(response.text, encoding="utf-8")
    return temp_file


def resolve_credentials(token, org_uuid):
    """Resolve token and org_uuid from arguments or auto-detect.

    Returns (token, org_uuid) tuple.
    Raises click.ClickException if credentials cannot be resolved.
    """
    # Get token
    if token is None:
        token = get_access_token_from_keychain()
        if token is None:
            if platform.system() == "Darwin":
                raise click.ClickException(
                    "Could not retrieve access token from macOS keychain. "
                    "Make sure you are logged into Claude Code, or provide --token."
                )
            else:
                raise click.ClickException(
                    "On non-macOS platforms, you must provide --token with your access token."
                )

    # Get org UUID
    if org_uuid is None:
        org_uuid = get_org_uuid_from_config()
        if org_uuid is None:
            raise click.ClickException(
                "Could not find organization UUID in ~/.claude.json. "  # path-privacy: ignore
                "Provide --org-uuid with your organization UUID."
            )

    return token, org_uuid


def format_session_for_display(session_data):
    """Format a session for display in the list or picker.

    Shows repo first (if available), then date, then title.
    Returns a formatted string.
    """
    title = session_data.get("title", "Untitled")
    created_at = session_data.get("created_at", "")
    repo = session_data.get("repo")
    # Truncate title if too long
    if len(title) > 50:
        title = title[:47] + "..."
    repo_display = repo if repo else "(no repo)"
    date_display = created_at[:19] if created_at else "N/A"
    return f"{repo_display:30}  {date_display:19}  {title}"


def maybe_open_browser(output_dir):
    """Open the index.html in the output directory in the default browser.

    Args:
        output_dir: Path to the output directory containing index.html.
    """
    index_url = (output_dir / "index.html").resolve().as_uri()
    webbrowser.open(index_url)


def run_embedding_pipeline(conn, embed_model=None, quiet=False):
    """Run ColBERT embedding pipeline on a star schema connection.

    Args:
        conn: DuckDB connection with star schema.
        embed_model: Model name override, or None for default.
        quiet: Suppress output.
    """
    try:
        from ..schemas.star.embeddings import EmbeddingPipeline

        if not quiet:
            click.echo("Running ColBERT embedding pipeline...")
        pipeline = EmbeddingPipeline(model_name=embed_model)
        result = pipeline.embed_sessions(conn)
        if not quiet:
            click.echo(f"  Embedded {result['sessions_embedded']} sessions")
        match_result = pipeline.match_delegations(conn)
        if not quiet and match_result["delegations_rescored"] > 0:
            click.echo(
                f"  Re-scored {match_result['delegations_rescored']} delegations"
            )
    except ImportError:
        click.echo(
            "Warning: pylate not installed. Install with: uv add ccutils[colbert]"
        )
