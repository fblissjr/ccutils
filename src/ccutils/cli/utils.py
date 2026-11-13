"""Utility functions for CLI commands."""

import webbrowser
from pathlib import Path

import click


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


def default_lake_root():
    """Where `ccutils lake` writes when the user passes no ``-o/--output``.

    Home-anchored for the same reason as `default_archive_output`, and kept
    apart from it on purpose: a warehouse output dir is disposable (rebuild
    instead of migrate), while a harness lake is an archive that may hold the
    only copy of conversations the app has since deleted. Deleting a
    warehouse to rebuild it must not be able to take the archive with it.
    """
    return Path.home() / ".ccutils" / "lake"


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


def maybe_open_browser(output_dir):
    """Open the index.html in the output directory in the default browser.

    Args:
        output_dir: Path to the output directory containing index.html.
    """
    index_url = (output_dir / "index.html").resolve().as_uri()
    webbrowser.open(index_url)
