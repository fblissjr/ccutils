"""Explore command -- launches harlequin for interactive DuckDB exploration."""

import subprocess

import click


@click.command("explore")
@click.argument("database", type=click.Path(exists=True))
def explore_cmd(database):
    """Open a DuckDB database in harlequin for interactive SQL exploration.

    Requires the 'explore' optional dependency:

        uv pip install ccutils[explore]

    Examples:

        ccutils explore ./analytics/archive.duckdb
    """
    try:
        result = subprocess.run(["harlequin", database])
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    except FileNotFoundError:
        raise click.ClickException(
            "harlequin is not installed. Install it with:\n\n"
            "    uv pip install ccutils[explore]"
        )
