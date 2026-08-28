"""`ccutils open` -- launch DuckDB's SQL UI against a built warehouse.

It launches, it does not build. The README draws that boundary deliberately:
ccutils produces a warehouse and hands it to whatever you want to query it
with, rather than growing a query surface of its own. So this command
resolves a path, checks a file is there, and execs `duckdb -ui`.

Failing loudly when there is no warehouse matters more than it looks. The
silent no-op has shipped three times in this codebase, and "opened a SQL UI
on an empty database" is exactly the shape of a failure a user reads as
"there is no data" rather than "you have not built it yet".
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import click

from .utils import default_archive_output

WAREHOUSE_NAME = "archive.duckdb"


def resolve_warehouse(output: str | None) -> Path:
    """The warehouse a bare `ccutils open` should target.

    ``-o`` names the DIRECTORY holding it, matching every other command --
    ``-o`` is a directory everywhere as of 0.20.0. A path pointing straight
    at a ``.duckdb`` file is accepted too, because typing it is the obvious
    thing to try.
    """
    if output is None:
        return default_archive_output() / WAREHOUSE_NAME
    path = Path(output).expanduser()
    if path.suffix == ".duckdb":
        return path
    return path / WAREHOUSE_NAME


@click.command("open")
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help=(
        "Directory holding the warehouse (default: the archive directory "
        "ccutils writes to). A path to a .duckdb file also works."
    ),
)
def open_cmd(output):
    """Open a built warehouse in the DuckDB SQL UI.

    Launches `duckdb -ui` against the warehouse. It does not build one --
    run a conversion with `--format duckdb` first.
    """
    warehouse = resolve_warehouse(output)

    if not warehouse.exists():
        raise click.ClickException(
            f"No warehouse at {warehouse}.\n"
            "Build one first, for example:\n"
            f"  ccutils --source --format duckdb -o {warehouse.parent}"
        )

    duckdb_bin = shutil.which("duckdb")
    if duckdb_bin is None:
        raise click.ClickException(
            "The `duckdb` CLI is not on PATH, so the SQL UI cannot be "
            "launched. Install it (https://duckdb.org/docs/installation/) "
            f"and re-run, or open {warehouse} with any DuckDB client."
        )

    click.echo(f"Opening {warehouse}")
    os.execv(duckdb_bin, [duckdb_bin, "-ui", str(warehouse)])
