"""`ccutils guide` -- regenerate the reader's guide beside a warehouse."""

from __future__ import annotations

import sys

import click
import duckdb

from ..guide import write_guide
from ..schemas.star.schema import expected_schema_fingerprint, schema_fingerprint
from .open_cmd import resolve_warehouse


@click.command("guide")
@click.option(
    "-o", "--output", type=click.Path(), default=None,
    help="Directory holding the warehouse (default: the archive directory). "
         "A path to a .duckdb file also works.",
)
def guide_cmd(output):
    """Write README.md beside a built warehouse, generated from the warehouse.

    The guide says what this file holds: projects, sessions, every table
    and view with its status and row count, join paths, and the audit
    findings that were accepted with a reason.
    """
    warehouse = resolve_warehouse(output)
    if not warehouse.exists():
        click.echo(f"No warehouse at {warehouse}.", err=True)
        sys.exit(2)
    conn = duckdb.connect(str(warehouse), read_only=True)
    try:
        if schema_fingerprint(conn) != expected_schema_fingerprint():
            click.echo(
                f"{warehouse} holds a schema this version did not write; rebuild it first.",
                err=True,
            )
            sys.exit(2)
        path = write_guide(conn, warehouse.parent)
    finally:
        conn.close()
    click.echo(f"Wrote {path}")
