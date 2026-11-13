"""`ccutils audit` -- plausibility checks against a built warehouse.

Exits 0 when clean, 1 on any unallowlisted error (or warning, under
--strict), 2 when there is no warehouse or it was written by a different
schema. Nonzero on errors is the point: this command gates releases and
rewrites. The column profile is a warning by default because it fires on
every build scoped to one project; gate a full-corpus build with --strict.
"""

from __future__ import annotations

import sys

import click
import duckdb

from ..audit import format_report, run_audit
from ..schemas.star.schema import expected_schema_fingerprint, schema_fingerprint
from .open_cmd import resolve_warehouse


@click.command("audit")
@click.option(
    "-o", "--output", type=click.Path(), default=None,
    help="Directory holding the warehouse (default: the archive directory). "
         "A path to a .duckdb file also works.",
)
@click.option(
    "--strict", is_flag=True, default=False,
    help="Fail on warnings too (the column profile). For full-corpus builds.",
)
def audit_cmd(output, strict):
    """Check a built warehouse for the defect classes that have shipped.

    Errors: natural keys, API-response grain, foreign keys, declared
    coverage against what runs actually wrote, views that execute, runs
    left running or failed, and the fact grains against the Parquet lake
    beside the warehouse. Warnings: columns that are always NULL or
    single-valued.
    """
    warehouse = resolve_warehouse(output)
    if not warehouse.exists():
        click.echo(f"No warehouse at {warehouse}.", err=True)
        sys.exit(2)

    conn = duckdb.connect(str(warehouse), read_only=True)
    try:
        if schema_fingerprint(conn) != expected_schema_fingerprint():
            click.echo(
                f"{warehouse} holds a schema this version did not write; "
                "rebuild it before auditing.", err=True,
            )
            sys.exit(2)
        report = run_audit(conn, lake_root=warehouse.parent / "parquet_lake")
    finally:
        conn.close()

    click.echo(format_report(report, warehouse))
    if not (report.strict_clean if strict else report.clean):
        sys.exit(1)
