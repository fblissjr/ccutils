"""`ccutils lake <harness>` -- mirror a harness's native store into the Parquet lake.

EXPERIMENTAL. Tier 1 only: nothing here touches a warehouse. The lake layout
may change before it is declared stable and is not covered by semver; see
docs/HARNESS_ARCHITECTURE.md for where it is going.

Exits 0 when every unit was written or unchanged, 1 when any unit errored (or
another run holds the lock), 2 on a usage error such as an unknown harness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from ..parsers.lake import LakeLockedError, write_lake
from .utils import default_lake_root

LAKE_HARNESSES = ("antigravity",)

_EXPERIMENTAL = (
    "EXPERIMENTAL: the lake layout may change before it is declared stable "
    "and is not covered by semver."
)


def _source(harness: str, stores: tuple[str, ...]):
    if harness == "antigravity":
        from ..parsers.antigravity import AntigravitySource
        from ..parsers.antigravity.stores import KNOWN_STORES
        unknown = [s for s in stores if s not in KNOWN_STORES]
        if unknown:
            raise click.UsageError(
                f"not an Antigravity store: {', '.join(unknown)} (known: {', '.join(KNOWN_STORES)})")
        return AntigravitySource(stores=stores or None)
    raise click.UsageError(f"no lake source for {harness}")  # unreachable: click.Choice


@click.command("lake")
@click.argument("harness", type=click.Choice(LAKE_HARNESSES))
@click.option(
    "-o", "--output", type=click.Path(file_okay=False, path_type=Path), default=None,
    help="Lake root (default: ~/.ccutils/lake). The harness gets a subdirectory.",  # path-privacy: ignore
)
@click.option(
    "--source", "source_root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
    help="The harness's data root (antigravity: ~/.gemini).",  # path-privacy: ignore
)
@click.option(
    "--store", "stores", multiple=True,
    help="Only this store (repeatable), e.g. antigravity-cli.",
)
@click.option("--force", is_flag=True, help="Rewrite every unit even if its source is unchanged.")
def lake_cmd(harness, output, source_root, stores, force):
    """EXPERIMENTAL: mirror another harness's transcript store into Parquet.

    A byte-faithful Tier 1 archive: every source table and file, nothing
    decoded or interpreted, no warehouse. Units whose source is unchanged are
    skipped; conversations deleted from the app are kept and marked. Reads
    are an allowlist: credential files beside the data are never opened.
    The lake is private data (transcripts can contain secrets agents
    printed); it is never written under the current directory.
    """
    lake_root = output if output is not None else default_lake_root()
    source = _source(harness, stores)
    try:
        result = write_lake(source, lake_root=lake_root, root=source_root, force=force)
    except LakeLockedError as e:
        click.echo(f"{e}; wait for it to finish.", err=True)
        sys.exit(1)

    click.echo(_EXPERIMENTAL)
    click.echo(f"Lake: {result.harness_dir}  (run {result.lake_run_id})")
    if result.context.get("app_version"):
        click.echo(f"Installed app version: {result.context['app_version']}")

    by_store: dict[str, dict[str, int]] = {}
    rows: dict[str, int] = {}
    for o in result.outcomes:
        c = by_store.setdefault(o.store, {})
        c[o.status] = c.get(o.status, 0) + 1
        for table, n in o.tables.items():
            rows[table] = rows.get(table, 0) + n
    width = max((len(s) for s in by_store), default=0)
    for store in sorted(by_store):
        c = by_store[store]
        click.echo(
            f"  {store:<{width}}  written {c.get('written', 0)}  unchanged {c.get('unchanged', 0)}"
            f"  superseded {c.get('superseded', 0)}  error {c.get('error', 0)}"
        )
    if result.missing:
        click.echo(f"  source gone, kept in the archive: {len(result.missing)}")
    if rows:
        click.echo("Rows written: " + ", ".join(f"{t} {n:,}" for t, n in sorted(rows.items())))
    drift = [f"{o.store}/{o.unit_id}: {n}" for o in result.outcomes for n in o.notes]
    for heading, lines in (("Notes", result.notes), ("Schema drift", drift)):
        if lines:
            click.echo(f"{heading}:")
            for line in lines:
                click.echo(f"  - {line}")
    errors = [o for o in result.outcomes if o.status == "error"]
    if errors:
        click.echo("Errors (previous lake copy kept; retried next run):", err=True)
        for o in errors:
            click.echo(f"  - {o.store}/{o.unit_kind}/{o.unit_id}: {o.error}", err=True)
        sys.exit(1)
