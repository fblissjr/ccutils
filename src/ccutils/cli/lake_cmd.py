"""`ccutils lake <harness>` -- mirror a harness's native store into the Parquet lake.

EXPERIMENTAL. Tier 1 only: nothing here touches a warehouse. The lake layout
may change before it is declared stable and is not covered by semver; see
docs/HARNESS_ARCHITECTURE.md for where it is going.

Exits 0 when every unit was written or unchanged, 1 when any unit errored (or
another run holds the lock), 2 on a usage error such as an unknown harness.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from typing import Callable, TextIO

import click

from ..parsers.lake import LakeLockedError, UnitEvent, write_lake
from .utils import default_lake_root

LAKE_HARNESSES = ("antigravity",)

_EXPERIMENTAL = (
    "EXPERIMENTAL: the lake layout may change before it is declared stable "
    "and is not covered by semver."
)


class Reporter:
    """Shows a run as it happens, on stderr so the summary on stdout is the
    same with or without it.

    `verbose` writes one line for every unit that was worked on (never for an
    unchanged one). Otherwise `live`, which the command sets when stderr is a
    terminal, redraws a single line in place naming the unit being worked on;
    `clear` removes it before anything else is printed. With neither, nothing
    is written, so a run under a scheduler logs what it always did.
    """

    _LABELS = {"mirror": "Mirroring", "gone": "Marking gone", "decode": "Decoding"}
    _REDRAW_SECONDS = 0.1

    def __init__(self, stream: TextIO, *, live: bool, verbose: bool,
                 width: Callable[[], int] | None = None, clock: Callable[[], float] = time.monotonic):
        self._stream = stream
        self._live = live and not verbose
        self._verbose = verbose
        self._width = width or (lambda: shutil.get_terminal_size().columns)
        self._clock = clock
        self._drawn = False
        self._last_draw = float("-inf")

    def phase(self, text: str) -> None:
        """A stretch of work that reports no units, such as scanning the source."""
        if self._verbose:
            self._stream.write(f"{text}\n")
            self._stream.flush()
        elif self._live:
            self._draw(text)

    def unit(self, event: UnitEvent) -> None:
        if self._verbose:
            if event.status is None or event.status == "unchanged":
                return
            count = f"[{event.index:>{len(str(event.total))}}/{event.total}]"
            detail = event.error or ", ".join(
                f"{table.replace('_', ' ')} {n:,}" for table, n in sorted(event.rows.items()) if n)
            status = "source gone, marked" if event.status == "source_missing" else event.status
            line = f"  {count} {event.phase} {event.store}/{event.unit_kind}/{event.unit_id}  {status}"
            self._stream.write(f"{line}  {detail}".rstrip() + "\n")
            self._stream.flush()
        elif self._live:
            # A unit starting work is always drawn, because the wait follows it.
            # Units that end at once (unchanged) would redraw faster than a
            # terminal shows, so those are drawn at intervals.
            if event.status is not None and self._clock() - self._last_draw < self._REDRAW_SECONDS:
                return
            label = self._LABELS.get(event.phase, event.phase)
            self._draw(f"{label} [{event.index}/{event.total}] {event.store} {event.unit_kind} {event.unit_id}")

    def _draw(self, text: str) -> None:
        room = max(self._width() - 1, 10)
        if len(text) > room:
            text = text[:room - 3] + "..."
        self._stream.write(f"\r{text}\x1b[K")
        self._stream.flush()
        self._drawn = True
        self._last_draw = self._clock()

    def clear(self) -> None:
        if self._drawn:
            self._stream.write("\r\x1b[K")
            self._stream.flush()
            self._drawn = False


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
@click.option(
    "--decode", is_flag=True,
    help="After mirroring, derive readable tables and views beside the lake "
         "(<lake root>/antigravity-decoded/, re-derivable, never inside the archive).",
)
@click.option(
    "--no-thinking", "no_thinking", is_flag=True,
    help="With --decode: write no thinking text into anything derived. The lake keeps it.",
)
@click.option(
    "-v", "--verbose", is_flag=True,
    help="Print a line on stderr for every unit as it is written, marked or decoded. "
         "Without it, a terminal shows one progress line that is redrawn in place.",
)
def lake_cmd(harness, output, source_root, stores, force, decode, no_thinking, verbose):
    """EXPERIMENTAL: mirror another harness's transcript store into Parquet.

    A byte-faithful Tier 1 archive: every source table and file, nothing
    decoded or interpreted, no warehouse. Units whose source is unchanged are
    skipped; conversations deleted from the app are kept and marked. Reads
    are an allowlist: credential files beside the data are never opened.
    The lake is private data (transcripts can contain secrets agents
    printed); it is never written under the current directory.
    """
    if no_thinking and not decode:
        raise click.UsageError(
            "--no-thinking applies to --decode. The lake is a byte-faithful archive and keeps "
            "thinking text; there is nothing for the flag to do without --decode.")
    lake_root = output if output is not None else default_lake_root()
    source = _source(harness, stores)
    reporter = Reporter(sys.stderr, live=sys.stderr.isatty(), verbose=verbose)
    reporter.phase(f"Scanning the {harness} store")
    try:
        result = write_lake(source, lake_root=lake_root, root=source_root, force=force, on_unit=reporter.unit)
    except LakeLockedError as e:
        reporter.clear()
        click.echo(f"{e}; wait for it to finish.", err=True)
        sys.exit(1)
    finally:
        reporter.clear()

    click.echo(_EXPERIMENTAL)
    click.echo(f"Lake: {result.harness_dir}  (run {result.lake_run_id})")
    if result.context.get("app_version"):
        click.echo(f"Installed app version: {result.context['app_version']}")

    by_store: dict[str, dict[str, int]] = {}
    rows: dict[str, int] = {}
    carried: dict[str, int] = {}
    for o in result.outcomes:
        c = by_store.setdefault(o.store, {})
        c[o.status] = c.get(o.status, 0) + 1
        for table, n in o.tables.items():
            rows[table] = rows.get(table, 0) + n
        for table, n in o.carried.items():
            carried[table] = carried.get(table, 0) + n
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
    if carried:
        click.echo("Carried forward (gone from or emptied in the source, kept marked source_present=false): "
                   + ", ".join(f"{t} {n:,}" for t, n in sorted(carried.items())))
    unit_notes = [f"{o.store}/{o.unit_kind}/{o.unit_id}: {n}" for o in result.outcomes for n in o.notes]
    for heading, lines in (("Notes", result.notes), ("Unit notes (schema drift, supersedes)", unit_notes)):
        if lines:
            click.echo(f"{heading}:")
            for line in lines:
                click.echo(f"  - {line}")
    errors = [o for o in result.outcomes if o.status == "error"]
    if errors:
        click.echo("Errors (previous lake copy kept; retried next run):", err=True)
        for o in errors:
            click.echo(f"  - {o.store}/{o.unit_kind}/{o.unit_id}: {o.error}", err=True)
    decode_errors = _decode(lake_root, include_thinking=not no_thinking, reporter=reporter) if decode else []
    if errors or decode_errors:
        sys.exit(1)


def _decode(lake_root: Path, *, include_thinking: bool, reporter: Reporter) -> list[str]:
    """Derive the readable tables and the views database; returns the units that failed."""
    from ..parsers.antigravity import shaped

    reporter.phase("Decoding the lake")
    try:
        shaped_result = shaped.shape_lake(lake_root, include_thinking=include_thinking, on_unit=reporter.unit)
    finally:
        reporter.clear()
    click.echo(
        f"Decoded: {shaped_result.tree}  units written {shaped_result.written}  "
        f"unchanged {shaped_result.unchanged}  steps {shaped_result.steps:,}  "
        f"transcript lines {shaped_result.transcript_lines:,}"
    )
    if shaped_result.undecoded_steps or shaped_result.unparsed_lines:
        click.echo(
            f"  not decoded (kept as rows with decode_error, or counted): steps "
            f"{shaped_result.undecoded_steps:,}, transcript lines {shaped_result.unparsed_lines:,}"
        )
    if not include_thinking:
        click.echo("  no thinking text was written here; the lake itself keeps thinking text.")
    reporter.phase("Writing the views database")
    try:
        try:
            database = shaped.write_views_database(lake_root)
        finally:
            reporter.clear()
        click.echo(f"Views: {database}  ({', '.join(shaped.VIEW_NAMES)})")
    except FileNotFoundError as e:
        click.echo(f"Views not written: {e}", err=True)
    if shaped_result.errors:
        click.echo("Decode errors (the unit keeps its previous derived copy, if any):", err=True)
        for line in shaped_result.errors:
            click.echo(f"  - {line}", err=True)
    return shaped_result.errors
