"""Harness-generic Tier 1 lake runner.

A harness's native store (Tier 0) is mirrored into Parquet under
``<lake_root>/<harness>/`` by a `LakeSource`, which knows the harness's files.
Everything that is not harness-specific lives here:

- **Change detection.** A unit is skipped when the stat of every file it
  reads (size, mtime_ns, inode) and the source's ``lake_format_version`` are
  unchanged. There is deliberately no content hash: a SQLite checkpoint
  rewrites bytes without changing content, and a WAL write changes the
  ``-wal`` file without touching the main one, so the unit lists both.
  ``lake_format_version`` exists because a fingerprint of the SOURCE cannot
  see a change in what the WRITER emits; bump it whenever the output of a
  source changes, and the drift test beside each source makes forgetting fail.
- **Atomic replacement.** A unit is written to ``.<name>.tmp-<run>/`` and
  swapped in with renames, so a crash never leaves a unit holding files from
  two runs. Leftovers are recovered at the start of the next run.
- **Archive semantics.** The lake is an archive, not a cache: nothing it
  once held leaves the current tree except into ``_superseded/``. Every row
  carries ``source_present``, true when the run that wrote it read it from
  the source. When a unit is rewritten, rows of a keyed table
  (``LakeUnit.carry_keys``) whose key the new snapshot lacks, and whole tables
  the new snapshot lacks, are carried forward with ``source_present = false``
  and the envelope of the run that actually read them. A unit whose source
  disappears is kept and its rows are marked the same way. When the source
  says the new snapshot is not a continuation of the old one
  (`LakeSource.supersedes`, e.g. a conversation reverted and re-used step
  indices), or a carry could not keep every old column, the old unit moves to
  ``_superseded/<name>/<run>/`` whole instead of being overwritten.
- **Scope.** Only units in the stores a discovery says it scanned
  (``Discovery.scanned_stores``) can be marked missing; a run restricted to
  one store says nothing about the others.
- **Run metadata.** ``_manifest.parquet`` has one row per unit;
  ``_runs.parquet`` one row per invocation, written ``running`` at start and
  closed ``complete`` or ``failed``, so a crashed run is distinguishable from
  one that found nothing.
- **Exclusion.** An exclusive ``flock`` on ``<lake_root>/<harness>/.lock``.

The `LakeSource` interface is provisional: Claude Code joins it in Phase 2
of docs/HARNESS_ARCHITECTURE.md, and it will be revised against that second
real harness.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Protocol

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ccutils._version import PARSER_VERSION
from ccutils.provenance import record_source_label

MANIFEST_FILE = "_manifest.parquet"
RUNS_FILE = "_runs.parquet"
LOCK_FILE = ".lock"
SUPERSEDED_DIR = "_superseded"

# Row groups are capped by bytes, not rows: one Antigravity generation row is
# ~97 MB (a full prompt snapshot with images) while most are a few KB.
DEFAULT_ROW_GROUP_BYTES = 64 * 1024 * 1024


class LakeLockedError(RuntimeError):
    """Another lake run holds the lock for this harness."""


# ---------------------------------------------------------------------------
# Contract between the runner and a source
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LakeUnit:
    """One independently replaceable piece of the lake.

    source_relpaths lists EVERY file the unit reads, relative to source_root;
    their stat is the unit's fingerprint. carry_keys lists ``(table, key
    column)`` pairs: rows of that table whose key vanishes from the source are
    carried forward rather than dropped. A table not listed is carried only
    when it vanishes whole; its rows otherwise follow the source (which is
    what `LakeSource.supersedes` is for)."""

    store: str
    unit_kind: str
    unit_id: str
    source_root: Path
    source_relpaths: tuple[str, ...]
    out_relpath: str
    carry_keys: tuple[tuple[str, str], ...] = ()

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.store, self.unit_kind, self.unit_id)


@dataclass
class Discovery:
    units: list[LakeUnit]
    notes: list[str] = field(default_factory=list)
    context: dict[str, str] = field(default_factory=dict)
    # Stores this discovery looked in. A previous unit outside them is left
    # alone rather than marked missing. None means "every store".
    scanned_stores: frozenset[str] | None = None


@dataclass
class UnitWrite:
    tables: dict[str, int]
    notes: list[str] = field(default_factory=list)
    snapshot: str | None = None


def envelope_fields() -> list[pa.Field]:
    """Columns every lake row carries, first, in this order."""
    return [
        pa.field("harness", pa.string(), nullable=False),
        pa.field("store", pa.string(), nullable=False),
        pa.field("record_source", pa.string(), nullable=False),
        pa.field("source_relpath", pa.string(), nullable=False),
        pa.field("lake_run_id", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("parser_version", pa.string(), nullable=False),
        pa.field("lake_format_version", pa.int32(), nullable=False),
        pa.field("source_present", pa.bool_(), nullable=False),
    ]


@dataclass(frozen=True)
class Envelope:
    harness: str
    store: str
    lake_run_id: str
    ingested_at: datetime
    parser_version: str
    lake_format_version: int
    allowed_record_sources: frozenset[str]

    def row(self, record_source: str, source_relpath: str) -> dict:
        record_source_label(record_source)
        if record_source not in self.allowed_record_sources:
            raise ValueError(
                f"record_source {record_source!r} is allow-listed but not declared "
                f"by the {self.harness!r} lake source"
            )
        return {
            "harness": self.harness,
            "store": self.store,
            "record_source": record_source,
            "source_relpath": source_relpath,
            "lake_run_id": self.lake_run_id,
            "ingested_at": self.ingested_at,
            "parser_version": self.parser_version,
            "lake_format_version": self.lake_format_version,
            "source_present": True,
        }


class LakeSource(Protocol):
    harness_id: str
    lake_format_version: int
    record_sources: frozenset[str]

    def discover(self, root: Path | None) -> Discovery: ...

    def write_unit(self, unit: LakeUnit, out_dir: Path, envelope: Envelope) -> UnitWrite: ...

    def supersedes(self, unit: LakeUnit, old_dir: Path, new_dir: Path) -> str | None:
        """Why the new snapshot must not overwrite the old one, or None.

        Called with both finished units before the swap. Only the source
        knows what continuity means for its data (an ordered log that lost
        entries, an entry re-created under an old key); bias toward a reason,
        since a spurious one costs disk and a missed one costs data."""
        ...


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------


class TableWriter:
    """Stream rows into one Parquet file with byte-capped row groups.

    Binary columns skip dictionary encoding (a dictionary of unique blobs is
    pure overhead). A table with no rows still gets a file, so a reader can
    tell "mirrored, empty" from "never mirrored"."""

    def __init__(self, path: Path, schema: pa.Schema, row_group_bytes: int = DEFAULT_ROW_GROUP_BYTES):
        self.path = path
        self.schema = schema
        self.row_group_bytes = row_group_bytes
        self._pending: list[dict] = []
        self._pending_bytes = 0
        self._rows = 0
        dictionary_cols = [
            f.name for f in schema
            if not (pa.types.is_binary(f.type) or pa.types.is_large_binary(f.type))
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(
            str(path), schema, compression="zstd",
            use_dictionary=dictionary_cols,  # type: ignore[arg-type]  # pyarrow accepts a column list
        )

    def add(self, row: dict) -> None:
        self._pending.append(row)
        self._pending_bytes += sum(
            len(v) if isinstance(v, (bytes, str)) else 8 for v in row.values() if v is not None
        )
        if self._pending_bytes >= self.row_group_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        table = pa.Table.from_pylist(self._pending, schema=self.schema)
        self._writer.write_table(table, row_group_size=len(self._pending))
        self._rows += len(self._pending)
        self._pending = []
        self._pending_bytes = 0

    def close(self) -> int:
        self._flush()
        if self._rows == 0:
            self._writer.write_table(self.schema.empty_table())
        self._writer.close()
        return self._rows


def _write_small_table(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), str(tmp), compression="zstd")
    os.replace(tmp, path)


_MANIFEST_SCHEMA = pa.schema([
    pa.field("store", pa.string()),
    pa.field("unit_kind", pa.string()),
    pa.field("unit_id", pa.string()),
    pa.field("out_relpath", pa.string()),
    pa.field("fingerprint", pa.string()),
    pa.field("lake_format_version", pa.int32()),
    pa.field("parser_version", pa.string()),
    pa.field("status", pa.string()),
    pa.field("source_present", pa.bool_()),
    pa.field("snapshot", pa.string()),
    pa.field("tables", pa.string()),
    pa.field("carried", pa.string()),
    pa.field("notes", pa.string()),
    pa.field("error", pa.string()),
    pa.field("first_ingested_at", pa.timestamp("us", tz="UTC")),
    pa.field("last_ingested_at", pa.timestamp("us", tz="UTC")),
    pa.field("last_seen_at", pa.timestamp("us", tz="UTC")),
    pa.field("lake_run_id", pa.string()),
])

_RUNS_SCHEMA = pa.schema([
    pa.field("lake_run_id", pa.string()),
    pa.field("harness", pa.string()),
    pa.field("status", pa.string()),
    pa.field("started_at", pa.timestamp("us", tz="UTC")),
    pa.field("finished_at", pa.timestamp("us", tz="UTC")),
    pa.field("parser_version", pa.string()),
    pa.field("lake_format_version", pa.int32()),
    pa.field("source_root", pa.string()),
    pa.field("force", pa.bool_()),
    pa.field("context", pa.string()),
    pa.field("counts", pa.string()),
    pa.field("notes", pa.string()),
    pa.field("error", pa.string()),
])


def read_manifest(harness_dir: Path) -> list[dict]:
    path = harness_dir / MANIFEST_FILE
    return pq.read_table(path).to_pylist() if path.exists() else []


def read_runs(harness_dir: Path) -> list[dict]:
    path = harness_dir / RUNS_FILE
    return pq.read_table(path).to_pylist() if path.exists() else []


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LakeLockedError(f"another lake run holds {path}") from None
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def fingerprint(unit: LakeUnit) -> str:
    h = hashlib.sha256()
    for rel in sorted(unit.source_relpaths):
        try:
            st = os.stat(unit.source_root / rel, follow_symlinks=False)
            h.update(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}\0{st.st_ino}\n".encode())
        except FileNotFoundError:
            h.update(f"{rel}\0missing\n".encode())
    return h.hexdigest()


def _recover_interrupted_swaps(harness_dir: Path) -> None:
    """Undo what a crash mid-swap left behind.

    ``.<name>.tmp-<run>`` is an incomplete write: the source still has its
    data, so it is discarded. ``.<name>.old-<run>`` is the previous unit
    moved aside; it is restored when the new one never landed."""
    for tmp in list(harness_dir.rglob(".*.tmp-*")):
        if tmp.is_dir():
            shutil.rmtree(tmp)
        else:
            tmp.unlink()
    for old in list(harness_dir.rglob(".*.old-*")):
        name = old.name[1:].rsplit(".old-", 1)[0]
        final = old.parent / name
        if final.exists():
            shutil.rmtree(old)
        else:
            os.replace(old, final)


def _conform(table: pa.Table, schema: pa.Schema) -> pa.Table | None:
    """`table` in `schema`'s shape, marked absent; None if a column would be lost.

    Columns the old table lacks (a later format version added them) are
    filled with nulls; a column the new schema lacks cannot be carried without
    dropping data, and the caller supersedes instead."""
    if any(name not in schema.names for name in table.column_names):
        return None
    columns = []
    for f in schema:
        if f.name == "source_present":
            columns.append(pa.array([False] * table.num_rows, pa.bool_()))
        elif f.name in table.column_names:
            columns.append(table.column(f.name).cast(f.type))
        else:
            columns.append(pa.nulls(table.num_rows, f.type))
    return pa.Table.from_arrays(columns, schema=schema)


def _mark_absent_schema(schema: pa.Schema) -> pa.Schema:
    if "source_present" in schema.names:
        return schema
    return schema.append(pa.field("source_present", pa.bool_(), nullable=False))


def _rewrite_with(path: Path, extra: pa.Table) -> None:
    """Append `extra` to the Parquet file at `path`, streaming the original."""
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    source = pq.ParquetFile(path)
    schema = source.schema_arrow
    dictionary_cols = [f.name for f in schema
                       if not (pa.types.is_binary(f.type) or pa.types.is_large_binary(f.type))]
    with pq.ParquetWriter(str(tmp), schema, compression="zstd",
                          use_dictionary=dictionary_cols) as writer:  # type: ignore[arg-type]
        for i in range(source.num_row_groups):
            writer.write_table(source.read_row_group(i))
        writer.write_table(extra)
    os.replace(tmp, path)


def _carry_forward(old_dir: Path, new_dir: Path, carry_keys: dict[str, str]) -> tuple[dict[str, int], bool]:
    """Carry what the new snapshot lost into it. Returns (rows carried per table, lossless).

    A table missing from `new_dir` is copied whole; a keyed table gets the
    old rows whose key is absent from the new one. Every carried row is
    marked ``source_present = false`` and keeps its original envelope."""
    carried: dict[str, int] = {}
    for old_file in sorted(old_dir.glob("*.parquet")):
        table = old_file.stem
        new_file = new_dir / old_file.name
        if not new_file.exists():
            old = pq.read_table(old_file)
            conformed = _conform(old, _mark_absent_schema(old.schema))
            if conformed is None:
                return carried, False
            pq.write_table(conformed, str(new_file), compression="zstd")
            carried[table] = old.num_rows
            continue
        key = carry_keys.get(table)
        if key is None:
            continue
        new_keys = pq.read_table(new_file, columns=[key]).column(0)
        old_keys = pq.read_table(old_file, columns=[key]).column(0)
        lost_mask = pc.invert(pc.is_in(old_keys, value_set=new_keys.combine_chunks()))
        if not pc.any(lost_mask).as_py():
            continue
        lost = pq.read_table(old_file).filter(lost_mask)
        conformed = _conform(lost, pq.ParquetFile(new_file).schema_arrow)
        if conformed is None:
            return carried, False
        _rewrite_with(new_file, conformed)
        carried[table] = lost.num_rows
    return carried, True


def _swap_in(tmp: Path, final: Path, run_id: str, supersede: bool) -> None:
    """Move a finished unit into place, moving the old one to _superseded/ if asked."""
    if final.exists():
        if supersede:
            target = final.parent / SUPERSEDED_DIR / final.name / run_id
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(final, target)
        else:
            old = final.parent / f".{final.name}.old-{run_id}"
            os.replace(final, old)
            os.replace(tmp, final)
            shutil.rmtree(old)
            return
    os.replace(tmp, final)


def _mark_unit_absent(final: Path, run_id: str) -> None:
    """Rewrite a kept unit so every row says its source is gone."""
    tmp = final.parent / f".{final.name}.tmp-{run_id}"
    tmp.mkdir()
    try:
        _, lossless = _carry_forward(final, tmp, {})
        if not lossless:  # unreachable: a table conforms to its own schema
            raise RuntimeError(f"cannot mark {final} absent without losing a column")
        _swap_in(tmp, final, run_id, supersede=False)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


@dataclass
class UnitOutcome:
    store: str
    unit_kind: str
    unit_id: str
    status: str                      # written | superseded | unchanged | error
    tables: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    carried: dict[str, int] = field(default_factory=dict)


@dataclass
class LakeRunResult:
    lake_run_id: str
    harness_dir: Path
    outcomes: list[UnitOutcome]
    missing: list[tuple[str, str, str]]
    notes: list[str]
    context: dict[str, str]

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for o in self.outcomes:
            counts[o.status] = counts.get(o.status, 0) + 1
        if self.missing:
            counts["source_missing"] = len(self.missing)
        return counts

    @property
    def has_errors(self) -> bool:
        return any(o.status == "error" for o in self.outcomes)


def _write_runs(harness_dir: Path, rows: list[dict]) -> None:
    _write_small_table(harness_dir / RUNS_FILE, rows, _RUNS_SCHEMA)


def write_lake(
    source: LakeSource,
    *,
    lake_root: Path,
    root: Path | None = None,
    force: bool = False,
) -> LakeRunResult:
    """Mirror one harness's native store into ``<lake_root>/<harness>/``."""
    harness_dir = Path(lake_root) / source.harness_id
    harness_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    with _exclusive_lock(harness_dir / LOCK_FILE):
        _recover_interrupted_swaps(harness_dir)
        runs = read_runs(harness_dir)
        run_row = {
            "lake_run_id": run_id, "harness": source.harness_id, "status": "running",
            "started_at": _now(), "finished_at": None, "parser_version": PARSER_VERSION,
            "lake_format_version": source.lake_format_version,
            "source_root": str(root) if root else None, "force": force,
            "context": None, "counts": None, "notes": None, "error": None,
        }
        _write_runs(harness_dir, runs + [run_row])
        try:
            result = _run(source, harness_dir, run_id, root, force)
        except BaseException as e:
            run_row.update(status="failed", finished_at=_now(), error=f"{type(e).__name__}: {e}")
            _write_runs(harness_dir, runs + [run_row])
            raise
        run_row.update(
            status="complete", finished_at=_now(),
            context=json.dumps(result.context, sort_keys=True),
            counts=json.dumps(result.counts(), sort_keys=True),
            notes=json.dumps(result.notes),
        )
        _write_runs(harness_dir, runs + [run_row])
        return result


def _run(source: LakeSource, harness_dir: Path, run_id: str, root: Path | None, force: bool) -> LakeRunResult:
    previous = {(r["store"], r["unit_kind"], r["unit_id"]): r for r in read_manifest(harness_dir)}
    discovery = source.discover(root)
    manifest: dict[tuple[str, str, str], dict] = {}
    outcomes: list[UnitOutcome] = []
    for unit in discovery.units:
        now = _now()
        fp = fingerprint(unit)
        prev = previous.get(unit.key)
        final = harness_dir / unit.out_relpath
        if (
            not force and prev is not None and prev["status"] not in ("error", "source_missing")
            and prev["fingerprint"] == fp
            and prev["lake_format_version"] == source.lake_format_version
            and final.is_dir()
        ):
            manifest[unit.key] = {**prev, "status": "unchanged", "source_present": True,
                                  "last_seen_at": now, "lake_run_id": run_id}
            outcomes.append(UnitOutcome(*unit.key, status="unchanged"))
            continue

        envelope = Envelope(
            harness=source.harness_id, store=unit.store, lake_run_id=run_id,
            ingested_at=now, parser_version=PARSER_VERSION,
            lake_format_version=source.lake_format_version,
            allowed_record_sources=source.record_sources,
        )
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.parent / f".{final.name}.tmp-{run_id}"
        try:
            tmp.mkdir()
            written = source.write_unit(unit, tmp, envelope)
            carried: dict[str, int] = {}
            reason = None
            if final.is_dir():
                reason = source.supersedes(unit, final, tmp)
                carried, lossless = _carry_forward(final, tmp, dict(unit.carry_keys))
                if not lossless:
                    reason = reason or "carry-forward would drop an old column"
            if reason:
                written.notes.append(f"superseded: {reason}")
            _swap_in(tmp, final, run_id, supersede=reason is not None)
            superseded = reason is not None
        except Exception as e:  # one unit failing must not stop the others
            shutil.rmtree(tmp, ignore_errors=True)
            error = f"{type(e).__name__}: {e}"
            base = prev or {
                "store": unit.store, "unit_kind": unit.unit_kind, "unit_id": unit.unit_id,
                "out_relpath": unit.out_relpath, "fingerprint": None,
                "lake_format_version": None, "parser_version": None, "snapshot": None,
                "tables": None, "carried": None, "notes": None, "first_ingested_at": None,
                "last_ingested_at": None,
            }
            manifest[unit.key] = {**base, "status": "error", "source_present": True,
                                  "error": error, "last_seen_at": now, "lake_run_id": run_id}
            outcomes.append(UnitOutcome(*unit.key, status="error", error=error))
            continue

        status = "superseded" if superseded else "written"
        manifest[unit.key] = {
            "store": unit.store, "unit_kind": unit.unit_kind, "unit_id": unit.unit_id,
            "out_relpath": unit.out_relpath, "fingerprint": fp,
            "lake_format_version": source.lake_format_version,
            "parser_version": PARSER_VERSION, "status": status, "source_present": True,
            "snapshot": written.snapshot, "tables": json.dumps(written.tables, sort_keys=True),
            "carried": json.dumps(carried, sort_keys=True) if carried else None,
            "notes": json.dumps(written.notes) if written.notes else None, "error": None,
            "first_ingested_at": (prev or {}).get("first_ingested_at") or now,
            "last_ingested_at": now, "last_seen_at": now, "lake_run_id": run_id,
        }
        outcomes.append(UnitOutcome(*unit.key, status=status, tables=written.tables,
                                    notes=written.notes, carried=carried))

    missing = []
    scanned = discovery.scanned_stores
    for key, prev in previous.items():
        if key in manifest:
            continue
        if scanned is not None and prev["store"] not in scanned:
            manifest[key] = prev  # not looked at this run; nothing to say about it
            continue
        missing.append(key)
        if prev["status"] != "source_missing":
            final = harness_dir / prev["out_relpath"]
            try:
                if final.is_dir():
                    _mark_unit_absent(final, run_id)
            except Exception as e:  # the unit is untouched; retried next run
                manifest[key] = {**prev, "status": "error", "source_present": False,
                                 "error": f"marking absent: {type(e).__name__}: {e}"}
                continue
        manifest[key] = {**prev, "status": "source_missing", "source_present": False, "error": None}

    rows = [manifest[k] for k in sorted(manifest)]
    _write_small_table(harness_dir / MANIFEST_FILE, rows, _MANIFEST_SCHEMA)
    return LakeRunResult(
        lake_run_id=run_id, harness_dir=harness_dir, outcomes=outcomes,
        missing=sorted(missing), notes=discovery.notes, context=discovery.context,
    )
