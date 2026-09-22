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
- **Archive semantics.** The lake is an archive, not a cache: a unit whose
  source disappears is kept and marked ``source_present = false``; a new
  snapshot whose ``idx`` set is not a superset of the old one moves the old
  one to ``_superseded/<name>/<run>/`` instead of overwriting it.
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
    their stat is the unit's fingerprint. primary_table, when set, names a
    table with an ``idx`` column whose value set must never shrink."""

    store: str
    unit_kind: str
    unit_id: str
    source_root: Path
    source_relpaths: tuple[str, ...]
    out_relpath: str
    primary_table: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.store, self.unit_kind, self.unit_id)


@dataclass
class Discovery:
    units: list[LakeUnit]
    notes: list[str] = field(default_factory=list)
    context: dict[str, str] = field(default_factory=dict)


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
        }


class LakeSource(Protocol):
    harness_id: str
    lake_format_version: int
    record_sources: frozenset[str]

    def discover(self, root: Path | None) -> Discovery: ...

    def write_unit(self, unit: LakeUnit, out_dir: Path, envelope: Envelope) -> UnitWrite: ...


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


def _idx_set(table_path: Path) -> set[int] | None:
    if not table_path.exists():
        return None
    table = pq.read_table(table_path, columns=["idx"])
    return set(table.column("idx").to_pylist())


def _swap_in(tmp: Path, final: Path, run_id: str, primary_table: str | None) -> bool:
    """Move a finished unit into place. Returns True when the old one was superseded."""
    superseded = False
    if final.exists():
        if primary_table:
            old_idx = _idx_set(final / f"{primary_table}.parquet")
            new_idx = _idx_set(tmp / f"{primary_table}.parquet") or set()
            superseded = old_idx is not None and not old_idx <= new_idx
        if superseded:
            target = final.parent / SUPERSEDED_DIR / final.name / run_id
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(final, target)
        else:
            old = final.parent / f".{final.name}.old-{run_id}"
            os.replace(final, old)
            os.replace(tmp, final)
            shutil.rmtree(old)
            return False
    os.replace(tmp, final)
    return superseded


@dataclass
class UnitOutcome:
    store: str
    unit_kind: str
    unit_id: str
    status: str                      # written | superseded | unchanged | error
    tables: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str | None = None


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
            not force and prev is not None and prev["status"] != "error"
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
            superseded = _swap_in(tmp, final, run_id, unit.primary_table)
        except Exception as e:  # one unit failing must not stop the others
            shutil.rmtree(tmp, ignore_errors=True)
            error = f"{type(e).__name__}: {e}"
            base = prev or {
                "store": unit.store, "unit_kind": unit.unit_kind, "unit_id": unit.unit_id,
                "out_relpath": unit.out_relpath, "fingerprint": None,
                "lake_format_version": None, "parser_version": None, "snapshot": None,
                "tables": None, "notes": None, "first_ingested_at": None,
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
            "notes": json.dumps(written.notes) if written.notes else None, "error": None,
            "first_ingested_at": (prev or {}).get("first_ingested_at") or now,
            "last_ingested_at": now, "last_seen_at": now, "lake_run_id": run_id,
        }
        outcomes.append(UnitOutcome(*unit.key, status=status, tables=written.tables, notes=written.notes))

    missing = []
    for key, prev in previous.items():
        if key not in manifest:
            manifest[key] = {**prev, "status": "source_missing", "source_present": False}
            missing.append(key)

    rows = [manifest[k] for k in sorted(manifest)]
    _write_small_table(harness_dir / MANIFEST_FILE, rows, _MANIFEST_SCHEMA)
    return LakeRunResult(
        lake_run_id=run_id, harness_dir=harness_dir, outcomes=outcomes,
        missing=sorted(missing), notes=discovery.notes, context=discovery.context,
    )
