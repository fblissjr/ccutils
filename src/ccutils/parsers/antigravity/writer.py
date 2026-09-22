"""Read one Antigravity unit and write its lake tables.

Two jobs: take a consistent snapshot of a SQLite file that the app may be
writing, and mirror files byte-for-byte (or as an inventory row).

Snapshot method. The files are in WAL mode (header bytes 18/19 are 2), so
recent writes can live only in ``<name>.db-wal``. Opening the original, even
``mode=ro``, may create ``-wal``/``-shm`` files in the app's directory, and
``immutable=1`` ignores the WAL outright and can read a torn file during a
checkpoint. So the writer copies ``.db`` and ``-wal`` (never ``-shm``, which
SQLite rebuilds from the WAL) into the unit's scratch dir and opens the copy.
The copy is taken only when the stat of both files is identical before and
after copying; otherwise it retries. That is sufficient because WAL frames are
checksummed and commit-marked: a copy taken while no write happened carries
every committed transaction, and recovery on open discards any uncommitted
tail.
"""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import sqlite3
import stat as stat_mod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ccutils.parsers.lake import Envelope, TableWriter
from ccutils.parsers.antigravity import schema as ag_schema
from ccutils.parsers.antigravity import stores

FETCH_ROWS = 64
SNAPSHOT_ATTEMPTS = 5


class LakeWriteError(RuntimeError):
    """A unit cannot be mirrored faithfully (drift, type mismatch, unstable source)."""


def _stat_key(path: Path):
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return (st.st_size, st.st_mtime_ns, st.st_ino)


def snapshot_sqlite(db: Path, scratch: Path, attempts: int = SNAPSHOT_ATTEMPTS) -> tuple[Path, str]:
    """Copy a SQLite file and its WAL into `scratch`; returns (copy, method)."""
    wal = db.with_name(db.name + "-wal")
    scratch.mkdir(parents=True, exist_ok=True)
    dst = scratch / db.name
    dst_wal = dst.with_name(dst.name + "-wal")
    for attempt in range(1, attempts + 1):
        before = (_stat_key(db), _stat_key(wal))
        if before[0] is None:
            raise FileNotFoundError(db)
        for p in (dst, dst_wal):
            if p.exists():
                p.unlink()
        try:
            shutil.copyfile(db, dst)
            if before[1] is not None:
                shutil.copyfile(wal, dst_wal)
        except FileNotFoundError:
            continue  # the WAL vanished mid-copy (a checkpoint); take it again
        if (_stat_key(db), _stat_key(wal)) == before:
            return dst, f"copy (attempt {attempt}{', with wal' if before[1] else ''})"
    raise LakeWriteError(f"{db.name} kept changing while being copied ({attempts} attempts)")


def _check(value, kind: str, where: str):
    if value is None:
        return value
    expected = ag_schema.PYTHON_TYPES[kind]
    if type(value) is not expected:
        raise LakeWriteError(f"{where}: stored as {type(value).__name__}, contract says {kind}")
    return value


def mirror_sqlite(
    db_copy: Path,
    out_dir: Path,
    expected: dict[str, tuple[tuple[str, str], ...]],
    envelope: Envelope,
    record_source: str,
    source_relpath: str,
    conversation_id: str | None,
) -> tuple[dict[str, int], list[str]]:
    """Mirror every table of a SQLite copy to `<table>.parquet` in `out_dir`."""
    counts: dict[str, int] = {}
    notes: list[str] = []
    base_row = envelope.row(record_source, source_relpath)
    if conversation_id is not None:
        base_row["conversation_id"] = conversation_id
    conn = sqlite3.connect(db_copy)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        missing_tables = sorted(set(expected) - set(tables))
        if missing_tables:
            raise LakeWriteError(f"missing table(s): {', '.join(missing_tables)}")
        for table in tables:
            actual = [(r[1], r[2]) for r in conn.execute(f"PRAGMA table_info(`{table}`)")]
            declared = dict(expected.get(table, ()))
            if table not in expected:
                notes.append(f"drift: unknown table {table} mirrored")
            missing = [c for c in declared if c not in {name for name, _ in actual}]
            if missing:
                raise LakeWriteError(f"{table}: missing column(s) {', '.join(missing)}")
            columns = []
            for name, decl in actual:
                if name in declared:
                    columns.append((name, declared[name]))
                else:
                    columns.append((name, ag_schema.storage_kind(decl)))
                    if table in expected:
                        notes.append(f"drift: unknown column {table}.{name} ({decl or 'no type'}) mirrored")
            schema = ag_schema.table_schema(columns, with_conversation_id=conversation_id is not None)
            writer = TableWriter(out_dir / f"{table}.parquet", schema)
            select = ", ".join(f"`{name}`" for name, _ in columns)
            cursor = conn.execute(f"SELECT {select} FROM `{table}` ORDER BY rowid")
            try:
                while True:
                    batch = cursor.fetchmany(FETCH_ROWS)
                    if not batch:
                        break
                    for values in batch:
                        row = dict(base_row)
                        for (name, kind), value in zip(columns, values):
                            row[name] = _check(value, kind, f"{table}.{name}")
                        writer.add(row)
            finally:
                counts[table] = writer.close()
    finally:
        conn.close()
    return counts, notes


@dataclass(frozen=True)
class FileSpec:
    relpath: str          # relative to the unit's source root
    kind: str
    keep: str             # stores.BYTES | INVENTORY | CAPPED, or "symlink"
    record_source: str


def _utc(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def file_row(source_root: Path, spec: FileSpec) -> dict | None:
    """One files-table row; None if the file vanished since discovery."""
    path = source_root / spec.relpath
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    row = {"relpath": spec.relpath, "kind": spec.kind, "size_bytes": None,
           "mtime": _utc(st.st_mtime_ns), "sha256": None, "content": None}
    if stat_mod.S_ISLNK(st.st_mode) or spec.keep == "symlink":
        row["kind"] = "symlink"
        return row  # never followed, never opened
    if not stat_mod.S_ISREG(st.st_mode):
        return None
    keep_bytes = spec.keep == stores.BYTES or (
        spec.keep == stores.CAPPED and st.st_size <= stores.UNCLASSIFIED_MAX_BYTES)
    digest = hashlib.sha256()
    chunks = [] if keep_bytes else None
    size = 0
    # O_NOFOLLOW: lstat and open are two calls, and a symlink planted between
    # them must not be followed. O_NONBLOCK keeps a FIFO from hanging the open.
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError as e:
        if e.errno == errno.ELOOP:
            row["kind"] = "symlink"
            return row
        raise
    if not stat_mod.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        return None
    with os.fdopen(fd, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
    row["size_bytes"] = size
    row["sha256"] = digest.hexdigest()
    if chunks is not None:
        row["content"] = b"".join(chunks)
    return row


def write_files(
    out_path: Path,
    source_root: Path,
    specs: list[FileSpec],
    envelope: Envelope,
    conversation_id: str | None,
) -> int:
    writer = TableWriter(out_path, ag_schema.files_schema())
    try:
        for spec in specs:
            row = file_row(source_root, spec)
            if row is None:
                continue
            writer.add({**envelope.row(spec.record_source, spec.relpath),
                        "conversation_id": conversation_id, **row})
    finally:
        rows = writer.close()
    return rows
