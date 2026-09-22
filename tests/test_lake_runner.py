"""The harness-generic lake runner (parsers/lake.py), through a toy source.

The runner owns everything that is not specific to a harness: change
detection, atomic unit replacement, archive semantics, the manifest, the run
log and the lock. Antigravity is its only real source so far; this toy source
is the second implementation, so the runner's contract is exercised without
Antigravity's details leaking into it.

Toy contract: every `<store>/<name>.txt` under the root is one unit, written
as `lines.parquet` with one row per line and `idx` = line number.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from ccutils.parsers import lake
from ccutils.parsers.lake import (
    Discovery,
    Envelope,
    LakeLockedError,
    LakeUnit,
    TableWriter,
    UnitWrite,
    envelope_fields,
    read_manifest,
    read_runs,
    write_lake,
)
import pyarrow as pa


class ToySource:
    harness_id = "toy"
    lake_format_version = 1
    record_sources = frozenset({"claude_code_jsonl"})

    def __init__(self, fail_on: str | None = None, discover_raises: bool = False):
        self.fail_on = fail_on
        self.discover_raises = discover_raises
        self.written: list[str] = []

    def discover(self, root):
        if self.discover_raises:
            raise RuntimeError("discovery exploded")
        units = []
        for p in sorted(root.glob("*/*.txt")):
            store = p.parent.name
            units.append(LakeUnit(
                store=store, unit_kind="doc", unit_id=p.stem, source_root=root / store,
                source_relpaths=(p.name,), out_relpath=f"{store}/docs/{p.stem}",
                primary_table="lines",
            ))
        return Discovery(units=units, notes=["toy note"], context={"app_version": "9.9"})

    def write_unit(self, unit, out_dir, envelope):
        self.written.append(unit.unit_id)
        schema = pa.schema(envelope_fields() + [pa.field("idx", pa.int64()), pa.field("text", pa.string())])
        w = TableWriter(out_dir / "lines.parquet", schema)
        text = (unit.source_root / unit.source_relpaths[0]).read_text()
        for i, line in enumerate(text.splitlines()):
            w.add({**envelope.row("claude_code_jsonl", unit.source_relpaths[0]), "idx": i, "text": line})
            if self.fail_on == unit.unit_id and i == 0:
                raise RuntimeError("crash mid-unit")
        rows = w.close()
        return UnitWrite(tables={"lines": rows}, snapshot="read")


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "src"
    (r / "s1").mkdir(parents=True)
    (r / "s1" / "a.txt").write_text("one\ntwo\n")
    (r / "s1" / "b.txt").write_text("x\n")
    return r


@pytest.fixture
def lake_root(tmp_path):
    return tmp_path / "lake"


def _status(result):
    return {o.unit_id: o.status for o in result.outcomes}


def _lines(lake_root, unit="a"):
    return pq.read_table(lake_root / "toy" / "s1" / "docs" / unit / "lines.parquet").to_pylist()


class TestWriteAndSkip:
    def test_first_run_writes_every_unit_with_envelope(self, root, lake_root):
        result = write_lake(ToySource(), lake_root=lake_root, root=root)
        assert _status(result) == {"a": "written", "b": "written"}
        rows = _lines(lake_root)
        assert [r["text"] for r in rows] == ["one", "two"]
        r0 = rows[0]
        assert r0["harness"] == "toy" and r0["store"] == "s1"
        assert r0["record_source"] == "claude_code_jsonl"
        assert r0["source_relpath"] == "a.txt"
        assert r0["lake_run_id"] == result.lake_run_id
        assert r0["lake_format_version"] == 1

    def test_second_run_is_a_no_op(self, root, lake_root):
        write_lake(ToySource(), lake_root=lake_root, root=root)
        path = lake_root / "toy" / "s1" / "docs" / "a" / "lines.parquet"
        before = path.stat().st_mtime_ns
        src = ToySource()
        result = write_lake(src, lake_root=lake_root, root=root)
        assert _status(result) == {"a": "unchanged", "b": "unchanged"}
        assert src.written == []
        assert path.stat().st_mtime_ns == before

    def test_force_rewrites(self, root, lake_root):
        write_lake(ToySource(), lake_root=lake_root, root=root)
        result = write_lake(ToySource(), lake_root=lake_root, root=root, force=True)
        assert _status(result) == {"a": "written", "b": "written"}

    def test_changed_source_is_rewritten(self, root, lake_root):
        write_lake(ToySource(), lake_root=lake_root, root=root)
        (root / "s1" / "a.txt").write_text("one\ntwo\nthree\n")
        result = write_lake(ToySource(), lake_root=lake_root, root=root)
        assert _status(result) == {"a": "written", "b": "unchanged"}
        assert len(_lines(lake_root)) == 3

    def test_format_version_bump_reingests(self, root, lake_root):
        # A widened output contract is invisible to a source fingerprint; the
        # format version is what makes an existing lake re-derive.
        write_lake(ToySource(), lake_root=lake_root, root=root)
        bumped = ToySource()
        bumped.lake_format_version = 2
        result = write_lake(bumped, lake_root=lake_root, root=root)
        assert _status(result) == {"a": "written", "b": "written"}
        manifest = {r["unit_id"]: r for r in read_manifest(lake_root / "toy")}
        assert manifest["a"]["lake_format_version"] == 2


class TestArchiveSemantics:
    def test_vanished_source_keeps_its_unit(self, root, lake_root):
        write_lake(ToySource(), lake_root=lake_root, root=root)
        (root / "s1" / "b.txt").unlink()
        result = write_lake(ToySource(), lake_root=lake_root, root=root)
        assert (lake_root / "toy" / "s1" / "docs" / "b" / "lines.parquet").exists()
        manifest = {r["unit_id"]: r for r in read_manifest(lake_root / "toy")}
        assert manifest["b"]["source_present"] is False
        assert manifest["b"]["status"] == "source_missing"
        assert ("s1", "doc", "b") in result.missing

    def test_idx_loss_supersedes_instead_of_overwriting(self, root, lake_root):
        write_lake(ToySource(), lake_root=lake_root, root=root)
        (root / "s1" / "a.txt").write_text("only\n")  # idx {0,1} -> {0}
        result = write_lake(ToySource(), lake_root=lake_root, root=root)
        assert _status(result)["a"] == "superseded"
        assert [r["text"] for r in _lines(lake_root)] == ["only"]
        kept = list((lake_root / "toy" / "s1" / "docs" / "_superseded" / "a").glob("*/lines.parquet"))
        assert len(kept) == 1
        assert [r["text"] for r in pq.read_table(kept[0]).to_pylist()] == ["one", "two"]

    def test_growth_does_not_supersede(self, root, lake_root):
        write_lake(ToySource(), lake_root=lake_root, root=root)
        (root / "s1" / "a.txt").write_text("one\ntwo\nthree\n")
        write_lake(ToySource(), lake_root=lake_root, root=root)
        assert not (lake_root / "toy" / "s1" / "docs" / "_superseded").exists()


class TestFailureHandling:
    def test_crash_mid_unit_leaves_previous_unit_intact(self, root, lake_root):
        write_lake(ToySource(), lake_root=lake_root, root=root)
        (root / "s1" / "a.txt").write_text("changed\nlines\n")
        result = write_lake(ToySource(fail_on="a"), lake_root=lake_root, root=root)
        assert _status(result)["a"] == "error"
        assert [r["text"] for r in _lines(lake_root)] == ["one", "two"]
        leftovers = [p for p in (lake_root / "toy" / "s1" / "docs").iterdir() if p.name.startswith(".")]
        assert leftovers == []

    def test_errored_unit_is_retried_next_run(self, root, lake_root):
        write_lake(ToySource(fail_on="a"), lake_root=lake_root, root=root)
        result = write_lake(ToySource(), lake_root=lake_root, root=root)
        assert _status(result)["a"] == "written"

    def test_stale_old_dir_is_recovered(self, root, lake_root):
        # A crash between the two renames of the swap leaves only `.a.old-*`.
        write_lake(ToySource(), lake_root=lake_root, root=root)
        docs = lake_root / "toy" / "s1" / "docs"
        os.replace(docs / "a", docs / ".a.old-deadbeef")
        (docs / ".b.tmp-deadbeef").mkdir()
        write_lake(ToySource(), lake_root=lake_root, root=root)
        assert (docs / "a" / "lines.parquet").exists()
        assert not any(p.name.startswith(".") for p in docs.iterdir())

    def test_run_row_is_opened_and_closed(self, root, lake_root):
        result = write_lake(ToySource(), lake_root=lake_root, root=root)
        runs = read_runs(lake_root / "toy")
        assert [r["status"] for r in runs] == ["complete"]
        assert runs[0]["lake_run_id"] == result.lake_run_id
        assert '"written": 2' in runs[0]["counts"]
        assert '"app_version": "9.9"' in runs[0]["context"]

    def test_exception_closes_the_run_as_failed(self, root, lake_root):
        with pytest.raises(RuntimeError):
            write_lake(ToySource(discover_raises=True), lake_root=lake_root, root=root)
        runs = read_runs(lake_root / "toy")
        assert runs[-1]["status"] == "failed"
        assert "discovery exploded" in runs[-1]["error"]

    def test_concurrent_run_refuses(self, root, lake_root):
        (lake_root / "toy").mkdir(parents=True)
        with lake._exclusive_lock(lake_root / "toy" / ".lock"):
            with pytest.raises(LakeLockedError):
                write_lake(ToySource(), lake_root=lake_root, root=root)

    def test_record_source_outside_the_sources_declaration_is_refused(self, root, lake_root):
        env = Envelope(harness="toy", store="s1", lake_run_id="r", ingested_at=lake._now(),
                       parser_version="x", lake_format_version=1,
                       allowed_record_sources=frozenset({"claude_code_jsonl"}))
        with pytest.raises(ValueError):
            env.row("history_jsonl", "a.txt")  # allow-listed globally, not declared by this source
        with pytest.raises(ValueError):
            env.row("not_a_label", "a.txt")


class TestTableWriter:
    def test_large_rows_get_their_own_row_groups(self, tmp_path):
        schema = pa.schema([pa.field("idx", pa.int64()), pa.field("blob", pa.large_binary())])
        w = TableWriter(tmp_path / "t.parquet", schema, row_group_bytes=1000)
        for i in range(5):
            w.add({"idx": i, "blob": b"x" * 600})
        assert w.close() == 5
        meta = pq.ParquetFile(tmp_path / "t.parquet").metadata
        assert meta.num_rows == 5
        assert meta.num_row_groups >= 3

    def test_empty_table_still_writes_a_file_with_schema(self, tmp_path):
        schema = pa.schema([pa.field("idx", pa.int64())])
        w = TableWriter(tmp_path / "t.parquet", schema)
        assert w.close() == 0
        assert pq.read_table(tmp_path / "t.parquet").schema.names == ["idx"]
