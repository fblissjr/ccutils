"""The harness-generic lake runner (parsers/lake.py), through a toy source.

The runner owns everything that is not specific to a harness: change
detection, atomic unit replacement, archive semantics, the manifest, the run
log and the lock. Antigravity is its only real source so far; this toy source
is the second implementation, so the runner's contract is exercised without
Antigravity's details leaking into it.

Toy contract: every `<store>/<name>.txt` under the root is one unit, written
as `lines.parquet` with one row per line and `idx` = line number. An optional
`<store>/<name>.tags` sidecar becomes `tags.parquet`, one row per tag, keyed on
`tag` for carry-forward. A unit is superseded when its `idx` set shrinks.
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

    def __init__(self, fail_on: str | None = None, discover_raises: bool = False,
                 stores: tuple[str, ...] | None = None):
        self.fail_on = fail_on
        self.discover_raises = discover_raises
        self.stores = stores
        self.written: list[str] = []

    def discover(self, root):
        if self.discover_raises:
            raise RuntimeError("discovery exploded")
        units = []
        scanned = sorted({p.name for p in root.iterdir() if p.is_dir()
                          and (self.stores is None or p.name in self.stores)})
        for store in scanned:
            for p in sorted((root / store).glob("*.txt")):
                rels = [p.name] + ([f"{p.stem}.tags"] if p.with_suffix(".tags").exists() else [])
                units.append(LakeUnit(
                    store=store, unit_kind="doc", unit_id=p.stem, source_root=root / store,
                    source_relpaths=tuple(rels), out_relpath=f"{store}/docs/{p.stem}",
                    carry_keys=(("tags", "tag"),),
                ))
        return Discovery(units=units, notes=["toy note"], context={"app_version": "9.9"},
                         scanned_stores=frozenset(scanned))

    def supersedes(self, unit, old_dir, new_dir):
        old, new = old_dir / "lines.parquet", new_dir / "lines.parquet"
        if not (old.exists() and new.exists()):
            return None
        lost = set(pq.read_table(old, columns=["idx"]).column(0).to_pylist()) - set(
            pq.read_table(new, columns=["idx"]).column(0).to_pylist())
        return f"lines: idx {sorted(lost)} gone" if lost else None

    def write_unit(self, unit, out_dir, envelope):
        self.written.append(unit.unit_id)
        schema = pa.schema(envelope_fields() + [pa.field("idx", pa.int64()), pa.field("text", pa.string())])
        w = TableWriter(out_dir / "lines.parquet", schema)
        text = (unit.source_root / unit.source_relpaths[0]).read_text()
        for i, line in enumerate(text.splitlines()):
            w.add({**envelope.row("claude_code_jsonl", unit.source_relpaths[0]), "idx": i, "text": line})
            if self.fail_on == unit.unit_id and i == 0:
                raise RuntimeError("crash mid-unit")
        tables = {"lines": w.close()}
        if len(unit.source_relpaths) > 1:
            tag_rel = unit.source_relpaths[1]
            tw = TableWriter(out_dir / "tags.parquet", pa.schema(envelope_fields() + [pa.field("tag", pa.string())]))
            for tag in (unit.source_root / tag_rel).read_text().split():
                tw.add({**envelope.row("claude_code_jsonl", tag_rel), "tag": tag})
            tables["tags"] = tw.close()
        return UnitWrite(tables=tables, snapshot="read")


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "src"
    (r / "s1").mkdir(parents=True)
    (r / "s1" / "a.txt").write_text("one\ntwo\n")
    (r / "s1" / "b.txt").write_text("x\n")
    (r / "s1" / "a.tags").write_text("red green\n")
    (r / "s2").mkdir()
    (r / "s2" / "c.txt").write_text("other store\n")
    return r


@pytest.fixture
def lake_root(tmp_path):
    return tmp_path / "lake"


def _status(result):
    return {o.unit_id: o.status for o in result.outcomes}


def _lines(lake_root, unit="a"):
    return pq.read_table(lake_root / "toy" / "s1" / "docs" / unit / "lines.parquet").to_pylist()


def _tags(lake_root, unit="a"):
    return {r["tag"]: r for r in pq.read_table(lake_root / "toy" / "s1" / "docs" / unit / "tags.parquet").to_pylist()}


class TestWriteAndSkip:
    def test_first_run_writes_every_unit_with_envelope(self, root, lake_root):
        result = write_lake(ToySource(), lake_root=lake_root, root=root)
        assert _status(result) == {"a": "written", "b": "written", "c": "written"}
        rows = _lines(lake_root)
        assert [r["text"] for r in rows] == ["one", "two"]
        r0 = rows[0]
        assert r0["harness"] == "toy" and r0["store"] == "s1"
        assert r0["record_source"] == "claude_code_jsonl"
        assert r0["source_relpath"] == "a.txt"
        assert r0["lake_run_id"] == result.lake_run_id
        assert r0["lake_format_version"] == 1
        assert r0["source_present"] is True

    def test_second_run_is_a_no_op(self, root, lake_root):
        write_lake(ToySource(), lake_root=lake_root, root=root)
        path = lake_root / "toy" / "s1" / "docs" / "a" / "lines.parquet"
        before = path.stat().st_mtime_ns
        src = ToySource()
        result = write_lake(src, lake_root=lake_root, root=root)
        assert _status(result) == {"a": "unchanged", "b": "unchanged", "c": "unchanged"}
        assert src.written == []
        assert path.stat().st_mtime_ns == before

    def test_force_rewrites(self, root, lake_root):
        write_lake(ToySource(), lake_root=lake_root, root=root)
        result = write_lake(ToySource(), lake_root=lake_root, root=root, force=True)
        assert _status(result) == {"a": "written", "b": "written", "c": "written"}

    def test_changed_source_is_rewritten(self, root, lake_root):
        write_lake(ToySource(), lake_root=lake_root, root=root)
        (root / "s1" / "a.txt").write_text("one\ntwo\nthree\n")
        result = write_lake(ToySource(), lake_root=lake_root, root=root)
        assert _status(result) == {"a": "written", "b": "unchanged", "c": "unchanged"}
        assert len(_lines(lake_root)) == 3

    def test_format_version_bump_reingests(self, root, lake_root):
        # A widened output contract is invisible to a source fingerprint; the
        # format version is what makes an existing lake re-derive.
        write_lake(ToySource(), lake_root=lake_root, root=root)
        bumped = ToySource()
        bumped.lake_format_version = 2
        result = write_lake(bumped, lake_root=lake_root, root=root)
        assert _status(result) == {"a": "written", "b": "written", "c": "written"}
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

    def test_missing_units_rows_are_marked_absent(self, root, lake_root):
        # Row-level source_present stays truthful: a kept unit whose source is
        # gone does not keep claiming its rows are still there.
        write_lake(ToySource(), lake_root=lake_root, root=root)
        (root / "s1" / "b.txt").unlink()
        write_lake(ToySource(), lake_root=lake_root, root=root)
        rows = _lines(lake_root, "b")
        assert [r["text"] for r in rows] == ["x"]
        assert {r["source_present"] for r in rows} == {False}
        # ...and a third run neither rewrites it nor flips it back.
        path = lake_root / "toy" / "s1" / "docs" / "b" / "lines.parquet"
        before = path.stat().st_mtime_ns
        write_lake(ToySource(), lake_root=lake_root, root=root)
        assert path.stat().st_mtime_ns == before

    def test_a_failure_marking_a_unit_absent_is_recorded_and_retried(self, root, lake_root, monkeypatch):
        write_lake(ToySource(), lake_root=lake_root, root=root)
        (root / "s1" / "b.txt").unlink()
        real = lake._mark_unit_absent
        monkeypatch.setattr(lake, "_mark_unit_absent", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
        result = write_lake(ToySource(), lake_root=lake_root, root=root)  # does not raise
        row = {r["unit_id"]: r for r in read_manifest(lake_root / "toy")}["b"]
        assert row["status"] == "error" and "disk full" in row["error"]
        assert ("s1", "doc", "b") in result.missing
        monkeypatch.setattr(lake, "_mark_unit_absent", real)
        write_lake(ToySource(), lake_root=lake_root, root=root)
        assert {r["source_present"] for r in _lines(lake_root, "b")} == {False}

    def test_a_returning_source_is_rewritten_even_with_the_same_stat(self, root, lake_root, monkeypatch):
        # A restored file (backup, re-mount) can come back with the stat it
        # had; the unchanged-skip must not leave its rows saying "absent".
        write_lake(ToySource(), lake_root=lake_root, root=root)
        hidden = root / "hidden-b.txt"
        os.replace(root / "s1" / "b.txt", hidden)
        write_lake(ToySource(), lake_root=lake_root, root=root)
        os.replace(hidden, root / "s1" / "b.txt")  # same inode, size and mtime
        result = write_lake(ToySource(), lake_root=lake_root, root=root)
        assert _status(result)["b"] == "written"
        assert {r["source_present"] for r in _lines(lake_root, "b")} == {True}

    def test_an_unscanned_store_is_not_marked_missing(self, root, lake_root):
        # `--store s1` says nothing about s2; its units must not be reported gone.
        write_lake(ToySource(), lake_root=lake_root, root=root)
        result = write_lake(ToySource(stores=("s1",)), lake_root=lake_root, root=root)
        assert result.missing == []
        manifest = {r["unit_id"]: r for r in read_manifest(lake_root / "toy")}
        assert manifest["c"]["source_present"] is True
        assert manifest["c"]["status"] != "source_missing"

    def test_a_lost_keyed_row_is_carried_forward_marked_absent(self, root, lake_root):
        write_lake(ToySource(), lake_root=lake_root, root=root)
        (root / "s1" / "a.tags").write_text("green blue\n")
        result = write_lake(ToySource(), lake_root=lake_root, root=root)
        assert _status(result)["a"] == "written"
        tags = _tags(lake_root)
        assert set(tags) == {"red", "green", "blue"}
        assert tags["red"]["source_present"] is False
        assert tags["green"]["source_present"] is True and tags["blue"]["source_present"] is True
        # The carried row keeps the envelope of the run that actually read it.
        first_run = read_runs(lake_root / "toy")[0]["lake_run_id"]
        assert tags["red"]["lake_run_id"] == first_run
        assert {o.unit_id: o.carried for o in result.outcomes}["a"] == {"tags": 1}

    def test_a_carried_row_that_reappears_is_present_again(self, root, lake_root):
        write_lake(ToySource(), lake_root=lake_root, root=root)
        (root / "s1" / "a.tags").write_text("green\n")
        write_lake(ToySource(), lake_root=lake_root, root=root)
        (root / "s1" / "a.tags").write_text("green red\n")
        write_lake(ToySource(), lake_root=lake_root, root=root)
        tags = _tags(lake_root)
        assert len(tags) == 2 and tags["red"]["source_present"] is True

    def test_a_vanished_table_is_carried_whole(self, root, lake_root):
        write_lake(ToySource(), lake_root=lake_root, root=root)
        (root / "s1" / "a.tags").unlink()
        (root / "s1" / "a.txt").write_text("one\ntwo\nthree\n")
        result = write_lake(ToySource(), lake_root=lake_root, root=root)
        assert _status(result)["a"] == "written"
        tags = _tags(lake_root)
        assert set(tags) == {"red", "green"}
        assert {r["source_present"] for r in tags.values()} == {False}
        assert {r["source_present"] for r in _lines(lake_root)} == {True}

    def test_carry_across_a_format_bump_fills_the_new_columns(self, root, lake_root):
        # A v1 table written before source_present existed must still carry.
        write_lake(ToySource(), lake_root=lake_root, root=root)
        path = lake_root / "toy" / "s1" / "docs" / "a" / "tags.parquet"
        pq.write_table(pq.read_table(path).drop_columns(["source_present"]), path)
        bumped = ToySource()
        bumped.lake_format_version = 2
        (root / "s1" / "a.tags").write_text("green\n")
        write_lake(bumped, lake_root=lake_root, root=root)
        tags = _tags(lake_root)
        assert tags["red"]["source_present"] is False and tags["red"]["lake_format_version"] == 1

    def test_a_carry_that_would_drop_a_column_supersedes_instead(self, root, lake_root):
        write_lake(ToySource(), lake_root=lake_root, root=root)
        path = lake_root / "toy" / "s1" / "docs" / "a" / "tags.parquet"
        old = pq.read_table(path)
        pq.write_table(old.append_column("legacy", pa.array(["kept"] * old.num_rows)), path)
        (root / "s1" / "a.tags").write_text("green\n")
        result = write_lake(ToySource(), lake_root=lake_root, root=root)
        assert _status(result)["a"] == "superseded"
        kept = list((lake_root / "toy" / "s1" / "docs" / "_superseded" / "a").glob("*/tags.parquet"))
        assert len(kept) == 1 and "legacy" in pq.read_table(kept[0]).column_names

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
        assert '"written": 3' in runs[0]["counts"]
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
