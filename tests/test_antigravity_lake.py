"""The Antigravity raw lake (parsers/antigravity/), against a realistic fixture tree.

`helpers_antigravity.make_gemini_home` builds the real layout with the real
DDL, a WAL-mode database, brain dirs with excluded subtrees, encrypted files,
an unknown and a backup store, and credential decoys locked `chmod 000`. The
lake is Tier 1: a byte-faithful mirror, so most assertions compare lake bytes
to source bytes rather than to what a test author expected them to mean.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from ccutils.parsers.antigravity import AntigravitySource
from ccutils.parsers.antigravity import schema as ag_schema
from ccutils.parsers.antigravity import stores
from ccutils.parsers.lake import read_manifest, write_lake

from helpers_antigravity import (
    CONV_A,
    CONV_B,
    CONV_CLI,
    CONV_EMPTY,
    CONV_PB,
    IMPLICIT,
    Step,
    append_steps,
    make_gemini_home,
    unlock,
    write_conversation_db,
)


@pytest.fixture
def home(tmp_path):
    h = make_gemini_home(tmp_path / "gemini")
    yield h
    h.close()
    unlock(h)


@pytest.fixture
def lake_root(tmp_path):
    return tmp_path / "lake"


def run(home, lake_root, **kw):
    src = AntigravitySource(app_bundle=None, **{k: v for k, v in kw.items() if k == "stores"})
    return write_lake(src, lake_root=lake_root, root=home.root,
                      **{k: v for k, v in kw.items() if k != "stores"})


def conv_dir(lake_root, conv, store="antigravity"):
    return lake_root / "antigravity" / store / "conversations" / conv


def table(path):
    return pq.read_table(path).to_pylist()


def manifest(lake_root):
    return {(r["store"], r["unit_kind"], r["unit_id"]): r for r in read_manifest(lake_root / "antigravity")}


# ---------------------------------------------------------------------------
# Mirror fidelity
# ---------------------------------------------------------------------------


class TestConversationMirror:
    def test_every_sqlite_table_is_mirrored_with_every_row(self, home, lake_root):
        run(home, lake_root)
        d = conv_dir(lake_root, CONV_A)
        names = {p.stem for p in d.glob("*.parquet")}
        assert set(ag_schema.CONVERSATION_TABLES) | {"files"} == names
        assert len(table(d / "steps.parquet")) == 3
        assert len(table(d / "gen_metadata.parquet")) == 2
        assert len(table(d / "trajectory_meta.parquet")) == 1
        assert len(table(d / "battle_mode_infos.parquet")) == 0  # mirrored empty, not missing

    def test_blobs_are_byte_identical_to_the_source(self, home, lake_root):
        run(home, lake_root)
        src = sqlite3.connect(home.store() / "conversations" / f"{CONV_A}.db")
        expected = {idx: (payload, meta) for idx, payload, meta in
                    src.execute("SELECT idx, step_payload, metadata FROM steps")}
        src.close()
        rows = table(conv_dir(lake_root, CONV_A) / "steps.parquet")
        assert {r["idx"]: (r["step_payload"], r["metadata"]) for r in rows} == expected

    def test_rows_carry_the_envelope_and_conversation_id(self, home, lake_root):
        result = run(home, lake_root)
        row = table(conv_dir(lake_root, CONV_A) / "steps.parquet")[0]
        assert row["harness"] == "antigravity"
        assert row["store"] == "antigravity"
        assert row["record_source"] == "antigravity_conversation_db"
        assert row["source_relpath"] == f"conversations/{CONV_A}.db"
        assert row["conversation_id"] == CONV_A
        assert row["lake_run_id"] == result.lake_run_id
        assert row["lake_format_version"] == ag_schema.LAKE_FORMAT_VERSION

    def test_nothing_is_promoted_at_tier_1(self, home, lake_root):
        # Tier 1 interprets nothing: the columns are the source's columns plus
        # the envelope and the conversation key. Decoding is Phase 1.
        run(home, lake_root)
        cols = pq.read_schema(conv_dir(lake_root, CONV_A) / "steps.parquet").names
        expected = [f.name for f in ag_schema.envelope_fields()] + ["conversation_id"] + [
            c for c, _ in ag_schema.CONVERSATION_TABLES["steps"]]
        assert cols == expected

    def test_empty_conversation_is_mirrored(self, home, lake_root):
        result = run(home, lake_root)
        assert {o.unit_id: o.status for o in result.outcomes}[CONV_EMPTY] == "written"
        assert table(conv_dir(lake_root, CONV_EMPTY) / "steps.parquet") == []

    def test_summaries_are_mirrored_per_store(self, home, lake_root):
        run(home, lake_root)
        main = table(lake_root / "antigravity" / "antigravity" / "summaries" / "conversation_summaries.parquet")
        cli = table(lake_root / "antigravity" / "antigravity-cli" / "summaries" / "conversation_summaries.parquet")
        assert {r["conversation_id"] for r in main} == {CONV_A, CONV_B, CONV_EMPTY, CONV_PB, CONV_CLI}
        assert [r["app_data_dir"] for r in cli] == ["antigravity-cli"]
        sub = next(r for r in main if r["conversation_id"] == CONV_B)
        assert (sub["parent_conversation_id"], sub["nesting_depth"]) == (CONV_A, 1)
        assert main[0]["record_source"] == "antigravity_summaries_db"


class TestWal:
    def test_uncheckpointed_wal_rows_are_captured(self, tmp_path, lake_root):
        h = make_gemini_home(tmp_path / "gemini", wal=True)
        try:
            db = h.store("antigravity-cli") / "conversations" / f"{CONV_CLI}.db"
            wal = db.with_name(db.name + "-wal")
            # Positive control: the third step exists ONLY in the -wal. Without
            # this, a fixture whose WAL was checkpointed would pass while the
            # writer ignored the WAL entirely.
            assert wal.exists() and wal.stat().st_size > 0
            ro = sqlite3.connect(f"file:{db}?immutable=1", uri=True)
            assert ro.execute("SELECT count(*) FROM steps").fetchone()[0] == 2
            ro.close()

            write_lake(AntigravitySource(app_bundle=None), lake_root=lake_root, root=h.root)
            rows = table(conv_dir(lake_root, CONV_CLI, "antigravity-cli") / "steps.parquet")
            assert sorted(r["idx"] for r in rows) == [0, 1, 2]
        finally:
            h.close()
            unlock(h)

    def test_a_wal_only_change_is_detected(self, tmp_path, lake_root):
        h = make_gemini_home(tmp_path / "gemini", wal=True)
        try:
            src = AntigravitySource(app_bundle=None)
            write_lake(src, lake_root=lake_root, root=h.root)
            db = h.store("antigravity-cli") / "conversations" / f"{CONV_CLI}.db"
            db_stat = db.stat()
            append_steps(h.wal_conn, [Step(3, 15)])
            assert (db.stat().st_size, db.stat().st_mtime_ns) == (db_stat.st_size, db_stat.st_mtime_ns)
            result = write_lake(src, lake_root=lake_root, root=h.root)
            status = {o.unit_id: o.status for o in result.outcomes}
            assert status[CONV_CLI] == "written"
            rows = table(conv_dir(lake_root, CONV_CLI, "antigravity-cli") / "steps.parquet")
            assert sorted(r["idx"] for r in rows) == [0, 1, 2, 3]
        finally:
            h.close()
            unlock(h)

    def test_snapshot_never_writes_beside_the_source(self, home, lake_root):
        before = sorted(os.listdir(home.store() / "conversations"))
        run(home, lake_root)
        assert sorted(os.listdir(home.store() / "conversations")) == before


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


class TestSchemaDrift:
    def test_extra_column_is_ingested_and_recorded(self, home, lake_root):
        db = home.store() / "conversations" / f"{CONV_B}.db"
        db.unlink()
        write_conversation_db(db, CONV_B, [Step(0, 14)], extra_steps_column="new_col")
        run(home, lake_root)
        rows = table(conv_dir(lake_root, CONV_B) / "steps.parquet")
        assert rows[0]["new_col"] == "extra-0"
        notes = json.loads(manifest(lake_root)[("antigravity", "conversation", CONV_B)]["notes"])
        assert any("steps.new_col" in n for n in notes)

    def test_extra_table_is_mirrored_and_recorded(self, home, lake_root):
        db = home.store() / "conversations" / f"{CONV_B}.db"
        db.unlink()
        write_conversation_db(db, CONV_B, [Step(0, 14)], extra_table="future_table")
        run(home, lake_root)
        assert len(table(conv_dir(lake_root, CONV_B) / "future_table.parquet")) == 1
        notes = json.loads(manifest(lake_root)[("antigravity", "conversation", CONV_B)]["notes"])
        assert any("future_table" in n for n in notes)

    def test_missing_column_errors_that_unit_only(self, home, lake_root):
        db = home.store() / "conversations" / f"{CONV_B}.db"
        db.unlink()
        write_conversation_db(db, CONV_B, [Step(0, 14)], drop_steps_column="render_info")
        result = run(home, lake_root)
        status = {o.unit_id: o.status for o in result.outcomes}
        assert status[CONV_B] == "error"
        assert status[CONV_A] == "written"
        err = next(o.error for o in result.outcomes if o.unit_id == CONV_B)
        assert "render_info" in err
        assert result.has_errors

    def test_storage_type_mismatch_errors_loudly(self, home, lake_root):
        # SQLite does not enforce declared types; a value stored as the wrong
        # type must fail the unit, not be coerced into a plausible column.
        db = home.store() / "conversations" / f"{CONV_B}.db"
        conn = sqlite3.connect(db)
        conn.execute("UPDATE steps SET step_type = 'fifteen' WHERE idx = 1")
        conn.commit()
        conn.close()
        result = run(home, lake_root)
        err = next(o.error for o in result.outcomes if o.unit_id == CONV_B)
        assert "steps.step_type" in err


# ---------------------------------------------------------------------------
# Files around the database
# ---------------------------------------------------------------------------


def files_by_relpath(lake_root, conv=CONV_A, store="antigravity"):
    return {r["relpath"]: r for r in table(conv_dir(lake_root, conv, store) / "files.parquet")}


class TestConversationFiles:
    def test_classified_text_is_kept_as_bytes(self, home, lake_root):
        run(home, lake_root)
        files = files_by_relpath(lake_root)
        b = f"brain/{CONV_A}"
        expect = {
            f"{b}/.system_generated/logs/transcript_full.jsonl": "transcript_full",
            f"{b}/.system_generated/logs/transcript.jsonl": "transcript",
            f"{b}/.system_generated/messages/77777777-7777-4777-8777-777777777777.json": "agent_message",
            f"{b}/.system_generated/steps/2/output.txt": "step_output",
            f"{b}/.system_generated/tasks/task-1.log": "task_log",
            f"{b}/task.md": "artifact",
            f"{b}/task.md.metadata.json": "artifact_metadata",
            f"{b}/task.md.resolved.0": "artifact_resolved",
            f"{b}/.agents/agents/explorer/agent.md": "agent_definition",
            f"{b}/notes.txt": "unclassified",
            f"{b}/.system_generated/steps/5/content.md": "step_output",
            f"{b}/.system_generated/messages/undelivered/88888888-8888-4888-8888-888888888888": "agent_message",
            f"{b}/artifacts/plan.md": "artifact",
            f"browser_recordings/{CONV_A}/metadata.json": "recording_metadata",
        }
        for rel, kind in expect.items():
            assert files[rel]["kind"] == kind, rel
            src = (home.store() / rel).read_bytes()
            assert files[rel]["content"] == src, rel
            assert files[rel]["sha256"] == hashlib.sha256(src).hexdigest(), rel

    def test_media_uploads_and_big_unknowns_are_inventory_only(self, home, lake_root):
        run(home, lake_root)
        files = files_by_relpath(lake_root)
        b = f"brain/{CONV_A}"
        for rel, kind in {
            f"{b}/shot.png": "media",
            f"{b}/.user_uploaded/photo.png": "user_upload",
            f"{b}/.tempmediaStorage/media_1789395530474.img": "media",
            f"{b}/big.bin": "unclassified",
            f"browser_recordings/{CONV_A}/1790000000000000000.jpg": "recording_frame",
        }.items():
            assert files[rel]["kind"] == kind, rel
            assert files[rel]["content"] is None, rel
            assert files[rel]["sha256"] == hashlib.sha256((home.store() / rel).read_bytes()).hexdigest()
            assert files[rel]["size_bytes"] == (home.store() / rel).stat().st_size

    def test_git_and_chunks_are_excluded(self, home, lake_root):
        run(home, lake_root)
        rels = files_by_relpath(lake_root)
        assert not any("/.git/" in r or "/chunks/" in r for r in rels)

    def test_symlink_is_recorded_never_followed(self, home, lake_root):
        run(home, lake_root)
        leak = files_by_relpath(lake_root)[f"brain/{CONV_A}/leak"]
        assert leak["kind"] == "symlink"
        assert leak["content"] is None and leak["sha256"] is None

    def test_legacy_pb_is_archived_as_ciphertext(self, home, lake_root):
        result = run(home, lake_root)
        assert {o.unit_id: o.status for o in result.outcomes}[CONV_PB] == "written"
        files = files_by_relpath(lake_root, CONV_PB)
        pb = files[f"conversations/{CONV_PB}.pb"]
        assert pb["kind"] == "legacy_pb"
        assert pb["record_source"] == "antigravity_encrypted"
        assert pb["content"] == (home.store() / "conversations" / f"{CONV_PB}.pb").read_bytes()
        assert f"brain/{CONV_PB}/implementation_plan.md" in files
        assert not (conv_dir(lake_root, CONV_PB) / "steps.parquet").exists()

    def test_store_files_implicit_and_config(self, home, lake_root):
        run(home, lake_root)
        base = lake_root / "antigravity"
        sf = {r["relpath"]: r for r in table(base / "antigravity" / "store_files" / "store_files.parquet")}
        assert sf[f"annotations/{CONV_A}.pbtxt"]["kind"] == "annotation"
        assert sf["agyhub_summaries_proto.pb"]["kind"] == "summaries_proto"
        cli = {r["relpath"]: r for r in table(base / "antigravity-cli" / "store_files" / "store_files.parquet")}
        assert cli["history.jsonl"]["kind"] == "cli_history"
        imp = table(base / "antigravity" / "implicit" / "implicit_files.parquet")
        assert [r["relpath"] for r in imp] == [f"implicit/{IMPLICIT}.pb"]
        assert imp[0]["record_source"] == "antigravity_encrypted"
        cfg = table(base / "_global" / "config_projects" / "config_projects.parquet")
        assert [r["relpath"] for r in cfg] == ["config/projects/project-1.json"]


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


class TestStores:
    def test_backup_and_unknown_stores_are_not_ingested(self, home, lake_root):
        result = run(home, lake_root)
        assert {o.store for o in result.outcomes} == {"antigravity", "antigravity-cli", "_global"}
        assert any("weird-store" in n for n in result.notes)
        assert any("antigravity-backup" in n for n in result.notes)

    def test_store_filter(self, home, lake_root):
        result = run(home, lake_root, stores=("antigravity-cli",))
        assert {o.store for o in result.outcomes} == {"antigravity-cli", "_global"}

    def test_non_uuid_brain_dirs_are_noted_not_ingested(self, home, lake_root):
        result = run(home, lake_root)
        assert "tempmediaStorage" not in {o.unit_id for o in result.outcomes}
        assert any("tempmediaStorage" in n for n in result.notes)


# ---------------------------------------------------------------------------
# Re-runs
# ---------------------------------------------------------------------------


class TestReruns:
    def test_second_run_is_a_no_op(self, home, lake_root):
        run(home, lake_root)
        steps = conv_dir(lake_root, CONV_A) / "steps.parquet"
        before = steps.stat().st_mtime_ns
        result = run(home, lake_root)
        assert {o.status for o in result.outcomes} == {"unchanged"}
        assert steps.stat().st_mtime_ns == before

    def test_a_new_brain_file_rewrites_its_conversation(self, home, lake_root):
        run(home, lake_root)
        (home.store() / "brain" / CONV_A / "walkthrough.md").write_text("# Walkthrough\n")
        result = run(home, lake_root)
        status = {o.unit_id: o.status for o in result.outcomes}
        assert status[CONV_A] == "written"
        assert status[CONV_B] == "unchanged"
        assert f"brain/{CONV_A}/walkthrough.md" in files_by_relpath(lake_root)

    def test_a_deleted_conversation_stays_in_the_archive(self, home, lake_root):
        run(home, lake_root)
        (home.store() / "conversations" / f"{CONV_B}.db").unlink()
        run(home, lake_root)
        assert (conv_dir(lake_root, CONV_B) / "steps.parquet").exists()
        row = manifest(lake_root)[("antigravity", "conversation", CONV_B)]
        assert row["source_present"] is False

    def test_dropped_steps_supersede_the_old_snapshot(self, home, lake_root):
        run(home, lake_root)
        db = home.store() / "conversations" / f"{CONV_A}.db"
        conn = sqlite3.connect(db)
        conn.execute("DELETE FROM steps WHERE idx = 2")
        conn.commit()
        conn.close()
        result = run(home, lake_root)
        assert {o.unit_id: o.status for o in result.outcomes}[CONV_A] == "superseded"
        kept = list((lake_root / "antigravity" / "antigravity" / "conversations" / "_superseded" / CONV_A)
                    .glob("*/steps.parquet"))
        assert len(kept) == 1 and len(table(kept[0])) == 3


# ---------------------------------------------------------------------------
# Format version
# ---------------------------------------------------------------------------

# One entry per LAKE_FORMAT_VERSION. When this test fails you changed what the
# lake writer emits (a schema, a classification rule, the allowlist, a store
# name): bump LAKE_FORMAT_VERSION in parsers/antigravity/schema.py and add the
# new fingerprint here. Without the bump, every existing lake would keep its
# old shape forever, because the source fingerprint cannot see a writer change.
PINNED_FORMAT_FINGERPRINTS = {
    1: "e2f0012235d8d639a6f82710ed1649958ab5f198db3eb27d46b945212b3ab650",
}


def test_lake_format_version_is_pinned_to_the_writer_contract():
    assert PINNED_FORMAT_FINGERPRINTS.get(ag_schema.LAKE_FORMAT_VERSION) == ag_schema.format_fingerprint(), (
        "The Antigravity lake writer's output contract changed. Bump LAKE_FORMAT_VERSION "
        f"and pin {ag_schema.format_fingerprint()!r} for it."
    )


def test_format_fingerprint_sees_a_rule_change(monkeypatch):
    before = ag_schema.format_fingerprint()
    monkeypatch.setattr(stores, "UNCLASSIFIED_MAX_BYTES", stores.UNCLASSIFIED_MAX_BYTES + 1)
    assert ag_schema.format_fingerprint() != before


def test_every_record_source_is_allow_listed():
    from ccutils.provenance import _RECORD_SOURCES
    assert AntigravitySource.record_sources <= _RECORD_SOURCES


# ---------------------------------------------------------------------------
# The read allowlist
# ---------------------------------------------------------------------------

_TRACE: list[tuple[str, str]] | None = None
_HOOKED = False


def _audit(event, args):
    if _TRACE is None:
        return
    if event in ("open", "sqlite3.connect", "shutil.copyfile", "shutil.copy", "os.open"):
        path = args[0] if args else None
        if isinstance(path, (str, bytes, os.PathLike)):
            _TRACE.append((event, os.fsdecode(os.fspath(path))))
    elif event == "subprocess.Popen":
        _TRACE.append((event, repr(args)))


@contextmanager
def traced():
    """Record every file-opening audit event while active.

    Audit hooks cannot be removed once added, so one hook is installed for the
    session and switched on and off with a module global."""
    global _TRACE, _HOOKED
    if not _HOOKED:
        sys.addaudithook(_audit)
        _HOOKED = True
    _TRACE = []
    try:
        yield _TRACE
    finally:
        _TRACE = None


def violations(trace, gemini_root: Path):
    """Opened paths under the Antigravity root that the allowlist does not permit."""
    root = gemini_root.resolve()
    bad = []
    for event, path in trace:
        if event == "subprocess.Popen":
            bad.append(path)
            continue
        p = Path(path)
        try:
            resolved = (p if p.is_absolute() else Path.cwd() / p).resolve()
        except OSError:
            continue
        if resolved == root or root in resolved.parents:
            if not stores.is_allowed(root, resolved):
                bad.append(str(resolved))
    return bad


class TestReadAllowlist:
    def test_nothing_outside_the_allowlist_is_opened(self, home, lake_root):
        with traced() as trace:
            run(home, lake_root)
        assert violations(trace, home.root) == []
        opened_under_root = [p for _e, p in trace if str(home.root.resolve()) in str(Path(p).resolve())]
        assert len(opened_under_root) > 10  # the tracer saw the reads at all
        for decoy in home.decoys:
            assert not any(Path(p).resolve() == decoy.resolve() for _e, p in trace), decoy

    def test_positive_control_a_decoy_read_is_caught(self, home):
        decoy = home.root / "oauth_creds.json"
        with traced() as trace:
            with pytest.raises(PermissionError):
                open(decoy, "rb")
        assert violations(trace, home.root) == [str(decoy.resolve())]

    def test_allowlist_is_anchored(self, home):
        root = home.root.resolve()
        assert stores.is_allowed(root, root / "antigravity" / "conversations" / f"{CONV_A}.db")
        assert not stores.is_allowed(root, root / "antigravity" / "conversations" / f"{CONV_A}.db.bak")
        assert not stores.is_allowed(root, root / "oauth_creds.json")
        assert not stores.is_allowed(root, root / "antigravity-backup" / "conversations" / f"{CONV_A}.db")
        assert not stores.is_allowed(root, root / "config" / "mcp_config.json")
        # brain/<id>/.git is inside an allowed subtree; the classifier, not the
        # allowlist, excludes it -- and the tracer test proves it is never opened.
        assert stores.classify_conversation_file(".git/HEAD", "brain") is None
