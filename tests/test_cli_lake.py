"""`ccutils lake antigravity`: assert what the command does, not only its exit code.

An exit-code-only test of a CLI flag once stayed green while the flag did
nothing; every test here checks files on disk or the manifest as well.
"""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from ccutils.cli import cli
from ccutils.parsers.lake import _exclusive_lock, read_manifest

from helpers_antigravity import CONV_A, CONV_B, Step, make_gemini_home, unlock, write_conversation_db


@pytest.fixture
def home(tmp_path):
    h = make_gemini_home(tmp_path / "gemini")
    yield h
    h.close()
    unlock(h)


def invoke(*args):
    return CliRunner().invoke(cli, ["lake", *args])


def test_writes_the_lake_under_output(home, tmp_path):
    out = tmp_path / "out"
    result = invoke("antigravity", "-o", str(out), "--source", str(home.root))
    assert result.exit_code == 0, result.output
    steps = out / "antigravity" / "antigravity" / "conversations" / CONV_A / "steps.parquet"
    assert pq.read_table(steps).num_rows == 3
    units = {(r["store"], r["unit_id"]): r["status"] for r in read_manifest(out / "antigravity")}
    assert units[("antigravity", CONV_A)] == "written"
    assert "antigravity-cli" in result.output
    assert "weird-store" in result.output  # skipped stores are reported, not silent


def test_second_run_reports_unchanged(home, tmp_path):
    out = tmp_path / "out"
    invoke("antigravity", "-o", str(out), "--source", str(home.root))
    result = invoke("antigravity", "-o", str(out), "--source", str(home.root))
    assert result.exit_code == 0
    assert "written 0" in result.output
    assert {r["status"] for r in read_manifest(out / "antigravity")} == {"unchanged"}


def test_default_root_is_home_anchored_never_cwd(home, tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.chdir(cwd)
    result = invoke("antigravity", "--source", str(home.root))
    assert result.exit_code == 0, result.output
    assert (fake_home / ".ccutils" / "lake" / "antigravity" / "_manifest.parquet").exists()
    assert list(cwd.iterdir()) == []


def test_store_filter(home, tmp_path):
    out = tmp_path / "out"
    result = invoke("antigravity", "-o", str(out), "--source", str(home.root), "--store", "antigravity-cli")
    assert result.exit_code == 0, result.output
    assert {r["store"] for r in read_manifest(out / "antigravity")} == {"antigravity-cli", "_global"}


def test_unit_error_exits_1_and_names_the_unit(home, tmp_path):
    db = home.store() / "conversations" / f"{CONV_B}.db"
    db.unlink()
    write_conversation_db(db, CONV_B, [Step(0, 14)], drop_steps_column="render_info")
    result = invoke("antigravity", "-o", str(tmp_path / "out"), "--source", str(home.root))
    assert result.exit_code == 1
    assert CONV_B in result.output and "render_info" in result.output


def test_unknown_store_exits_2_before_touching_anything(home, tmp_path):
    out = tmp_path / "out"
    result = invoke("antigravity", "-o", str(out), "--source", str(home.root), "--store", "antigravity-backup")
    assert result.exit_code == 2
    assert "not an Antigravity store" in result.output
    assert not out.exists()


def test_unknown_harness_exits_2():
    result = invoke("claude_code")
    assert result.exit_code == 2


def test_held_lock_exits_1(home, tmp_path):
    out = tmp_path / "out"
    (out / "antigravity").mkdir(parents=True)
    with _exclusive_lock(out / "antigravity" / ".lock"):
        result = invoke("antigravity", "-o", str(out), "--source", str(home.root))
    assert result.exit_code == 1
    assert "another lake run" in result.output


def test_help_says_experimental():
    result = invoke("--help")
    assert result.exit_code == 0
    assert "EXPERIMENTAL" in result.output
