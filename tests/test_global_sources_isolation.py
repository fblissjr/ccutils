"""The test suite never reads the developer's real Claude home through a global source.

A warehouse-building test that goes through a batch driver calls
`run_global_sources` with no `claude_home`, and the fallback was the real
`~/.claude`. So every such test imported the developer's actual prompt
history into a throwaway warehouse under pytest's temp directory: measured on
2026-10-13, 14,439 prompts from 121 project paths per warehouse, in eleven
tests, about fifteen seconds each. The copies outlived the run.

The fallback is one function now, `default_claude_home`, and
`tests/conftest.py` points it at an empty temp directory for every test. A
test that wants a home passes one, or writes into the isolated one.

Delete this and the suite goes back to copying real prompts into temp files,
on every machine that has any, with nothing to show it happened.
"""

from __future__ import annotations

import json
from pathlib import Path

from helpers_ccutils import write_minimal_session

from ccutils import create_star_schema
from ccutils.etl import global_sources
from ccutils.export.duckdb_archive import generate_duckdb_archive


def test_no_test_resolves_the_real_home():
    """The autouse fixture is in force: the fallback is not the real home."""
    resolved = global_sources.default_claude_home()
    assert resolved != Path.home() / ".claude"
    assert Path.home() not in resolved.parents or "pytest" in str(resolved)


def test_outside_the_suite_the_fallback_is_the_claude_home(monkeypatch):
    """What production gets: undo the suite's isolation and ask again."""
    monkeypatch.undo()
    assert global_sources.default_claude_home() == Path.home() / ".claude"


def _one_project(tmp_path):
    source = tmp_path / "projects"
    project = source / "-work-demo"
    project.mkdir(parents=True)
    write_minimal_session(project / "sess-A.jsonl", "sess-A")
    return source


def _prompts(output_dir):
    conn = create_star_schema(output_dir / "archive.duckdb")
    try:
        return conn.execute("SELECT display_text FROM dim_prompt ORDER BY 1").fetchall()
    finally:
        conn.close()


def test_a_batch_build_reads_history_from_the_isolated_home(tmp_path):
    """A positive control: the import still runs, and it reads the home it was given.

    Without this the test below passes just as well when history import is
    broken outright.
    """
    home = global_sources.default_claude_home()
    home.mkdir(parents=True, exist_ok=True)
    history = home / "history.jsonl"
    history.write_text(json.dumps(
        {"display": "a prompt written by this test", "project": "/work/demo",
         "sessionId": "sess-A", "timestamp": 1745052000000}))
    try:
        generate_duckdb_archive(_one_project(tmp_path), tmp_path / "out")
        assert _prompts(tmp_path / "out") == [("a prompt written by this test",)]
    finally:
        history.unlink()


def test_a_batch_build_holds_no_prompt_it_was_not_given(tmp_path):
    """The isolated home holds no history, so the warehouse holds no prompts.

    On a machine with a real history file this held thousands of rows.
    """
    generate_duckdb_archive(_one_project(tmp_path), tmp_path / "out")
    assert _prompts(tmp_path / "out") == []


def test_a_direct_call_with_no_home_is_isolated_too(tmp_path):
    """`run_global_sources(conn)` with no argument is the other way in."""
    conn = create_star_schema(tmp_path / "direct.duckdb")
    try:
        global_sources.run_global_sources(conn)
        assert conn.execute("SELECT COUNT(*) FROM dim_prompt").fetchone()[0] == 0
        # The import was still attempted and recorded, not skipped.
        assert conn.execute(
            "SELECT COUNT(*) FROM etl.runs WHERE run_kind = 'global_source'"
        ).fetchone()[0] >= 1
    finally:
        conn.close()
