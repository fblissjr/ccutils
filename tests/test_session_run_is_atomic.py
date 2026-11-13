"""A session run that fails leaves the warehouse as it was before the run.

On 2026-11-12 every session run of a Linux build failed at its first fact
populator under DuckDB 2.0. Each run had already committed its stub
dimensions and `dim_session`, so the warehouse listed 65 sessions and held
no facts, and only `etl.runs` said otherwise. Every statement autocommitted.

A run is now one transaction after its `etl.runs` row: the staging load,
the dimensions and every fact commit together or not at all, and the
failure is recorded after the rollback, with the step that raised.

The oracles are the warehouse's own earlier state, not the rule restated:
a session that loaded cleanly and then fails a re-run must read back
byte-identical, and a session whose first run fails must leave no row
anywhere but its failed `etl.runs` and `etl.steps` records.

What breaks without it: a failed run leaves dimensions that describe
sessions with no facts, and a failed re-run half-replaces a good session.
"""

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl import orchestrator
from ccutils.etl.orchestrator import run_v15_etl

DATA_TABLES = ("dim_session", "fact_messages", "fact_tool_calls",
               "fact_token_usage", "dim_tool", "dim_model", "dim_project")


def _session(path, sid, text="go"):
    lines = [
        {"type": "user", "uuid": f"{sid}-u1", "sessionId": sid,
         "timestamp": "2026-04-19T10:00:00Z", "cwd": "/p",
         "message": {"role": "user", "content": text}},
        {"type": "assistant", "uuid": f"{sid}-a1", "parentUuid": f"{sid}-u1",
         "sessionId": sid, "timestamp": "2026-04-19T10:00:05Z",
         "message": {"id": f"msg-{sid}", "role": "assistant",
                     "model": "claude-opus-4-7",
                     "usage": {"input_tokens": 10, "output_tokens": 5},
                     "content": [{"type": "tool_use", "id": f"toolu-{sid}",
                                  "name": "Bash", "input": {"command": "ls"}}]}},
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines))
    return path


def _snapshot(conn, sid):
    """Every row the session owns in the data tables, as plain tuples."""
    snap = {}
    for t in DATA_TABLES:
        cols = [r[0] for r in conn.execute(f"DESCRIBE {t}").fetchall()]
        where = "WHERE session_id = ?" if "session_id" in cols else ""
        params = [sid] if where else []
        snap[t] = sorted(map(repr, conn.execute(
            f"SELECT * FROM {t} {where}", params).fetchall()))
    return snap


def _break(monkeypatch, name="populate_fact_tool_calls"):
    def boom(conn, *, run):
        raise RuntimeError("simulated populator failure")
    monkeypatch.setattr(orchestrator, name, boom)


@pytest.fixture
def setup(tmp_path):
    proj = tmp_path / "projects" / "-p"
    proj.mkdir(parents=True)
    conn = create_star_schema(tmp_path / "w.duckdb")
    return conn, proj, tmp_path / "lake"


def test_a_failed_first_run_leaves_no_rows(setup, monkeypatch):
    conn, proj, lake = setup
    run_v15_etl(conn, _session(proj / "good.jsonl", "good"), parquet_lake_root=lake)
    before = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in DATA_TABLES}

    _break(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated"):
        run_v15_etl(conn, _session(proj / "bad.jsonl", "bad"), parquet_lake_root=lake)

    after = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
             for t in DATA_TABLES}
    assert after == before
    assert conn.execute(
        "SELECT COUNT(*) FROM dim_session WHERE session_id = 'bad'"
    ).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM etl.log_entries").fetchone()[0] == 0


def test_the_failure_is_recorded_with_the_step_that_raised(setup, monkeypatch):
    conn, proj, lake = setup
    _break(monkeypatch)
    with pytest.raises(RuntimeError):
        run_v15_etl(conn, _session(proj / "bad.jsonl", "bad"), parquet_lake_root=lake)
    runs = conn.execute(
        "SELECT status, error_message FROM etl.runs WHERE run_kind = 'session'"
    ).fetchall()
    assert runs == [("failed", "simulated populator failure")]


def test_a_failing_step_is_recorded_after_the_rollback(setup, monkeypatch):
    conn, proj, lake = setup

    def fails_inside_a_step(conn, *, run):
        with run.step("upsert:fact_tool_calls", kind="upsert", table="fact_tool_calls"):
            raise RuntimeError("Binder Error: simulated")
    monkeypatch.setattr(orchestrator, "populate_fact_tool_calls", fails_inside_a_step)
    with pytest.raises(RuntimeError):
        run_v15_etl(conn, _session(proj / "bad.jsonl", "bad"), parquet_lake_root=lake)
    steps = conn.execute(
        "SELECT step_name, status, error_message FROM etl.steps"
    ).fetchall()
    # Only the step that raised survives the rollback; the steps that had
    # succeeded wrote rows that no longer exist, so their records go too.
    assert steps == [("upsert:fact_tool_calls", "failed", "Binder Error: simulated")]


def test_a_failed_rerun_leaves_the_earlier_load_untouched(setup, monkeypatch):
    conn, proj, lake = setup
    path = _session(proj / "s.jsonl", "s")
    run_v15_etl(conn, path, parquet_lake_root=lake)
    before = _snapshot(conn, "s")

    # The session grows (a new entry would change rows) and the re-run fails.
    _session(path, "s", text="go, then more")
    _break(monkeypatch)
    with pytest.raises(RuntimeError):
        run_v15_etl(conn, path, parquet_lake_root=lake)

    assert _snapshot(conn, "s") == before


def test_a_run_after_a_failure_loads_normally(setup, monkeypatch):
    conn, proj, lake = setup
    path = _session(proj / "s.jsonl", "s")
    _break(monkeypatch)
    with pytest.raises(RuntimeError):
        run_v15_etl(conn, path, parquet_lake_root=lake)
    monkeypatch.undo()
    run_v15_etl(conn, path, parquet_lake_root=lake)
    assert conn.execute(
        "SELECT COUNT(*) FROM fact_messages WHERE session_id = 's'"
    ).fetchone()[0] == 2
    assert [r[0] for r in conn.execute(
        "SELECT status FROM etl.runs WHERE run_kind = 'session' ORDER BY started_at"
    ).fetchall()] == ["failed", "success"]
