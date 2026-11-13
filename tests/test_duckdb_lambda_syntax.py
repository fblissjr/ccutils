"""The ETL's SQL binds when DuckDB refuses the single-arrow lambda.

DuckDB 2.0 rejects `x -> expr` lambdas by default ("Deprecated lambda
arrow (->) detected"). The committed lockfile pinned 1.5.x, which still
accepted them, so the suite stayed green while a fresh install on another
machine resolved 2.0 and every session run failed at its first fact
populator: dimensions written, every fact table empty, exit 0.

This test does not depend on which DuckDB the lockfile resolves. It turns
the arrow off on the connection (`lambda_syntax = 'DISABLE_SINGLE_ARROW'`,
accepted since DuckDB 1.3) and runs the whole per-session ETL, so any arrow
lambda reintroduced anywhere in a populator fails here on 1.5 as on 2.0.
It asserts the content the lambdas compute, not only that the run finished.

What breaks without it: a populator written with `x -> ...` passes the
suite on a 1.5 lockfile and empties every fact on DuckDB 2.0.
"""

import json

from ccutils import create_star_schema
from ccutils.etl.orchestrator import run_v15_etl


def _session_with_array_content(path):
    """A user turn, an assistant text+tool_use, and an array tool_result.

    fact_messages.content_text and fact_tool_calls.result_content_text are
    the two lambda sites; both need ARRAY content to compute anything.
    """
    lines = [
        {"type": "user", "uuid": "u1", "sessionId": "lam-session",
         "timestamp": "2026-04-19T10:00:00Z", "cwd": "/p", "gitBranch": "main",
         "message": {"role": "user", "content": "go"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "sessionId": "lam-session", "timestamp": "2026-04-19T10:00:05Z",
         "cwd": "/p",
         "message": {"role": "assistant", "model": "claude-opus-4-7",
                     "id": "msg_1",
                     "content": [
                         {"type": "text", "text": "listing"},
                         {"type": "tool_use", "id": "toolu_lam1",
                          "name": "Bash", "input": {"command": "ls"}},
                     ],
                     "usage": {"input_tokens": 1, "output_tokens": 1}}},
        {"type": "user", "uuid": "u2", "parentUuid": "a1",
         "sessionId": "lam-session", "timestamp": "2026-04-19T10:00:06Z",
         "cwd": "/p",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "toolu_lam1",
              "content": [{"type": "text", "text": "file-a"},
                          {"type": "image", "source": {}},
                          {"type": "text", "text": "file-b"}]}]}},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines))
    return path


def test_etl_binds_with_single_arrow_lambdas_disabled(tmp_path):
    project = tmp_path / "projects" / "-p"
    project.mkdir(parents=True)
    session = _session_with_array_content(project / "lam-session.jsonl")

    conn = create_star_schema(tmp_path / "w.duckdb")
    conn.execute("SET lambda_syntax = 'DISABLE_SINGLE_ARROW'")

    run_v15_etl(conn, session, parquet_lake_root=tmp_path / "lake")

    status = conn.execute(
        "SELECT status, error_message FROM etl.runs WHERE run_kind = 'session'"
    ).fetchall()
    assert status == [("success", None)]

    assistant_text = conn.execute(
        "SELECT content_text FROM fact_messages WHERE message_id = 'a1'"
    ).fetchone()
    assert assistant_text == ("listing",)

    result_text = conn.execute(
        "SELECT result_content_text FROM fact_tool_calls "
        "WHERE tool_use_id = 'toolu_lam1'"
    ).fetchone()
    assert result_text == ("file-a file-b",)
