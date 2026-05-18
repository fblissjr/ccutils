"""Tests for the v0.15 fact_agent_delegations populator (Phase D).

Grain: one row per Task tool_use (parent-side subagent spawn).

The legacy populator tried to cross-link parent and subagent sessions
via heuristic matching. In v0.15 the agent rollup metrics (totalDurationMs,
totalTokens, totalToolUseCount, status, subagent_type) are captured
structurally on fact_tool_results.agent_* columns from the R1 toolUseResult
payload, so the parent-side fact stands on its own.

agent_session_key / parent_session_key are NULL for now -- cross-session
subagent linkage (reading .meta.json sidecars to mark dim_session.is_agent
and parent_session_key) is a separate Phase D follow-up. session_id on
this fact is the PARENT session that delegated.

Run AFTER populate_fact_tool_uses + populate_fact_tool_results.
"""

from __future__ import annotations

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.fact_agent_delegations import populate_fact_agent_delegations
from ccutils.etl.fact_messages import populate_fact_messages
from ccutils.etl.fact_tool_calls import (
    populate_fact_tool_results,
    populate_fact_tool_uses,
)
from ccutils.etl.lineage import EtlRun
from ccutils.etl.staging import load_session_to_staging
from ccutils.parsers.parquet_writer import write_session_to_parquet


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


@pytest.fixture
def agent_session(tmp_path):
    """Two Task delegations: one completed, one interrupted."""
    jsonl = tmp_path / "agents.jsonl"
    lines = [
        {"type": "user", "uuid": "u1", "sessionId": "agent-s",
         "timestamp": "2026-04-19T10:00:00Z", "cwd": "/p",
         "gitBranch": "main", "version": "2.1.114",
         "message": {"role": "user", "content": "delegate two tasks"}},
        # Task 1: completed
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "sessionId": "agent-s", "timestamp": "2026-04-19T10:00:01Z",
         "requestId": "r1",
         "message": {"role": "assistant", "model": "claude-opus-4-7",
                     "content": [{"type": "tool_use", "id": "tu_t1",
                                  "name": "Task",
                                  "input": {
                                      "description": "Explore stuff",
                                      "subagent_type": "Explore",
                                      "prompt": "go look at thing 1",
                                  }}]}},
        {"type": "user", "uuid": "u2", "parentUuid": "a1",
         "sessionId": "agent-s", "timestamp": "2026-04-19T10:01:00Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_t1",
              "content": [{"type": "text", "text": "Findings report ..."}]},
         ]},
         "toolUseResult": {
             "agentId": "ag-001", "agentType": "Explore",
             "status": "completed",
             "totalDurationMs": 38695, "totalTokens": 70817,
             "totalToolUseCount": 9,
             "prompt": "go look at thing 1",
             "content": [{"type": "text", "text": "Findings report ..."}],
         }},
        # Task 2: interrupted
        {"type": "assistant", "uuid": "a2", "parentUuid": "u2",
         "sessionId": "agent-s", "timestamp": "2026-04-19T10:02:00Z",
         "requestId": "r2",
         "message": {"role": "assistant", "model": "claude-opus-4-7",
                     "content": [{"type": "tool_use", "id": "tu_t2",
                                  "name": "Task",
                                  "input": {
                                      "description": "Big plan",
                                      "subagent_type": "Plan",
                                      "prompt": "plan it",
                                  }}]}},
        {"type": "user", "uuid": "u3", "parentUuid": "a2",
         "sessionId": "agent-s", "timestamp": "2026-04-19T10:02:30Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_t2",
              "content": "Interrupted"},
         ]},
         "toolUseResult": {
             "agentId": "ag-002", "agentType": "Plan",
             "status": "interrupted",
             "totalDurationMs": 5000, "totalTokens": 8000,
             "totalToolUseCount": 2,
             "wasInterrupted": True,
         }},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


def _populate(conn, jsonl_path, tmp_path):
    run = EtlRun.start(conn, source_path=str(jsonl_path))
    log_path, _ = write_session_to_parquet(
        jsonl_path, tmp_path / "lake",
        etl_run_id=run.etl_run_id, project_slug="test-project",
    )
    load_session_to_staging(conn, log_path)
    conn.execute(
        """
        INSERT INTO dim_tool (tool_key, tool_name, tool_category)
        SELECT DISTINCT md5(tool_name), tool_name, 'unknown' FROM (
            SELECT json_extract_string(b.block, '$.name') AS tool_name
            FROM stg_log_entries sle, LATERAL (
                SELECT unnest(json_extract(sle.message_json, '$.content')::JSON[]) AS block
            ) b
            WHERE sle.type = 'assistant'
              AND json_type(sle.message_json, '$.content') = 'ARRAY'
              AND json_extract_string(b.block, '$.type') = 'tool_use'
        ) WHERE tool_name IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM dim_tool dt WHERE dt.tool_key = md5(tool_name))
        """
    )
    populate_fact_messages(conn, run=run)
    populate_fact_tool_uses(conn, run=run)
    populate_fact_tool_results(conn, run=run)
    populate_fact_agent_delegations(conn, run=run)
    return run


class TestFactAgentDelegations:
    def test_one_row_per_task_tool_use(self, conn, agent_session, tmp_path):
        _populate(conn, agent_session, tmp_path)
        n = conn.execute(
            "SELECT COUNT(*) FROM fact_agent_delegations"
        ).fetchone()[0]
        assert n == 2

    def test_captures_task_input(self, conn, agent_session, tmp_path):
        _populate(conn, agent_session, tmp_path)
        row = conn.execute(
            """
            SELECT task_description, task_prompt, subagent_type
            FROM fact_agent_delegations
            WHERE tool_use_id = 'tu_t1'
            """
        ).fetchone()
        assert row == ("Explore stuff", "go look at thing 1", "Explore")

    def test_captures_agent_rollup(self, conn, agent_session, tmp_path):
        _populate(conn, agent_session, tmp_path)
        row = conn.execute(
            """
            SELECT agent_status, agent_total_duration_ms, agent_total_tokens,
                   agent_total_tool_use_count
            FROM fact_agent_delegations
            WHERE tool_use_id = 'tu_t1'
            """
        ).fetchone()
        assert row[0] == "completed"
        assert row[1] == 38695.0
        assert row[2] == 70817
        assert row[3] == 9

    def test_captures_interrupted_status(
        self, conn, agent_session, tmp_path
    ):
        _populate(conn, agent_session, tmp_path)
        row = conn.execute(
            """
            SELECT agent_status, agent_was_interrupted
            FROM fact_agent_delegations
            WHERE tool_use_id = 'tu_t2'
            """
        ).fetchone()
        assert row[0] == "interrupted"
        assert row[1] is True

    def test_seconds_to_completion_computed(
        self, conn, agent_session, tmp_path
    ):
        _populate(conn, agent_session, tmp_path)
        row = conn.execute(
            """
            SELECT seconds_to_completion FROM fact_agent_delegations
            WHERE tool_use_id = 'tu_t1'
            """
        ).fetchone()
        # Delegation at 10:00:01, completion at 10:01:00 -> 59s
        assert row[0] == 59.0

    def test_agent_session_key_null_for_now(
        self, conn, agent_session, tmp_path
    ):
        """Cross-session subagent linkage is a separate Phase D follow-up;
        for now both agent_session_key and parent_session_key are NULL."""
        _populate(conn, agent_session, tmp_path)
        rows = conn.execute(
            """
            SELECT agent_session_key, parent_session_key
            FROM fact_agent_delegations
            """
        ).fetchall()
        for row in rows:
            assert row[0] is None
            assert row[1] is None

    def test_lineage_block_populated(self, conn, agent_session, tmp_path):
        _populate(conn, agent_session, tmp_path)
        row = conn.execute(
            """
            SELECT created_at, last_updated_at, etl_run_id, record_source,
                   hash_diff, is_deleted
            FROM fact_agent_delegations LIMIT 1
            """
        ).fetchone()
        assert row[0] is not None
        assert row[1] is not None
        assert row[2] is not None
        assert row[3] == "claude_code_jsonl"
        assert row[4] is not None
        assert row[5] is False

    def test_idempotent_reetl(self, conn, agent_session, tmp_path):
        _populate(conn, agent_session, tmp_path)
        first = conn.execute(
            "SELECT last_updated_at FROM fact_agent_delegations ORDER BY tool_use_id"
        ).fetchall()
        _populate(conn, agent_session, tmp_path)
        second = conn.execute(
            "SELECT last_updated_at FROM fact_agent_delegations ORDER BY tool_use_id"
        ).fetchall()
        assert first == second

    def test_non_task_tool_uses_ignored(self, conn, tmp_path):
        """Only Task / Agent tool uses become delegations."""
        jsonl = tmp_path / "mixed.jsonl"
        lines = [
            {"type": "user", "uuid": "u1", "sessionId": "mix-s",
             "timestamp": "2026-04-19T10:00:00Z", "cwd": "/p",
             "gitBranch": "main", "version": "2.1.114",
             "message": {"role": "user", "content": "do stuff"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "sessionId": "mix-s", "timestamp": "2026-04-19T10:00:01Z",
             "requestId": "r1",
             "message": {"role": "assistant", "model": "claude-opus-4-7",
                         "content": [{"type": "tool_use", "id": "tu_bash",
                                      "name": "Bash",
                                      "input": {"command": "ls"}}]}},
            {"type": "user", "uuid": "u2", "parentUuid": "a1",
             "sessionId": "mix-s", "timestamp": "2026-04-19T10:00:02Z",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "tu_bash",
                  "content": "files\n"},
             ]},
             "toolUseResult": {"stdout": "files\n",
                               "interrupted": False, "exitCode": 0}},
        ]
        jsonl.write_text("\n".join(json.dumps(d) for d in lines))
        _populate(conn, jsonl, tmp_path)
        n = conn.execute(
            "SELECT COUNT(*) FROM fact_agent_delegations"
        ).fetchone()[0]
        assert n == 0
