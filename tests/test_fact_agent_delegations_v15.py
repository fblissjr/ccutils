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

import hashlib
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
             "resolvedModel": "claude-sonnet-5",
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
             "resolvedModel": "claude-sonnet-5",
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

    def test_parent_session_key_set_to_delegating_session(
        self, conn, agent_session, tmp_path
    ):
        """parent_session_key points at the session that did the
        delegating -- same as session_key on this fact."""
        _populate(conn, agent_session, tmp_path)
        rows = conn.execute(
            """
            SELECT parent_session_key, session_key
            FROM fact_agent_delegations
            """
        ).fetchall()
        assert rows  # smoke
        for parent_sk, session_sk in rows:
            assert parent_sk == session_sk
            assert parent_sk is not None

    def test_agent_session_key_derived_without_the_subagent_loaded(
        self, conn, agent_session, tmp_path
    ):
        """The key is derived from the natural key, not looked up.

        It previously resolved through a correlated subquery on
        dim_session.agent_id, which only finds a row if the agent's OWN
        transcript happened to be ETL'd already. ETL is per-session and a
        parent is typically processed before its agents, so on the real
        corpus this produced agent_session_key NULL for 941 of 941
        delegations -- while 826 of them carried a subagent_type and 936
        agent sessions sat unlinked. Excluding the column from _HASH_COLS
        meant a later run never repaired it either: hash unchanged, no
        update, NULL forever.

        session_key is md5(session_id) and an agent's session_id is
        'agent-<agent_id>' (verified: holds for all 2,046 agent sessions),
        so the key needs no lookup and no ordering guarantee.
        """
        _populate(conn, agent_session, tmp_path)
        rows = conn.execute(
            """
            SELECT ftr.agent_id, fad.agent_session_key
            FROM fact_agent_delegations fad
            JOIN fact_tool_results ftr USING (tool_use_id)
            ORDER BY ftr.agent_id
            """
        ).fetchall()
        assert rows, "no delegations produced"
        for agent_id, agent_sk in rows:
            assert agent_sk is not None, f"{agent_id} left unlinked"
            expected = hashlib.md5(
                f"agent-{agent_id}".encode()
            ).hexdigest()
            assert agent_sk == expected

    def test_agent_session_key_survives_parent_first_ordering(
        self, conn, tmp_path
    ):
        """The real-world ordering: parent ETL'd before the agent exists.

        Claim: delete this and the lookup-based implementation passes every
        other test in this file (its fixtures load both sides, or neither)
        while producing zero linkage on a real corpus, because nothing here
        exercises parent-then-agent ordering.
        """
        from ccutils.etl.orchestrator import run_v15_etl

        parent = tmp_path / "p.jsonl"
        parent.write_text("\n".join(json.dumps(d) for d in [
            {"type": "user", "uuid": "u1", "sessionId": "ord-parent",
             "timestamp": "2026-04-19T10:00:00Z", "cwd": "/p",
             "gitBranch": "main", "version": "2.1.114",
             "message": {"role": "user", "content": "delegate"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "sessionId": "ord-parent", "timestamp": "2026-04-19T10:00:01Z",
             "requestId": "r1",
             "message": {"role": "assistant", "model": "claude-opus-4-7",
                         "content": [{"type": "tool_use", "id": "tu_ord",
                                      "name": "Task",
                                      "input": {"description": "go",
                                                "subagent_type": "Explore",
                                                "prompt": "x"}}]}},
            {"type": "user", "uuid": "u2", "parentUuid": "a1",
             "sessionId": "ord-parent", "timestamp": "2026-04-19T10:00:30Z",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "tu_ord",
                  "content": [{"type": "text", "text": "done"}]}]},
             "toolUseResult": {"agentId": "ord-agent-1",
                               "agentType": "Explore", "status": "completed",
                               "totalDurationMs": 10, "totalTokens": 5,
                               "totalToolUseCount": 1}},
        ]))
        # Parent first, agent transcript never ingested at all.
        run_v15_etl(conn, parent, project_name="test-project",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT agent_session_key FROM fact_agent_delegations "
            "WHERE tool_use_id = 'tu_ord'"
        ).fetchone()
        assert row[0] == hashlib.md5(b"agent-ord-agent-1").hexdigest()

    def test_resolved_model_captured_from_tool_use_result(
        self, conn, agent_session, tmp_path
    ):
        """toolUseResult.resolvedModel is the model the subagent ACTUALLY ran
        on, and it is the only place that fact appears.

        Claim: delete this and per-delegation model attribution is
        unrecoverable. A subagent's own transcript records the model on its
        assistant entries, but 894 of 2,046 agent sessions on the real corpus
        have no ingestible transcript at all -- and the parent's delegation
        row is the only other place the model is stated. 815 resolvedModel
        values sit in the corpus today, captured nowhere.
        """
        _populate(conn, agent_session, tmp_path)
        rows = conn.execute(
            "SELECT tool_use_id, agent_resolved_model "
            "FROM fact_agent_delegations ORDER BY tool_use_id"
        ).fetchall()
        assert rows, "no delegations produced"
        models = {r[1] for r in rows}
        assert models == {"claude-sonnet-5"}, models

    def test_resolved_model_null_when_absent(self, conn, tmp_path):
        """Older transcripts predate resolvedModel; absence must not error."""
        from ccutils.etl.orchestrator import run_v15_etl

        jsonl = tmp_path / "nomodel.jsonl"
        jsonl.write_text("\n".join(json.dumps(d) for d in [
            {"type": "user", "uuid": "u1", "sessionId": "nomodel-s",
             "timestamp": "2026-04-19T10:00:00Z", "cwd": "/p",
             "gitBranch": "main", "version": "2.1.114",
             "message": {"role": "user", "content": "delegate"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "sessionId": "nomodel-s", "timestamp": "2026-04-19T10:00:01Z",
             "requestId": "r1",
             "message": {"role": "assistant", "model": "claude-opus-5",
                         "content": [{"type": "tool_use", "id": "tu_nm",
                                      "name": "Task",
                                      "input": {"description": "go",
                                                "subagent_type": "Explore",
                                                "prompt": "x"}}]}},
            {"type": "user", "uuid": "u2", "parentUuid": "a1",
             "sessionId": "nomodel-s", "timestamp": "2026-04-19T10:00:30Z",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "tu_nm",
                  "content": [{"type": "text", "text": "done"}]}]},
             "toolUseResult": {"agentId": "nm-agent", "agentType": "Explore",
                               "status": "completed", "totalDurationMs": 10,
                               "totalTokens": 5, "totalToolUseCount": 1}},
        ]))
        run_v15_etl(conn, jsonl, project_name="test-project",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT agent_resolved_model FROM fact_agent_delegations "
            "WHERE tool_use_id = 'tu_nm'"
        ).fetchone()
        assert row == (None,)

class TestAsyncLaunchIsNotACompletion:
    """A background launch acknowledgment must not masquerade as a result.

    Since Claude Code v2.1.198+ subagents run in the background by default:
    the tool result returned at spawn time is an acknowledgment, not the
    agent's output. On a real corpus 719 of 941 delegations (76%) are
    `async_launched`, and on those rows three columns held values that read
    as valid and were not --

      completion_timestamp   the acknowledgment's timestamp, milliseconds
                             after the spawn
      seconds_to_completion  median 2.05s, versus 102.45s on the 192 rows
                             that really completed
      agent_output_text      literally "Async agent launched successfully."

    Claim: delete these and any aggregate over seconds_to_completion
    silently blends acknowledgment latency with real duration, with nothing
    in the row marking which is which -- and the bias runs the wrong way,
    because async is what long-running expensive delegations use. NULL is
    honest; a plausible wrong number is not.

    This is the honesty half only. Re-deriving the real metrics from the
    agent's own transcript is separate work -- see
    internal/plans/2026-08-01_agent_delegation_capture_gap.md.
    """

    def test_async_launch_nulls_the_misleading_columns(self, conn, tmp_path):
        from ccutils.etl.orchestrator import run_v15_etl

        jsonl = tmp_path / "async.jsonl"
        jsonl.write_text("\n".join(json.dumps(d) for d in [
            {"type": "user", "uuid": "u1", "sessionId": "async-s",
             "timestamp": "2026-04-19T10:00:00Z", "cwd": "/p",
             "gitBranch": "main", "version": "2.1.114",
             "message": {"role": "user", "content": "delegate"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "sessionId": "async-s", "timestamp": "2026-04-19T10:00:01Z",
             "requestId": "r1",
             "message": {"role": "assistant", "model": "claude-opus-5",
                         "content": [{"type": "tool_use", "id": "tu_async",
                                      "name": "Task",
                                      "input": {"description": "go",
                                                "subagent_type": "Explore",
                                                "prompt": "x"}}]}},
            {"type": "user", "uuid": "u2", "parentUuid": "a1",
             "sessionId": "async-s", "timestamp": "2026-04-19T10:00:03Z",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "tu_async",
                  "content": "Async agent launched successfully."}]},
             "toolUseResult": {
                 "isAsync": True, "status": "async_launched",
                 "agentId": "async-agent-1",
                 "resolvedModel": "claude-sonnet-5",
                 "description": "go"}},
        ]))
        run_v15_etl(conn, jsonl, project_name="test-project",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            """
            SELECT agent_is_async, completion_timestamp,
                   seconds_to_completion, agent_output_text,
                   agent_status, agent_resolved_model
            FROM fact_agent_delegations WHERE tool_use_id = 'tu_async'
            """
        ).fetchone()
        assert row[0] is True, "isAsync is stated in the payload; capture it"
        # The three that lied are NULL...
        assert row[1] is None, "completion_timestamp was the ack timestamp"
        assert row[2] is None, "seconds_to_completion was ack latency"
        assert row[3] is None, "agent_output_text was the ack text"
        # ...while everything genuinely stated at spawn time survives.
        assert row[4] == "async_launched"
        assert row[5] == "claude-sonnet-5"

    def test_synchronous_completion_keeps_its_metrics(
        self, conn, agent_session, tmp_path
    ):
        """The nulling must be gated on isAsync, not applied to everything."""
        _populate(conn, agent_session, tmp_path)
        rows = conn.execute(
            "SELECT agent_is_async, completion_timestamp, "
            "seconds_to_completion, agent_output_text "
            "FROM fact_agent_delegations"
        ).fetchall()
        assert rows
        for is_async, completion_ts, secs, output in rows:
            assert not is_async
            assert completion_ts is not None
            assert secs is not None
            assert output is not None


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

    def test_agent_session_key_resolves_when_subagent_also_loaded(
        self, conn, tmp_path
    ):
        """When both the parent session (with the Task tool_use) AND the
        subagent JSONL (which gets is_agent=TRUE + agent_id set) are
        loaded, agent_session_key on fact_agent_delegations resolves
        via dim_session.agent_id."""
        from ccutils.etl.orchestrator import run_v15_etl

        # Parent session with one Task whose toolUseResult carries agentId
        parent_jsonl = tmp_path / "parent.jsonl"
        parent_lines = [
            {"type": "user", "uuid": "u1", "sessionId": "parent-s",
             "timestamp": "2026-04-19T10:00:00Z", "cwd": "/p",
             "gitBranch": "main", "version": "2.1.114",
             "message": {"role": "user", "content": "delegate"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "sessionId": "parent-s", "timestamp": "2026-04-19T10:00:01Z",
             "requestId": "r1",
             "message": {"role": "assistant", "model": "claude-opus-4-7",
                         "content": [{"type": "tool_use", "id": "tu_link",
                                      "name": "Task",
                                      "input": {"description": "go",
                                                "subagent_type": "Explore",
                                                "prompt": "explore"}}]}},
            {"type": "user", "uuid": "u2", "parentUuid": "a1",
             "sessionId": "parent-s", "timestamp": "2026-04-19T10:00:30Z",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "tu_link",
                  "content": [{"type": "text", "text": "done"}]},
             ]},
             "toolUseResult": {
                 "agentId": "subagent-xyz", "agentType": "Explore",
                 "status": "completed",
                 "totalDurationMs": 1000, "totalTokens": 100,
                 "totalToolUseCount": 1,
             }},
        ]
        parent_jsonl.write_text("\n".join(json.dumps(d) for d in parent_lines))

        # Subagent JSONL on disk at the canonical layout
        sub_dir = (
            tmp_path / "projects" / "-Users-dev-myrepo"
            / "parent-s" / "subagents"
        )
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub_jsonl = sub_dir / "agent-subagent-xyz.jsonl"
        sub_lines = [
            {"type": "user", "uuid": "us1",
             "sessionId": "agent-subagent-xyz",
             "timestamp": "2026-04-19T10:00:05Z",
             "cwd": "/p", "gitBranch": "main", "version": "2.1.114",
             "message": {"role": "user", "content": "explore"}},
        ]
        sub_jsonl.write_text("\n".join(json.dumps(d) for d in sub_lines))

        run_v15_etl(conn, sub_jsonl, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        run_v15_etl(conn, parent_jsonl, project_name="test",
                    parquet_lake_root=tmp_path / "lake")

        row = conn.execute(
            """
            SELECT fad.agent_session_key, ds.session_id AS agent_session_id
            FROM fact_agent_delegations fad
            JOIN dim_session ds ON fad.agent_session_key = ds.session_key
            WHERE fad.tool_use_id = 'tu_link'
            """
        ).fetchone()
        assert row is not None, "agent_session_key did not resolve"
        assert row[1] == "agent-subagent-xyz"

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


class TestAsyncCompletionReconciliation:
    """Roadmap 0e step 2: re-derive async rollups from the AGENT's transcript.

    Claim each test encodes -- delete it and this breaks silently:

    - Since Claude Code v2.1.198+ the parent's tool result for a background
      spawn is a launch acknowledgment, so the parent side has no metrics at
      all. 721 of 943 delegations on the real corpus. The honesty fix NULLed
      the misleading values; these tests assert the values come BACK, derived
      from the agent's own session rows, which already carry usage and
      stop_reason.
    - The agent session is normally ETL'd AFTER its parent, so a per-session
      populator cannot see it. That ordering is what left agent_session_key
      NULL on 941/941 rows before it was derived instead of joined. These
      tests drive the parent FIRST and the agent SECOND on purpose: if the
      reconciliation is ever moved back into the per-session populator, they
      fail.
    - completion_state must distinguish a refused spawn from a finished one.
      29 of 30 NULL-agent_status rows on the real corpus never spawned an
      agent at all (fork-inside-fork, depth limit 3 of 3, cancellation).
    """

    def _parent_with_async_spawn(self, tmp_path, agent_id="abc123def"):
        """Parent session that background-launches one agent."""
        proj = tmp_path / "projects" / "-Users-dev-myrepo"
        parent_dir = proj / "parent-uuid"
        parent_dir.mkdir(parents=True, exist_ok=True)
        jsonl = proj / "parent-uuid.jsonl"
        lines = [
            {"type": "user", "uuid": "u1", "sessionId": "parent-uuid",
             "timestamp": "2026-04-19T10:00:00Z", "cwd": "/work",
             "gitBranch": "main", "version": "2.1.198",
             "message": {"role": "user", "content": "delegate it"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "sessionId": "parent-uuid", "timestamp": "2026-04-19T10:00:01Z",
             "requestId": "r1",
             "message": {"role": "assistant", "model": "claude-opus-5",
                         "content": [{"type": "tool_use", "id": "tu_async1",
                                      "name": "Agent",
                                      "input": {
                                          "description": "Audit the docs",
                                          "prompt": "audit",
                                          "subagent_type": "general-purpose",
                                      }}]}},
            # The acknowledgment: lands 2s after the spawn, describes nothing
            # about the agent. This is the shape 76% of the corpus has.
            {"type": "user", "uuid": "u2", "parentUuid": "a1",
             "sessionId": "parent-uuid", "timestamp": "2026-04-19T10:00:03Z",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "tu_async1",
                  "content": "Async agent launched successfully."}]},
             "toolUseResult": {"isAsync": True, "status": "async_launched",
                               "agentId": agent_id,
                               "description": "Audit the docs",
                               "resolvedModel": "claude-sonnet-5"}},
        ]
        jsonl.write_text("\n".join(json.dumps(d) for d in lines))
        return jsonl

    def _agent_transcript(self, tmp_path, agent_id="abc123def",
                          *, stop_reason: str | None = "end_turn"):
        """The agent's OWN file. Entries carry the PARENT's sessionId --
        that is the real contract (CLAUDE.md); identity is the filename."""
        d = (tmp_path / "projects" / "-Users-dev-myrepo"
             / "parent-uuid" / "subagents")
        d.mkdir(parents=True, exist_ok=True)
        agent_jsonl = d / f"agent-{agent_id}.jsonl"
        lines = [
            {"type": "user", "uuid": "au1", "sessionId": "parent-uuid",
             "timestamp": "2026-04-19T10:02:11Z", "cwd": "/work",
             "gitBranch": "main", "version": "2.1.198",
             "message": {"role": "user", "content": "audit"}},
            {"type": "assistant", "uuid": "aa1", "parentUuid": "au1",
             "sessionId": "parent-uuid", "timestamp": "2026-04-19T10:03:57Z",
             "requestId": "ar1",
             "message": {"role": "assistant", "model": "claude-sonnet-5",
                         "stop_reason": stop_reason,
                         "usage": {"input_tokens": 20, "output_tokens": 9788},
                         "content": [{"type": "text",
                                      "text": "audit complete"}]}},
        ]
        agent_jsonl.write_text("\n".join(json.dumps(d) for d in lines))
        return agent_jsonl

    def _reconcile(self, conn):
        from ccutils.etl.fact_agent_delegations import (
            populate_delegation_completion,
        )
        run = EtlRun.start(conn, source_path="reconcile")
        populate_delegation_completion(conn, run=run)

    def test_async_rollup_rederived_from_agent_transcript(self, conn, tmp_path):
        """The whole point: an async delegation gets REAL metrics back."""
        from ccutils.etl.orchestrator import run_v15_etl

        parent = self._parent_with_async_spawn(tmp_path)
        agent = self._agent_transcript(tmp_path)
        # Parent FIRST, agent SECOND -- the ordering that defeats a
        # per-session populator.
        run_v15_etl(conn, parent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")
        run_v15_etl(conn, agent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")
        self._reconcile(conn)

        row = conn.execute("""
            SELECT completion_state, agent_derived_io_tokens,
                   completion_timestamp, seconds_to_completion
            FROM fact_agent_delegations WHERE tool_use_id = 'tu_async1'
        """).fetchone()
        assert row[0] == "completed", "terminal stop_reason => completed"
        # Reads agent_derived_io_tokens, NOT agent_total_tokens: the derived
        # sum does not reproduce the API's stated totalTokens (12/188 within
        # 10% on ground truth) so the two are kept in separate columns.
        assert row[1] == 9808, "20 in + 9788 out, summed from the agent file"
        # 10:03:57 is the agent's last timestamp, NOT 10:00:03 (the ack).
        assert row[2].strftime("%H:%M:%S") == "10:03:57"
        # 10:00:01 spawn -> 10:03:57 done = 236s, not the 2s ack latency.
        assert row[3] == pytest.approx(236.0, abs=1.0)

    def test_in_flight_agent_leaves_rollups_null(self, conn, tmp_path):
        """No terminal stop_reason => in_flight, and a partial sum is NOT
        written. A partial sum is indistinguishable from a fast agent."""
        from ccutils.etl.orchestrator import run_v15_etl

        parent = self._parent_with_async_spawn(tmp_path)
        agent = self._agent_transcript(tmp_path, stop_reason=None)
        run_v15_etl(conn, parent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")
        run_v15_etl(conn, agent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")
        self._reconcile(conn)

        row = conn.execute("""
            SELECT completion_state, agent_total_tokens, seconds_to_completion
            FROM fact_agent_delegations WHERE tool_use_id = 'tu_async1'
        """).fetchone()
        assert row[0] == "no_completion_recorded"
        assert row[1] is None, "no partial sums -- NULL is honest"
        assert row[2] is None

    def test_api_stated_and_derived_tokens_never_share_a_column(
        self, conn, tmp_path
    ):
        """`agent_total_tokens` is the API's number or nothing.

        The derived input+output sum does NOT reproduce the API's stated
        `totalTokens`. Ground truth: 188 synchronous delegations carry both a
        stated rollup and an ingested agent transcript, so the async
        derivation can be scored where the answer is known. Duration matched
        188/188 and tool count 188/188 exactly -- but tokens matched only
        12/188 within 10%, with the per-row ratio spanning p10 0.063 to p90
        1.004. No formula reconciles them (in+out, out only, +cache_creation,
        +cache_read, total_uncached_equivalent, the 5m/1h splits), it is not a
        capture gap (median 23 assistant records, 23 carrying usage), and not
        a nested-agent rollup (all 188 spawned none). They measure different
        things.

        Writing the derived sum into the stated column made async delegations
        look 3x cheaper than sync ones (median 19,444 vs 61,362) while
        measuring 2x longer -- backwards, and silently, in the one column a
        cost analysis would reach for.

        Delete this and the two definitions merge back into one column.
        """
        from ccutils.etl.orchestrator import run_v15_etl

        parent = self._parent_with_async_spawn(tmp_path)
        agent = self._agent_transcript(tmp_path)
        run_v15_etl(conn, parent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")
        run_v15_etl(conn, agent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")
        self._reconcile(conn)

        row = conn.execute("""
            SELECT completion_state, agent_total_tokens,
                   agent_derived_io_tokens, agent_total_duration_ms,
                   agent_total_tool_use_count
            FROM fact_agent_delegations WHERE tool_use_id = 'tu_async1'
        """).fetchone()
        assert row[0] == "completed"
        assert row[1] is None, (
            "the API never stated a token count for a background launch; "
            "the derived sum is a different measure and must not stand in"
        )
        assert row[2] == 9808, (
            "20 in + 9788 out, summed from the agent's own transcript, kept "
            "in its own column"
        )
        # Non-vacuity: the derivations that DID validate against ground truth
        # stay in the shared columns. If this drifts to None the test above
        # would pass for the wrong reason -- a reconciliation that wrote
        # nothing at all.
        assert row[3] is not None, "duration derivation validated 188/188"
        # row[4] (tool count) is deliberately not asserted here: this
        # fixture's agent calls no tools, so there is no fact_tool_uses row
        # to count and the value is NULL for a reason unrelated to tokens.

    def test_sync_delegation_keeps_api_stated_tokens(self, conn, tmp_path):
        """The other half: a synchronous row's stated tokens are untouched,
        and it gets no derived value invented for it."""
        from ccutils.etl.orchestrator import run_v15_etl

        proj = tmp_path / "projects" / "-Users-dev-myrepo"
        proj.mkdir(parents=True, exist_ok=True)
        jsonl = proj / "syncdel.jsonl"
        lines = [
            {"type": "user", "uuid": "u1", "sessionId": "syncdel",
             "timestamp": "2026-04-19T10:00:00Z", "cwd": "/work",
             "gitBranch": "main", "version": "2.1.198",
             "message": {"role": "user", "content": "delegate"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "sessionId": "syncdel", "timestamp": "2026-04-19T10:00:01Z",
             "requestId": "r1",
             "message": {"role": "assistant", "model": "claude-opus-5",
                         "content": [{"type": "tool_use", "id": "tu_sync1",
                                      "name": "Agent",
                                      "input": {"description": "d",
                                                "prompt": "p"}}]}},
            {"type": "user", "uuid": "u2", "parentUuid": "a1",
             "sessionId": "syncdel", "timestamp": "2026-04-19T10:05:00Z",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "tu_sync1",
                  "content": "done"}]},
             "toolUseResult": {"status": "completed", "agentId": "zzz999",
                               "totalTokens": 62150,
                               "totalDurationMs": 91747,
                               "totalToolUseCount": 14}},
        ]
        jsonl.write_text("\n".join(json.dumps(d) for d in lines))
        run_v15_etl(conn, jsonl, project_name="p",
                    parquet_lake_root=tmp_path / "lake")
        self._reconcile(conn)

        row = conn.execute("""
            SELECT agent_total_tokens, agent_derived_io_tokens
            FROM fact_agent_delegations WHERE tool_use_id = 'tu_sync1'
        """).fetchone()
        assert row[0] == 62150, "the API's stated number, preserved verbatim"
        assert row[1] is None, (
            "no agent transcript in the warehouse => nothing to derive; a "
            "zero here would read as 'the agent used no tokens'"
        )

    def test_successful_background_launch_is_not_spawn_failed(
        self, conn, tmp_path
    ):
        """A launch that SUCCEEDED but named no agent is not a failed spawn.

        Corpus evidence: 30 rows matched the spawn_failed branch, but one of
        them reads "Fork started - processing in background" -- a successful
        background launch that simply carried no agentId. The 29 genuine
        failures all carry a stated `is_error: true`; this one omits the
        field. Without gating on that stated signal the branch over-matches
        and reports a launch that worked as a failure.

        Delete this and spawn_failed silently absorbs successful launches
        whose agent id is not stated.
        """
        from ccutils.etl.orchestrator import run_v15_etl

        proj = tmp_path / "projects" / "-Users-dev-myrepo"
        proj.mkdir(parents=True, exist_ok=True)
        jsonl = proj / "forked.jsonl"
        lines = [
            {"type": "user", "uuid": "u1", "sessionId": "forked",
             "timestamp": "2026-04-19T10:00:00Z", "cwd": "/work",
             "gitBranch": "main", "version": "2.1.198",
             "message": {"role": "user", "content": "delegate"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "sessionId": "forked", "timestamp": "2026-04-19T10:00:01Z",
             "requestId": "r1",
             "message": {"role": "assistant", "model": "claude-opus-5",
                         "content": [{"type": "tool_use", "id": "tu_forked",
                                      "name": "Agent",
                                      "input": {"description": "fork",
                                                "prompt": "go"}}]}},
            # No agentId, and critically NO is_error -- the launch worked.
            {"type": "user", "uuid": "u2", "parentUuid": "a1",
             "sessionId": "forked", "timestamp": "2026-04-19T10:00:02Z",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "tu_forked",
                  "content": "Fork started - processing in background"}]}},
        ]
        jsonl.write_text("\n".join(json.dumps(d) for d in lines))
        run_v15_etl(conn, jsonl, project_name="p",
                    parquet_lake_root=tmp_path / "lake")
        self._reconcile(conn)

        state = conn.execute("""
            SELECT completion_state FROM fact_agent_delegations
            WHERE tool_use_id = 'tu_forked'
        """).fetchone()[0]
        assert state != "spawn_failed", (
            "no stated is_error => the spawn did not fail; with no agent to "
            "reconcile against, NULL (not reconciled) is the honest value"
        )
        assert state is None

    def test_no_completion_state_asserts_liveness(self, conn, tmp_path):
        """No emitted state may claim the agent was running. It is not
        decidable from the transcript.

        The state formerly called `in_flight_at_ingest` asserted exactly
        that, and the corpus contradicts it: of 101 rows carrying it, 98
        ended mid-tool-loop a median of 15.7 DAYS before the ETL ran and
        only 2 had a last message recent enough to still be running. What
        is actually known is narrower -- the agent's transcript records no
        completion -- and `no_completion_recorded` says only that.

        This mirrors the `semantic_session_behavior` guard that forbids
        `archetype`/`label`/`bucket` column names: a claim the data cannot
        support must not re-enter the schema through a value name.
        """
        from ccutils.etl.orchestrator import run_v15_etl

        parent = self._parent_with_async_spawn(tmp_path)
        agent = self._agent_transcript(tmp_path, stop_reason=None)
        run_v15_etl(conn, parent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")
        run_v15_etl(conn, agent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")
        self._reconcile(conn)

        states = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT completion_state FROM fact_agent_delegations "
                "WHERE completion_state IS NOT NULL"
            ).fetchall()
        ]
        # Non-vacuity: the unfinished-agent case must actually be present,
        # or this passes on an empty set.
        assert "no_completion_recorded" in states
        for s in states:
            for banned in ("in_flight", "running", "active", "live"):
                assert banned not in s.lower(), (
                    f"completion_state {s!r} asserts liveness, which the "
                    "transcript cannot establish"
                )

    def test_refused_spawn_is_spawn_failed_not_completed(self, conn, tmp_path):
        """29 of 30 NULL-status rows on the real corpus: the agent was never
        created. Distinct from completed, in_flight, and from plain NULL."""
        from ccutils.etl.orchestrator import run_v15_etl

        proj = tmp_path / "projects" / "-Users-dev-myrepo"
        proj.mkdir(parents=True, exist_ok=True)
        jsonl = proj / "refused.jsonl"
        lines = [
            {"type": "user", "uuid": "u1", "sessionId": "refused",
             "timestamp": "2026-04-19T10:00:00Z", "cwd": "/work",
             "gitBranch": "main", "version": "2.1.198",
             "message": {"role": "user", "content": "delegate"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "sessionId": "refused", "timestamp": "2026-04-19T10:00:01Z",
             "requestId": "r1",
             "message": {"role": "assistant", "model": "claude-opus-5",
                         "content": [{"type": "tool_use", "id": "tu_refused",
                                      "name": "Agent",
                                      "input": {"description": "nested",
                                                "prompt": "go"}}]}},
            {"type": "user", "uuid": "u2", "parentUuid": "a1",
             "sessionId": "refused", "timestamp": "2026-04-19T10:00:02Z",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "tu_refused",
                  "content": "Subagent nesting limit reached (depth 3 of 3). "
                             "Complete this task directly using your tools.",
                  "is_error": True}]}},
        ]
        jsonl.write_text("\n".join(json.dumps(d) for d in lines))
        run_v15_etl(conn, jsonl, project_name="p",
                    parquet_lake_root=tmp_path / "lake")
        self._reconcile(conn)

        row = conn.execute("""
            SELECT completion_state, agent_session_key, agent_total_tokens
            FROM fact_agent_delegations WHERE tool_use_id = 'tu_refused'
        """).fetchone()
        assert row[0] == "spawn_failed", (
            "no agentId in the payload and an error result => never spawned"
        )
        assert row[1] is None, "no agent session to link to"
        assert row[2] is None

    def test_sync_delegation_keeps_its_parent_side_metrics(self, conn, tmp_path):
        """Non-vacuity guard: the 222 synchronous rows already work. The
        reconciliation must not overwrite or NULL them."""
        from ccutils.etl.orchestrator import run_v15_etl

        proj = tmp_path / "projects" / "-Users-dev-myrepo"
        proj.mkdir(parents=True, exist_ok=True)
        jsonl = proj / "sync.jsonl"
        lines = [
            {"type": "user", "uuid": "u1", "sessionId": "sync-s",
             "timestamp": "2026-04-19T10:00:00Z", "cwd": "/work",
             "gitBranch": "main", "version": "2.1.114",
             "message": {"role": "user", "content": "delegate"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "sessionId": "sync-s", "timestamp": "2026-04-19T10:00:01Z",
             "requestId": "r1",
             "message": {"role": "assistant", "model": "claude-opus-5",
                         "content": [{"type": "tool_use", "id": "tu_sync",
                                      "name": "Agent",
                                      "input": {"description": "quick",
                                                "prompt": "go"}}]}},
            {"type": "user", "uuid": "u2", "parentUuid": "a1",
             "sessionId": "sync-s", "timestamp": "2026-04-19T10:01:43Z",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "tu_sync",
                  "content": "done"}]},
             "toolUseResult": {"status": "completed", "totalTokens": 4242,
                               "totalDurationMs": 102450,
                               "totalToolUseCount": 7,
                               "agentId": "syncagent1"}},
        ]
        jsonl.write_text("\n".join(json.dumps(d) for d in lines))
        run_v15_etl(conn, jsonl, project_name="p",
                    parquet_lake_root=tmp_path / "lake")
        self._reconcile(conn)

        row = conn.execute("""
            SELECT completion_state, agent_total_tokens, seconds_to_completion
            FROM fact_agent_delegations WHERE tool_use_id = 'tu_sync'
        """).fetchone()
        assert row[0] == "completed"
        assert row[1] == 4242, "parent-side metric preserved, not clobbered"
        assert row[2] == pytest.approx(102.0, abs=1.0)


class TestSoftDeletedResultsDoNotFanOut:
    """A soft-deleted fact_tool_results twin must not fan out the inbound.

    Same defect class as the file-operations test of the same name: this
    populator derives from fact_tool_uses JOIN fact_tool_results, and a
    repaired (soft-deleted) duplicate result row must not re-enter through
    the join. Observed on a real pre-R23 warehouse: 1 session failed with
    2 duplicate delegation_keys after the open-time repair had run.
    """

    def test_repaired_twin_does_not_duplicate_the_inbound(
        self, conn, agent_session, tmp_path
    ):
        _populate(conn, agent_session, tmp_path)
        before = conn.execute(
            "SELECT COUNT(*) FROM fact_agent_delegations WHERE NOT is_deleted"
        ).fetchone()[0]
        assert before > 0, "fixture must produce at least one delegation"

        conn.execute(
            "INSERT INTO fact_tool_results SELECT * REPLACE ("
            "  'e_twin' AS entry_id, TRUE AS is_deleted,"
            "  current_timestamp AS deleted_at)"
            "FROM fact_tool_results WHERE tool_name = 'Task' "
            "ORDER BY tool_use_id LIMIT 1"
        )

        _populate(conn, agent_session, tmp_path)  # must not raise

        dup = conn.execute(
            "SELECT COUNT(*) FROM (SELECT delegation_key FROM "
            "fact_agent_delegations WHERE NOT is_deleted GROUP BY 1 "
            "HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        assert dup == 0
