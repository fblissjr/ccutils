"""semantic_agent_delegations: the delegation edge, as a view.

Grain: one row per Agent / Task tool call in `fact_tool_calls` (the
parent-side spawn), joined to the agent's own session where its transcript
is in the warehouse.

Until 1.0.0 this was a table, `semantic_agent_delegations`, written by a
per-session populator and then repaired by a post-loop reconciliation pass,
because a parent is normally loaded before its agents and a per-session
populator cannot see rows that do not exist yet. Every column in it was a
projection of `fact_tool_calls` or a rollup of the agent's own facts, so it
is a view now: nothing to order, nothing to repair, nothing that can go
stale or be skipped by one entry point.

These tests were ported from the table's suite. What each one guards did
not change with the storage: a stated value and a derived one never share a
column where they measure different things, a background launch's
acknowledgment is never reported as the agent's outcome, and no state claims
more than the transcript records. The fixtures still load the parent FIRST
and the agent SECOND, because that ordering is what the table could not
survive without its repair pass.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.orchestrator import run_v15_etl


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
    """Load one transcript through the real pipeline. There is no delegation
    step to call: the view reads what the ordinary populators wrote."""
    run_v15_etl(conn, jsonl_path, project_name="test-project",
                parquet_lake_root=tmp_path / "lake")


class TestOneRowPerSpawn:
    def test_one_row_per_task_tool_use(self, conn, agent_session, tmp_path):
        _populate(conn, agent_session, tmp_path)
        n = conn.execute(
            "SELECT COUNT(*) FROM semantic_agent_delegations"
        ).fetchone()[0]
        assert n == 2

    def test_captures_task_input(self, conn, agent_session, tmp_path):
        _populate(conn, agent_session, tmp_path)
        row = conn.execute(
            """
            SELECT task_description, task_prompt, subagent_type
            FROM semantic_agent_delegations
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
            FROM semantic_agent_delegations
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
            SELECT agent_status
            FROM semantic_agent_delegations
            WHERE tool_use_id = 'tu_t2'
            """
        ).fetchone()
        assert row[0] == "interrupted"

    def test_result_time_is_stated_and_nothing_is_derived_without_the_agent(
        self, conn, agent_session, tmp_path
    ):
        """The parent's result time is stated; with no agent transcript in
        the warehouse there is nothing to derive a completion from."""
        _populate(conn, agent_session, tmp_path)
        row = conn.execute(
            """
            SELECT EXTRACT(EPOCH FROM (result_timestamp - delegation_timestamp)),
                   derived_completion_timestamp, derived_seconds_to_completion
            FROM semantic_agent_delegations WHERE tool_use_id = 'tu_t1'
            """
        ).fetchone()
        # Delegation at 10:00:01, result at 10:01:00 -> 59s, as the parent saw it.
        assert row == (59.0, None, None)

    def test_parent_session_key_is_the_delegating_session(
        self, conn, agent_session, tmp_path
    ):
        _populate(conn, agent_session, tmp_path)
        rows = conn.execute(
            "SELECT parent_session_key, parent_session_id "
            "FROM semantic_agent_delegations"
        ).fetchall()
        assert len(rows) == 2
        for parent_sk, parent_id in rows:
            assert parent_id == "agent-s"
            assert parent_sk == hashlib.md5(b"agent-s").hexdigest()

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
            FROM semantic_agent_delegations fad
            JOIN fact_tool_calls ftr USING (tool_use_id)
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
            "SELECT agent_session_key FROM semantic_agent_delegations "
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
            "FROM semantic_agent_delegations ORDER BY tool_use_id"
        ).fetchall()
        assert rows, "no delegations produced"
        models = {r[1] for r in rows}
        assert models == {"claude-sonnet-5"}, models

    def test_resolved_model_null_when_absent(self, conn, tmp_path):
        """Older transcripts predate resolvedModel; absence must not error."""
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
            "SELECT agent_resolved_model FROM semantic_agent_delegations "
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

    def test_async_launch_ack_is_never_reported_as_a_completion(self, conn, tmp_path):
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
        _populate(conn, jsonl, tmp_path)
        row = conn.execute(
            """
            SELECT agent_is_async, derived_completion_timestamp,
                   derived_seconds_to_completion, agent_output_text,
                   agent_status, agent_resolved_model, result_timestamp
            FROM semantic_agent_delegations WHERE tool_use_id = 'tu_async'
            """
        ).fetchone()
        assert row[0] is True, "isAsync is stated in the payload; capture it"
        # Nothing under a completion name holds the acknowledgment...
        assert row[1] is None, "no agent transcript, so no completion to derive"
        assert row[2] is None, "and no duration: 2s here would be ack latency"
        assert row[3] is None, "agent_output_text was the ack text"
        # ...while everything genuinely stated at spawn time survives, the
        # acknowledgment's own time included, under a name that says what it is.
        assert row[4] == "async_launched"
        assert row[5] == "claude-sonnet-5"
        assert row[6].strftime("%H:%M:%S") == "10:00:03"
        columns = {r[0] for r in conn.execute(
            "DESCRIBE semantic_agent_delegations").fetchall()}
        assert not {"completion_timestamp", "seconds_to_completion"} & columns, (
            "an unqualified completion column would hold the ack on async rows"
        )

    def test_synchronous_delegation_keeps_what_its_parent_stated(
        self, conn, agent_session, tmp_path
    ):
        """The withholding is gated on isAsync, not applied to everything."""
        _populate(conn, agent_session, tmp_path)
        rows = conn.execute(
            "SELECT agent_is_async, result_timestamp, "
            "agent_total_duration_ms, agent_output_text "
            "FROM semantic_agent_delegations"
        ).fetchall()
        assert len(rows) == 2
        for is_async, result_ts, duration, output in rows:
            assert not is_async
            assert result_ts is not None
            assert duration is not None
            assert output is not None

    def test_agent_session_key_resolves_when_subagent_also_loaded(
        self, conn, tmp_path
    ):
        """When both the parent session (with the Task tool_use) AND the
        subagent JSONL (which gets is_agent=TRUE + agent_id set) are
        loaded, agent_session_key on semantic_agent_delegations resolves
        via dim_session.agent_id."""
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
            FROM semantic_agent_delegations fad
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
            "SELECT COUNT(*) FROM semantic_agent_delegations"
        ).fetchone()[0]
        assert n == 0


class TestAsyncCompletionIsDerivedFromTheAgent:
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
      tests drive the parent FIRST and the agent SECOND on purpose, and
      run no step after the second load: a view sees the agent's rows the
      moment they exist. Under the table this needed a post-loop pass, and a
      warehouse built by an entry point that skipped the pass was wrong.
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

    def test_async_rollup_derived_from_agent_transcript(self, conn, tmp_path):
        """The whole point: an async delegation gets REAL metrics."""
        parent = self._parent_with_async_spawn(tmp_path)
        agent = self._agent_transcript(tmp_path)
        # Parent FIRST, agent SECOND -- the ordering that defeated a
        # per-session populator. Nothing runs after the second load.
        run_v15_etl(conn, parent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")
        run_v15_etl(conn, agent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")

        row = conn.execute("""
            SELECT completion_state, derived_io_tokens,
                   derived_completion_timestamp, derived_seconds_to_completion,
                   derived_duration_ms
            FROM semantic_agent_delegations WHERE tool_use_id = 'tu_async1'
        """).fetchone()
        assert row[0] == "completed", "terminal stop_reason => completed"
        assert row[1] == 9808, "20 in + 9788 out, summed from the agent file"
        # 10:03:57 is the agent's last timestamp, NOT 10:00:03 (the ack).
        assert row[2].strftime("%H:%M:%S") == "10:03:57"
        # 10:00:01 spawn -> 10:03:57 done = 236s, not the 2s ack latency.
        assert row[3] == pytest.approx(236.0, abs=1.0)
        # 10:02:11 first entry -> 10:03:57 last = 106s of agent time.
        assert row[4] == 106000.0

    def test_agent_final_report_is_derived(self, conn, tmp_path):
        """The agent's final report is the value of the delegation.

        On a background launch `agent_output_text` would hold the launch
        acknowledgment, so it is NULL -- which alone would leave the
        warehouse recording that work was delegated and nothing about what
        came back. The agent's own terminal assistant message is in the
        warehouse already; the report is derived from it, in its own column.
        """
        parent = self._parent_with_async_spawn(tmp_path)
        agent = self._agent_transcript(tmp_path)
        run_v15_etl(conn, parent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")
        run_v15_etl(conn, agent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")

        stated, derived = conn.execute("""
            SELECT agent_output_text, derived_output_text
            FROM semantic_agent_delegations WHERE tool_use_id = 'tu_async1'
        """).fetchone()
        assert stated is None, "the async ack is not the agent's output"
        assert derived == "audit complete"

    def test_unfinished_agent_output_is_not_derived(self, conn, tmp_path):
        """A partial answer is worse than no answer.

        An unfinished agent's last message is indistinguishable from a
        finished one's, so reporting it would present a half-done delegation
        as complete.
        """
        parent = self._parent_with_async_spawn(tmp_path)
        agent = self._agent_transcript(tmp_path, stop_reason=None)
        run_v15_etl(conn, parent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")
        run_v15_etl(conn, agent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")

        state, derived = conn.execute("""
            SELECT completion_state, derived_output_text
            FROM semantic_agent_delegations WHERE tool_use_id = 'tu_async1'
        """).fetchone()
        assert state != "completed"
        assert derived is None

    def test_synchronous_delegation_keeps_its_stated_output(
        self, conn, agent_session, tmp_path
    ):
        """A sync delegation's tool result IS the agent's output, and stays.

        The table's version of this test queried a warehouse nothing had been
        loaded into, so it passed on an empty set whatever the code did.
        """
        _populate(conn, agent_session, tmp_path)
        rows = dict(conn.execute(
            "SELECT tool_use_id, agent_output_text FROM semantic_agent_delegations"
        ).fetchall())
        assert "Findings report" in rows["tu_t1"]
        assert rows["tu_t2"] == "Interrupted"

    def test_no_recorded_completion_leaves_every_derived_value_null(self, conn, tmp_path):
        """No terminal stop_reason => no partial sums. A partial sum is
        indistinguishable from a fast agent."""
        parent = self._parent_with_async_spawn(tmp_path)
        agent = self._agent_transcript(tmp_path, stop_reason=None)
        run_v15_etl(conn, parent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")
        run_v15_etl(conn, agent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")

        row = conn.execute("""
            SELECT completion_state, derived_io_tokens, derived_duration_ms,
                   derived_tool_use_count, derived_seconds_to_completion,
                   derived_completion_timestamp, derived_output_text
            FROM semantic_agent_delegations WHERE tool_use_id = 'tu_async1'
        """).fetchone()
        assert row[0] == "no_completion_recorded"
        assert row[1:] == (None,) * 6, "no partial sums -- NULL is honest"

    def test_stated_and_derived_never_share_a_column(self, conn, tmp_path):
        """A column holds what the API stated or what was derived, never both.

        Tokens are why. The derived input+output sum does NOT reproduce the
        API's stated `totalTokens`. Ground truth: 188 synchronous delegations
        carry both a stated rollup and an ingested agent transcript, so the
        derivation can be scored where the answer is known. Duration matched
        188/188 and tool count 188/188 exactly -- but tokens matched only
        12/188 within 10%, with the per-row ratio spanning p10 0.063 to p90
        1.004. No formula reconciles them (in+out, out only, +cache_creation,
        +cache_read, total_uncached_equivalent, the 5m/1h splits), it is not a
        capture gap (median 23 assistant records, 23 carrying usage), and not
        a nested-agent rollup (all 188 spawned none). They measure different
        things. Writing the derived sum into the stated column made async
        delegations look 3x cheaper than sync ones (median 19,444 vs 61,362)
        while measuring 2x longer.

        Duration and tool count DID validate, and the table kept those two
        blended under one name (stated on sync rows, derived on async). That
        ended with the view: a reader of `agent_total_duration_ms` should
        not need `agent_is_async` to know whose number it is.

        Delete this and the two provenances merge back into one column.
        """
        parent = self._parent_with_async_spawn(tmp_path)
        agent = self._agent_transcript(tmp_path)
        run_v15_etl(conn, parent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")
        run_v15_etl(conn, agent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")

        row = conn.execute("""
            SELECT completion_state,
                   agent_total_tokens, agent_total_duration_ms,
                   agent_total_tool_use_count,
                   derived_io_tokens, derived_duration_ms,
                   derived_tool_use_count
            FROM semantic_agent_delegations WHERE tool_use_id = 'tu_async1'
        """).fetchone()
        assert row[0] == "completed"
        assert row[1:4] == (None, None, None), (
            "the API states no rollup for a background launch, and a derived "
            "value must not stand in under a stated name"
        )
        # Non-vacuity: the derivation ran. If these drift to None the
        # assertion above would pass for the wrong reason -- a view that
        # derived nothing at all.
        assert row[4] == 9808, "20 in + 9788 out, from the agent's own usage"
        assert row[5] == 106000.0
        assert row[6] == 0, "this fixture's agent calls no tools"

    def test_sync_delegation_keeps_api_stated_tokens(self, conn, tmp_path):
        """The other half: a synchronous row's stated tokens are untouched,
        and it gets no derived value invented for it."""
        proj = tmp_path / "projects" / "-Users-dev-myrepo"
        proj.mkdir(parents=True, exist_ok=True)
        jsonl = proj / "syncdel.jsonl"
        jsonl.write_text("\n".join(json.dumps(d) for d in self._sync_parent(
            "syncdel", "tu_sync1", agent_id="zzz999", tokens=62150,
            duration_ms=91747, tool_uses=14)))
        run_v15_etl(conn, jsonl, project_name="p",
                    parquet_lake_root=tmp_path / "lake")

        row = conn.execute("""
            SELECT agent_total_tokens, derived_io_tokens
            FROM semantic_agent_delegations WHERE tool_use_id = 'tu_sync1'
        """).fetchone()
        assert row[0] == 62150, "the API's stated number, preserved verbatim"
        assert row[1] is None, (
            "no agent transcript in the warehouse => nothing to derive; a "
            "zero here would read as 'the agent used no tokens'"
        )

    @staticmethod
    def _sync_parent(session_id, tool_use_id, *, agent_id, tokens, duration_ms,
                     tool_uses):
        """A parent that ran one agent synchronously and saw it complete."""
        return [
            {"type": "user", "uuid": "u1", "sessionId": session_id,
             "timestamp": "2026-04-19T10:00:00Z", "cwd": "/work",
             "gitBranch": "main", "version": "2.1.114",
             "message": {"role": "user", "content": "delegate"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "sessionId": session_id, "timestamp": "2026-04-19T10:00:01Z",
             "requestId": "r1",
             "message": {"role": "assistant", "model": "claude-opus-5",
                         "content": [{"type": "tool_use", "id": tool_use_id,
                                      "name": "Agent",
                                      "input": {"description": "quick",
                                                "prompt": "go"}}]}},
            {"type": "user", "uuid": "u2", "parentUuid": "a1",
             "sessionId": session_id, "timestamp": "2026-04-19T10:05:00Z",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": tool_use_id,
                  "content": "done"}]},
             "toolUseResult": {"status": "completed", "agentId": agent_id,
                               "totalTokens": tokens,
                               "totalDurationMs": duration_ms,
                               "totalToolUseCount": tool_uses}},
        ]

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

        state = conn.execute("""
            SELECT completion_state FROM semantic_agent_delegations
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
        parent = self._parent_with_async_spawn(tmp_path)
        agent = self._agent_transcript(tmp_path, stop_reason=None)
        run_v15_etl(conn, parent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")
        run_v15_etl(conn, agent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")

        states = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT completion_state FROM semantic_agent_delegations "
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

        row = conn.execute("""
            SELECT completion_state, agent_session_key, agent_total_tokens
            FROM semantic_agent_delegations WHERE tool_use_id = 'tu_refused'
        """).fetchone()
        assert row[0] == "spawn_failed", (
            "no agentId in the payload and an error result => never spawned"
        )
        assert row[1] is None, "no agent session to link to"
        assert row[2] is None

    def test_sync_delegation_without_its_agent_keeps_stated_and_derives_nothing(
        self, conn, tmp_path
    ):
        """What the parent saw stays stated; what nobody can derive stays NULL.

        This is the known cost of separating the two, accepted on
        2026-09-10: a synchronous delegation whose agent transcript was
        pruned upstream has no derived outcome at all. The table answered
        'completed' here by copying the parent's stated status into the
        derived state. The stated status is still on the row, under its own
        name, and a reader who wants either reading can have it.
        """
        proj = tmp_path / "projects" / "-Users-dev-myrepo"
        proj.mkdir(parents=True, exist_ok=True)
        jsonl = proj / "sync.jsonl"
        jsonl.write_text("\n".join(json.dumps(d) for d in self._sync_parent(
            "sync-s", "tu_sync", agent_id="syncagent1", tokens=4242,
            duration_ms=102450, tool_uses=7)))
        run_v15_etl(conn, jsonl, project_name="p",
                    parquet_lake_root=tmp_path / "lake")

        row = conn.execute("""
            SELECT agent_status, agent_total_tokens, agent_total_duration_ms,
                   agent_total_tool_use_count,
                   completion_state, derived_duration_ms, derived_tool_use_count
            FROM semantic_agent_delegations WHERE tool_use_id = 'tu_sync'
        """).fetchone()
        assert row[:4] == ("completed", 4242, 102450.0, 7), "stated, untouched"
        assert row[4:] == (None, None, None), "no transcript, nothing derived"

    def test_sync_delegation_with_its_agent_carries_both_side_by_side(
        self, conn, tmp_path
    ):
        """Where both exist they sit in different columns and can be compared.

        This is the shape `ccutils audit` scores: the stated tool count
        against the one counted from the agent's own calls.
        """
        proj = tmp_path / "projects" / "-Users-dev-myrepo"
        (proj / "parent-uuid").mkdir(parents=True, exist_ok=True)
        jsonl = proj / "parent-uuid.jsonl"
        jsonl.write_text("\n".join(json.dumps(d) for d in self._sync_parent(
            "parent-uuid", "tu_both", agent_id="abc123def", tokens=5000,
            duration_ms=106000, tool_uses=0)))
        agent = self._agent_transcript(tmp_path)
        run_v15_etl(conn, jsonl, project_name="p",
                    parquet_lake_root=tmp_path / "lake")
        run_v15_etl(conn, agent, project_name="p",
                    parquet_lake_root=tmp_path / "lake")

        row = conn.execute("""
            SELECT agent_status, completion_state,
                   agent_total_duration_ms, derived_duration_ms,
                   agent_total_tool_use_count, derived_tool_use_count,
                   agent_total_tokens, derived_io_tokens
            FROM semantic_agent_delegations WHERE tool_use_id = 'tu_both'
        """).fetchone()
        assert row[:2] == ("completed", "completed")
        assert row[2:4] == (106000.0, 106000.0)
        assert row[4:6] == (0, 0)
        assert row[6:] == (5000, 9808), "two measures, not one corrected by the other"

        from ccutils.audit import check_delegation_ground_truth

        assert list(check_delegation_ground_truth(conn)) == []
        # The oracle can fail: a child tool call the parent never counted.
        conn.execute(
            "UPDATE fact_tool_calls SET session_id = 'agent-abc123def', "
            "session_key = md5('agent-abc123def') WHERE tool_use_id = 'tu_both'"
        )
        conn.execute(
            "INSERT INTO fact_tool_calls SELECT * REPLACE ("
            "  'parent-uuid' AS session_id, md5('parent-uuid') AS session_key,"
            "  'tu_both2' AS tool_use_id, 'e_both2' AS entry_id) "
            "FROM fact_tool_calls WHERE tool_use_id = 'tu_both'"
        )
        findings = list(check_delegation_ground_truth(conn))
        assert findings and findings[0].check == "delegation_ground_truth"

class TestSoftDeletedToolCallsStayOut:
    """A soft-deleted `fact_tool_calls` row is not a delegation.

    The table had this as a fan-out test: a repaired (soft-deleted) duplicate
    re-entering its populator's join raised on the natural key. A view has no
    key assertion to catch that, so the filter is all there is, and without
    it the twin would simply be a second row for the same spawn.
    """

    def test_a_soft_deleted_twin_is_not_a_second_delegation(
        self, conn, agent_session, tmp_path
    ):
        _populate(conn, agent_session, tmp_path)
        before = conn.execute(
            "SELECT COUNT(*) FROM semantic_agent_delegations"
        ).fetchone()[0]
        assert before == 2, "fixture must produce its two delegations"

        conn.execute(
            "INSERT INTO fact_tool_calls SELECT * REPLACE ("
            "  'e_twin' AS entry_id, TRUE AS is_deleted,"
            "  current_timestamp AS deleted_at)"
            "FROM fact_tool_calls WHERE tool_name = 'Task' "
            "ORDER BY tool_use_id LIMIT 1"
        )

        rows = conn.execute(
            "SELECT delegation_key FROM semantic_agent_delegations"
        ).fetchall()
        assert len(rows) == before
        assert len({r[0] for r in rows}) == before


class TestItIsAViewAndNothingRepairsIt:
    """The table, its populator and the reconciliation run are gone.

    Guards against a half-restoration: a table that is created and never
    filled reads as "no delegations", and a reconciliation run kind with no
    pass behind it reads as "reconciled".
    """

    def test_the_table_and_its_machinery_do_not_exist(self, conn):
        import importlib

        kinds = dict(conn.execute(
            "SELECT table_name, table_type FROM information_schema.tables"
        ).fetchall())
        assert kinds.get("semantic_agent_delegations") == "VIEW"
        assert "fact_agent_delegations" not in kinds

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("ccutils.etl.fact_agent_delegations")
        orchestrator = importlib.import_module("ccutils.etl.orchestrator")
        assert not hasattr(orchestrator, "run_post_session_reconciliation")

        from ccutils.export.duckdb_archive import _PROGRESS_TABLES
        from ccutils.schemas.star.schema import NATURAL_KEYS, TABLE_COVERAGE

        assert "fact_agent_delegations" not in NATURAL_KEYS
        assert "fact_agent_delegations" not in TABLE_COVERAGE
        assert "fact_agent_delegations" not in _PROGRESS_TABLES

    def test_a_build_records_no_reconciliation_run(self, conn, agent_session, tmp_path):
        from ccutils.etl.global_sources import run_global_sources

        _populate(conn, agent_session, tmp_path)
        run_global_sources(conn)
        kinds = {r[0] for r in conn.execute("SELECT DISTINCT run_kind FROM etl.runs").fetchall()}
        assert "reconciliation" not in kinds
        assert "session" in kinds


class TestToolCountIsZeroNotNull:
    """An agent that finished without calling a tool used ZERO tools.

    The table left its (blended) tool count NULL there (a LEFT JOIN over
    the agent's tool calls found no rows), which made "used none" read the
    same as "unknown". `derived_tool_use_count` comes from the agent's
    session summary, where no rows is a counted zero.
    """

    def test_completed_async_agent_with_no_tools_counts_zero(self, conn, tmp_path):
        helper = TestAsyncCompletionIsDerivedFromTheAgent()
        parent = helper._parent_with_async_spawn(tmp_path)
        agent = helper._agent_transcript(tmp_path)
        _populate(conn, parent, tmp_path)
        _populate(conn, agent, tmp_path)
        state, tools = conn.execute(
            "SELECT completion_state, derived_tool_use_count "
            "FROM semantic_agent_delegations WHERE tool_use_id = 'tu_async1'"
        ).fetchone()
        assert state == "completed"
        assert tools == 0

    def test_an_unfinished_agent_still_reports_no_count(self, conn, tmp_path):
        helper = TestAsyncCompletionIsDerivedFromTheAgent()
        parent = helper._parent_with_async_spawn(tmp_path)
        agent = helper._agent_transcript(tmp_path, stop_reason=None)
        _populate(conn, parent, tmp_path)
        _populate(conn, agent, tmp_path)
        assert conn.execute(
            "SELECT derived_tool_use_count FROM semantic_agent_delegations "
            "WHERE tool_use_id = 'tu_async1'"
        ).fetchone() == (None,)


class TestSessionSummaryCarriesDelegationFeatures:
    """The parent's summary row says how much it delegated.

    Read from `fact_tool_calls` and the child sessions directly, never from
    `semantic_agent_delegations`: that view reads the summary for the child,
    so the reverse reference would be a cycle.
    """

    def _summary(self, conn, session_id):
        return conn.execute(
            "SELECT total_delegations, total_spawn_failures, max_child_depth, "
            "delegated_output_tokens FROM semantic_session_summary "
            "WHERE session_id = ?", [session_id]
        ).fetchone()

    def test_a_session_that_delegated_nothing_reads_zero(self, conn, tmp_path):
        jsonl = tmp_path / "plain.jsonl"
        jsonl.write_text(json.dumps(
            {"type": "user", "uuid": "u1", "sessionId": "plain-s",
             "timestamp": "2026-04-19T10:00:00Z", "cwd": "/p",
             "message": {"role": "user", "content": "hello"}}))
        _populate(conn, jsonl, tmp_path)
        assert self._summary(conn, "plain-s") == (0, 0, None, 0)

    def test_spawns_and_child_output_are_counted_on_the_parent(self, conn, tmp_path):
        helper = TestAsyncCompletionIsDerivedFromTheAgent()
        parent = helper._parent_with_async_spawn(tmp_path)
        agent = helper._agent_transcript(tmp_path)
        _populate(conn, parent, tmp_path)
        # Before the child is loaded: one spawn, nothing known about it.
        assert self._summary(conn, "parent-uuid") == (1, 0, None, 0)
        # The sidecar states the child's depth; nothing here infers one.
        agent.with_suffix(".meta.json").write_text(
            json.dumps({"agentType": "general-purpose", "spawnDepth": 1}))
        _populate(conn, agent, tmp_path)
        delegations, failures, depth, out_tokens = self._summary(conn, "parent-uuid")
        assert (delegations, failures) == (1, 0)
        assert out_tokens == 9788, "the child's output tokens, from its own usage"
        assert depth == 1
        # The child's own summary row counts nothing as delegated.
        assert self._summary(conn, "agent-abc123def")[0] == 0

    def test_a_refused_spawn_is_a_spawn_failure(self, conn, tmp_path):
        jsonl = tmp_path / "refused.jsonl"
        jsonl.write_text("\n".join(json.dumps(d) for d in [
            {"type": "user", "uuid": "u1", "sessionId": "refused-s",
             "timestamp": "2026-04-19T10:00:00Z", "cwd": "/work",
             "message": {"role": "user", "content": "delegate"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "sessionId": "refused-s", "timestamp": "2026-04-19T10:00:01Z",
             "requestId": "r1",
             "message": {"role": "assistant", "model": "claude-opus-5",
                         "content": [{"type": "tool_use", "id": "tu_r",
                                      "name": "Agent",
                                      "input": {"description": "nested",
                                                "prompt": "go"}}]}},
            {"type": "user", "uuid": "u2", "parentUuid": "a1",
             "sessionId": "refused-s", "timestamp": "2026-04-19T10:00:02Z",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "tu_r",
                  "content": "Subagent nesting limit reached (depth 3 of 3).",
                  "is_error": True}]}},
        ]))
        _populate(conn, jsonl, tmp_path)
        assert self._summary(conn, "refused-s")[:2] == (1, 1)
