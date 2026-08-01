"""Tests for the v0.15 fact_tool_chain_steps populator (Phase D).

Grain: one row per (session, tool_use, step_position). A chain is an
"agentic run" -- the contiguous block of tool_uses following one human
turn, up to the next human turn. Derived from fact_tool_uses +
fact_tool_results + fact_messages (for the human-turn boundaries).

Run AFTER populate_fact_tool_uses + populate_fact_tool_results +
populate_fact_messages.

The grain must not depend on how content blocks are packed into JSONL
entries -- see TestAgenticRunGrain, which runs the same assertions over
both packings.
"""

from __future__ import annotations

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.fact_messages import populate_fact_messages
from ccutils.etl.fact_tool_calls import (
    populate_fact_tool_results,
    populate_fact_tool_uses,
)
from ccutils.etl.fact_tool_chain_steps import populate_fact_tool_chain_steps
from ccutils.etl.lineage import EtlRun
from ccutils.etl.staging import load_session_to_staging
from ccutils.parsers.parquet_writer import write_session_to_parquet


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


@pytest.fixture
def chain_session(tmp_path):
    """Two assistant turns, each with a chain of tools.

    Turn 1: Read, Edit, Bash (3-step chain)
    Turn 2: Read (1-step chain)
    """
    jsonl = tmp_path / "chain.jsonl"
    lines = [
        {"type": "user", "uuid": "u1", "sessionId": "chain-s",
         "timestamp": "2026-04-19T10:00:00Z", "cwd": "/p",
         "gitBranch": "main", "version": "2.1.114",
         "message": {"role": "user", "content": "do a multi-tool turn"}},
        # Turn 1: Read, Edit, Bash within one assistant message
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "sessionId": "chain-s", "timestamp": "2026-04-19T10:00:01Z",
         "requestId": "r1",
         "message": {"role": "assistant", "model": "claude-opus-4-7",
                     "content": [
                         {"type": "tool_use", "id": "tu_r",
                          "name": "Read",
                          "input": {"file_path": "/p/a.py"}},
                         {"type": "tool_use", "id": "tu_e",
                          "name": "Edit",
                          "input": {"file_path": "/p/a.py",
                                    "old_string": "x",
                                    "new_string": "y"}},
                         {"type": "tool_use", "id": "tu_b",
                          "name": "Bash",
                          "input": {"command": "pytest"}},
                     ]}},
        # tool_results follow (separate user message)
        {"type": "user", "uuid": "u2", "parentUuid": "a1",
         "sessionId": "chain-s", "timestamp": "2026-04-19T10:00:02Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_r", "content": "data"},
             {"type": "tool_result", "tool_use_id": "tu_e", "content": "ok"},
             {"type": "tool_result", "tool_use_id": "tu_b", "content": "fail",
              "is_error": True},
         ]},
         "toolUseResult": {"interrupted": False, "exitCode": 1}},
        # Turn 2: just one Read (new chain)
        {"type": "user", "uuid": "u3", "parentUuid": "u2",
         "sessionId": "chain-s", "timestamp": "2026-04-19T10:00:05Z",
         "message": {"role": "user", "content": "one more"}},
        {"type": "assistant", "uuid": "a2", "parentUuid": "u3",
         "sessionId": "chain-s", "timestamp": "2026-04-19T10:00:06Z",
         "requestId": "r2",
         "message": {"role": "assistant", "model": "claude-opus-4-7",
                     "content": [
                         {"type": "tool_use", "id": "tu_r2",
                          "name": "Read",
                          "input": {"file_path": "/p/b.py"}},
                     ]}},
        {"type": "user", "uuid": "u4", "parentUuid": "a2",
         "sessionId": "chain-s", "timestamp": "2026-04-19T10:00:07Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_r2", "content": "data"},
         ]}},
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
    populate_fact_tool_chain_steps(conn, run=run)
    return run


class TestFactToolChainSteps:
    def test_one_row_per_tool_use(self, conn, chain_session, tmp_path):
        _populate(conn, chain_session, tmp_path)
        # 3 tools in chain 1, 1 tool in chain 2 = 4 chain steps
        n = conn.execute(
            "SELECT COUNT(*) FROM fact_tool_chain_steps"
        ).fetchone()[0]
        assert n == 4

    def test_step_position_starts_at_one_per_chain(
        self, conn, chain_session, tmp_path
    ):
        _populate(conn, chain_session, tmp_path)
        rows = conn.execute(
            """
            SELECT chain_id, step_position, tool_use_id
            FROM fact_tool_chain_steps
            ORDER BY chain_id, step_position
            """
        ).fetchall()
        # Group by chain_id; each chain should start at position 1
        by_chain = {}
        for chain_id, pos, tu_id in rows:
            by_chain.setdefault(chain_id, []).append((pos, tu_id))
        for steps in by_chain.values():
            positions = [pos for pos, _ in steps]
            assert positions[0] == 1
            assert positions == list(range(1, len(positions) + 1))

    def test_prev_and_next_tool_key_linked(
        self, conn, chain_session, tmp_path
    ):
        _populate(conn, chain_session, tmp_path)
        # Find Edit's chain step; its prev should be Read, next should be Bash
        rows = conn.execute(
            """
            SELECT dt_cur.tool_name AS cur_name,
                   dt_prev.tool_name AS prev_name,
                   dt_next.tool_name AS next_name
            FROM fact_tool_chain_steps fcs
            JOIN dim_tool dt_cur  ON fcs.tool_key      = dt_cur.tool_key
            LEFT JOIN dim_tool dt_prev ON fcs.prev_tool_key = dt_prev.tool_key
            LEFT JOIN dim_tool dt_next ON fcs.next_tool_key = dt_next.tool_key
            WHERE fcs.tool_use_id = 'tu_e'
            """
        ).fetchone()
        assert rows == ("Edit", "Read", "Bash")

    def test_first_step_has_null_prev(self, conn, chain_session, tmp_path):
        _populate(conn, chain_session, tmp_path)
        row = conn.execute(
            """
            SELECT prev_tool_key, next_tool_key FROM fact_tool_chain_steps
            WHERE tool_use_id = 'tu_r'
            """
        ).fetchone()
        assert row[0] is None  # No previous tool in chain
        assert row[1] is not None  # next is Edit

    def test_last_step_has_null_next(self, conn, chain_session, tmp_path):
        _populate(conn, chain_session, tmp_path)
        row = conn.execute(
            """
            SELECT prev_tool_key, next_tool_key FROM fact_tool_chain_steps
            WHERE tool_use_id = 'tu_b'
            """
        ).fetchone()
        assert row[0] is not None  # Prev is Edit
        assert row[1] is None  # End of chain

    def test_is_error_captured_from_tool_result(
        self, conn, chain_session, tmp_path
    ):
        _populate(conn, chain_session, tmp_path)
        row = conn.execute(
            "SELECT is_error FROM fact_tool_chain_steps WHERE tool_use_id = 'tu_b'"
        ).fetchone()
        assert row[0] is True

    def test_separate_chains_per_assistant_message(
        self, conn, chain_session, tmp_path
    ):
        _populate(conn, chain_session, tmp_path)
        # Should have 2 distinct chain_ids
        n_chains = conn.execute(
            "SELECT COUNT(DISTINCT chain_id) FROM fact_tool_chain_steps"
        ).fetchone()[0]
        assert n_chains == 2

    def test_lineage_block_populated(self, conn, chain_session, tmp_path):
        _populate(conn, chain_session, tmp_path)
        row = conn.execute(
            """
            SELECT created_at, last_updated_at, etl_run_id, record_source,
                   hash_diff, is_deleted
            FROM fact_tool_chain_steps LIMIT 1
            """
        ).fetchone()
        assert row[0] is not None
        assert row[1] is not None
        assert row[2] is not None
        assert row[3] == "claude_code_jsonl"
        assert row[4] is not None
        assert row[5] is False

    def test_idempotent_reetl(self, conn, chain_session, tmp_path):
        _populate(conn, chain_session, tmp_path)
        first = conn.execute(
            "SELECT last_updated_at FROM fact_tool_chain_steps ORDER BY chain_step_id"
        ).fetchall()
        _populate(conn, chain_session, tmp_path)
        second = conn.execute(
            "SELECT last_updated_at FROM fact_tool_chain_steps ORDER BY chain_step_id"
        ).fetchall()
        assert first == second


@pytest.fixture
def realistic_chain_session(tmp_path):
    """The same two agentic runs as `chain_session`, in the REAL JSONL shape.

    Claude Code writes one content block per assistant entry: parallel tool
    calls become separate entries that share an API message id but carry
    distinct uuids. Verified against real transcripts -- content-block shapes
    observed are ('tool_use',), ('thinking',), ('text',), never mixed.

    `chain_session` packs three tool_use blocks into one entry, which real
    transcripts never produce. Both fixtures must yield identical chains:
    run 1 = Read -> Edit -> Bash, run 2 = Read.
    """
    jsonl = tmp_path / "realistic.jsonl"

    def assistant(uuid, parent, ts, tool_id, name, tool_input, req):
        return {
            "type": "assistant", "uuid": uuid, "parentUuid": parent,
            "sessionId": "real-s", "timestamp": ts, "requestId": req,
            "message": {"role": "assistant", "model": "claude-opus-4-7",
                        "id": "msg_shared_1",
                        "content": [{"type": "tool_use", "id": tool_id,
                                     "name": name, "input": tool_input}]},
        }

    def result(uuid, parent, ts, tool_id, content, is_error=False):
        block = {"type": "tool_result", "tool_use_id": tool_id,
                 "content": content}
        if is_error:
            block["is_error"] = True
        return {"type": "user", "uuid": uuid, "parentUuid": parent,
                "sessionId": "real-s", "timestamp": ts,
                "message": {"role": "user", "content": [block]}}

    lines = [
        {"type": "user", "uuid": "u1", "sessionId": "real-s",
         "timestamp": "2026-04-19T10:00:00Z", "cwd": "/p",
         "gitBranch": "main", "version": "2.1.114",
         "message": {"role": "user", "content": "do a multi-tool turn"}},
        # Agentic run 1 -- three tools, one per assistant entry.
        assistant("a1", "u1", "2026-04-19T10:00:01Z", "tu_rr", "Read",
                  {"file_path": "/p/a.py"}, "r1"),
        result("u2", "a1", "2026-04-19T10:00:02Z", "tu_rr", "data"),
        assistant("a2", "u2", "2026-04-19T10:00:03Z", "tu_ee", "Edit",
                  {"file_path": "/p/a.py", "old_string": "x",
                   "new_string": "y"}, "r2"),
        result("u3", "a2", "2026-04-19T10:00:04Z", "tu_ee", "ok"),
        assistant("a3", "u3", "2026-04-19T10:00:05Z", "tu_bb", "Bash",
                  {"command": "pytest"}, "r3"),
        result("u4", "a3", "2026-04-19T10:00:06Z", "tu_bb", "fail",
               is_error=True),
        # A new human turn opens agentic run 2.
        {"type": "user", "uuid": "u5", "parentUuid": "u4",
         "sessionId": "real-s", "timestamp": "2026-04-19T10:00:10Z",
         "message": {"role": "user", "content": "one more"}},
        assistant("a4", "u5", "2026-04-19T10:00:11Z", "tu_rr2", "Read",
                  {"file_path": "/p/b.py"}, "r4"),
        result("u6", "a4", "2026-04-19T10:00:12Z", "tu_rr2", "data"),
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


class TestAgenticRunGrain:
    """A chain is the human-turn-delimited run, not the assistant entry.

    Claim: delete these and the chain grain can silently regress to
    one-step-per-entry, which makes prev/next_tool_key universally NULL and
    empties semantic_tool_patterns and facet F07 without failing anything.
    That is exactly what shipped: on a 2,250-session corpus, 71,175 of 71,216
    chain steps were step_position 1 and F07 was [] for 99.5% of sessions.
    """

    def test_chain_spans_assistant_entries(
        self, conn, realistic_chain_session, tmp_path
    ):
        _populate(conn, realistic_chain_session, tmp_path)
        n_chains = conn.execute(
            "SELECT COUNT(DISTINCT chain_id) FROM fact_tool_chain_steps"
        ).fetchone()[0]
        assert n_chains == 2, "one chain per human turn, not per entry"

    def test_prev_next_linked_across_entries(
        self, conn, realistic_chain_session, tmp_path
    ):
        _populate(conn, realistic_chain_session, tmp_path)
        row = conn.execute(
            """
            SELECT dt_cur.tool_name, dt_prev.tool_name, dt_next.tool_name
            FROM fact_tool_chain_steps fcs
            JOIN dim_tool dt_cur ON fcs.tool_key = dt_cur.tool_key
            LEFT JOIN dim_tool dt_prev ON fcs.prev_tool_key = dt_prev.tool_key
            LEFT JOIN dim_tool dt_next ON fcs.next_tool_key = dt_next.tool_key
            WHERE fcs.tool_use_id = 'tu_ee'
            """
        ).fetchone()
        assert row == ("Edit", "Read", "Bash")

    def test_step_positions_run_the_length_of_the_run(
        self, conn, realistic_chain_session, tmp_path
    ):
        _populate(conn, realistic_chain_session, tmp_path)
        rows = conn.execute(
            """
            SELECT fcs.step_position, dt.tool_name
            FROM fact_tool_chain_steps fcs
            JOIN dim_tool dt ON fcs.tool_key = dt.tool_key
            WHERE fcs.tool_use_id IN ('tu_rr', 'tu_ee', 'tu_bb')
            ORDER BY fcs.step_position
            """
        ).fetchall()
        assert rows == [(1, "Read"), (2, "Edit"), (3, "Bash")]

    def test_new_human_turn_starts_a_new_chain(
        self, conn, realistic_chain_session, tmp_path
    ):
        _populate(conn, realistic_chain_session, tmp_path)
        row = conn.execute(
            """
            SELECT step_position, prev_tool_key, next_tool_key
            FROM fact_tool_chain_steps WHERE tool_use_id = 'tu_rr2'
            """
        ).fetchone()
        assert row == (1, None, None), "run 2 is a fresh single-step chain"

    def test_time_since_prev_is_real_elapsed_time(
        self, conn, realistic_chain_session, tmp_path
    ):
        """Was always NULL under the per-entry grain (no prev row existed)."""
        _populate(conn, realistic_chain_session, tmp_path)
        row = conn.execute(
            "SELECT time_since_prev_seconds FROM fact_tool_chain_steps "
            "WHERE tool_use_id = 'tu_ee'"
        ).fetchone()
        assert row[0] == pytest.approx(2.0)  # 10:00:01 -> 10:00:03

    def test_grain_is_independent_of_block_packing(
        self, conn, realistic_chain_session, chain_session, tmp_path
    ):
        """Both JSONL packings must produce the same chain shape."""
        _populate(conn, realistic_chain_session, tmp_path)
        _populate(conn, chain_session, tmp_path)
        shapes = conn.execute(
            """
            SELECT session_id, chain_id, COUNT(*) AS n
            FROM fact_tool_chain_steps
            GROUP BY session_id, chain_id
            """
        ).fetchall()
        per_session = {}
        for session_id, _chain, n in shapes:
            per_session.setdefault(session_id, []).append(n)
        assert sorted(per_session["real-s"]) == [1, 3]
        assert sorted(per_session["chain-s"]) == [1, 3]


@pytest.fixture
def repeated_pattern_session(tmp_path):
    """Turn 1: Read, Edit, Bash. Turn 2: Read, Edit.

    Read->Edit occurs twice (crosses the semantic_tool_patterns
    HAVING COUNT(*) >= 2 threshold); Edit->Bash occurs once (stays below).

    Written in the real JSONL packing -- one content block per assistant
    entry -- so the view is exercised against transitions that must span
    entries. Under the old per-entry chain grain this fixture yields no
    transitions at all, which is the corpus-wide failure it now guards.
    """
    jsonl = tmp_path / "pattern.jsonl"

    def assistant(uuid, parent, ts, tool_id, name, tool_input, req):
        return {
            "type": "assistant", "uuid": uuid, "parentUuid": parent,
            "sessionId": "pattern-s", "timestamp": ts, "requestId": req,
            "message": {"role": "assistant", "model": "claude-opus-4-7",
                        "content": [{"type": "tool_use", "id": tool_id,
                                     "name": name, "input": tool_input}]},
        }

    def result(uuid, parent, ts, tool_id, content):
        return {"type": "user", "uuid": uuid, "parentUuid": parent,
                "sessionId": "pattern-s", "timestamp": ts,
                "message": {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": tool_id,
                     "content": content}]}}

    lines = [
        {"type": "user", "uuid": "u1", "sessionId": "pattern-s",
         "timestamp": "2026-04-19T11:00:00Z", "cwd": "/p",
         "gitBranch": "main", "version": "2.1.114",
         "message": {"role": "user", "content": "repeat a tool pattern"}},
        assistant("a1", "u1", "2026-04-19T11:00:01Z", "tu_r1", "Read",
                  {"file_path": "/p/a.py"}, "r1"),
        result("u2", "a1", "2026-04-19T11:00:02Z", "tu_r1", "data"),
        assistant("a2", "u2", "2026-04-19T11:00:03Z", "tu_e1", "Edit",
                  {"file_path": "/p/a.py", "old_string": "x",
                   "new_string": "y"}, "r2"),
        result("u3", "a2", "2026-04-19T11:00:04Z", "tu_e1", "ok"),
        assistant("a3", "u3", "2026-04-19T11:00:05Z", "tu_b1", "Bash",
                  {"command": "pytest"}, "r3"),
        result("u4", "a3", "2026-04-19T11:00:06Z", "tu_b1", "ok"),
        {"type": "user", "uuid": "u5", "parentUuid": "u4",
         "sessionId": "pattern-s", "timestamp": "2026-04-19T11:00:10Z",
         "message": {"role": "user", "content": "again"}},
        assistant("a4", "u5", "2026-04-19T11:00:11Z", "tu_r2", "Read",
                  {"file_path": "/p/b.py"}, "r4"),
        result("u6", "a4", "2026-04-19T11:00:12Z", "tu_r2", "data"),
        assistant("a5", "u6", "2026-04-19T11:00:13Z", "tu_e2", "Edit",
                  {"file_path": "/p/b.py", "old_string": "x",
                   "new_string": "y"}, "r5"),
        result("u7", "a5", "2026-04-19T11:00:14Z", "tu_e2", "ok"),
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


class TestSemanticToolPatterns:
    """Behavior coverage for the semantic_tool_patterns view (it had none --
    DDL binding alone doesn't prove the aggregation is right)."""

    def test_repeated_transition_visible_with_frequency(
        self, conn, repeated_pattern_session, tmp_path
    ):
        _populate(conn, repeated_pattern_session, tmp_path)
        rows = conn.execute(
            "SELECT tool_name, next_tool_name, frequency, error_count "
            "FROM semantic_tool_patterns"
        ).fetchall()
        assert ("Read", "Edit", 2, 0) in rows

    def test_single_occurrence_excluded_by_threshold(
        self, conn, repeated_pattern_session, tmp_path
    ):
        _populate(conn, repeated_pattern_session, tmp_path)
        rows = conn.execute(
            "SELECT tool_name, next_tool_name FROM semantic_tool_patterns"
        ).fetchall()
        assert ("Edit", "Bash") not in rows
