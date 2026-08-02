"""A projection must emit the grain its populator declares.

Claim: `fact_tool_uses` and `fact_tool_results` declare
`natural_key="tool_use_id"`, so their projections must produce ONE row per
tool_use_id. They did not. A real Claude Code session records a single
tool_use_id under two distinct entry uuids, so unnesting content blocks
across entries yielded two rows for one key.

That was first patched in `lineage_upsert` -- a generic collapse in machinery
shared by 13 facts, applying a tool-call-specific judgment to all of them and
resolving it silently. These tests pin the fix at the layer that owns the
grain instead: the projection that declares the key produces the key.

Evidence the collapse is safe, measured on a 2,344-session corpus: all 29
violating keys are content-identical on every extracted column
(`is_error`, `result_content_text`, `timestamp`, `tool_name`, and every
typed `bash_*`/`edit_*`/`read_*`/`agent_*` field). Only `result_payload_json`
-- the raw blob -- ever differs, on 8 of 29. Nothing is lost by keeping one.

Delete these and duplicate keys return to the shared helper, where the reason
they are safe to collapse is not visible.
"""

import json

import pytest

from ccutils import create_star_schema


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


def _session_with_repeated_tool_use_id(tmp_path):
    """One tool_use_id recorded under TWO distinct entry uuids.

    This is the real corpus shape: 0 entries carry two tool_result blocks for
    one id, but 29 tool_use_ids span two entries within a session.
    """
    proj = tmp_path / "projects" / "-Users-dev-myrepo"
    proj.mkdir(parents=True, exist_ok=True)
    jsonl = proj / "dup.jsonl"
    lines = [
        {"type": "user", "uuid": "u1", "sessionId": "dup",
         "timestamp": "2026-04-19T10:00:00Z", "cwd": "/work",
         "gitBranch": "main", "version": "2.1.198",
         "message": {"role": "user", "content": "go"}},
        # Two assistant entries, distinct uuids, SAME tool_use id.
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "sessionId": "dup", "timestamp": "2026-04-19T10:00:01Z",
         "requestId": "r1",
         "message": {"role": "assistant", "model": "claude-opus-5",
                     "content": [{"type": "tool_use", "id": "tu_dup",
                                  "name": "Bash",
                                  "input": {"command": "ls"}}]}},
        {"type": "assistant", "uuid": "a2", "parentUuid": "u1",
         "sessionId": "dup", "timestamp": "2026-04-19T10:00:01Z",
         "requestId": "r1",
         "message": {"role": "assistant", "model": "claude-opus-5",
                     "content": [{"type": "tool_use", "id": "tu_dup",
                                  "name": "Bash",
                                  "input": {"command": "ls"}}]}},
        # Two user entries, distinct uuids, SAME tool_result id.
        {"type": "user", "uuid": "u2", "parentUuid": "a1",
         "sessionId": "dup", "timestamp": "2026-04-19T10:00:02Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_dup",
              "content": "ok", "is_error": False}]}},
        {"type": "user", "uuid": "u3", "parentUuid": "a2",
         "sessionId": "dup", "timestamp": "2026-04-19T10:00:02Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_dup",
              "content": "ok", "is_error": False}]}},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


class TestProjectionEmitsDeclaredGrain:
    def _run(self, conn, tmp_path):
        from ccutils.etl.orchestrator import run_v15_etl

        run_v15_etl(conn, _session_with_repeated_tool_use_id(tmp_path),
                    project_name="p", parquet_lake_root=tmp_path / "lake")

    def test_fact_tool_uses_one_row_per_tool_use_id(self, conn, tmp_path):
        self._run(conn, tmp_path)
        n = conn.execute(
            "SELECT COUNT(*) FROM fact_tool_uses "
            "WHERE tool_use_id = 'tu_dup' AND NOT is_deleted"
        ).fetchone()[0]
        assert n == 1

    def test_fact_tool_results_one_row_per_tool_use_id(self, conn, tmp_path):
        self._run(conn, tmp_path)
        n = conn.execute(
            "SELECT COUNT(*) FROM fact_tool_results "
            "WHERE tool_use_id = 'tu_dup' AND NOT is_deleted"
        ).fetchone()[0]
        assert n == 1

    def test_downstream_facts_inherit_uniqueness(self, conn, tmp_path):
        """`fact_file_operations` projects FROM `fact_tool_uses`, and
        `fact_tool_chain_steps` from the same grain -- neither should need
        its own collapse once the source projection is correct."""
        self._run(conn, tmp_path)
        for table, key in (
            ("fact_file_operations", "tool_use_id"),
            ("fact_tool_chain_steps", "chain_step_id"),
            ("fact_errors", "error_id"),
        ):
            dups = conn.execute(
                f"SELECT COUNT(*) FROM (SELECT {key} FROM {table} "
                f"WHERE NOT is_deleted GROUP BY 1 HAVING COUNT(*) > 1)"
            ).fetchone()[0]
            assert dups == 0, f"{table}.{key} has {dups} duplicate keys"

    def test_the_surviving_row_carries_real_content(self, conn, tmp_path):
        """Non-vacuity: collapsing must not produce an empty husk. A test
        asserting only COUNT(*) == 1 would pass if the projection emitted
        nothing at all."""
        self._run(conn, tmp_path)
        row = conn.execute(
            "SELECT tool_name, input_json FROM fact_tool_uses "
            "WHERE tool_use_id = 'tu_dup' AND NOT is_deleted"
        ).fetchone()
        assert row[0] == "Bash"
        assert "ls" in row[1]

        res = conn.execute(
            "SELECT result_content_text, is_error FROM fact_tool_results "
            "WHERE tool_use_id = 'tu_dup' AND NOT is_deleted"
        ).fetchone()
        assert res[0] == "ok"
        assert res[1] is False

    def test_earliest_entry_wins_when_they_differ(self, conn, tmp_path):
        """Pin the tie-break DIRECTION, not just the row count.

        The shared fixture gives both duplicates the same timestamp and
        identical content -- which models the real corpus, but means flipping
        `ORDER BY timestamp, entry_id` to DESC would leave every other
        assertion in this file passing. That is the "test written from the
        same premise as its code" pattern this project has been bitten by.
        This one distinguishes them, so the ordering is falsifiable.
        """
        from ccutils.etl.orchestrator import run_v15_etl

        proj = tmp_path / "projects" / "-Users-dev-myrepo"
        proj.mkdir(parents=True, exist_ok=True)
        jsonl = proj / "ordered.jsonl"
        lines = [
            {"type": "user", "uuid": "u1", "sessionId": "ordered",
             "timestamp": "2026-04-19T10:00:00Z", "cwd": "/work",
             "gitBranch": "main", "version": "2.1.198",
             "message": {"role": "user", "content": "go"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "sessionId": "ordered", "timestamp": "2026-04-19T10:00:01Z",
             "requestId": "r1",
             "message": {"role": "assistant", "model": "claude-opus-5",
                         "content": [{"type": "tool_use", "id": "tu_ord",
                                      "name": "Bash",
                                      "input": {"command": "EARLIER"}}]}},
            {"type": "assistant", "uuid": "a2", "parentUuid": "u1",
             "sessionId": "ordered", "timestamp": "2026-04-19T10:00:09Z",
             "requestId": "r1",
             "message": {"role": "assistant", "model": "claude-opus-5",
                         "content": [{"type": "tool_use", "id": "tu_ord",
                                      "name": "Bash",
                                      "input": {"command": "LATER"}}]}},
        ]
        jsonl.write_text("\n".join(json.dumps(d) for d in lines))
        run_v15_etl(conn, jsonl, project_name="p",
                    parquet_lake_root=tmp_path / "lake")

        row = conn.execute(
            "SELECT input_json FROM fact_tool_uses "
            "WHERE tool_use_id = 'tu_ord' AND NOT is_deleted"
        ).fetchall()
        assert len(row) == 1
        assert "EARLIER" in row[0][0], (
            "ORDER BY timestamp ASC means the first-recorded entry survives"
        )

    def test_agent_delegations_inherit_uniqueness(self, conn, tmp_path):
        """`fact_agent_delegations` is named as an inheriting fact but the
        shared fixture uses Bash, which its `tool_name IN ('Task','Agent')`
        filter excludes -- so the claim was asserted only in prose."""
        from ccutils.etl.orchestrator import run_v15_etl

        proj = tmp_path / "projects" / "-Users-dev-myrepo"
        proj.mkdir(parents=True, exist_ok=True)
        jsonl = proj / "deleg.jsonl"
        spawn = {"type": "tool_use", "id": "tu_agent", "name": "Agent",
                 "input": {"description": "d", "prompt": "p"}}
        lines = [
            {"type": "user", "uuid": "u1", "sessionId": "deleg",
             "timestamp": "2026-04-19T10:00:00Z", "cwd": "/work",
             "gitBranch": "main", "version": "2.1.198",
             "message": {"role": "user", "content": "go"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "sessionId": "deleg", "timestamp": "2026-04-19T10:00:01Z",
             "requestId": "r1",
             "message": {"role": "assistant", "model": "claude-opus-5",
                         "content": [spawn]}},
            # Same spawn recorded again under a distinct entry uuid.
            {"type": "assistant", "uuid": "a2", "parentUuid": "u1",
             "sessionId": "deleg", "timestamp": "2026-04-19T10:00:01Z",
             "requestId": "r1",
             "message": {"role": "assistant", "model": "claude-opus-5",
                         "content": [spawn]}},
        ]
        jsonl.write_text("\n".join(json.dumps(d) for d in lines))
        run_v15_etl(conn, jsonl, project_name="p",
                    parquet_lake_root=tmp_path / "lake")

        dups = conn.execute(
            "SELECT COUNT(*) FROM (SELECT delegation_key FROM "
            "fact_agent_delegations WHERE NOT is_deleted "
            "GROUP BY 1 HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        assert dups == 0
        # Non-vacuity: the fixture must actually produce a delegation.
        n = conn.execute(
            "SELECT COUNT(*) FROM fact_agent_delegations WHERE NOT is_deleted"
        ).fetchone()[0]
        assert n == 1

    def test_distinct_tool_use_ids_are_not_collapsed(self, conn, tmp_path):
        """Non-vacuity: the collapse must be per-key, not global."""
        from ccutils.etl.orchestrator import run_v15_etl

        proj = tmp_path / "projects" / "-Users-dev-myrepo"
        proj.mkdir(parents=True, exist_ok=True)
        jsonl = proj / "multi.jsonl"
        lines = [
            {"type": "user", "uuid": "u1", "sessionId": "multi",
             "timestamp": "2026-04-19T10:00:00Z", "cwd": "/work",
             "gitBranch": "main", "version": "2.1.198",
             "message": {"role": "user", "content": "go"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "sessionId": "multi", "timestamp": "2026-04-19T10:00:01Z",
             "requestId": "r1",
             "message": {"role": "assistant", "model": "claude-opus-5",
                         "content": [{"type": "tool_use", "id": "tu_a",
                                      "name": "Bash",
                                      "input": {"command": "ls"}}]}},
            {"type": "assistant", "uuid": "a2", "parentUuid": "u1",
             "sessionId": "multi", "timestamp": "2026-04-19T10:00:02Z",
             "requestId": "r2",
             "message": {"role": "assistant", "model": "claude-opus-5",
                         "content": [{"type": "tool_use", "id": "tu_b",
                                      "name": "Read",
                                      "input": {"file_path": "/x"}}]}},
        ]
        jsonl.write_text("\n".join(json.dumps(d) for d in lines))
        run_v15_etl(conn, jsonl, project_name="p",
                    parquet_lake_root=tmp_path / "lake")

        ids = [r[0] for r in conn.execute(
            "SELECT tool_use_id FROM fact_tool_uses WHERE NOT is_deleted "
            "ORDER BY tool_use_id").fetchall()]
        assert ids == ["tu_a", "tu_b"]
