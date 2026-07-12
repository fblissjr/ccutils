"""Tests for the v0.15 fact_plan_revisions populator (Phase D).

Grain: one row per ExitPlanMode tool_use in a session.

The original rethink driver was that v0.14 classified plan-revision
outcome by string-matching against tool_result text (truncated to 2000
chars), even though structural signals existed: tool_use_id has a
matching tool_result whose is_error tristate distinguishes accepted
(False) from rejected (True) cleanly. v0.15's fact_tool_results carries
is_error as a nullable BOOLEAN (R16), so the classification is now
deterministic.

Outcome classification:
- 'superseded' -- a later ExitPlanMode exists in the same session
- 'accepted'   -- tool_result.is_error = FALSE
- 'rejected'   -- tool_result.is_error = TRUE
- 'pending'    -- no tool_result yet (session in flight)
- 'unknown'    -- tool_result present but is_error is NULL

parent_revision_key chains revisions within a session by timestamp.
user_feedback_text captures the next user text message after a rejection.
"""

from __future__ import annotations

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.fact_messages import populate_fact_messages
from ccutils.etl.fact_plan_revisions import populate_fact_plan_revisions
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


def _user(uid, parent_uid, session, ts, text):
    return {
        "type": "user", "uuid": uid, "parentUuid": parent_uid,
        "sessionId": session, "timestamp": ts,
        "cwd": "/p", "gitBranch": "main", "version": "2.1.114",
        "message": {"role": "user", "content": text},
    }


def _plan_call(uid, parent_uid, session, ts, tool_use_id, plan_text,
               request_id="r", plan_file_path=None):
    tool_input = {"plan": plan_text}
    if plan_file_path is not None:
        tool_input["planFilePath"] = plan_file_path
    return {
        "type": "assistant", "uuid": uid, "parentUuid": parent_uid,
        "sessionId": session, "timestamp": ts, "requestId": request_id,
        "message": {"role": "assistant", "model": "claude-opus-4-7",
                    "content": [{"type": "tool_use", "id": tool_use_id,
                                 "name": "ExitPlanMode",
                                 "input": tool_input}]},
    }


def _plan_result(uid, parent_uid, session, ts, tool_use_id, *,
                 is_error, content="The user has approved your plan"):
    return {
        "type": "user", "uuid": uid, "parentUuid": parent_uid,
        "sessionId": session, "timestamp": ts,
        "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tool_use_id,
             "content": content, "is_error": is_error},
        ]},
    }


@pytest.fixture
def accepted_session(tmp_path):
    """One ExitPlanMode that gets accepted."""
    jsonl = tmp_path / "accepted.jsonl"
    lines = [
        _user("u1", None, "accepted-s", "2026-04-19T10:00:00Z", "plan it"),
        _plan_call("a1", "u1", "accepted-s", "2026-04-19T10:00:01Z",
                   "tu_plan_1", "Step 1\nStep 2"),
        _plan_result("u2", "a1", "accepted-s", "2026-04-19T10:00:30Z",
                     "tu_plan_1", is_error=False,
                     content="The user has approved your plan. Proceed."),
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def rejected_session(tmp_path):
    """One ExitPlanMode that gets rejected, with user feedback after."""
    jsonl = tmp_path / "rejected.jsonl"
    lines = [
        _user("u1", None, "rejected-s", "2026-04-19T10:00:00Z", "plan it"),
        _plan_call("a1", "u1", "rejected-s", "2026-04-19T10:00:01Z",
                   "tu_plan_r", "Bad plan"),
        _plan_result("u2", "a1", "rejected-s", "2026-04-19T10:00:30Z",
                     "tu_plan_r", is_error=True,
                     content="The user doesn't want to proceed with this plan."),
        _user("u3", "u2", "rejected-s", "2026-04-19T10:00:45Z",
              "actually do it differently"),
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def superseded_session(tmp_path):
    """Two ExitPlanModes -- first is superseded, second is accepted."""
    jsonl = tmp_path / "superseded.jsonl"
    lines = [
        _user("u1", None, "superseded-s", "2026-04-19T10:00:00Z", "plan v1"),
        _plan_call("a1", "u1", "superseded-s", "2026-04-19T10:00:01Z",
                   "tu_plan_1", "Plan v1", request_id="r1"),
        _plan_result("u2", "a1", "superseded-s", "2026-04-19T10:00:30Z",
                     "tu_plan_1", is_error=True, content="rejected"),
        _user("u3", "u2", "superseded-s", "2026-04-19T10:00:45Z",
              "revise the plan"),
        _plan_call("a2", "u3", "superseded-s", "2026-04-19T10:01:00Z",
                   "tu_plan_2", "Plan v2", request_id="r2"),
        _plan_result("u4", "a2", "superseded-s", "2026-04-19T10:01:30Z",
                     "tu_plan_2", is_error=False, content="approved"),
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def pending_session(tmp_path):
    """ExitPlanMode with no resolution -- session ended mid-flight."""
    jsonl = tmp_path / "pending.jsonl"
    lines = [
        _user("u1", None, "pending-s", "2026-04-19T10:00:00Z", "plan it"),
        _plan_call("a1", "u1", "pending-s", "2026-04-19T10:00:01Z",
                   "tu_plan_p", "In progress"),
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
    populate_fact_plan_revisions(conn, run=run)
    return run


class TestFactPlanRevisionsOutcome:
    def test_accepted_when_is_error_false(self, conn, accepted_session, tmp_path):
        _populate(conn, accepted_session, tmp_path)
        rows = conn.execute(
            "SELECT outcome, outcome_signal FROM fact_plan_revisions"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "accepted"
        assert rows[0][1] == "is_error=FALSE"

    def test_rejected_when_is_error_true(self, conn, rejected_session, tmp_path):
        _populate(conn, rejected_session, tmp_path)
        rows = conn.execute(
            "SELECT outcome, outcome_signal FROM fact_plan_revisions"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "rejected"
        assert rows[0][1] == "is_error=TRUE"

    def test_superseded_when_later_plan_exists(
        self, conn, superseded_session, tmp_path
    ):
        _populate(conn, superseded_session, tmp_path)
        rows = conn.execute(
            """
            SELECT tool_use_id, outcome, outcome_signal
            FROM fact_plan_revisions
            ORDER BY revision_number
            """
        ).fetchall()
        assert len(rows) == 2
        # First plan was rejected by tool_result BUT a later plan exists,
        # so it's marked 'superseded'.
        assert rows[0] == ("tu_plan_1", "superseded", "later_plan_exists")
        # Second plan was accepted.
        assert rows[1] == ("tu_plan_2", "accepted", "is_error=FALSE")

    def test_pending_when_no_resolution(self, conn, pending_session, tmp_path):
        _populate(conn, pending_session, tmp_path)
        rows = conn.execute(
            "SELECT outcome, outcome_signal FROM fact_plan_revisions"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "pending"

    def test_accepted_via_content_signature_when_is_error_null(
        self, conn, tmp_path
    ):
        """Real Claude Code sessions emit accepted plan results without
        an explicit is_error: false -- the field is just absent. The v0.15
        populator falls back to matching the documented approval signature
        in result_content_text, but reads the FULL untruncated content
        from fact_tool_results (unlike v0.14)."""
        jsonl = tmp_path / "sig.jsonl"
        lines = [
            _user("u1", None, "sig-s", "2026-04-19T10:00:00Z", "plan"),
            _plan_call("a1", "u1", "sig-s", "2026-04-19T10:00:01Z",
                       "tu_sig", "do x"),
            # is_error field intentionally absent -> parses as None
            {"type": "user", "uuid": "u2", "parentUuid": "a1",
             "sessionId": "sig-s", "timestamp": "2026-04-19T10:00:30Z",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "tu_sig",
                  "content": "User has approved your plan. Go."},
             ]}},
        ]
        jsonl.write_text("\n".join(json.dumps(d) for d in lines))
        _populate(conn, jsonl, tmp_path)
        row = conn.execute(
            "SELECT outcome, outcome_signal FROM fact_plan_revisions"
        ).fetchone()
        assert row == ("accepted", "approval_signature")


class TestFactPlanRevisionsPlanFilePath:
    """Claude Code emits input.planFilePath alongside the plan text."""

    def test_plan_file_path_captured(self, conn, tmp_path):
        jsonl = tmp_path / "planfile.jsonl"
        pfp = "<HOME>/.claude/plans/resilient-wiggling-meteor.md"
        lines = [
            _user("u1", None, "pf-s", "2026-04-19T10:00:00Z", "plan it"),
            _plan_call("a1", "u1", "pf-s", "2026-04-19T10:00:01Z",
                       "tu_pf_1", "Step 1", plan_file_path=pfp),
            _plan_result("u2", "a1", "pf-s", "2026-04-19T10:00:30Z",
                         "tu_pf_1", is_error=False),
        ]
        jsonl.write_text("\n".join(json.dumps(d) for d in lines))
        _populate(conn, jsonl, tmp_path)
        row = conn.execute(
            "SELECT plan_file_path FROM fact_plan_revisions"
        ).fetchone()
        assert row[0] == pfp

    def test_plan_file_path_null_when_absent(self, conn, accepted_session,
                                             tmp_path):
        """Older sessions predate planFilePath -- NULL, not error."""
        _populate(conn, accepted_session, tmp_path)
        row = conn.execute(
            "SELECT plan_file_path FROM fact_plan_revisions"
        ).fetchone()
        assert row[0] is None


class TestFactPlanRevisionsMigration:
    def test_existing_warehouse_gains_plan_file_path(self, tmp_path):
        """The warehouse is persistent: CREATE TABLE IF NOT EXISTS never
        widens an existing table, so create_star_schema must carry an
        explicit ADD COLUMN migration for columns added after 0.17.0."""
        db = tmp_path / "old.duckdb"
        conn = create_star_schema(db)
        # Simulate a pre-plan_file_path warehouse.
        conn.execute("ALTER TABLE fact_plan_revisions DROP COLUMN plan_file_path")
        conn.close()

        conn = create_star_schema(db)
        cols = {
            r[0] for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'fact_plan_revisions'"
            ).fetchall()
        }
        assert "plan_file_path" in cols


class TestFactPlanRevisionsChain:
    def test_revision_number_starts_at_one_per_session(
        self, conn, superseded_session, tmp_path
    ):
        _populate(conn, superseded_session, tmp_path)
        rows = conn.execute(
            "SELECT revision_number FROM fact_plan_revisions ORDER BY revision_number"
        ).fetchall()
        assert [r[0] for r in rows] == [1, 2]

    def test_parent_revision_key_links_chain(
        self, conn, superseded_session, tmp_path
    ):
        _populate(conn, superseded_session, tmp_path)
        rows = conn.execute(
            """
            SELECT revision_number, revision_key, parent_revision_key
            FROM fact_plan_revisions ORDER BY revision_number
            """
        ).fetchall()
        # Revision 1 has no parent; revision 2's parent is revision 1.
        assert rows[0][2] is None
        assert rows[1][2] == rows[0][1]


class TestFactPlanRevisionsContent:
    def test_plan_text_captured(self, conn, accepted_session, tmp_path):
        _populate(conn, accepted_session, tmp_path)
        row = conn.execute(
            "SELECT plan_text, plan_char_count FROM fact_plan_revisions"
        ).fetchone()
        assert row[0] == "Step 1\nStep 2"
        assert row[1] == len("Step 1\nStep 2")

    def test_seconds_to_resolution_computed(
        self, conn, accepted_session, tmp_path
    ):
        _populate(conn, accepted_session, tmp_path)
        seconds = conn.execute(
            "SELECT seconds_to_resolution FROM fact_plan_revisions"
        ).fetchone()[0]
        # Plan at 10:00:01, resolved at 10:00:30 -> 29 seconds.
        assert seconds == 29.0

    def test_user_feedback_captured_on_rejection(
        self, conn, rejected_session, tmp_path
    ):
        _populate(conn, rejected_session, tmp_path)
        row = conn.execute(
            "SELECT user_feedback_text FROM fact_plan_revisions"
        ).fetchone()
        assert row[0] == "actually do it differently"


class TestFactPlanRevisionsLineage:
    def test_lineage_block_populated(self, conn, accepted_session, tmp_path):
        _populate(conn, accepted_session, tmp_path)
        row = conn.execute(
            """
            SELECT created_at, last_updated_at, etl_run_id, record_source,
                   hash_diff, is_deleted
            FROM fact_plan_revisions LIMIT 1
            """
        ).fetchone()
        assert row[0] is not None
        assert row[1] is not None
        assert row[2] is not None
        assert row[3] == "claude_code_jsonl"
        assert row[4] is not None
        assert row[5] is False

    def test_idempotent_reetl(self, conn, accepted_session, tmp_path):
        _populate(conn, accepted_session, tmp_path)
        first = conn.execute(
            "SELECT last_updated_at FROM fact_plan_revisions"
        ).fetchall()
        _populate(conn, accepted_session, tmp_path)
        second = conn.execute(
            "SELECT last_updated_at FROM fact_plan_revisions"
        ).fetchall()
        assert first == second
