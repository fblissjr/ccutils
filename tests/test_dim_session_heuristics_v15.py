"""Tests for the v0.15 dim_session heuristic enrichment populator.

The legacy heuristics module classified intent / complexity / outcome /
domain per session using zero-dep keyword + scoring rules. v0.15 moves
them under ccutils.etl.heuristics and runs them as a post-fact populator
that UPDATEs dim_session in place. Inputs are pulled from the v0.15 facts
(fact_messages, fact_tool_uses, fact_tool_results, bridge_session_file +
dim_file) rather than from the legacy fact_tool_calls / fact_errors path.

This unblocks the NULL columns in semantic_project_context /
semantic_sessions for v0.15 databases.
"""

from __future__ import annotations

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.heuristics import (
    classify_complexity,
    classify_domain,
    classify_intent,
    classify_outcome,
)
from ccutils.etl.lineage import EtlRun
from ccutils.etl.orchestrator import run_v15_etl


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


class TestClassifierPureFunctions:
    """The classifiers are pure -- exercise their decision tables directly."""

    def test_classify_intent_bug_fix(self):
        assert classify_intent("fix the broken thing") == "bug_fix"

    def test_classify_intent_feature(self):
        assert classify_intent("add a new dashboard") == "feature"

    def test_classify_intent_refactor(self):
        assert classify_intent("refactor and clean up the auth layer") == "refactor"

    def test_classify_intent_test(self):
        assert classify_intent("write tests for the parser") == "test"

    def test_classify_intent_docs(self):
        assert classify_intent("update the readme") == "docs"

    def test_classify_intent_empty_defaults_to_explore(self):
        assert classify_intent("") == "explore"
        assert classify_intent(None) == "explore"

    def test_classify_complexity_trivial(self):
        # 0 tools, 1 message, no agents, no errors
        assert classify_complexity(0, 1, 0, 0) == "trivial"

    def test_classify_complexity_complex(self):
        # lots of tools + lots of messages + agent + errors
        assert classify_complexity(50, 100, 1, 5) == "complex"

    def test_classify_outcome_success(self):
        assert classify_outcome("Done -- all tests pass.") == "success"

    def test_classify_outcome_failure_via_text(self):
        assert classify_outcome("Sorry, I couldn't fix it.") == "failure"

    def test_classify_outcome_failure_via_error_rate(self):
        assert classify_outcome("Some progress made.", error_rate=0.9) == "failure"

    def test_classify_outcome_unknown_on_empty(self):
        assert classify_outcome("") == "unknown"
        assert classify_outcome(None) == "unknown"

    def test_classify_domain_backend(self):
        assert classify_domain([".py", ".py", ".rs"]) == "backend"

    def test_classify_domain_mixed(self):
        # equal weight on multiple domains
        assert classify_domain([".py", ".css"]) == "mixed"

    def test_classify_domain_unknown(self):
        assert classify_domain([]) == "unknown"
        assert classify_domain([".foo"]) == "unknown"


# ---------------------------------------------------------------------------
# Integration: full v0.15 ETL run, then verify dim_session columns populated
# ---------------------------------------------------------------------------


@pytest.fixture
def bug_fix_session(tmp_path):
    """Session anchored on a 'fix' prompt with an Edit + an error."""
    jsonl = tmp_path / "bugfix.jsonl"
    lines = [
        {"type": "user", "uuid": "u1", "sessionId": "bug-s",
         "timestamp": "2026-04-19T10:00:00Z", "cwd": "/work",
         "gitBranch": "main", "version": "2.1.114",
         "message": {"role": "user",
                     "content": "fix the broken parser bug"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "sessionId": "bug-s", "timestamp": "2026-04-19T10:00:01Z",
         "requestId": "r1",
         "message": {"role": "assistant", "model": "claude-opus-4-7",
                     "content": [
                         {"type": "text", "text": "On it"},
                         {"type": "tool_use", "id": "tu_edit",
                          "name": "Edit",
                          "input": {"file_path": "/work/parser.py",
                                    "old_string": "x", "new_string": "y"}},
                     ]}},
        {"type": "user", "uuid": "u2", "parentUuid": "a1",
         "sessionId": "bug-s", "timestamp": "2026-04-19T10:00:02Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_edit",
              "content": "ok", "is_error": False},
         ]},
         "toolUseResult": {"filePath": "/work/parser.py",
                           "userModified": False, "replaceAll": False,
                           "structuredPatch": []}},
        {"type": "assistant", "uuid": "a2", "parentUuid": "u2",
         "sessionId": "bug-s", "timestamp": "2026-04-19T10:00:03Z",
         "requestId": "r2",
         "message": {"role": "assistant", "model": "claude-opus-4-7",
                     "content": [
                         {"type": "text",
                          "text": "Done -- the bug is fixed."},
                     ]}},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


class TestDimSessionEnrichmentIntegration:
    def test_intent_classified_from_first_user_message(
        self, conn, bug_fix_session, tmp_path
    ):
        run_v15_etl(conn, bug_fix_session, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        intent = conn.execute(
            "SELECT intent FROM dim_session WHERE session_id = 'bug-s'"
        ).fetchone()[0]
        assert intent == "bug_fix"

    def test_outcome_classified_from_last_assistant_message(
        self, conn, bug_fix_session, tmp_path
    ):
        run_v15_etl(conn, bug_fix_session, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        outcome = conn.execute(
            "SELECT outcome FROM dim_session WHERE session_id = 'bug-s'"
        ).fetchone()[0]
        assert outcome == "success"

    def test_domain_classified_from_files_touched(
        self, conn, bug_fix_session, tmp_path
    ):
        run_v15_etl(conn, bug_fix_session, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        domain = conn.execute(
            "SELECT domain FROM dim_session WHERE session_id = 'bug-s'"
        ).fetchone()[0]
        # parser.py -> backend
        assert domain == "backend"

    def test_complexity_classified_from_session_metrics(
        self, conn, bug_fix_session, tmp_path
    ):
        run_v15_etl(conn, bug_fix_session, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        complexity = conn.execute(
            "SELECT complexity FROM dim_session WHERE session_id = 'bug-s'"
        ).fetchone()[0]
        # 1 tool, ~3 messages, no agent, no errors -> simple
        assert complexity in ("trivial", "simple")

    def test_first_user_message_captured_and_truncated(
        self, conn, bug_fix_session, tmp_path
    ):
        run_v15_etl(conn, bug_fix_session, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT first_user_message FROM dim_session "
            "WHERE session_id = 'bug-s'"
        ).fetchone()
        assert row[0] is not None
        assert "fix the broken parser bug" in row[0]
        # truncation cap
        assert len(row[0]) <= 500

    def test_last_assistant_message_captured(
        self, conn, bug_fix_session, tmp_path
    ):
        run_v15_etl(conn, bug_fix_session, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT last_assistant_message FROM dim_session "
            "WHERE session_id = 'bug-s'"
        ).fetchone()
        assert row[0] is not None
        assert "Done" in row[0]

    def test_semantic_project_context_returns_classified_session(
        self, conn, bug_fix_session, tmp_path
    ):
        run_v15_etl(conn, bug_fix_session, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            """
            SELECT intent, complexity, outcome, domain
            FROM semantic_project_context WHERE session_id = 'bug-s'
            """
        ).fetchone()
        assert row is not None
        assert row[0] == "bug_fix"
        assert row[2] == "success"
        assert row[3] == "backend"

    def test_idempotent_reetl(self, conn, bug_fix_session, tmp_path):
        run_v15_etl(conn, bug_fix_session, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        first = conn.execute(
            "SELECT intent, complexity, outcome, domain FROM dim_session"
        ).fetchall()
        run_v15_etl(conn, bug_fix_session, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        second = conn.execute(
            "SELECT intent, complexity, outcome, domain FROM dim_session"
        ).fetchall()
        assert first == second
