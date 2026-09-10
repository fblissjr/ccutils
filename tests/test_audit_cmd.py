"""`ccutils audit`: the warehouse checks itself, and can gate.

Claim: every structural bug this project shipped (a join key 100% NULL, a
category 100% 'unknown', a chain grain stuck at position 1, a natural key
violated, a column nothing writes) passed the unit suite and was found by
a person eyeballing corpus numbers. These checks are that eyeballing as a
command. Each test plants one defect class in an otherwise clean warehouse
and asserts the audit names it; the clean case asserts it stays quiet, so
the checks cannot pass by flagging everything.

Delete these and the audit can silently stop seeing a class, which is the
exact failure it exists to prevent.
"""

import json

import duckdb
import pytest
from click.testing import CliRunner

from ccutils import cli, create_star_schema
from ccutils.audit import run_audit
from ccutils.etl.orchestrator import run_v15_etl


def _session_with_a_tool_call(path, sid="audit-s"):
    """User -> assistant Bash call (with usage) -> tool result."""
    lines = [
        {"type": "user", "uuid": f"{sid}-u1", "sessionId": sid,
         "timestamp": "2026-04-19T10:00:00Z", "cwd": "/p",
         "message": {"role": "user", "content": "go"}},
        {"type": "assistant", "uuid": f"{sid}-a1", "parentUuid": f"{sid}-u1",
         "sessionId": sid, "timestamp": "2026-04-19T10:00:05Z",
         "requestId": "req_1",
         "message": {"id": "msg_1", "role": "assistant", "model": "claude-opus-4-7",
                     "usage": {"input_tokens": 10, "output_tokens": 5},
                     "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash",
                                  "input": {"command": "ls"}}]}},
        {"type": "user", "uuid": f"{sid}-u2", "parentUuid": f"{sid}-a1",
         "sessionId": sid, "timestamp": "2026-04-19T10:00:06Z", "cwd": "/p",
         "toolUseResult": {"stdout": "ok", "stderr": "", "interrupted": False},
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]}},
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines))
    return path


@pytest.fixture
def warehouse(tmp_path):
    """A small warehouse with one real session ETL'd through the pipeline."""
    db = tmp_path / "wh" / "archive.duckdb"
    db.parent.mkdir()
    conn = create_star_schema(db)
    run_v15_etl(
        conn, _session_with_a_tool_call(tmp_path / "s.jsonl"),
        project_name="test-project", parquet_lake_root=tmp_path / "lake",
    )
    conn.close()
    return db


def _open(db):
    return duckdb.connect(str(db))


def _findings(db, check=None):
    conn = _open(db)
    try:
        report = run_audit(conn)
    finally:
        conn.close()
    out = [f for f in report.findings if check is None or f.check == check]
    return out, report


class TestCleanWarehouseIsQuiet:
    def test_no_findings_on_a_clean_small_warehouse(self, warehouse):
        findings, report = _findings(warehouse)
        assert findings == [], [str(f) for f in findings]
        assert report.checks_run, "no check ran -- a green with no checks is no verdict"

    def test_sparse_column_checks_skip_below_threshold(self, warehouse):
        """One session cannot tell a NULL column from an unwritten one;
        the audit says it skipped rather than guessing either way."""
        _, report = _findings(warehouse)
        assert "null_column" in report.skipped
        assert "single_valued_column" in report.skipped


class TestPlantedDefectsAreFound:
    def test_natural_key_violation(self, warehouse):
        conn = _open(warehouse)
        conn.execute(
            "INSERT INTO fact_tool_calls SELECT * REPLACE ('dup-entry' AS entry_id, "
            "'other-hash' AS hash_diff) FROM fact_tool_calls LIMIT 1"
        )
        conn.close()
        findings, _ = _findings(warehouse, "natural_key_unique")
        assert [f.object for f in findings] == ["fact_tool_calls"]

    def test_declared_populated_but_no_step_wrote_it(self, warehouse):
        conn = _open(warehouse)
        conn.execute("DELETE FROM etl.steps WHERE table_name = 'fact_messages'")
        conn.close()
        findings, _ = _findings(warehouse, "declared_but_unwritten")
        assert [f.object for f in findings] == ["fact_messages"]

    def test_rows_present_but_no_step_names_the_table(self, warehouse):
        conn = _open(warehouse)
        conn.execute("DELETE FROM etl.steps WHERE table_name = 'dim_file'")
        conn.execute(
            "INSERT INTO dim_file (file_key, file_path, file_name) "
            "VALUES ('k', '/p/f.py', 'f.py')"
        )
        conn.close()
        findings, _ = _findings(warehouse, "rows_without_step")
        assert "dim_file" in [f.object for f in findings]

    def test_stuck_run(self, warehouse):
        conn = _open(warehouse)
        conn.execute(
            "INSERT INTO etl.runs (etl_run_id, version_key, source_path, status) "
            "VALUES ('stuck', '0/1', '/x', 'running')"
        )
        conn.close()
        findings, _ = _findings(warehouse, "stuck_runs")
        assert findings and findings[0].object == "etl.runs"

    def test_unresolved_foreign_key(self, warehouse):
        conn = _open(warehouse)
        conn.execute(
            "INSERT INTO fact_messages SELECT * REPLACE ('orphan' AS entry_id, "
            "'no-such-session' AS session_key) FROM fact_messages LIMIT 1"
        )
        conn.close()
        findings, _ = _findings(warehouse, "fk_unresolved")
        assert ("fact_messages", "session_key") in [(f.object, f.detail) for f in findings]

    def test_api_response_grain(self, warehouse):
        conn = _open(warehouse)
        conn.execute(
            "INSERT INTO fact_token_usage SELECT * REPLACE ('dup-entry' AS entry_id) "
            "FROM fact_token_usage LIMIT 1"
        )
        conn.close()
        findings, _ = _findings(warehouse, "api_response_grain")
        assert findings

    def test_null_column_above_threshold(self, warehouse):
        """Sixty-plus results and a column NULL on every one of them is a
        promise the source never keeps, and that is a finding. (This is
        the shape bash_exit_code had on 52,974 real rows before 1.0.0.)"""
        conn = _open(warehouse)
        conn.execute(
            "INSERT INTO fact_tool_calls SELECT r.* REPLACE "
            "('e' || i AS entry_id, 'toolu_' || i AS tool_use_id) "
            "FROM fact_tool_calls r, range(60) t(i) LIMIT 60"
        )
        conn.close()
        findings, _ = _findings(warehouse, "null_column")
        assert ("fact_tool_calls", "webfetch_http_code") in [
            (f.object, f.detail) for f in findings
        ]

    def test_view_that_fails_to_execute(self, warehouse):
        """DuckDB binds views at query time, so dropping a source table
        is how a view dies in practice."""
        conn = _open(warehouse)
        conn.execute("DROP TABLE fact_plan_revisions")
        conn.close()
        findings, _ = _findings(warehouse, "view_executes")
        assert [f.object for f in findings] == ["semantic_decisions"]


class TestExceptionsAreDataNotDeletedChecks:
    def test_allowlisted_finding_is_reported_as_allowed_not_failed(self, warehouse):
        conn = _open(warehouse)
        conn.execute(
            "INSERT INTO etl.runs (etl_run_id, version_key, source_path, status) "
            "VALUES ('stuck', '0/1', '/x', 'running')"
        )
        conn.execute(
            "INSERT INTO etl.audit_exceptions VALUES "
            "('stuck_runs', 'etl.runs', NULL, 'known: interrupted 2026-09-10')"
        )
        conn.close()
        findings, report = _findings(warehouse)
        assert findings == []
        assert [a.check for a in report.allowed] == ["stuck_runs"]


class TestCli:
    def test_clean_exits_zero(self, warehouse):
        result = CliRunner().invoke(cli, ["audit", "-o", str(warehouse.parent)])
        assert result.exit_code == 0, result.output
        assert "0 findings" in result.output

    def test_findings_exit_one_and_are_listed(self, warehouse):
        conn = _open(warehouse)
        conn.execute(
            "INSERT INTO etl.runs (etl_run_id, version_key, source_path, status) "
            "VALUES ('stuck', '0/1', '/x', 'running')"
        )
        conn.close()
        result = CliRunner().invoke(cli, ["audit", "-o", str(warehouse.parent)])
        assert result.exit_code == 1
        assert "stuck_runs" in result.output

    def test_foreign_schema_exits_two(self, tmp_path):
        db = tmp_path / "old.duckdb"
        c = duckdb.connect(str(db))
        c.execute("CREATE TABLE dim_session (session_key VARCHAR)")
        c.close()
        result = CliRunner().invoke(cli, ["audit", "-o", str(db)])
        assert result.exit_code == 2
        assert "rebuild" in result.output.lower()

    def test_missing_warehouse_exits_two(self, tmp_path):
        result = CliRunner().invoke(cli, ["audit", "-o", str(tmp_path)])
        assert result.exit_code == 2
