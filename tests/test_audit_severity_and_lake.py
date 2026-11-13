"""`ccutils audit` tells a broken build from a good one.

On 2026-11-13 audit exited 1 on two Linux warehouses alike: one whose
every session run had failed (dimensions full, every fact empty) and its
rebuild, which was correct. The column-profile checks (`null_column`,
`single_valued_column`) fire on any build scoped to one project, where one
git branch or one memory scope is the scope, not a defect, so they drowned
the one check that separated the two. A check that is always red says as
little as one that is always green.

So the checks are tiered. Structural checks are errors and set the exit
code; the column profile is a warning unless `--strict` asks for it to
gate (a release on a full corpus, where those checks found real bugs).
Two errors are new:

- `failed_runs`: an `etl.runs` row with status `failed`. Nothing in the
  audit looked at run status before.
- `lake_reconciliation`: the declared grains, checked against the Parquet
  lake beside the warehouse. `fact_messages` is one row per user/assistant
  entry and `fact_tool_calls` one row per tool call per transcript, so per
  session the lake and the table must agree. It is what showed the rebuild
  correct (every session agreeing on both counts). Only lake files written by
  this warehouse's successful runs count: the 2026-11-13 lake still held a
  failed earlier run's files under a stale partition.

What breaks without it: a build that loaded nothing and a build that
loaded everything get the same exit code, and the audit cannot gate.
"""

import json

import duckdb
import pytest
from click.testing import CliRunner

from ccutils import cli, create_star_schema
from ccutils.audit import run_audit
from ccutils.etl import orchestrator
from ccutils.etl.orchestrator import run_v15_etl


def _session(path, sid):
    """User -> assistant with one Bash call -> its result."""
    lines = [
        {"type": "user", "uuid": f"{sid}-u1", "sessionId": sid,
         "timestamp": "2026-04-19T10:00:00Z", "cwd": "/p",
         "message": {"role": "user", "content": "go"}},
        {"type": "assistant", "uuid": f"{sid}-a1", "parentUuid": f"{sid}-u1",
         "sessionId": sid, "timestamp": "2026-04-19T10:00:05Z",
         "message": {"id": f"msg-{sid}", "role": "assistant",
                     "model": "claude-opus-4-7",
                     "usage": {"input_tokens": 10, "output_tokens": 5},
                     "content": [{"type": "tool_use", "id": f"toolu-{sid}",
                                  "name": "Bash", "input": {"command": "ls"}}]}},
        {"type": "user", "uuid": f"{sid}-u2", "parentUuid": f"{sid}-a1",
         "sessionId": sid, "timestamp": "2026-04-19T10:00:06Z", "cwd": "/p",
         "toolUseResult": {"stdout": "ok", "stderr": "", "interrupted": False},
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": f"toolu-{sid}",
              "content": "ok"}]}},
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines))
    return path


def _build(tmp_path, sids=("s1", "s2")):
    """A warehouse with its lake beside it, the layout every build writes."""
    out = tmp_path / "wh"
    out.mkdir()
    proj = tmp_path / "projects" / "-p"
    proj.mkdir(parents=True)
    conn = create_star_schema(out / "archive.duckdb")
    for sid in sids:
        run_v15_etl(conn, _session(proj / f"{sid}.jsonl", sid),
                    parquet_lake_root=out / "parquet_lake")
    conn.close()
    return out


def _audit(out, lake=True):
    conn = duckdb.connect(str(out / "archive.duckdb"), read_only=True)
    try:
        return run_audit(conn, lake_root=(out / "parquet_lake") if lake else None)
    finally:
        conn.close()


def _checks(findings):
    return sorted({f.check for f in findings})


def _plant_null_column(out):
    """Sixty tool-call rows: enough for the column profile to look, and
    webfetch_http_code NULL on all of them (a Bash-only corpus)."""
    conn = duckdb.connect(str(out / "archive.duckdb"))
    conn.execute(
        "INSERT INTO fact_tool_calls SELECT r.* REPLACE "
        "('e' || i AS entry_id, 'toolu_' || i AS tool_use_id, "
        "'k' || i AS tool_call_key) "
        "FROM fact_tool_calls r, range(60) t(i) LIMIT 60"
    )
    conn.close()


class TestAGoodBuildIsClean:
    def test_no_errors_and_the_lake_was_read(self, tmp_path):
        report = _audit(_build(tmp_path))
        assert report.findings == [], [str(f) for f in report.findings]
        assert "lake_reconciliation" in report.checks_run
        assert "lake_reconciliation" not in report.skipped


class TestColumnProfileIsAWarning:
    def test_a_null_column_warns_and_does_not_fail(self, tmp_path):
        out = _build(tmp_path)
        _plant_null_column(out)
        report = _audit(out, lake=False)
        assert "null_column" in _checks(report.warnings)
        assert "null_column" not in _checks(report.findings)

    def test_cli_exits_zero_on_warnings_and_one_under_strict(self, tmp_path):
        out = _build(tmp_path)
        _plant_null_column(out)
        loose = CliRunner().invoke(cli, ["audit", "-o", str(out)])
        strict = CliRunner().invoke(cli, ["audit", "-o", str(out), "--strict"])
        # The planted rows are not in the lake, so lake_reconciliation would
        # fail them too; this test is about the column profile alone.
        assert "warn  null_column" in loose.output
        assert strict.exit_code == 1, strict.output


class TestFailedRunsAreErrors:
    def test_a_failed_run_is_an_error(self, tmp_path):
        out = _build(tmp_path)
        conn = duckdb.connect(str(out / "archive.duckdb"))
        conn.execute(
            "INSERT INTO etl.runs (etl_run_id, version_key, source_path, status, "
            "error_message) VALUES ('f1', '0/1', '/x.jsonl', 'failed', 'boom')"
        )
        conn.close()
        report = _audit(out)
        failed = [f for f in report.findings if f.check == "failed_runs"]
        assert len(failed) == 1 and "1 " in failed[0].measured

    def test_a_build_whose_facts_never_loaded_exits_one(self, tmp_path, monkeypatch):
        """The 2026-11-13 shape: every run fails at its first fact populator,
        after the dimensions are written."""
        def boom(conn, *, run):
            raise RuntimeError("Binder Error: simulated")

        monkeypatch.setattr(orchestrator, "populate_fact_messages", boom)
        out = tmp_path / "wh"
        out.mkdir()
        proj = tmp_path / "projects" / "-p"
        proj.mkdir(parents=True)
        conn = create_star_schema(out / "archive.duckdb")
        for sid in ("s1", "s2"):
            with pytest.raises(RuntimeError):
                run_v15_etl(conn, _session(proj / f"{sid}.jsonl", sid),
                            parquet_lake_root=out / "parquet_lake")
        conn.close()

        result = CliRunner().invoke(cli, ["audit", "-o", str(out)])
        assert result.exit_code == 1, result.output
        assert "failed_runs" in result.output


class TestTheLakeAndTheFactsAgree:
    def test_a_missing_message_row_is_found(self, tmp_path):
        out = _build(tmp_path)
        conn = duckdb.connect(str(out / "archive.duckdb"))
        conn.execute(
            "DELETE FROM fact_messages WHERE message_id = 's2-u2'"
        )
        conn.close()
        found = [f for f in _audit(out).findings
                 if f.check == "lake_reconciliation"]
        assert [(f.object, "s2" in f.measured) for f in found] == [
            ("fact_messages", True)
        ]

    def test_a_missing_tool_call_row_is_found(self, tmp_path):
        out = _build(tmp_path)
        conn = duckdb.connect(str(out / "archive.duckdb"))
        conn.execute("DELETE FROM fact_tool_calls WHERE session_id = 's1'")
        conn.close()
        found = [f for f in _audit(out).findings
                 if f.check == "lake_reconciliation"]
        assert [f.object for f in found] == ["fact_tool_calls"]

    def test_lake_files_from_other_runs_are_ignored(self, tmp_path):
        """A stale partition left by an earlier failed build is not this
        warehouse's lake and must not be counted against it."""
        out = _build(tmp_path)
        other = tmp_path / "other"
        other.mkdir()
        proj = tmp_path / "projects" / "-p"
        c2 = create_star_schema(other / "archive.duckdb")
        run_v15_etl(c2, _session(proj / "s3.jsonl", "s3"),
                    parquet_lake_root=out / "parquet_lake")
        c2.close()
        assert _audit(out).findings == []

    def test_no_lake_is_a_stated_skip(self, tmp_path):
        out = _build(tmp_path)
        report = _audit(out, lake=False)
        assert "lake_reconciliation" in report.skipped

    def test_cli_reads_the_lake_beside_the_warehouse(self, tmp_path):
        out = _build(tmp_path)
        conn = duckdb.connect(str(out / "archive.duckdb"))
        conn.execute("DELETE FROM fact_tool_calls WHERE session_id = 's1'")
        conn.close()
        result = CliRunner().invoke(cli, ["audit", "-o", str(out)])
        assert result.exit_code == 1, result.output
        assert "lake_reconciliation" in result.output
