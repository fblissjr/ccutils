"""What the first full-corpus `ccutils audit` run flagged, fixed.

Claim: each test here pins one finding from the 2026-09-10 calibration run
to the fix that closes it. The fixtures model the payload shapes the corpus
actually carries (a Bash result has stdout/stderr/interrupted and no exit
code; a failed call says "Exit code N" in its text; a meta entry has no
timestamp; a file-history snapshot carries one inside the snapshot), not the
shapes an earlier fixture wished for.

Delete these and the columns can go back to being promises the source never
keeps.
"""

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.orchestrator import run_v15_etl


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


def _session(path, sid="fix-s"):
    lines = [
        {"type": "custom-title", "customTitle": "first title", "sessionId": sid},
        {"type": "permission-mode", "permissionMode": "default", "sessionId": sid},
        {"type": "user", "uuid": f"{sid}-u1", "sessionId": sid,
         "timestamp": "2026-04-19T10:00:00Z", "cwd": "/p", "permissionMode": "plan",
         "message": {"role": "user", "content": "go"}},
        {"type": "file-history-snapshot", "messageId": f"{sid}-u1",
         "snapshot": {"messageId": f"{sid}-u1", "trackedFileBackups": {},
                      "timestamp": "2026-04-19T10:00:00.500Z"},
         "isSnapshotUpdate": False},
        {"type": "assistant", "uuid": f"{sid}-a1", "parentUuid": f"{sid}-u1",
         "sessionId": sid, "timestamp": "2026-04-19T10:00:05Z", "requestId": "req_1",
         "message": {"id": "msg_1", "role": "assistant", "model": "claude-opus-4-7",
                     "usage": {"input_tokens": 10, "output_tokens": 5},
                     "content": [
                         {"type": "tool_use", "id": "toolu_ok", "name": "Bash",
                          "input": {"command": "ls"}},
                         ]}},
        {"type": "user", "uuid": f"{sid}-u2", "parentUuid": f"{sid}-a1",
         "sessionId": sid, "timestamp": "2026-04-19T10:00:06Z", "cwd": "/p",
         "toolUseResult": {"stdout": "ok", "stderr": "", "interrupted": False,
                           "isImage": False, "noOutputExpected": False},
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "toolu_ok", "content": "ok"}]}},
        {"type": "assistant", "uuid": f"{sid}-a2", "parentUuid": f"{sid}-u2",
         "sessionId": sid, "timestamp": "2026-04-19T10:00:07Z", "requestId": "req_2",
         "message": {"id": "msg_2", "role": "assistant", "model": "claude-opus-4-7",
                     "usage": {"input_tokens": 10, "output_tokens": 5},
                     "content": [
                         {"type": "tool_use", "id": "toolu_fail", "name": "Bash",
                          "input": {"command": "false"}},
                         ]}},
        {"type": "user", "uuid": f"{sid}-u3", "parentUuid": f"{sid}-a2",
         "sessionId": sid, "timestamp": "2026-04-19T10:00:08Z", "cwd": "/p",
         "toolUseResult": "Exit code 2\nboom",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "toolu_fail", "is_error": True,
              "content": "Exit code 2\nboom"}]}},
        {"type": "assistant", "uuid": f"{sid}-a3", "parentUuid": f"{sid}-u3",
         "sessionId": sid, "timestamp": "2026-04-19T10:00:09Z", "requestId": "req_3",
         "message": {"id": "msg_3", "role": "assistant", "model": "claude-opus-4-7",
                     "usage": {"input_tokens": 10, "output_tokens": 5},
                     "content": [
                         {"type": "tool_use", "id": "toolu_blocked", "name": "Bash",
                          "input": {"command": "rm -rf /", "dangerouslyDisableSandbox": True}},
                         ]}},
        {"type": "user", "uuid": f"{sid}-u4", "parentUuid": f"{sid}-a3",
         "sessionId": sid, "timestamp": "2026-04-19T10:00:10Z", "cwd": "/p",
         "toolUseResult": {"stdout": "", "stderr": "", "interrupted": False,
                           "dangerouslyDisableSandbox": True},
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "toolu_blocked", "is_error": True,
              "content": "<tool_use_error>Blocked: PreToolUse hook denied</tool_use_error>"}]}},
        {"type": "custom-title", "customTitle": "final title", "sessionId": sid},
        {"type": "permission-mode", "permissionMode": "auto", "sessionId": sid},
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines))
    return path


@pytest.fixture
def loaded(conn, tmp_path):
    run_v15_etl(conn, _session(tmp_path / "s.jsonl"), project_name="test-project",
                parquet_lake_root=tmp_path / "lake")
    return conn


class TestSessionMetaColumnsArePopulated:
    """custom_title and permission_mode were NULL on every one of 2,994
    sessions while fact_meta_events held the values."""

    def test_custom_title_is_the_last_one_stated(self, loaded):
        row = loaded.execute(
            "SELECT custom_title FROM dim_session WHERE session_id = 'fix-s'"
        ).fetchone()
        assert row == ("final title",)

    def test_permission_mode_is_the_last_one_stated(self, loaded):
        row = loaded.execute(
            "SELECT permission_mode FROM dim_session WHERE session_id = 'fix-s'"
        ).fetchone()
        assert row == ("auto",)


class TestFileHistorySnapshotsCarryTheirTimestamp:
    """The entry has no timestamp; the snapshot inside it does. date_key
    and time_key were NULL on all 6,081 rows."""

    def test_timestamp_and_keys_come_from_the_snapshot(self, loaded):
        row = loaded.execute(
            "SELECT timestamp IS NOT NULL, date_key, time_key "
            "FROM fact_file_history_snapshots"
        ).fetchone()
        assert row[0] is True
        assert row[1] == 20260419
        assert row[2] is not None


class TestBashColumnsSayWhatTheSourceStates:
    """exitCode / durationMs / wasInterrupted never appear in a real
    payload (0 of 52,974 Bash results). What the source states is is_error
    and the error text, so the failure is derived from those and named as
    derived."""

    def test_dropped_columns_are_gone(self, conn):
        cols = {r[0] for r in conn.execute("DESCRIBE fact_tool_calls").fetchall()}
        assert not {"bash_exit_code", "bash_duration_ms", "bash_interrupted",
                    "agent_was_interrupted"} & cols
        assert "caller_type" not in {
            r[0] for r in conn.execute("DESCRIBE fact_tool_calls").fetchall()}

    def test_exit_code_is_derived_from_the_error_text(self, loaded):
        rows = dict(loaded.execute(
            "SELECT tool_use_id, derived_exit_code FROM fact_tool_calls"
        ).fetchall())
        assert rows == {"toolu_ok": None, "toolu_fail": 2, "toolu_blocked": None}

    def test_failure_kind_is_classified(self, loaded):
        rows = dict(loaded.execute(
            "SELECT tool_use_id, derived_failure_kind FROM fact_tool_calls"
        ).fetchall())
        assert rows == {"toolu_ok": None, "toolu_fail": "nonzero_exit",
                        "toolu_blocked": "blocked_by_hook"}

    def test_sandbox_bypass_is_captured_as_stated(self, loaded):
        rows = dict(loaded.execute(
            "SELECT tool_use_id, sandbox_disabled FROM fact_tool_calls"
        ).fetchall())
        assert rows == {"toolu_ok": False, "toolu_fail": False, "toolu_blocked": True}


class TestEveryWriterHasAStep:
    def test_bridge_memory_link_step_is_recorded(self, conn, tmp_path):
        from ccutils.etl.dim_memory import run_memory_import

        root = tmp_path / "projects" / "-home-user-proj" / "memory"
        root.mkdir(parents=True)
        (root / "MEMORY.md").write_text("# index\n\n- [T](topic.md) -- hook\n")
        (root / "topic.md").write_text("---\nname: topic\n---\n\nbody [[other]]\n")
        run_memory_import(conn, projects_root=tmp_path / "projects")
        steps = dict(conn.execute(
            "SELECT table_name, rows_inserted FROM etl.steps WHERE table_name LIKE '%memory%'"
        ).fetchall())
        assert steps.get("dim_memory")
        assert steps.get("bridge_memory_link") == 2


class TestMemoryDatesResolve:
    def test_memory_without_stated_modified_still_has_a_dim_date_row(self, conn, tmp_path):
        """date_key falls back to the file mtime when frontmatter states no
        `modified`; dim_date was seeded from modified_at only, so those
        rows dangled (1 of 143 on the corpus)."""
        from ccutils.etl.dim_memory import run_memory_import

        root = tmp_path / "projects" / "-home-user-proj" / "memory"
        root.mkdir(parents=True)
        (root / "MEMORY.md").write_text("# index\n")
        (root / "topic.md").write_text("---\nname: topic\n---\n\nbody\n")
        run_memory_import(conn, projects_root=tmp_path / "projects")
        dangling = conn.execute(
            "SELECT COUNT(*) FROM dim_memory m WHERE m.date_key IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 FROM dim_date d WHERE d.date_key = m.date_key)"
        ).fetchone()[0]
        assert dangling == 0
