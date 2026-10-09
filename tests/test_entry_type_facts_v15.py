"""Tests for the entry-type facts.

Three facts capture the top-level entry types that are not messages:

  fact_entry_events     one row per attachment, permission-mode,
                        custom-title, agent-name, last-prompt,
                        file-history-snapshot, queue-operation or pr-link
                        entry: (entry_type, subtype, value_text, payload_json)
  fact_progress_events  the progress data variants, with typed columns
  fact_system_events    the system subtypes, with typed columns

`fact_entry_events` was five tables until 1.0.0 (`fact_attachments`,
`fact_meta_events`, `fact_file_history_snapshots`, `fact_queue_operations`,
`fact_pr_links`). Each held one or two columns of its own beside the same
entry id, session and timestamp, and every reader filtered on one entry
type, so they are one table keyed on that type. What each entry type
contributes is asserted below per type, because that mapping is the whole
of what the collapse could have got wrong: a value left out of `value_text`
or `payload_json` is a value the warehouse no longer has.
"""

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.entry_type_facts import (
    ENTRY_EVENT_TYPES,
    populate_fact_entry_events,
    populate_fact_progress_events,
    populate_fact_system_events,
)
from ccutils.etl.lineage import EtlRun
from ccutils.etl.staging import load_session_to_staging
from ccutils.parsers.parquet_writer import write_session_to_parquet


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


def _stage(conn, jsonl_path, tmp_path, run):
    log_path, _ = write_session_to_parquet(
        jsonl_path, tmp_path / "lake",
        etl_run_id=run.etl_run_id, project_slug="test-project",
    )
    load_session_to_staging(conn, log_path)


@pytest.fixture
def attachment_session(tmp_path):
    jsonl = tmp_path / "att.jsonl"
    lines = [
        {"type": "attachment", "uuid": "att1", "sessionId": "att-s",
         "timestamp": "2026-04-19T10:00:00Z",
         "attachment": {"type": "diagnostics", "files": [
             {"uri": "/p/x.py", "diagnostics": [
                 {"message": "unused import", "severity": 4,
                  "range": {"start": {"line": 1, "character": 0},
                            "end": {"line": 1, "character": 12}}},
             ]},
         ]}},
        {"type": "attachment", "uuid": "att2", "sessionId": "att-s",
         "timestamp": "2026-04-19T10:00:01Z",
         "attachment": {"type": "hook_success", "hookName": "ruff",
                        "hookEvent": "PostToolUse", "toolUseID": "tu_x",
                        "stdout": "ok"}},
        {"type": "attachment", "uuid": "att3", "sessionId": "att-s",
         "timestamp": "2026-04-19T10:00:02Z",
         "attachment": {"type": "invoked_skills", "skills": [
             {"name": "x", "path": "/s/x.md", "content": "..."},
         ]}},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def progress_session(tmp_path):
    jsonl = tmp_path / "prog.jsonl"
    lines = [
        {"type": "progress", "uuid": "p1", "sessionId": "prog-s",
         "timestamp": "2026-04-19T10:00:00Z",
         "toolUseID": "tu_x", "parentToolUseID": "tu_x",
         "data": {"type": "hook_progress", "hookName": "ruff",
                  "hookEvent": "PreToolUse", "command": "hooks/ruff.py"}},
        {"type": "progress", "uuid": "p2", "sessionId": "prog-s",
         "timestamp": "2026-04-19T10:00:01Z",
         "toolUseID": "tu_y", "parentToolUseID": "tu_y",
         "data": {"type": "bash_progress", "stdout": "running..."}},
        {"type": "progress", "uuid": "p3", "sessionId": "prog-s",
         "timestamp": "2026-04-19T10:00:02Z",
         "data": {"type": "agent_progress", "agentId": "ag-1"}},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def system_session(tmp_path):
    jsonl = tmp_path / "sys.jsonl"
    lines = [
        {"type": "system", "subtype": "turn_duration", "uuid": "s1",
         "sessionId": "sys-s", "timestamp": "2026-04-19T10:00:00Z",
         "durationMs": 1234, "messageCount": 3},
        {"type": "system", "subtype": "stop_hook_summary", "uuid": "s2",
         "sessionId": "sys-s", "timestamp": "2026-04-19T10:00:01Z",
         "hookCount": 1, "preventedContinuation": False,
         "stopReason": "end_turn", "hasOutput": True, "level": "suggestion"},
        {"type": "system", "subtype": "api_error", "uuid": "s3",
         "sessionId": "sys-s", "timestamp": "2026-04-19T10:00:02Z",
         "error": {"status": 503, "type": "overloaded_error"},
         "retryInMs": 1000, "retryAttempt": 1, "maxRetries": 3, "level": "error"},
        {"type": "system", "subtype": "compact_boundary", "uuid": "s4",
         "sessionId": "sys-s", "timestamp": "2026-04-19T10:00:03Z",
         "content": "Conversation compacted",
         "compactMetadata": {"trigger": "auto", "preTokens": 100000},
         "logicalParentUuid": "u1"},
        {"type": "system", "subtype": "local_command", "uuid": "s5",
         "sessionId": "sys-s", "timestamp": "2026-04-19T10:00:04Z",
         "content": "<local-command-stdout>x</local-command-stdout>"},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def meta_session(tmp_path):
    """Meta entries the way Claude Code writes them.

    They carry NO timestamp and no uuid: only `type`, `sessionId` and the
    value. Measured over the whole corpus on 2026-10-09, none of the
    permission-mode, custom-title, agent-name or last-prompt entries has one
    (the record is in CHANGELOG.md). They sit between timestamped entries,
    and a permission-mode entry RESTATES the current mode far more often
    than it changes it.

    This fixture used to give every meta entry a timestamp. The tests then
    ordered by a column that is empty on real data, and passed.
    """
    jsonl = tmp_path / "meta.jsonl"
    msg = lambda kind, uuid, ts: {  # noqa: E731
        "type": kind, "uuid": uuid, "sessionId": "meta-s", "timestamp": ts,
        "cwd": "/p", "message": {"role": kind, "content": "x"}}
    lines = [
        {"type": "permission-mode", "sessionId": "meta-s", "permissionMode": "default"},
        msg("user", "u1", "2026-04-19T10:00:00Z"),
        {"type": "custom-title", "sessionId": "meta-s", "customTitle": "the title"},
        msg("assistant", "a1", "2026-04-19T10:00:05Z"),
        {"type": "permission-mode", "sessionId": "meta-s", "permissionMode": "default"},
        msg("user", "u2", "2026-04-19T10:01:00Z"),
        {"type": "permission-mode", "sessionId": "meta-s", "permissionMode": "plan"},
        {"type": "agent-name", "sessionId": "meta-s", "agentName": "Explore"},
        msg("assistant", "a2", "2026-04-19T10:01:30Z"),
        {"type": "permission-mode", "sessionId": "meta-s", "permissionMode": "plan"},
        msg("user", "u3", "2026-04-19T10:02:00Z"),
        {"type": "permission-mode", "sessionId": "meta-s", "permissionMode": "acceptEdits"},
        {"type": "last-prompt", "sessionId": "meta-s", "lastPrompt": "make it faster"},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def file_history_session(tmp_path):
    jsonl = tmp_path / "fh.jsonl"
    lines = [
        {"type": "file-history-snapshot", "uuid": "fh1", "sessionId": "fh-s",
         "timestamp": "2026-04-19T10:00:00Z",
         "messageId": "m1", "isSnapshotUpdate": False,
         "snapshot": {"messageId": "m1", "trackedFileBackups": {},
                      "timestamp": "2026-04-19T10:00:00Z"}},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def queue_op_session(tmp_path):
    jsonl = tmp_path / "qo.jsonl"
    lines = [
        {"type": "queue-operation", "uuid": "qo1", "sessionId": "qo-s",
         "timestamp": "2026-04-19T10:00:00Z",
         "operation": "enqueue", "content": "queued prompt"},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def pr_session(tmp_path):
    jsonl = tmp_path / "pr.jsonl"
    lines = [
        {"type": "pr-link", "uuid": "pr1", "sessionId": "pr-s",
         "timestamp": "2026-04-19T10:00:00Z",
         "prNumber": 42, "prUrl": "https://github.com/o/r/pull/42",
         "prRepository": "o/r"},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


# --- DDL tests ---


class TestNewFactDdl:
    """Every new fact must have the standard lineage block + degenerate dims."""

    REQUIRED_LINEAGE = (
        "created_at", "last_updated_at",
        "created_by_version_key", "last_updated_by_version_key",
        "etl_run_id", "record_source", "hash_diff",
        "is_deleted", "deleted_at",
        "entry_id", "session_id",
    )

    @pytest.mark.parametrize("table", [
        "fact_entry_events", "fact_progress_events", "fact_system_events",
    ])
    def test_table_exists_with_lineage(self, conn, table):
        result = conn.execute(
            f"SELECT name FROM sqlite_master WHERE name='{table}'"
        ).fetchone()
        assert result is not None, f"Table missing: {table}"
        cols = {c[0] for c in conn.execute(f"DESCRIBE {table}").fetchall()}
        for required in self.REQUIRED_LINEAGE:
            assert required in cols, f"{table} missing {required}"


# --- Populator tests ---


class TestFactEntryEvents:
    """One row per entry of a collapsed type, each carrying what its own
    table used to."""

    def _load(self, conn, jsonl, tmp_path):
        run = EtlRun.start(conn, source_path=str(jsonl))
        _stage(conn, jsonl, tmp_path, run)
        populate_fact_entry_events(conn, run=run)

    def test_attachments_carry_their_type_and_payload(self, conn, attachment_session, tmp_path):
        self._load(conn, attachment_session, tmp_path)
        rows = conn.execute(
            "SELECT entry_type, subtype, value_text, payload_json "
            "FROM fact_entry_events ORDER BY subtype"
        ).fetchall()
        assert [(r[0], r[1], r[2]) for r in rows] == [
            ("attachment", "diagnostics", None),
            ("attachment", "hook_success", None),
            ("attachment", "invoked_skills", None),
        ]
        assert json.loads(rows[1][3])["hookName"] == "ruff"

    def test_meta_entries_keep_their_values(self, conn, meta_session, tmp_path):
        self._load(conn, meta_session, tmp_path)
        assert conn.execute("SELECT COUNT(*) FROM fact_entry_events").fetchone()[0] == 8
        others = dict(conn.execute(
            "SELECT entry_type, value_text FROM fact_entry_events "
            "WHERE entry_type <> 'permission-mode'"
        ).fetchall())
        assert others == {
            "custom-title": "the title",
            "agent-name": "Explore",
            "last-prompt": "make it faster",
        }
        assert conn.execute(
            "SELECT COUNT(*) FROM fact_entry_events WHERE subtype IS NOT NULL"
        ).fetchone()[0] == 0, "a meta entry states no sub-kind"

    def test_meta_entries_are_ordered_by_position_not_by_time(
        self, conn, meta_session, tmp_path
    ):
        """The file position is the only order a meta entry has.

        `timestamp` is what the entry states, and these state none. Ordering
        the permission-mode rows by it is ordering by NULL. `sequence_num`
        is the entry's position in its transcript, which is stated by the
        file itself and is the same column `fact_messages` carries, so an
        entry can also be placed between two messages.
        """
        self._load(conn, meta_session, tmp_path)
        rows = conn.execute(
            "SELECT sequence_num, value_text, timestamp FROM fact_entry_events "
            "WHERE entry_type = 'permission-mode' ORDER BY sequence_num"
        ).fetchall()
        assert [r[1] for r in rows] == ["default", "default", "plan", "plan", "acceptEdits"]
        assert [r[0] for r in rows] == [0, 4, 6, 9, 11]
        assert all(r[2] is None for r in rows), "none is stated, so none is stored"

    def test_an_entry_with_no_time_is_placed_by_its_neighbours(
        self, conn, meta_session, tmp_path
    ):
        """`derived_timestamp` is the stated time of the entry before it in
        the file, or of the one after when nothing timestamped precedes it.
        Named derived because it is; `timestamp` stays NULL. The date and
        time keys follow it, so a date filter does not drop these rows."""
        self._load(conn, meta_session, tmp_path)
        rows = conn.execute(
            "SELECT sequence_num, strftime(derived_timestamp, '%H:%M:%S'), date_key, time_key "
            "FROM fact_entry_events ORDER BY sequence_num"
        ).fetchall()
        assert rows == [
            (0, "10:00:00", 20260419, 1000),   # nothing before it: the next entry's
            (2, "10:00:00", 20260419, 1000),
            (4, "10:00:05", 20260419, 1000),
            (6, "10:01:00", 20260419, 1001),
            (7, "10:01:00", 20260419, 1001),
            (9, "10:01:30", 20260419, 1001),
            (11, "10:02:00", 20260419, 1002),
            (12, "10:02:00", 20260419, 1002),
        ]

    def test_an_entry_that_states_its_time_derives_none(
        self, conn, attachment_session, tmp_path
    ):
        self._load(conn, attachment_session, tmp_path)
        rows = conn.execute(
            "SELECT timestamp IS NOT NULL, derived_timestamp, sequence_num "
            "FROM fact_entry_events ORDER BY sequence_num"
        ).fetchall()
        assert rows == [(True, None, 0), (True, None, 1), (True, None, 2)]

    def test_file_history_snapshot_keeps_its_link_flag_and_snapshot(
        self, conn, file_history_session, tmp_path
    ):
        self._load(conn, file_history_session, tmp_path)
        entry_type, value, payload = conn.execute(
            "SELECT entry_type, value_text, payload_json FROM fact_entry_events"
        ).fetchone()
        assert entry_type == "file-history-snapshot"
        assert value == "m1", "the message the snapshot belongs to"
        parsed = json.loads(payload)
        assert parsed["isSnapshotUpdate"] is False
        assert parsed["snapshot"]["messageId"] == "m1"
        assert "trackedFileBackups" in parsed["snapshot"]

    def test_file_history_snapshot_takes_its_time_from_the_snapshot(self, conn, tmp_path):
        """The entry itself carries no timestamp; the snapshot inside does."""
        jsonl = tmp_path / "fh-nots.jsonl"
        jsonl.write_text(json.dumps(
            {"type": "file-history-snapshot", "uuid": "fh2", "sessionId": "fh2-s",
             "messageId": "m2", "isSnapshotUpdate": True,
             "snapshot": {"messageId": "m2", "trackedFileBackups": {},
                          "timestamp": "2026-04-19T11:22:33Z"}}))
        self._load(conn, jsonl, tmp_path)
        ts, date_key = conn.execute(
            "SELECT timestamp, date_key FROM fact_entry_events"
        ).fetchone()
        assert ts.strftime("%H:%M:%S") == "11:22:33"
        assert date_key == 20260419

    def test_queue_operation_carries_operation_and_content(self, conn, queue_op_session, tmp_path):
        self._load(conn, queue_op_session, tmp_path)
        assert conn.execute(
            "SELECT entry_type, subtype, value_text FROM fact_entry_events"
        ).fetchone() == ("queue-operation", "enqueue", "queued prompt")

    def test_pr_link_carries_url_number_and_repository(self, conn, pr_session, tmp_path):
        self._load(conn, pr_session, tmp_path)
        row = conn.execute(
            "SELECT entry_type, value_text, "
            "       json_extract(payload_json, '$.prNumber')::INTEGER, "
            "       json_extract_string(payload_json, '$.prRepository') "
            "FROM fact_entry_events"
        ).fetchone()
        assert row == ("pr-link", "https://github.com/o/r/pull/42", 42, "o/r")

    def test_only_the_declared_entry_types_land(self, conn, tmp_path):
        """The scope is a list, not "everything that is not a message".

        Other top-level types exist in real transcripts (`ai-title`, `mode`,
        `cost-state` and more). Taking them in is a decision about what the
        warehouse holds, to be made per type, not a side effect of a
        collapse.
        """
        assert ENTRY_EVENT_TYPES == (
            "attachment", "permission-mode", "custom-title", "agent-name",
            "last-prompt", "file-history-snapshot", "queue-operation", "pr-link",
        )
        jsonl = tmp_path / "mixed.jsonl"
        jsonl.write_text("\n".join(json.dumps(d) for d in [
            {"type": "user", "uuid": "u1", "sessionId": "mix-s",
             "timestamp": "2026-04-19T10:00:00Z", "cwd": "/p",
             "message": {"role": "user", "content": "hi"}},
            {"type": "ai-title", "sessionId": "mix-s", "aiTitle": "a title"},
            {"type": "custom-title", "sessionId": "mix-s",
             "timestamp": "2026-04-19T10:00:01Z", "customTitle": "kept"},
            {"type": "pr-link", "uuid": "pr9", "sessionId": "mix-s",
             "timestamp": "2026-04-19T10:00:02Z", "prNumber": 9,
             "prUrl": "https://github.com/o/r/pull/9", "prRepository": "o/r"},
        ]))
        self._load(conn, jsonl, tmp_path)
        rows = conn.execute(
            "SELECT entry_type, value_text FROM fact_entry_events ORDER BY timestamp"
        ).fetchall()
        assert rows == [("custom-title", "kept"),
                        ("pr-link", "https://github.com/o/r/pull/9")]
        ids = conn.execute("SELECT entry_id FROM fact_entry_events").fetchall()
        assert len({r[0] for r in ids}) == 2


class TestTheFiveTablesAreGone:
    """A table that is created and never filled reads as "nothing happened"."""

    GONE = ("fact_attachments", "fact_meta_events", "fact_file_history_snapshots",
            "fact_queue_operations", "fact_pr_links")

    def test_they_do_not_exist_or_stay_declared(self, conn):
        from ccutils.etl import entry_type_facts
        from ccutils.export.duckdb_archive import _PROGRESS_TABLES
        from ccutils.schemas.star.schema import NATURAL_KEYS, TABLE_COVERAGE

        tables = {r[0] for r in conn.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()}
        for name in self.GONE:
            assert name not in tables, name
            assert name not in NATURAL_KEYS, name
            assert name not in TABLE_COVERAGE, name
            assert name not in _PROGRESS_TABLES, name
            assert not hasattr(entry_type_facts, f"populate_{name}"), name
        assert "fact_entry_events" in tables
        assert NATURAL_KEYS["fact_entry_events"] == "entry_id"
        assert "fact_entry_events" in TABLE_COVERAGE
        assert "fact_entry_events" in _PROGRESS_TABLES


class TestFactProgressEvents:
    def test_all_three_progress_variants_loaded(self, conn, progress_session, tmp_path):
        """The legacy ETL only kept agent_progress; we now keep all 6."""
        run = EtlRun.start(conn, source_path=str(progress_session))
        _stage(conn, progress_session, tmp_path, run)
        populate_fact_progress_events(conn, run=run)
        types = sorted(
            r[0] for r in conn.execute(
                "SELECT data_type FROM fact_progress_events"
            ).fetchall()
        )
        assert types == ["agent_progress", "bash_progress", "hook_progress"]

    def test_hook_event_extracted(self, conn, progress_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(progress_session))
        _stage(conn, progress_session, tmp_path, run)
        populate_fact_progress_events(conn, run=run)
        row = conn.execute(
            "SELECT hook_name, hook_event "
            "FROM fact_progress_events WHERE data_type = 'hook_progress'"
        ).fetchone()
        assert row[0] == "ruff"
        assert row[1] == "PreToolUse"


class TestFactSystemEvents:
    def test_all_five_subtypes_loaded(self, conn, system_session, tmp_path):
        """Legacy ETL only kept turn_duration + stop_hook_summary;
        we now keep all 5 distinct subtypes in the fixture."""
        run = EtlRun.start(conn, source_path=str(system_session))
        _stage(conn, system_session, tmp_path, run)
        populate_fact_system_events(conn, run=run)
        subtypes = sorted(
            r[0] for r in conn.execute(
                "SELECT subtype FROM fact_system_events"
            ).fetchall()
        )
        assert subtypes == [
            "api_error", "compact_boundary", "local_command",
            "stop_hook_summary", "turn_duration",
        ]

    def test_typed_columns_per_subtype(self, conn, system_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(system_session))
        _stage(conn, system_session, tmp_path, run)
        populate_fact_system_events(conn, run=run)
        # turn_duration: durationMs + messageCount
        td = conn.execute(
            "SELECT duration_ms, message_count "
            "FROM fact_system_events WHERE subtype = 'turn_duration'"
        ).fetchone()
        assert td[0] == 1234
        assert td[1] == 3
        # api_error: retry fields
        ae = conn.execute(
            "SELECT retry_in_ms, retry_attempt, max_retries "
            "FROM fact_system_events WHERE subtype = 'api_error'"
        ).fetchone()
        assert ae[0] == 1000.0
        assert ae[1] == 1
        assert ae[2] == 3
        # compact_boundary: trigger + preTokens
        cb = conn.execute(
            "SELECT compact_trigger, compact_pre_tokens "
            "FROM fact_system_events WHERE subtype = 'compact_boundary'"
        ).fetchone()
        assert cb[0] == "auto"
        assert cb[1] == 100000


class TestIdempotency:
    """Every populator must be idempotent under re-ETL, for each entry type."""

    @pytest.mark.parametrize("fixture_name,populator_name", [
        ("attachment_session", "populate_fact_entry_events"),
        ("progress_session", "populate_fact_progress_events"),
        ("system_session", "populate_fact_system_events"),
        ("meta_session", "populate_fact_entry_events"),
        ("file_history_session", "populate_fact_entry_events"),
        ("queue_op_session", "populate_fact_entry_events"),
        ("pr_session", "populate_fact_entry_events"),
    ])
    def test_reetl_does_not_bump_last_updated_at(
        self, request, conn, tmp_path, fixture_name, populator_name,
    ):
        from ccutils.etl import entry_type_facts
        populator = getattr(entry_type_facts, populator_name)
        fixture = request.getfixturevalue(fixture_name)

        run1 = EtlRun.start(conn, source_path=str(fixture))
        _stage(conn, fixture, tmp_path, run1)
        populator(conn, run=run1)
        table = populator_name.replace("populate_", "")
        first = sorted(
            (r[0], r[1])
            for r in conn.execute(
                f"SELECT entry_id, last_updated_at FROM {table}"
            ).fetchall()
        )

        run2 = EtlRun.start(conn, source_path=str(fixture))
        _stage(conn, fixture, tmp_path, run2)
        populator(conn, run=run2)
        second = sorted(
            (r[0], r[1])
            for r in conn.execute(
                f"SELECT entry_id, last_updated_at FROM {table}"
            ).fetchall()
        )
        assert first == second
