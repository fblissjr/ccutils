"""Tests for the new fact_messages populator (Phase C chunk 2).

New shape (v0.15):
- Grain: one row per user or assistant entry in a session.
- Lineage block on every row (created_at, last_updated_at, version keys,
  etl_run_id, record_source, hash_diff, is_deleted, deleted_at).
- Degenerate dims on every row (session_id, message_id, entry_id).
- Honest token semantics: input_tokens = post-last-cache-breakpoint;
  cache_creation split into _5m and _1h; total_uncached_equivalent_tokens
  is the derived "what would this have cost without caching" number.
- stop_reason, permission_mode_at_send, prompt_id, request_id,
  is_api_error_message captured (previously dropped).
- content_json dropped (lives in stg_log_entries + Parquet lake).
- hash_diff guards UPDATE: re-running ETL on unchanged source is a no-op.
"""

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.fact_messages import populate_fact_messages
from ccutils.etl.lineage import EtlRun
from ccutils.etl.staging import load_session_to_staging
from ccutils.parsers.parquet_writer import write_session_to_parquet


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


def _stage_session(conn, jsonl_path, tmp_path, run):
    log_path, _ = write_session_to_parquet(
        jsonl_path, tmp_path / "lake",
        etl_run_id=run.etl_run_id, project_slug="test-project",
    )
    load_session_to_staging(conn, log_path)


@pytest.fixture
def basic_session(tmp_path):
    """One user, one assistant with usage, one tool_result user, one system."""
    jsonl = tmp_path / "basic.jsonl"
    lines = [
        {"type": "user", "uuid": "u1", "sessionId": "basic-session",
         "timestamp": "2026-04-19T10:00:00Z", "cwd": "/p", "gitBranch": "main",
         "permissionMode": "auto", "promptId": "prompt-1",
         "message": {"role": "user", "content": "go"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "sessionId": "basic-session", "timestamp": "2026-04-19T10:00:05Z",
         "requestId": "req_xyz",
         "message": {"role": "assistant", "model": "claude-opus-4-7",
             "content": [
                 {"type": "text", "text": "ok"},
                 {"type": "tool_use", "id": "toolu_001", "name": "Bash",
                  "input": {"command": "ls"}},
             ],
             "stop_reason": "tool_use",
             "usage": {
                 "input_tokens": 10,
                 "output_tokens": 5,
                 "cache_creation_input_tokens": 1500,
                 "cache_read_input_tokens": 8000,
                 "cache_creation": {
                     "ephemeral_5m_input_tokens": 1200,
                     "ephemeral_1h_input_tokens": 300,
                 },
             }}},
        {"type": "user", "uuid": "u2", "parentUuid": "a1",
         "sessionId": "basic-session", "timestamp": "2026-04-19T10:00:06Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "toolu_001", "content": "out"},
         ]},
         "toolUseResult": {"stdout": "out", "interrupted": False, "exitCode": 0}},
        # A system entry should be IGNORED by fact_messages.
        {"type": "system", "subtype": "turn_duration", "uuid": "s1",
         "sessionId": "basic-session", "timestamp": "2026-04-19T10:00:07Z",
         "durationMs": 999, "messageCount": 3},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


class TestFactMessagesDdl:
    def test_table_exists(self, conn):
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_messages'"
        ).fetchone()
        assert result is not None

    def test_required_lineage_columns(self, conn):
        cols = {c[0] for c in conn.execute("DESCRIBE fact_messages").fetchall()}
        for required in (
            "created_at", "last_updated_at",
            "created_by_version_key", "last_updated_by_version_key",
            "etl_run_id", "record_source", "hash_diff",
            "is_deleted", "deleted_at",
            "session_id", "message_id", "entry_id",
        ):
            assert required in cols, f"Missing lineage/degenerate col: {required}"

    def test_required_native_columns(self, conn):
        cols = {c[0] for c in conn.execute("DESCRIBE fact_messages").fetchall()}
        for required in (
            "message_type", "parent_message_id", "timestamp", "sequence_num",
            "is_sidechain", "is_meta", "is_compact_summary", "is_api_error_message",
            "stop_reason", "permission_mode_at_send", "prompt_id", "request_id",
            "api_error_text",
        ):
            assert required in cols, f"Missing native col: {required}"

    def test_required_token_columns_split(self, conn):
        """R11 fix: cache creation must be split into _5m and _1h columns."""
        cols = {c[0] for c in conn.execute("DESCRIBE fact_messages").fetchall()}
        for required in (
            "input_tokens", "output_tokens",
            "cache_creation_5m_tokens", "cache_creation_1h_tokens",
            "cache_read_tokens", "total_uncached_equivalent_tokens",
        ):
            assert required in cols, f"Missing token col: {required}"

    def test_content_json_dropped(self, conn):
        """content_json is duplicated in staging + Parquet; dropped from facts."""
        cols = {c[0] for c in conn.execute("DESCRIBE fact_messages").fetchall()}
        assert "content_json" not in cols


class TestPopulateFactMessages:
    def test_only_user_and_assistant_entries_become_rows(self, conn, basic_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(basic_session))
        _stage_session(conn, basic_session, tmp_path, run)
        populate_fact_messages(conn, run=run)
        types = sorted(
            r[0] for r in conn.execute("SELECT message_type FROM fact_messages").fetchall()
        )
        assert types == ["assistant", "user", "user"]

    def test_lineage_stamped_on_every_row(self, conn, basic_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(basic_session))
        _stage_session(conn, basic_session, tmp_path, run)
        populate_fact_messages(conn, run=run)
        rows = conn.execute(
            "SELECT etl_run_id, record_source, hash_diff, "
            "created_by_version_key, last_updated_by_version_key, is_deleted "
            "FROM fact_messages"
        ).fetchall()
        for r in rows:
            assert r[0] == run.etl_run_id
            assert r[1] == "claude_code_jsonl"
            assert r[2] is not None and len(r[2]) == 32  # hash_diff is MD5 hex
            assert r[3] == run.version_key
            assert r[4] == run.version_key
            assert r[5] is False

    def test_degenerate_session_id_carried(self, conn, basic_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(basic_session))
        _stage_session(conn, basic_session, tmp_path, run)
        populate_fact_messages(conn, run=run)
        # SELECT by session_id alone (no dim_session join) should work.
        n = conn.execute(
            "SELECT COUNT(*) FROM fact_messages WHERE session_id = 'basic-session'"
        ).fetchone()[0]
        assert n == 3

    def test_assistant_captures_stop_reason_and_request_id(self, conn, basic_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(basic_session))
        _stage_session(conn, basic_session, tmp_path, run)
        populate_fact_messages(conn, run=run)
        row = conn.execute(
            "SELECT stop_reason, request_id, is_api_error_message "
            "FROM fact_messages WHERE message_id = 'a1'"
        ).fetchone()
        assert row[0] == "tool_use"
        assert row[1] == "req_xyz"
        assert row[2] is False

    def test_user_captures_permission_mode_and_prompt_id(self, conn, basic_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(basic_session))
        _stage_session(conn, basic_session, tmp_path, run)
        populate_fact_messages(conn, run=run)
        row = conn.execute(
            "SELECT permission_mode_at_send, prompt_id "
            "FROM fact_messages WHERE message_id = 'u1'"
        ).fetchone()
        assert row[0] == "auto"
        assert row[1] == "prompt-1"

    def test_tokens_split_per_cache_tier(self, conn, basic_session, tmp_path):
        """R11: cache_creation is additive across tiers; populate from
        usage.cache_creation.{ephemeral_5m,ephemeral_1h}_input_tokens."""
        run = EtlRun.start(conn, source_path=str(basic_session))
        _stage_session(conn, basic_session, tmp_path, run)
        populate_fact_messages(conn, run=run)
        row = conn.execute(
            "SELECT input_tokens, output_tokens, "
            "cache_creation_5m_tokens, cache_creation_1h_tokens, "
            "cache_read_tokens, total_uncached_equivalent_tokens "
            "FROM fact_messages WHERE message_id = 'a1'"
        ).fetchone()
        assert row[0] == 10
        assert row[1] == 5
        assert row[2] == 1200
        assert row[3] == 300
        assert row[4] == 8000
        # Derived: total = read + creation_5m + creation_1h + input_tokens
        #        = 8000 + 1200 + 300 + 10 = 9510
        assert row[5] == 9510

    def test_user_with_no_usage_has_null_token_cols(self, conn, basic_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(basic_session))
        _stage_session(conn, basic_session, tmp_path, run)
        populate_fact_messages(conn, run=run)
        row = conn.execute(
            "SELECT input_tokens, cache_read_tokens "
            "FROM fact_messages WHERE message_id = 'u1'"
        ).fetchone()
        assert row[0] is None
        assert row[1] is None

    def test_has_tool_use_and_result_flags(self, conn, basic_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(basic_session))
        _stage_session(conn, basic_session, tmp_path, run)
        populate_fact_messages(conn, run=run)
        rows = {
            r[0]: r
            for r in conn.execute(
                "SELECT message_id, has_tool_use, has_tool_result, has_thinking "
                "FROM fact_messages"
            ).fetchall()
        }
        # a1 has tool_use, u2 has tool_result, u1 has neither
        assert rows["a1"][1] is True and rows["a1"][2] is False
        assert rows["u2"][1] is False and rows["u2"][2] is True
        assert rows["u1"][1] is False and rows["u1"][2] is False

    def test_sequence_num_monotonic_per_session(self, conn, basic_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(basic_session))
        _stage_session(conn, basic_session, tmp_path, run)
        populate_fact_messages(conn, run=run)
        seqs = [
            r[0] for r in conn.execute(
                "SELECT sequence_num FROM fact_messages "
                "WHERE session_id = 'basic-session' ORDER BY sequence_num"
            ).fetchall()
        ]
        # Stage seq for the 4 entries is 0,1,2,3 -- fact_messages keeps 0,1,2
        # (system entry at seq 3 is filtered out).
        assert seqs == [0, 1, 2]


class TestIdempotentReETL:
    """hash_diff-gated UPDATE means re-running ETL is a no-op when content
    didn't change; an actual change produces an UPDATE that bumps
    last_updated_at without touching created_at."""

    def test_second_run_does_not_change_created_at(self, conn, basic_session, tmp_path):
        run1 = EtlRun.start(conn, source_path=str(basic_session))
        _stage_session(conn, basic_session, tmp_path, run1)
        populate_fact_messages(conn, run=run1)
        created_first = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT message_id, created_at FROM fact_messages"
            ).fetchall()
        }

        # Re-run the entire pipeline -- no source changes
        run2 = EtlRun.start(conn, source_path=str(basic_session))
        _stage_session(conn, basic_session, tmp_path, run2)
        populate_fact_messages(conn, run=run2)
        created_second = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT message_id, created_at FROM fact_messages"
            ).fetchall()
        }
        assert created_first == created_second, (
            "created_at must be preserved on unchanged rows"
        )

    def test_second_run_unchanged_source_does_not_bump_last_updated_at(
        self, conn, basic_session, tmp_path
    ):
        """If hash_diff is unchanged, no UPDATE fires -- last_updated_at
        stays put. Otherwise last_updated_at becomes 'last ETL touch'
        rather than 'last content change'."""
        run1 = EtlRun.start(conn, source_path=str(basic_session))
        _stage_session(conn, basic_session, tmp_path, run1)
        populate_fact_messages(conn, run=run1)
        updated_first = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT message_id, last_updated_at FROM fact_messages"
            ).fetchall()
        }

        run2 = EtlRun.start(conn, source_path=str(basic_session))
        _stage_session(conn, basic_session, tmp_path, run2)
        populate_fact_messages(conn, run=run2)
        updated_second = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT message_id, last_updated_at FROM fact_messages"
            ).fetchall()
        }
        assert updated_first == updated_second


class TestSoftDelete:
    def test_removed_source_marks_is_deleted(self, conn, basic_session, tmp_path):
        run1 = EtlRun.start(conn, source_path=str(basic_session))
        _stage_session(conn, basic_session, tmp_path, run1)
        populate_fact_messages(conn, run=run1)
        n_before = conn.execute("SELECT COUNT(*) FROM fact_messages").fetchone()[0]
        assert n_before == 3

        # Now simulate that one entry vanished from source -- truncate staging
        # and reload with one row gone.
        conn.execute("DELETE FROM stg_log_entries")
        truncated = tmp_path / "basic-truncated.jsonl"
        truncated.write_text(
            json.dumps({
                "type": "user", "uuid": "u1", "sessionId": "basic-session",
                "timestamp": "2026-04-19T10:00:00Z",
                "message": {"role": "user", "content": "go"},
            }) + "\n"
        )
        run2 = EtlRun.start(conn, source_path=str(truncated))
        _stage_session(conn, truncated, tmp_path, run2)
        populate_fact_messages(conn, run=run2)

        # u1 from the truncated source is still active; a1 and u2 should be
        # marked is_deleted (their staged rows from basic_session were the
        # only proof of their existence).
        # NOTE: scope of "deletion detection" is per-session, not global.
        # For now we keep this simple -- soft-delete is for the same session_id
        # losing one of its entries between runs.
        active = conn.execute(
            "SELECT message_id FROM fact_messages WHERE is_deleted = FALSE ORDER BY message_id"
        ).fetchall()
        deleted = conn.execute(
            "SELECT message_id FROM fact_messages WHERE is_deleted = TRUE ORDER BY message_id"
        ).fetchall()
        assert ("u1",) in active
        assert ("a1",) in deleted
        assert ("u2",) in deleted
