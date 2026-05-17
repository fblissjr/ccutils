"""Tests for fact_token_usage (Phase C5a).

Per-API-response token breakdown with R11 cache-arithmetic fix:
- cache_creation split into _5m and _1h columns (pricing tiers are
  1.25x and 2x respectively; uniform rollup mis-bills cache-heavy
  sessions by up to 12.5x).
- input_tokens honestly named (Anthropic semantics: post-last-cache-
  breakpoint, NOT total uncached input).
- total_uncached_equivalent_tokens derived: what the bill would have
  been with no caching at all.
- service_tier / speed / inference_geo captured (legacy ETL had
  service_tier + speed but not inference_geo or server-tool usage).
"""

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.fact_token_usage import populate_fact_token_usage
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
def cached_session(tmp_path):
    """Two assistant responses with usage. One cache-heavy (5m + 1h split),
    one cache-light (read-only). User message with no usage at all."""
    jsonl = tmp_path / "cached.jsonl"
    lines = [
        {"type": "user", "uuid": "u1", "sessionId": "cached-s",
         "timestamp": "2026-04-19T10:00:00Z",
         "message": {"role": "user", "content": "go"}},
        # Cache-heavy response: 1500 created (1200 5m + 300 1h), 8000 read
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "sessionId": "cached-s", "timestamp": "2026-04-19T10:00:05Z",
         "message": {"role": "assistant", "model": "claude-opus-4-7",
             "content": [{"type": "text", "text": "ok"}],
             "stop_reason": "end_turn",
             "usage": {
                 "input_tokens": 10,
                 "output_tokens": 5,
                 "cache_creation_input_tokens": 1500,
                 "cache_read_input_tokens": 8000,
                 "cache_creation": {
                     "ephemeral_5m_input_tokens": 1200,
                     "ephemeral_1h_input_tokens": 300,
                 },
                 "service_tier": "standard",
                 "speed": "standard",
                 "inference_geo": "not_available",
                 "server_tool_use": {
                     "web_search_requests": 0,
                     "web_fetch_requests": 0,
                 },
             }}},
        # Cache-light response: only reads, no creation
        {"type": "assistant", "uuid": "a2", "parentUuid": "a1",
         "sessionId": "cached-s", "timestamp": "2026-04-19T10:00:10Z",
         "message": {"role": "assistant", "model": "claude-opus-4-7",
             "content": [{"type": "text", "text": "more"}],
             "stop_reason": "end_turn",
             "usage": {
                 "input_tokens": 3,
                 "output_tokens": 2,
                 "cache_read_input_tokens": 9000,
                 "service_tier": "priority",
                 "speed": "standard",
                 "inference_geo": "us",
             }}},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def no_usage_session(tmp_path):
    """Pre-v0.13 session with no usage on assistant messages. Should produce
    zero rows (we only emit when usage is present)."""
    jsonl = tmp_path / "no_usage.jsonl"
    lines = [
        {"type": "user", "uuid": "u1", "sessionId": "no-u-s",
         "message": {"role": "user", "content": "hi"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "sessionId": "no-u-s",
         "message": {"role": "assistant", "model": "claude-opus-4-5-20251101",
             "content": [{"type": "text", "text": "ok"}]}},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


class TestFactTokenUsageDdl:
    def test_table_exists(self, conn):
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE name='fact_token_usage'"
        ).fetchone()
        assert result is not None

    def test_lineage_columns(self, conn):
        cols = {c[0] for c in conn.execute("DESCRIBE fact_token_usage").fetchall()}
        for required in (
            "created_at", "last_updated_at",
            "created_by_version_key", "last_updated_by_version_key",
            "etl_run_id", "record_source", "hash_diff",
            "is_deleted", "deleted_at",
            "entry_id", "session_id",
        ):
            assert required in cols, f"Missing lineage col: {required}"

    def test_cache_tiers_split(self, conn):
        """R11: cache_creation must be split into _5m and _1h columns."""
        cols = {c[0] for c in conn.execute("DESCRIBE fact_token_usage").fetchall()}
        for required in (
            "cache_creation_5m_tokens",
            "cache_creation_1h_tokens",
            "cache_creation_total_tokens",
            "cache_read_tokens",
        ):
            assert required in cols, f"Missing cache col: {required}"

    def test_total_uncached_equivalent_present(self, conn):
        cols = {c[0] for c in conn.execute("DESCRIBE fact_token_usage").fetchall()}
        assert "total_uncached_equivalent_tokens" in cols

    def test_metadata_columns(self, conn):
        """Web research R: capture service_tier, speed, inference_geo, server_tool_use."""
        cols = {c[0] for c in conn.execute("DESCRIBE fact_token_usage").fetchall()}
        for required in (
            "service_tier", "speed", "inference_geo",
            "server_tool_use_web_search_requests",
            "server_tool_use_web_fetch_requests",
        ):
            assert required in cols, f"Missing metadata col: {required}"


class TestPopulateFactTokenUsage:
    def test_only_assistant_with_usage_becomes_a_row(self, conn, cached_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(cached_session))
        _stage(conn, cached_session, tmp_path, run)
        populate_fact_token_usage(conn, run=run)
        n = conn.execute("SELECT COUNT(*) FROM fact_token_usage").fetchone()[0]
        # Two assistant entries with usage; user has none -> 2 rows
        assert n == 2

    def test_no_usage_produces_zero_rows(self, conn, no_usage_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(no_usage_session))
        _stage(conn, no_usage_session, tmp_path, run)
        populate_fact_token_usage(conn, run=run)
        n = conn.execute("SELECT COUNT(*) FROM fact_token_usage").fetchone()[0]
        assert n == 0

    def test_cache_tiers_correctly_split(self, conn, cached_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(cached_session))
        _stage(conn, cached_session, tmp_path, run)
        populate_fact_token_usage(conn, run=run)
        # a1: 1200 5m, 300 1h, total 1500
        row = conn.execute(
            "SELECT cache_creation_5m_tokens, cache_creation_1h_tokens, "
            "cache_creation_total_tokens, cache_read_tokens "
            "FROM fact_token_usage ftu "
            "JOIN stg_log_entries sle ON sle.entry_id = ftu.entry_id "
            "WHERE sle.uuid = 'a1'"
        ).fetchone()
        assert row[0] == 1200
        assert row[1] == 300
        assert row[2] == 1500
        assert row[3] == 8000

    def test_total_uncached_equivalent_derived(self, conn, cached_session, tmp_path):
        """R11: total = input + cache_creation_total + cache_read."""
        run = EtlRun.start(conn, source_path=str(cached_session))
        _stage(conn, cached_session, tmp_path, run)
        populate_fact_token_usage(conn, run=run)
        row = conn.execute(
            "SELECT total_uncached_equivalent_tokens "
            "FROM fact_token_usage ftu "
            "JOIN stg_log_entries sle ON sle.entry_id = ftu.entry_id "
            "WHERE sle.uuid = 'a1'"
        ).fetchone()
        # 10 input + 1500 cache_creation_total + 8000 cache_read = 9510
        assert row[0] == 9510

    def test_metadata_columns_populated(self, conn, cached_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(cached_session))
        _stage(conn, cached_session, tmp_path, run)
        populate_fact_token_usage(conn, run=run)
        a1 = conn.execute(
            "SELECT service_tier, speed, inference_geo, "
            "server_tool_use_web_search_requests, "
            "server_tool_use_web_fetch_requests "
            "FROM fact_token_usage ftu "
            "JOIN stg_log_entries sle ON sle.entry_id = ftu.entry_id "
            "WHERE sle.uuid = 'a1'"
        ).fetchone()
        assert a1[0] == "standard"
        assert a1[1] == "standard"
        assert a1[2] == "not_available"
        assert a1[3] == 0
        assert a1[4] == 0
        # a2 has priority + us
        a2 = conn.execute(
            "SELECT service_tier, inference_geo "
            "FROM fact_token_usage ftu "
            "JOIN stg_log_entries sle ON sle.entry_id = ftu.entry_id "
            "WHERE sle.uuid = 'a2'"
        ).fetchone()
        assert a2[0] == "priority"
        assert a2[1] == "us"

    def test_lineage_stamped(self, conn, cached_session, tmp_path):
        run = EtlRun.start(conn, source_path=str(cached_session))
        _stage(conn, cached_session, tmp_path, run)
        populate_fact_token_usage(conn, run=run)
        for r in conn.execute(
            "SELECT etl_run_id, record_source, hash_diff FROM fact_token_usage"
        ).fetchall():
            assert r[0] == run.etl_run_id
            assert r[1] == "claude_code_jsonl"
            assert len(r[2]) == 32


class TestIdempotency:
    def test_reetl_does_not_bump_last_updated_at(self, conn, cached_session, tmp_path):
        run1 = EtlRun.start(conn, source_path=str(cached_session))
        _stage(conn, cached_session, tmp_path, run1)
        populate_fact_token_usage(conn, run=run1)
        first = sorted(
            (r[0], r[1])
            for r in conn.execute(
                "SELECT entry_id, last_updated_at FROM fact_token_usage"
            ).fetchall()
        )
        run2 = EtlRun.start(conn, source_path=str(cached_session))
        _stage(conn, cached_session, tmp_path, run2)
        populate_fact_token_usage(conn, run=run2)
        second = sorted(
            (r[0], r[1])
            for r in conn.execute(
                "SELECT entry_id, last_updated_at FROM fact_token_usage"
            ).fetchall()
        )
        assert first == second
