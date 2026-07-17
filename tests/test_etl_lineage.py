"""Tests for the lineage helpers (Phase B chunk 2).

These helpers stamp facts with provenance + version + change-detection
columns so re-running ETL on unchanged source is a no-op and re-runs
after a business-rule change produce a clean audit trail.
"""

import pytest

from ccutils import create_star_schema
from ccutils.etl.lineage import (
    EtlRun,
    PARSER_VERSION,
    hash_diff,
    record_source_label,
)


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


class TestEtlRunIdGeneration:
    def test_etl_run_id_is_unique_per_run(self, conn):
        r1 = EtlRun.start(conn, source_path="x")
        r2 = EtlRun.start(conn, source_path="x")
        assert r1.etl_run_id != r2.etl_run_id

    def test_etl_run_id_is_a_hex_string(self, conn):
        r = EtlRun.start(conn, source_path="x")
        assert isinstance(r.etl_run_id, str)
        assert len(r.etl_run_id) == 32  # UUID4 hex


class TestDimEtlVersionResolution:
    def test_first_use_inserts_version_row(self, conn):
        r = EtlRun.start(conn, source_path="x")
        rows = conn.execute(
            "SELECT ccutils_version, business_rules_version FROM dim_etl_version"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == PARSER_VERSION
        assert rows[0][1] == "1"

    def test_repeated_use_returns_same_version_key(self, conn):
        r1 = EtlRun.start(conn, source_path="x")
        r2 = EtlRun.start(conn, source_path="x")
        assert r1.version_key == r2.version_key
        # And dim_etl_version still has only one row
        n = conn.execute("SELECT COUNT(*) FROM dim_etl_version").fetchone()[0]
        assert n == 1

    def test_business_rule_bump_inserts_new_version(self, conn):
        r1 = EtlRun.start(conn, source_path="x", business_rules_version="1")
        r2 = EtlRun.start(conn, source_path="x", business_rules_version="2")
        assert r1.version_key != r2.version_key
        n = conn.execute("SELECT COUNT(*) FROM dim_etl_version").fetchone()[0]
        assert n == 2


class TestFactEtlRunsInsert:
    def test_start_inserts_running_row(self, conn):
        r = EtlRun.start(conn, source_path="archive/x")
        row = conn.execute(
            "SELECT etl_run_id, status, source_path, version_key FROM fact_etl_runs"
        ).fetchone()
        assert row is not None
        assert row[0] == r.etl_run_id
        assert row[1] == "running"
        assert row[2] == "archive/x"
        assert row[3] == r.version_key

    def test_complete_marks_success(self, conn):
        r = EtlRun.start(conn, source_path="x")
        r.complete(sessions_inserted=5)
        row = conn.execute(
            "SELECT status, completed_at, sessions_inserted, facts_inserted "
            "FROM fact_etl_runs WHERE etl_run_id = ?",
            [r.etl_run_id],
        ).fetchone()
        assert row[0] == "success"
        assert row[1] is not None  # completed_at populated
        assert row[2] == 5
        assert row[3] == 0  # derived from fact_etl_steps; no steps ran

    def test_complete_derives_fact_counts_from_steps(self, conn):
        r = EtlRun.start(conn, source_path="x")
        with r.step("upsert:fact_demo") as st:
            st.rows_inserted = 7
            st.rows_updated = 2
        r.complete(sessions_inserted=1)
        row = conn.execute(
            "SELECT facts_inserted, facts_updated FROM fact_etl_runs "
            "WHERE etl_run_id = ?",
            [r.etl_run_id],
        ).fetchone()
        assert row == (7, 2)

    def test_fail_marks_failed_with_error(self, conn):
        r = EtlRun.start(conn, source_path="x")
        r.fail("disk full")
        row = conn.execute(
            "SELECT status, error_message FROM fact_etl_runs WHERE etl_run_id = ?",
            [r.etl_run_id],
        ).fetchone()
        assert row[0] == "failed"
        assert row[1] == "disk full"


class TestHashDiff:
    def test_same_inputs_produce_same_hash(self):
        assert hash_diff(a=1, b="x") == hash_diff(a=1, b="x")

    def test_key_order_does_not_matter(self):
        assert hash_diff(a=1, b="x") == hash_diff(b="x", a=1)

    def test_different_values_change_hash(self):
        assert hash_diff(a=1) != hash_diff(a=2)

    def test_none_attribute_is_skipped(self):
        # Skipping NULL is the convention used elsewhere in the project so
        # adding a new optional column doesn't change every existing row's
        # hash_diff (which would cause spurious UPDATEs on re-ETL).
        assert hash_diff(a=1) == hash_diff(a=1, b=None)

    def test_returns_32_char_hex(self):
        h = hash_diff(a=1)
        assert isinstance(h, str)
        assert len(h) == 32

    def test_empty_attributes_returns_known_hash(self):
        # All-None attributes (newly-added optional columns) should produce
        # a stable hash, not raise.
        assert hash_diff() == hash_diff()


class TestRecordSourceLabel:
    def test_known_sources(self):
        assert record_source_label("claude_code_jsonl") == "claude_code_jsonl"

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError):
            record_source_label("malformed source")
