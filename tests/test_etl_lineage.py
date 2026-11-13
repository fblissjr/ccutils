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


class TestDimEtlVersionResolution:

    def test_repeated_use_returns_same_version_key(self, conn):
        r1 = EtlRun.start(conn, source_path="x")
        r2 = EtlRun.start(conn, source_path="x")
        assert r1.version_key == r2.version_key
        # And etl.versions still has only one row
        n = conn.execute("SELECT COUNT(*) FROM etl.versions").fetchone()[0]
        assert n == 1

    def test_business_rule_bump_inserts_new_version(self, conn):
        r1 = EtlRun.start(conn, source_path="x", business_rules_version="1")
        r2 = EtlRun.start(conn, source_path="x", business_rules_version="2")
        assert r1.version_key != r2.version_key
        n = conn.execute("SELECT COUNT(*) FROM etl.versions").fetchone()[0]
        assert n == 2


class TestFactEtlRunsInsert:
    def test_start_inserts_running_row(self, conn):
        r = EtlRun.start(conn, source_path="archive/x")
        row = conn.execute(
            "SELECT etl_run_id, status, source_path, version_key FROM etl.runs"
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
            "FROM etl.runs WHERE etl_run_id = ?",
            [r.etl_run_id],
        ).fetchone()
        assert row[0] == "success"
        assert row[1] is not None  # completed_at populated
        assert row[2] == 5
        assert row[3] == 0  # derived from etl.steps; no steps ran

    def test_complete_derives_fact_counts_from_steps(self, conn):
        r = EtlRun.start(conn, source_path="x")
        with r.step("upsert:fact_demo", kind="upsert") as st:
            st.rows_inserted = 7
            st.rows_updated = 2
        r.complete(sessions_inserted=1)
        row = conn.execute(
            "SELECT facts_inserted, facts_updated FROM etl.runs "
            "WHERE etl_run_id = ?",
            [r.etl_run_id],
        ).fetchone()
        assert row == (7, 2)

    def test_fail_marks_failed_with_error(self, conn):
        r = EtlRun.start(conn, source_path="x")
        r.fail("disk full")
        row = conn.execute(
            "SELECT status, error_message FROM etl.runs WHERE etl_run_id = ?",
            [r.etl_run_id],
        ).fetchone()
        assert row[0] == "failed"
        assert row[1] == "disk full"


class TestHashDiff:

    def test_key_order_does_not_matter(self):
        assert hash_diff(a=1, b="x") == hash_diff(b="x", a=1)

    def test_different_values_change_hash(self):
        assert hash_diff(a=1) != hash_diff(a=2)

    def test_none_attribute_is_skipped(self):
        # Skipping NULL is the convention used elsewhere in the project so
        # adding a new optional column doesn't change every existing row's
        # hash_diff (which would cause spurious UPDATEs on re-ETL).
        assert hash_diff(a=1) == hash_diff(a=1, b=None)


    def test_empty_attributes_returns_known_hash(self):
        # All-None attributes (newly-added optional columns) should produce
        # a stable hash, not raise.
        assert hash_diff() == hash_diff()


