"""DDL tests for v0.15 lineage + meta tables (Phase B chunk 1)."""

import pytest

from ccutils import create_star_schema


@pytest.fixture
def conn(tmp_path):
    db = tmp_path / "test.duckdb"
    return create_star_schema(db)


class TestDimEtlVersion:
    def test_table_exists(self, conn):
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_etl_version'"
        ).fetchone()
        assert result is not None

    def test_columns(self, conn):
        cols = [c[0] for c in conn.execute("DESCRIBE dim_etl_version").fetchall()]
        for col in ("version_key", "ccutils_version", "business_rules_version", "description", "first_seen_at"):
            assert col in cols, f"Missing column: {col}"


class TestFactEtlRuns:
    def test_table_exists(self, conn):
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_etl_runs'"
        ).fetchone()
        assert result is not None

    def test_columns(self, conn):
        cols = [c[0] for c in conn.execute("DESCRIBE fact_etl_runs").fetchall()]
        for col in (
            "etl_run_id", "version_key", "started_at", "completed_at",
            "status", "source_path",
            "sessions_seen", "sessions_inserted", "sessions_updated",
            "sessions_unchanged", "sessions_soft_deleted",
            "facts_inserted", "facts_updated",
            "error_message",
        ):
            assert col in cols, f"Missing column: {col}"

    def test_default_status_is_running(self, conn):
        conn.execute("INSERT INTO fact_etl_runs (etl_run_id) VALUES ('test-run-1')")
        status = conn.execute("SELECT status FROM fact_etl_runs WHERE etl_run_id='test-run-1'").fetchone()[0]
        assert status == "running"


class TestMetaSchemaVersion:
    def test_table_exists(self, conn):
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta_schema_version'"
        ).fetchone()
        assert result is not None

    def test_columns(self, conn):
        cols = [c[0] for c in conn.execute("DESCRIBE meta_schema_version").fetchall()]
        for col in ("migration_id", "applied_at", "description", "ccutils_version"):
            assert col in cols, f"Missing column: {col}"

    def test_distinct_from_dim_etl_version(self, conn):
        """meta_schema_version (DDL migrations) and dim_etl_version (ETL
        business-rules versioning) are separate concerns -- both must exist."""
        ms = conn.execute("SELECT name FROM sqlite_master WHERE name='meta_schema_version'").fetchone()
        dv = conn.execute("SELECT name FROM sqlite_master WHERE name='dim_etl_version'").fetchone()
        assert ms is not None and dv is not None
