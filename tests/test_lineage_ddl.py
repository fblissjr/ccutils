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
            "SELECT table_name FROM information_schema.tables WHERE table_schema='etl' AND table_name='versions'"
        ).fetchone()
        assert result is not None

    def test_columns(self, conn):
        cols = [c[0] for c in conn.execute("DESCRIBE etl.versions").fetchall()]
        for col in ("version_key", "ccutils_version", "business_rules_version", "description", "first_seen_at"):
            assert col in cols, f"Missing column: {col}"


class TestFactEtlRuns:
    def test_table_exists(self, conn):
        result = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='etl' AND table_name='runs'"
        ).fetchone()
        assert result is not None

    def test_columns(self, conn):
        cols = [c[0] for c in conn.execute("DESCRIBE etl.runs").fetchall()]
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
        conn.execute("INSERT INTO etl.runs (etl_run_id) VALUES ('test-run-1')")
        status = conn.execute("SELECT status FROM etl.runs WHERE etl_run_id='test-run-1'").fetchone()[0]
        assert status == "running"


class TestMetaSchemaVersion:
    def test_table_exists(self, conn):
        result = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='etl' AND table_name='schema_version'"
        ).fetchone()
        assert result is not None

    def test_columns(self, conn):
        cols = [c[0] for c in conn.execute("DESCRIBE etl.schema_version").fetchall()]
        assert set(cols) == {"schema_fingerprint", "ccutils_version", "created_at"}

    def test_stamp_matches_the_live_schema(self, conn):
        """The stamp is what create_star_schema compares against on open.
        A stamp that disagrees with the file it sits in is a refusal
        waiting to happen on the next open."""
        from ccutils.schemas.star.schema import schema_fingerprint

        stored = conn.execute(
            "SELECT schema_fingerprint FROM etl.schema_version"
        ).fetchone()[0]
        assert stored == schema_fingerprint(conn)

    def test_distinct_from_etl_versions(self, conn):
        """etl.schema_version (which DDL wrote this file) and
        etl.versions (which business rules wrote each row) are separate
        concerns -- both must exist."""
        ms = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='etl' AND table_name='schema_version'").fetchone()
        dv = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='etl' AND table_name='versions'").fetchone()
        assert ms is not None and dv is not None
