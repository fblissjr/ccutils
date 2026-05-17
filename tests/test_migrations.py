"""Tests for the migration runner (Phase B chunk 3).

The runner reads applied migration ids from meta_schema_version, applies
pending migrations in id order, and records each as it succeeds.
Idempotent: re-running with no new migrations is a no-op.
"""

import pytest

from ccutils import create_star_schema
from ccutils.schemas.migrations import (
    Migration,
    all_migrations,
    applied_migration_ids,
    apply_pending_migrations,
)


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


class TestAppliedMigrationIds:
    def test_empty_when_no_migrations_applied(self, conn):
        assert applied_migration_ids(conn) == set()

    def test_returns_applied_set(self, conn):
        conn.execute(
            "INSERT INTO meta_schema_version (migration_id, description) "
            "VALUES ('20260419_0001_initial', 'test')"
        )
        assert applied_migration_ids(conn) == {"20260419_0001_initial"}


class TestApplyPendingMigrations:
    def test_no_migrations_no_op(self, conn):
        applied = apply_pending_migrations(conn, migrations=[])
        assert applied == []
        n = conn.execute("SELECT COUNT(*) FROM meta_schema_version").fetchone()[0]
        assert n == 0

    def test_applies_single_pending(self, conn):
        m = Migration(
            id="20260419_0001_test",
            description="Create test table",
            up=lambda c: c.execute("CREATE TABLE _m_test (x INTEGER)"),
        )
        applied = apply_pending_migrations(conn, migrations=[m])
        assert applied == ["20260419_0001_test"]
        assert applied_migration_ids(conn) == {"20260419_0001_test"}
        # The migration's up() actually ran:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name='_m_test'"
        ).fetchone() is not None

    def test_records_ccutils_version(self, conn):
        m = Migration(id="20260419_0002_x", description="x", up=lambda c: None)
        apply_pending_migrations(conn, migrations=[m])
        row = conn.execute(
            "SELECT description, ccutils_version FROM meta_schema_version "
            "WHERE migration_id = '20260419_0002_x'"
        ).fetchone()
        assert row[0] == "x"
        assert row[1] is not None

    def test_skips_already_applied(self, conn):
        m = Migration(id="20260419_0003_once", description="once", up=lambda c: None)
        apply_pending_migrations(conn, migrations=[m])
        # Apply again -- should be a no-op
        applied = apply_pending_migrations(conn, migrations=[m])
        assert applied == []

    def test_applies_in_id_order(self, conn):
        executed: list[str] = []
        ma = Migration(id="20260419_0002_a", description="a",
                       up=lambda c: executed.append("a"))
        mb = Migration(id="20260419_0001_b", description="b",
                       up=lambda c: executed.append("b"))
        apply_pending_migrations(conn, migrations=[ma, mb])
        # Order should be 0001 before 0002, regardless of input order
        assert executed == ["b", "a"]

    def test_applies_only_new_migrations_on_second_call(self, conn):
        m1 = Migration(id="20260419_0001", description="first", up=lambda c: None)
        m2 = Migration(id="20260419_0002", description="second", up=lambda c: None)
        apply_pending_migrations(conn, migrations=[m1])
        applied2 = apply_pending_migrations(conn, migrations=[m1, m2])
        assert applied2 == ["20260419_0002"]
        assert applied_migration_ids(conn) == {"20260419_0001", "20260419_0002"}

    def test_failure_does_not_record_migration(self, conn):
        def boom(c):
            raise RuntimeError("oops")

        m = Migration(id="20260419_0004_fail", description="fail", up=boom)
        with pytest.raises(RuntimeError, match="oops"):
            apply_pending_migrations(conn, migrations=[m])
        assert applied_migration_ids(conn) == set()


class TestAllMigrationsDiscovery:
    def test_discovers_baseline_migration(self):
        discovered = all_migrations()
        ids = {m.id for m in discovered}
        assert "20260419_0001_initial" in ids

    def test_apply_pending_to_fresh_db_runs_baseline(self, conn):
        applied = apply_pending_migrations(conn, all_migrations())
        assert "20260419_0001_initial" in applied
