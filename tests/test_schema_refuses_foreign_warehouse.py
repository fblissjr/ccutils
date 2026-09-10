"""`create_star_schema` refuses a warehouse whose schema it did not write.

Claim: 1.0.0 deleted the migration machinery (`_COLUMN_MIGRATIONS`, its
backfills, `_repair_duplicate_natural_keys`, the `schemas/migrations`
runner). Nothing heals an old warehouse any more, so the only honest thing
to do with one is refuse it and say "rebuild". Silently opening it would
reinstate the exact bug class the deletion removed: a populator INSERTing
into a column that is not there, a predicate NULL-blind over rows written
before a column existed, a view failing to bind.

The check compares the shape of every base table in the file against the
shape this code's DDL produces, so a warehouse built by an older ccutils, a
newer ccutils, or a hand ALTER all read as foreign. Delete these and an
old warehouse opens, the CREATEs no-op, and the first populator crashes on a
missing column -- or worse, does not.
"""

from pathlib import Path

import duckdb
import pytest
from click.testing import CliRunner

from ccutils import cli, create_star_schema
from ccutils.schemas.star.schema import SchemaMismatchError

FIXTURE = Path(__file__).parent / "sample_session.jsonl"


def _build(path):
    create_star_schema(path).close()


class TestOwnWarehouseOpens:
    def test_fresh_build_reopens(self, tmp_path):
        db = tmp_path / "w.duckdb"
        _build(db)
        conn = create_star_schema(db)
        assert conn.execute("SELECT COUNT(*) FROM dim_session").fetchone()[0] == 0
        conn.close()

    def test_in_memory_works(self):
        create_star_schema(":memory:").close()

    def test_stamp_is_written_once(self, tmp_path):
        """The stamp records who wrote the schema. One row, not one per open."""
        db = tmp_path / "w.duckdb"
        _build(db)
        _build(db)
        conn = duckdb.connect(str(db))
        rows = conn.execute(
            "SELECT schema_fingerprint, ccutils_version FROM meta_schema_version"
        ).fetchall()
        conn.close()
        assert len(rows) == 1
        fingerprint, version = rows[0]
        assert fingerprint and version


class TestForeignWarehouseIsRefused:
    def _reopen_expecting_refusal(self, db):
        with pytest.raises(SchemaMismatchError) as exc:
            create_star_schema(db)
        assert "rebuild" in str(exc.value).lower()

    def test_extra_column_is_refused(self, tmp_path):
        db = tmp_path / "w.duckdb"
        _build(db)
        conn = duckdb.connect(str(db))
        conn.execute("ALTER TABLE fact_errors ADD COLUMN stray VARCHAR")
        conn.close()
        self._reopen_expecting_refusal(db)

    def test_missing_column_is_refused(self, tmp_path):
        """The shape a pre-1.0 warehouse takes once this code adds a column
        to a CREATE: the file lacks it, and nothing will ever add it."""
        db = tmp_path / "w.duckdb"
        _build(db)
        conn = duckdb.connect(str(db))
        conn.execute("ALTER TABLE fact_errors DROP COLUMN error_type")
        conn.close()
        self._reopen_expecting_refusal(db)

    def test_pre_1_0_warehouse_is_refused(self, tmp_path):
        """A warehouse from before the stamp existed carries the old
        migration-ledger `meta_schema_version`, not a fingerprint."""
        db = tmp_path / "old.duckdb"
        conn = duckdb.connect(str(db))
        conn.execute(
            "CREATE TABLE meta_schema_version (migration_id VARCHAR, "
            "applied_at TIMESTAMP, description VARCHAR, ccutils_version VARCHAR)"
        )
        conn.execute("CREATE TABLE dim_session (session_key VARCHAR)")
        conn.close()
        self._reopen_expecting_refusal(db)

    def test_refusal_names_the_writer_when_known(self, tmp_path):
        """The message should say which ccutils wrote the file, so the user
        knows it is an old build and not corruption."""
        db = tmp_path / "w.duckdb"
        _build(db)
        conn = duckdb.connect(str(db))
        conn.execute("UPDATE meta_schema_version SET ccutils_version = '0.99.0'")
        conn.execute("ALTER TABLE fact_errors ADD COLUMN stray VARCHAR")
        conn.close()
        with pytest.raises(SchemaMismatchError) as exc:
            create_star_schema(db)
        assert "0.99.0" in str(exc.value)

    def test_nothing_is_written_before_refusing(self, tmp_path):
        """Refusal must not half-create objects in the foreign file."""
        db = tmp_path / "old.duckdb"
        conn = duckdb.connect(str(db))
        conn.execute("CREATE TABLE dim_session (session_key VARCHAR)")
        conn.close()
        self._reopen_expecting_refusal(db)
        conn = duckdb.connect(str(db))
        tables = {
            r[0] for r in conn.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
        conn.close()
        assert tables == {"dim_session"}


class TestCliSurfacesRefusal:
    def test_duckdb_export_into_a_foreign_warehouse_fails_cleanly(self, tmp_path):
        """Exit nonzero with the rebuild message, not a traceback."""
        out = tmp_path / "out"
        out.mkdir()
        db = out / "archive.duckdb"
        _build(db)
        conn = duckdb.connect(str(db))
        conn.execute("ALTER TABLE fact_errors ADD COLUMN stray VARCHAR")
        conn.close()

        result = CliRunner().invoke(
            cli, [str(FIXTURE), "--format", "duckdb", "-o", str(out)]
        )
        assert result.exit_code != 0
        assert "rebuild" in result.output.lower()
        assert "Traceback" not in result.output
