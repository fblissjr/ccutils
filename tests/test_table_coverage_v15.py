"""Declared coverage matches the DDL, and what is declared populated is
actually written.

Claim: `TABLE_COVERAGE` is the declared half of the coverage layer and
`etl.steps.table_name` the measured half. The first test pins the dict to
the DDL so a table cannot ship undeclared or linger after deletion; the
last pins every per-session table to a step that names it, so "declared
populated but nothing writes it" is a red test here before it is an audit
finding on a real warehouse.

Delete these and the stub list is back to being prose nobody checks.
"""

import pytest

from ccutils import create_star_schema
from ccutils.etl.orchestrator import run_v15_etl
from ccutils.schemas.star.schema import TABLE_COVERAGE
from helpers_ccutils import write_minimal_session


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


def _ddl_objects(conn):
    return {
        (name, "table" if kind == "BASE TABLE" else "view")
        for name, kind in conn.execute(
            "SELECT table_name, table_type FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
    }


class TestDeclaredCoverageMatchesDdl:
    def test_every_object_in_main_is_declared_and_nothing_else(self, conn):
        declared = {(n, spec[0]) for n, spec in TABLE_COVERAGE.items()}
        actual = _ddl_objects(conn)
        assert declared == actual, (
            f"undeclared: {actual - declared}; declared but absent: {declared - actual}"
        )

    def test_statuses_are_from_the_closed_set(self):
        for name, (kind, status, by, reason) in TABLE_COVERAGE.items():
            if kind == "table":
                assert status in ("populated", "conditional"), name
                assert by, name
            else:
                assert status in ("keep", "delete"), name
                assert by is None, name
            assert reason, name

    def test_seeded_into_the_warehouse(self, conn):
        rows = conn.execute(
            "SELECT object_name, object_type, status, populated_by, reason "
            "FROM etl.table_coverage"
        ).fetchall()
        assert {r[0]: r[1:] for r in rows} == TABLE_COVERAGE

    def test_reseeded_on_every_open_not_appended(self, tmp_path):
        db = tmp_path / "w.duckdb"
        create_star_schema(db).close()
        conn = create_star_schema(db)
        n = conn.execute("SELECT COUNT(*) FROM etl.table_coverage").fetchone()[0]
        assert n == len(TABLE_COVERAGE)


class TestDeclaredPopulatedIsWritten:
    def test_every_per_session_table_has_a_step_naming_it(self, conn, tmp_path):
        run_v15_etl(
            conn, write_minimal_session(tmp_path / "s.jsonl", "cov-s"),
            project_name="test-project", parquet_lake_root=tmp_path / "lake",
        )
        written = {
            r[0] for r in conn.execute(
                "SELECT DISTINCT table_name FROM etl.steps WHERE status = 'success'"
            ).fetchall()
        }
        declared = {
            n for n, (kind, status, by, _) in TABLE_COVERAGE.items()
            if kind == "table" and status == "populated" and by == "session"
        }
        assert declared <= written, f"declared populated per session, no step writes them: {declared - written}"
