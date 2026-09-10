"""Every step that writes a table names the table it wrote.

Claim: `etl.steps` is the measured half of the coverage layer. "Which
tables has this warehouse ever written" must be a GROUP BY over
`table_name`, not a parse of `step_name`. Until now table identity was
encoded inside the string `upsert:fact_messages` for facts and not at all
for dimensions (`upsert_dimensions` wrote four tables under one label), so
the audit could not tell a dimension nobody writes from one written by a
step with a vague name.

Delete these and a new step can ship unlabelled, and the audit's
"declared populated but never written" check goes blind for that table.
"""

import pytest

from ccutils import create_star_schema
from ccutils.etl.orchestrator import run_v15_etl
from helpers_ccutils import write_minimal_session


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


@pytest.fixture
def basic_session(tmp_path):
    return write_minimal_session(tmp_path / "basic.jsonl", "basic-s")


@pytest.fixture
def steps(conn, basic_session, tmp_path):
    run_v15_etl(
        conn, basic_session, project_name="test-project",
        parquet_lake_root=tmp_path / "lake",
    )
    return conn.execute(
        "SELECT step_name, step_kind, table_name FROM etl.steps ORDER BY step_order"
    ).fetchall()


class TestStepsNameTheirTable:
    def test_column_exists(self, conn):
        cols = {r[0] for r in conn.execute("DESCRIBE etl.steps").fetchall()}
        assert "table_name" in cols

    def test_every_writing_step_names_a_table(self, steps):
        unlabelled = [
            name for name, kind, table in steps
            if table is None and name != "write_parquet"
        ]
        assert unlabelled == [], f"steps with no table_name: {unlabelled}"

    def test_upsert_steps_agree_with_their_name(self, steps):
        for name, kind, table in steps:
            if kind == "upsert":
                assert name == f"upsert:{table}", (name, table)

    def test_dimensions_are_one_step_each(self, steps):
        """`upsert_dimensions` used to be one step over four tables."""
        names = {name for name, _, _ in steps}
        assert "upsert_dimensions" not in names
        stage_tables = {table for _, kind, table in steps if kind == "stage"}
        assert {
            "dim_session", "dim_project", "dim_model", "dim_tool", "dim_date",
            "etl.log_entries", "dim_session_chain",
        } <= stage_tables

    def test_table_written_is_a_group_by(self, conn, steps):
        """The whole point: one query answers what this run wrote."""
        written = {
            r[0] for r in conn.execute(
                "SELECT DISTINCT table_name FROM etl.steps "
                "WHERE table_name IS NOT NULL AND status = 'success'"
            ).fetchall()
        }
        assert "fact_messages" in written
        assert "dim_session" in written
