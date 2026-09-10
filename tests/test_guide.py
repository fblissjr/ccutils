"""The reader's guide: a warehouse explains itself to whoever opens it cold.

Claim: the primary consumer of a ccutils warehouse is an agent, not a
person at a SQL prompt. Two external auditors reported empty tables as ETL
defects because the only explanation lived in a skill file they never saw.
The guide is generated FROM the warehouse (`etl.table_coverage`, `etl.steps`,
row counts, `etl.audit_exceptions`, the FK convention) so it describes what
this file holds, not what the code could hold, and it is written beside the
archive every time a warehouse is built.

Delete these and the guide can drift from the warehouse it sits next to,
which is the failure the stub list already had once.
"""

import duckdb
import pytest
from click.testing import CliRunner

from ccutils import cli, create_star_schema
from ccutils.etl.orchestrator import run_v15_etl
from ccutils.guide import render_guide
from ccutils.schemas.star.schema import TABLE_COVERAGE
from helpers_ccutils import write_minimal_session

FIXTURE_FILE = "sample_session.jsonl"


@pytest.fixture
def warehouse(tmp_path):
    db = tmp_path / "wh" / "archive.duckdb"
    db.parent.mkdir()
    conn = create_star_schema(db)
    run_v15_etl(
        conn, write_minimal_session(tmp_path / "s.jsonl", "guide-s"),
        project_name="guide-project", parquet_lake_root=tmp_path / "lake",
    )
    conn.execute(
        "INSERT INTO etl.audit_exceptions VALUES "
        "('null_column', 'fact_meta_events', 'timestamp', 'no source timestamp')"
    )
    conn.close()
    return db


class TestGuideDescribesThisWarehouse:
    def test_every_object_appears_with_its_reason(self, warehouse):
        conn = duckdb.connect(str(warehouse), read_only=True)
        text = render_guide(conn)
        conn.close()
        for name, (kind, status, by, reason) in TABLE_COVERAGE.items():
            assert name in text, name
            assert reason.split(" (")[0][:40] in text, name

    def test_row_counts_and_scope_are_measured_not_declared(self, warehouse):
        conn = duckdb.connect(str(warehouse), read_only=True)
        text = render_guide(conn)
        project = conn.execute("SELECT project_name FROM dim_project").fetchone()[0]
        conn.close()
        assert project in text
        assert "| fact_messages |" in text
        # one session, two entries: the count line must carry a real number
        assert "| dim_session | populated | 1 |" in text

    def test_accepted_findings_are_listed(self, warehouse):
        conn = duckdb.connect(str(warehouse), read_only=True)
        text = render_guide(conn)
        conn.close()
        assert "fact_meta_events.timestamp" in text
        assert "no source timestamp" in text

    def test_join_paths_are_listed(self, warehouse):
        conn = duckdb.connect(str(warehouse), read_only=True)
        text = render_guide(conn)
        conn.close()
        assert "fact_messages.session_key" in text and "dim_session.session_key" in text


class TestGuideIsWrittenBesideTheArchive:
    def test_duckdb_build_writes_the_guide(self, tmp_path):
        from pathlib import Path

        fixture = Path(__file__).parent / FIXTURE_FILE
        out = tmp_path / "out"
        result = CliRunner().invoke(cli, [str(fixture), "--format", "duckdb", "-o", str(out)])
        assert result.exit_code == 0, result.output
        guide = out / "README.md"
        assert guide.exists()
        text = guide.read_text()
        assert "archive.duckdb" in text
        assert "etl.table_coverage" in text

    def test_guide_command_regenerates_it(self, warehouse):
        result = CliRunner().invoke(cli, ["guide", "-o", str(warehouse.parent)])
        assert result.exit_code == 0, result.output
        assert (warehouse.parent / "README.md").exists()
