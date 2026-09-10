"""Run metadata and staging live in an `etl` schema; the version key is a
string a reader can read.

Claim (1.0.0 step 5): the main schema holds only what a consumer should
query -- dims, facts, semantic views. Machinery (run metadata, staging, the
DDL stamp) sits in `etl.*` with no `dim_` / `fact_` / `stg_` / `meta_`
prefix, so an agent opening the file cold sees the signal and nothing else.
The lineage stamp on every row is `<ccutils_version>/<business_rules>`
rather than an md5 that has to be joined before it says anything.

Delete these and the old names can drift back one table at a time, and a
row's provenance goes back to being a hash.
"""

import json

import duckdb
import pytest

from ccutils import create_star_schema
from ccutils._version import PARSER_VERSION
from ccutils.etl.lineage import BatchRun, EtlRun
from ccutils.schemas.star.json_export import export_star_schema_to_json
from ccutils.schemas.star.schema import SchemaMismatchError

ETL_TABLES = {
    "runs", "batch_runs", "steps", "versions", "schema_version", "log_entries",
}
OLD_NAMES = {
    "fact_etl_runs", "fact_etl_batch_runs", "fact_etl_steps",
    "dim_etl_version", "meta_schema_version", "stg_log_entries",
}


def _tables(conn, schema):
    return {
        r[0] for r in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = ? AND table_type = 'BASE TABLE'", [schema]
        ).fetchall()
    }


@pytest.fixture
def conn():
    c = create_star_schema(":memory:")
    yield c
    c.close()


class TestEtlSchemaHoldsTheMachinery:
    def test_etl_schema_has_exactly_the_machinery(self, conn):
        assert _tables(conn, "etl") == ETL_TABLES

    def test_main_schema_has_no_machinery(self, conn):
        main = _tables(conn, "main")
        assert not (main & OLD_NAMES), main & OLD_NAMES
        assert not {t for t in main if t.startswith(("stg_", "meta_"))}
        assert not {t for t in main if t.endswith("_etl_version")}

    def test_semantic_etl_runs_still_reads_from_main(self, conn):
        """The observability view is a consumer surface, so it stays in
        main and reads the machinery underneath."""
        run = EtlRun.start(conn, source_path="/x")
        with run.step("load_staging") as st:
            st.rows_inserted = 1
        run.complete()
        row = conn.execute(
            "SELECT status FROM semantic_etl_runs WHERE etl_run_id = ?",
            [run.etl_run_id],
        ).fetchone()
        assert row == ("success",)


class TestVersionKeyIsReadable:
    def test_run_stamps_a_plain_version_string(self, conn):
        run = EtlRun.start(conn, source_path="/x")
        assert run.version_key == f"{PARSER_VERSION}/1"
        stored = conn.execute(
            "SELECT version_key, ccutils_version, business_rules_version "
            "FROM etl.versions"
        ).fetchall()
        assert stored == [(f"{PARSER_VERSION}/1", PARSER_VERSION, "1")]

    def test_batch_uses_the_same_key(self, conn):
        with BatchRun.start(conn, source_root="/s", output_format="duckdb") as b:
            b.complete()
        assert b.version_key == f"{PARSER_VERSION}/1"

    def test_business_rules_version_is_part_of_the_key(self, conn):
        r1 = EtlRun.start(conn, source_path="/x", business_rules_version="1")
        r2 = EtlRun.start(conn, source_path="/x", business_rules_version="2")
        assert r1.version_key == f"{PARSER_VERSION}/1"
        assert r2.version_key == f"{PARSER_VERSION}/2"


class TestFingerprintCoversEtlSchema:
    def test_stray_column_on_an_etl_table_is_refused(self, tmp_path):
        db = tmp_path / "w.duckdb"
        create_star_schema(db).close()
        c = duckdb.connect(str(db))
        c.execute("ALTER TABLE etl.runs ADD COLUMN stray VARCHAR")
        c.close()
        with pytest.raises(SchemaMismatchError):
            create_star_schema(db)

    def test_stamp_lives_in_etl_schema(self, tmp_path):
        db = tmp_path / "w.duckdb"
        create_star_schema(db).close()
        c = duckdb.connect(str(db))
        n = c.execute("SELECT COUNT(*) FROM etl.schema_version").fetchone()[0]
        c.close()
        assert n == 1


class TestJsonExportMirrorsTheSchemas:
    def test_etl_tables_export_under_their_own_directory(self, conn, tmp_path):
        run = EtlRun.start(conn, source_path="/x")
        run.complete()
        export_star_schema_to_json(conn, tmp_path)
        assert (tmp_path / "etl" / "runs.json").exists()
        assert not (tmp_path / "facts" / "fact_etl_runs.json").exists()
        rows = json.loads((tmp_path / "etl" / "runs.json").read_text())
        assert rows[0]["etl_run_id"] == run.etl_run_id
        meta = json.loads((tmp_path / "meta.json").read_text())
        # Staging is scratch: never in an export.
        assert {t["name"] for t in meta["tables"]["etl"]} == ETL_TABLES - {"log_entries"}
        assert not (tmp_path / "etl" / "log_entries.json").exists()
