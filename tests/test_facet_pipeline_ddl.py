"""`dim_facet_type` keeps the history of facet prompt versions across reopens.

`dim_facet_type` is created with `CREATE TABLE IF NOT EXISTS` and seeded
with `INSERT ... ON CONFLICT DO NOTHING`, so a reopen keeps the rows of
earlier prompt versions instead of reseeding them away (CLAUDE.md). The
registry's columns and seeds were tested by copying the DDL; those tests
were removed on 2026-11-13.
"""

import pytest

from ccutils import create_star_schema


_LINEAGE_COLS = (
    "created_at",
    "created_by_version_key",
    "last_updated_at",
    "last_updated_by_version_key",
    "etl_run_id",
    "record_source",
    "hash_diff",
    "is_deleted",
    "deleted_at",
)

# Tier 1 (method='computed') from FACET_CLUSTER_PIPELINE.md §3. F20 is the
# Tier 2 LLM facet, so the Tier 1 range is NOT contiguous -- F01-F19 plus the
# behavioral pair F30/F31.
_TIER1_FACET_IDS = tuple(f"F{i:02d}" for i in range(1, 20)) + ("F30", "F31")


@pytest.fixture
def conn(tmp_path):
    db = tmp_path / "test.duckdb"
    return create_star_schema(db)


class TestDimFacetType:


    def test_historical_prompt_versions_survive_rerun(self, tmp_path):
        # ON CONFLICT DO NOTHING: when create_star_schema() is called on an
        # existing DB (the normal CLI path), a prompt_version row added
        # post-seed must NOT be wiped. Otherwise re-running the CLI would
        # destroy the historical registry that fact_session_facets rows
        # already reference by facet_type_key.
        #
        # Uses F90 (hypothetical facet) per the F90+ convention for test
        # fixtures -- see `tests/test_fact_session_facets_v15.py::
        # two_f90_versions` for the canonical convention statement.
        from ccutils import create_star_schema

        db_path = tmp_path / "rerun.duckdb"
        conn = create_star_schema(db_path)
        conn.execute(
            """
            INSERT INTO dim_facet_type
                (facet_type_key, facet_id, facet_name, tier, method,
                 output_type, prompt_text, prompt_version)
            VALUES (md5('F90' || '|' || 'v1'), 'F90', 'hypothetical',
                    2, 'llm', 'text', 'prompt v1', 'v1')
            """
        )
        conn.close()

        # Reopen via create_star_schema as the CLI does.
        conn2 = create_star_schema(db_path)
        survivors = conn2.execute(
            "SELECT facet_id, prompt_version FROM dim_facet_type "
            "WHERE facet_id = 'F90'"
        ).fetchall()
        assert survivors == [("F90", "v1")], (
            f"F90 v1 should survive re-create_star_schema, got {survivors}"
        )
        # Tier 1 seeds should still be all there.
        n_tier1 = conn2.execute(
            "SELECT COUNT(*) FROM dim_facet_type WHERE tier = 1"
        ).fetchone()[0]
        assert n_tier1 == len(_TIER1_FACET_IDS)


