"""DDL tests for the facet & cluster pipeline (step 1 of build order).

Three new tables land in `create_star_schema()`:

    dim_facet_type        - registry of facet definitions; seeded with the
                            19 Tier 1 facets (F01-F19) defined in
                            docs/FACET_CLUSTER_PIPELINE.md §3.
    fact_session_facets   - one row per (session, facet_type, prompt_version).
                            Structured values only (text/json/numeric/bool);
                            NO embedding column.
    fact_facet_embeddings - one row per (session, facet_type, embedding_model,
                            embedding_model_version). Stores the vector as
                            FLOAT[384] so DuckDB's native array_cosine_similarity
                            works without a vector DB.

The schema split (embeddings live in their own table, not as a BLOB column on
fact_session_facets) is a deliberate departure from the original design doc.
Reasons: keep the EAV facet table lean for SQL scans, let DuckDB array ops
work natively, absorb future model-version additions as new rows rather than
destructive overwrites of the structured-value table.
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

# F01-F19 from FACET_CLUSTER_PIPELINE.md §3 (Tier 1, method='computed').
_TIER1_FACET_IDS = tuple(f"F{i:02d}" for i in range(1, 20))


@pytest.fixture
def conn(tmp_path):
    db = tmp_path / "test.duckdb"
    return create_star_schema(db)


class TestDimFacetType:
    def test_table_exists(self, conn):
        result = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='dim_facet_type'"
        ).fetchone()
        assert result is not None

    def test_columns(self, conn):
        cols = [c[0] for c in conn.execute("DESCRIBE dim_facet_type").fetchall()]
        for col in (
            "facet_type_key",
            "facet_id",
            "facet_name",
            "tier",
            "method",
            "output_type",
            "prompt_text",
            "prompt_version",
            "embedding_model",
            "notes",  # data-level caveats (e.g. F16's UTC-hour limitation)
            "created_at",
        ):
            assert col in cols, f"Missing column: {col}"

    def test_facet_type_key_is_primary_key(self, conn):
        # Without an enforced PK, a future non-OR-REPLACE migration path could
        # silently seed duplicates. DuckDB DESCRIBE reports the PK in the 'key'
        # column.
        info = conn.execute("DESCRIBE dim_facet_type").fetchall()
        pk_cols = [row[0] for row in info if row[3] == "PRI"]
        assert pk_cols == ["facet_type_key"], (
            f"facet_type_key should be the PRIMARY KEY, got {pk_cols}"
        )

    def test_facet_with_caveat_carries_a_note(self, conn):
        # F16 (local_hour) and F08 (loc_delta) both have known data-level
        # caveats. The note must live on the row so analytical queries see it,
        # not just in source-code comments.
        rows = conn.execute(
            "SELECT facet_id, notes FROM dim_facet_type "
            "WHERE facet_id IN ('F08', 'F16')"
        ).fetchall()
        for facet_id, notes in rows:
            assert notes is not None and len(notes) > 0, (
                f"{facet_id} should carry a caveat in `notes`"
            )

    def test_historical_prompt_versions_survive_rerun(self, tmp_path):
        # ON CONFLICT DO NOTHING: when create_star_schema() is called on an
        # existing DB (the normal CLI path), a prompt_version row added
        # post-seed must NOT be wiped. Otherwise re-running the CLI would
        # destroy the historical registry that fact_session_facets rows
        # already reference by facet_type_key.
        #
        # Uses F90 (a hypothetical facet, not in the catalog) so the test
        # exercises the survival contract without colliding with the
        # already-seeded F20 v1 row.
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
        assert n_tier1 == 19

    def test_seeded_with_tier1_facets(self, conn):
        rows = conn.execute(
            "SELECT facet_id, tier, method "
            "FROM dim_facet_type WHERE tier = 1 ORDER BY facet_id"
        ).fetchall()
        ids = [r[0] for r in rows]
        assert set(ids) == set(_TIER1_FACET_IDS), (
            f"Expected Tier 1 facets F01-F19, got {ids}"
        )
        # All Tier 1 facets are SQL aggregations -- no inference, no prompt.
        for facet_id, tier, method in rows:
            assert tier == 1, f"{facet_id} wrong tier {tier}"
            assert method == "computed", f"{facet_id} wrong method {method}"

    def test_tier1_seeds_have_no_prompt(self, conn):
        # Tier 1 is purely computed; LLM prompt + version must be NULL so the
        # registry encodes "this facet is deterministic".
        rows = conn.execute(
            "SELECT facet_id, prompt_text, prompt_version "
            "FROM dim_facet_type WHERE tier = 1"
        ).fetchall()
        for facet_id, prompt_text, prompt_version in rows:
            assert prompt_text is None, f"{facet_id} should not carry a prompt"
            assert prompt_version is None, f"{facet_id} should not carry a prompt_version"

    def test_facet_type_key_is_unique(self, conn):
        # Surrogate key must be 1:1 with (facet_id, prompt_version).
        distinct, total = conn.execute(
            "SELECT COUNT(DISTINCT facet_type_key), COUNT(*) FROM dim_facet_type"
        ).fetchone()
        assert distinct == total, "facet_type_key collisions detected"


class TestFactSessionFacets:
    def test_table_exists(self, conn):
        result = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='fact_session_facets'"
        ).fetchone()
        assert result is not None

    def test_lineage_envelope(self, conn):
        cols = [c[0] for c in conn.execute("DESCRIBE fact_session_facets").fetchall()]
        for col in _LINEAGE_COLS:
            assert col in cols, f"Missing lineage column: {col}"

    def test_columns(self, conn):
        cols = [c[0] for c in conn.execute("DESCRIBE fact_session_facets").fetchall()]
        for col in (
            "facet_row_key",  # synthesized natural key for lineage_upsert
            "session_key",
            "session_id",
            "facet_type_key",
            "prompt_version",
            "value_text",
            "value_json",
            "value_numeric",
            "value_bool",
            # Step 3: Tier 2 QA aids (NULL / FALSE for Tier 1 rows)
            "is_fallback",
            "extraction_metadata_json",
            "extracted_at",
            "date_key",
            "time_key",
        ):
            assert col in cols, f"Missing column: {col}"

    def test_is_fallback_defaults_false(self, conn):
        # Tier 2 QA aid: rows written without an explicit value should
        # default to "successful extraction" (is_fallback=FALSE).
        # Tier 1 populator never sets is_fallback so it must default
        # cleanly there too.
        conn.execute(
            """
            INSERT INTO fact_session_facets (
                created_by_version_key, last_updated_by_version_key,
                etl_run_id, record_source, hash_diff,
                facet_row_key, session_key, session_id, facet_type_key
            )
            VALUES ('vk', 'vk', 'r1', 'claude_code_jsonl', 'h1',
                    'rk1', 'sk1', 's1', 'ftk1')
            """
        )
        is_fallback = conn.execute(
            "SELECT is_fallback FROM fact_session_facets WHERE facet_row_key = 'rk1'"
        ).fetchone()[0]
        assert is_fallback is False

    def test_no_embedding_column(self, conn):
        # Schema-split contract: embeddings live in fact_facet_embeddings.
        # See docs/FACET_CLUSTER_PIPELINE.md §4.
        cols = [c[0] for c in conn.execute("DESCRIBE fact_session_facets").fetchall()]
        assert "embedding" not in cols, (
            "Embeddings must live in fact_facet_embeddings, "
            "not inline on fact_session_facets"
        )


class TestFactFacetEmbeddings:
    def test_table_exists(self, conn):
        result = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='fact_facet_embeddings'"
        ).fetchone()
        assert result is not None

    def test_lineage_envelope(self, conn):
        cols = [c[0] for c in conn.execute("DESCRIBE fact_facet_embeddings").fetchall()]
        for col in _LINEAGE_COLS:
            assert col in cols, f"Missing lineage column: {col}"

    def test_columns(self, conn):
        cols = [c[0] for c in conn.execute("DESCRIBE fact_facet_embeddings").fetchall()]
        for col in (
            "embedding_row_key",  # synthesized natural key for lineage_upsert
            "session_key",
            "session_id",
            "facet_type_key",
            "embedding_model",
            "embedding_model_version",
            "embedding",
            "embedded_at",
            "date_key",
            "time_key",
        ):
            assert col in cols, f"Missing column: {col}"

    def test_embedding_is_fixed_size_float_array(self, conn):
        # FLOAT[384] locks in BGE-small-en-v1.5 as the default embedder and
        # unlocks DuckDB's native array_cosine_similarity / array_inner_product
        # without bringing a vector DB.
        info = conn.execute(
            "SELECT column_name, column_type "
            "FROM (DESCRIBE fact_facet_embeddings) "
            "WHERE column_name = 'embedding'"
        ).fetchone()
        assert info is not None, "embedding column missing"
        assert info[1] == "FLOAT[384]", (
            f"Expected FLOAT[384] (bge-small-en-v1.5 dim), got {info[1]!r}"
        )

    def test_model_and_version_are_separate_columns(self, conn):
        # Split lets queries do `WHERE embedding_model = 'bge-small-en-v1.5'`
        # without LIKE patterns; supports multi-version coexistence.
        cols = [c[0] for c in conn.execute("DESCRIBE fact_facet_embeddings").fetchall()]
        assert "embedding_model" in cols
        assert "embedding_model_version" in cols
        # embedding_dim from the original spec is redundant once
        # (embedding_model, embedding_model_version) is captured -- model+version
        # uniquely determines dim.
        assert "embedding_dim" not in cols, (
            "embedding_dim is redundant given (embedding_model, embedding_model_version)"
        )
