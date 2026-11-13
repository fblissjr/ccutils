# path-privacy: skip-file -- generic placeholder paths only
"""Schema-wide invariants: no HUGEINT column reaches a consumer, the deleted
plain-join views stay deleted, and no view bakes in a row order.

Column-by-column copies of the DDL were removed on 2026-11-13: a renamed
column a view uses fails `create_star_schema` itself, and the documented
columns are held both ways against `docs/STAR_SCHEMA.md` by
`tests/test_star_schema_doc.py`.
"""


from ccutils import (
    create_star_schema,
)


class TestNoColumnIsHugeint:
    """No column a consumer reads is HUGEINT.

    DuckDB types `SUM` over an integer column as HUGEINT (a 128-bit integer),
    and a view takes whatever type its expression has. Nothing is stored that
    way: every table column is INTEGER or BIGINT. But the summed columns of
    every `semantic_` view came out HUGEINT, and that reaches Arrow, and so a
    dataframe, as `decimal128(38, 0)` instead of `int64`.

    It arrived unannounced. `semantic_session_summary` was a table with
    INTEGER and BIGINT columns until 1.0.0 made it a view, and the same
    `SUM(...)` that had been cast by the INSERT then defined the column
    type. The doc went on saying BIGINT for a month.

    The sums are cast to BIGINT in the views. A session's token total does
    not approach 2^63.
    """

    def test_main_has_no_hugeint_column(self, tmp_path):
        conn = create_star_schema(tmp_path / "t.duckdb")
        rows = conn.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = 'main' AND data_type = 'HUGEINT' "
            "ORDER BY 1, 2"
        ).fetchall()
        conn.close()
        assert rows == []

    def test_a_summed_view_column_reaches_arrow_as_int64(self, tmp_path):
        """The effect, not just the declared type."""
        conn = create_star_schema(tmp_path / "t.duckdb")
        schema = conn.execute(
            "SELECT total_input_tokens, total_tool_errors, total_delegations "
            "FROM semantic_session_summary"
        ).to_arrow_table().schema
        conn.close()
        assert [str(f.type) for f in schema] == ["int64", "int64", "int64"]


class TestPlainJoinViewsAreGone:
    """A view exists because it encodes something a consumer would get wrong.

    The first five joined a fact to its dimensions and added nothing: no
    filter, no derivation, no guard. `docs/ETL_ARCHITECTURE.md` rule 3 says
    an object does not earn its place by saving a JOIN, and each one was a
    second name to keep in step with its fact, which is how
    `semantic_project_context` came to overlap `semantic_sessions` on most
    of its columns. They were marked `delete` in `TABLE_COVERAGE` and are
    deleted at 1.0.0. The joins they saved are written out once in the
    query-warehouse skill's recipes.

    Two more followed on 2026-10-13, by the same rule. `semantic_sessions`
    had been kept for its soft-delete filter; once the session summary
    became a view there was nothing left to filter, and it was forty
    pass-through columns and `WHERE TRUE`. `semantic_cost_analysis`
    re-projected the summary and added one column, `cache_hit_rate_pct`,
    which the coverage registry itself said discriminates nothing.
    """

    GONE = ("semantic_messages", "semantic_file_operations",
            "semantic_session_chains", "semantic_plan_revisions",
            "semantic_project_context",
            "semantic_sessions", "semantic_cost_analysis")

    def test_they_do_not_exist(self, tmp_path):
        from ccutils.schemas.star.schema import TABLE_COVERAGE

        conn = create_star_schema(tmp_path / "t.duckdb")
        objects = {r[0] for r in conn.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()}
        conn.close()
        for name in self.GONE:
            assert name not in objects, name
            assert name not in TABLE_COVERAGE, name

    def test_nothing_is_left_marked_for_deletion(self):
        """`delete` was a status for a view on its way out. Once the last one
        is gone, an entry carrying it is a deletion somebody forgot."""
        from ccutils.schemas.star.schema import TABLE_COVERAGE

        assert sorted(n for n, v in TABLE_COVERAGE.items() if v[1] == "delete") == []


class TestNoViewBakesInAnOrder:
    """A view's row order is the caller's to choose.

    An `ORDER BY` ending a view is not a contract: the moment a caller adds
    a clause of their own, a join or a `LIMIT`, the engine is free to return
    rows in any order, so the ordering reads as dependable and is not. It
    also sorts the whole result for callers who wanted three rows of it.
    Two views carried one until 2026-10-13.

    Delete this and an ordering creeps back into a view, and the next query
    that relies on it is right until it grows a `WHERE`.
    """

    def test_no_semantic_view_ends_with_an_order_by(self, tmp_path):
        import re

        conn = create_star_schema(tmp_path / "t.duckdb")
        views = conn.execute(
            "SELECT view_name, sql FROM duckdb_views() WHERE view_name LIKE 'semantic_%'"
        ).fetchall()
        conn.close()
        assert views, "no semantic views found: the check ran against nothing"
        # After the last closing parenthesis there is only the outer query's tail.
        ordered = sorted(
            name for name, sql in views
            if re.search(r"\bORDER\s+BY\b", sql.rstrip().rstrip(";").rsplit(")", 1)[-1], re.I)
        )
        assert ordered == []

    def test_the_check_sees_an_ordered_view(self, tmp_path):
        """The oracle can fail: a view that does end with ORDER BY is caught."""
        import re

        conn = create_star_schema(tmp_path / "t.duckdb")
        conn.execute(
            "CREATE VIEW semantic_zz_ordered AS "
            "SELECT session_id, COUNT(*) AS n FROM dim_session GROUP BY session_id ORDER BY n DESC")
        sql = conn.execute(
            "SELECT sql FROM duckdb_views() WHERE view_name = 'semantic_zz_ordered'"
        ).fetchone()[0]
        conn.close()
        assert re.search(r"\bORDER\s+BY\b", sql.rstrip().rstrip(";").rsplit(")", 1)[-1], re.I)
