"""Tests for dim_tool.tool_category and the semantic_session_behavior view.

`tool_category` shipped in the DDL with the comment "categorization left
to a heuristic pass" and was never populated -- every row read 'unknown'
on a 2,250-session corpus. It is the star-schema home for the one
tool -> kind-of-work mapping; without it every behavioral query
re-invents its own `tool_name IN (...)` lists and they drift.

`semantic_session_behavior` exposes the per-session behavioral feature
vector built on those categories, plus corpus-relative percentile ranks.
It deliberately emits NO archetype label and NO threshold: bucketing
belongs in the analysis layer where cutoffs can be derived from the
distribution, not frozen into a view.
"""

from __future__ import annotations

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.orchestrator import run_v15_etl


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


@pytest.fixture
def mixed_tool_session(tmp_path):
    """One human turn driving Read, Edit, Bash, Grep, WebFetch and an MCP tool.

    Shares are then exactly known: 6 tool uses, one per category under test.
    """
    jsonl = tmp_path / "mixed.jsonl"
    tools = [
        ("tu1", "Read", {"file_path": "/p/a.py"}),
        ("tu2", "Grep", {"pattern": "x"}),
        ("tu3", "Edit", {"file_path": "/p/a.py", "old_string": "x",
                         "new_string": "y"}),
        ("tu4", "Bash", {"command": "pytest"}),
        ("tu5", "WebFetch", {"url": "https://example.invalid"}),
        ("tu6", "mcp__someserver__do_thing", {"arg": 1}),
    ]
    lines = [
        {"type": "user", "uuid": "u1", "sessionId": "mixed-s",
         "timestamp": "2026-04-19T17:00:00Z", "cwd": "/p",
         "gitBranch": "main", "version": "2.1.114",
         "message": {"role": "user", "content": "do a spread of work"}},
    ]
    prev = "u1"
    for i, (tid, name, tool_input) in enumerate(tools):
        auid = f"a{i}"
        lines.append(
            {"type": "assistant", "uuid": auid, "parentUuid": prev,
             "sessionId": "mixed-s",
             "timestamp": f"2026-04-19T17:00:{i * 2 + 1:02d}Z",
             "requestId": f"r{i}",
             "message": {"role": "assistant", "model": "claude-opus-4-7",
                         "content": [{"type": "tool_use", "id": tid,
                                      "name": name, "input": tool_input}],
                         "usage": {"input_tokens": 10, "output_tokens": 5}}}
        )
        ruid = f"ur{i}"
        lines.append(
            {"type": "user", "uuid": ruid, "parentUuid": auid,
             "sessionId": "mixed-s",
             "timestamp": f"2026-04-19T17:00:{i * 2 + 2:02d}Z",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": tid,
                  "content": "ok"}]}}
        )
        prev = ruid
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


def _categories(conn):
    return dict(
        conn.execute(
            "SELECT tool_name, tool_category FROM dim_tool"
        ).fetchall()
    )


class TestDimToolCategory:
    """Claim: delete these and tool_category silently reverts to 'unknown',
    which is how it shipped -- every behavioral query then hardcodes its own
    tool-name lists and they drift apart."""

    def test_core_tools_categorized(self, conn, mixed_tool_session, tmp_path):
        run_v15_etl(conn, mixed_tool_session, project_name="test-project",
                    parquet_lake_root=tmp_path / "lake")
        cats = _categories(conn)
        assert cats["Read"] == "read"
        assert cats["Grep"] == "search"
        assert cats["Edit"] == "mutate"
        assert cats["Bash"] == "execute"
        assert cats["WebFetch"] == "web"

    def test_mcp_tools_categorized_by_prefix_not_enumeration(
        self, conn, mixed_tool_session, tmp_path
    ):
        """A name-convention rule, so new MCP servers need no code change."""
        run_v15_etl(conn, mixed_tool_session, project_name="test-project",
                    parquet_lake_root=tmp_path / "lake")
        assert _categories(conn)["mcp__someserver__do_thing"] == "mcp"

    def test_no_tool_left_unknown(self, conn, mixed_tool_session, tmp_path):
        run_v15_etl(conn, mixed_tool_session, project_name="test-project",
                    parquet_lake_root=tmp_path / "lake")
        unknown = [n for n, c in _categories(conn).items() if c == "unknown"]
        assert unknown == [], f"uncategorized tools: {unknown}"

    def test_categorization_backfills_preexisting_unknown_rows(
        self, conn, mixed_tool_session, tmp_path
    ):
        """Warehouses built before this pass hold 'unknown'; a re-run fixes
        them rather than leaving the old rows stranded."""
        conn.execute(
            "INSERT INTO dim_tool (tool_key, tool_name, tool_category) "
            "VALUES (md5('Glob'), 'Glob', 'unknown')"
        )
        run_v15_etl(conn, mixed_tool_session, project_name="test-project",
                    parquet_lake_root=tmp_path / "lake")
        assert _categories(conn)["Glob"] == "search"


class TestSemanticSessionBehavior:
    """Claim: delete these and the behavioral feature vector loses its
    contract -- shares stop summing to the tool count, or a threshold quietly
    reappears in the view and freezes an archetype cutoff into the schema."""

    def test_category_counts_match_the_session(
        self, conn, mixed_tool_session, tmp_path
    ):
        run_v15_etl(conn, mixed_tool_session, project_name="test-project",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            """
            SELECT tool_uses, read_ops, search_ops, mutate_ops,
                   execute_ops, web_ops
            FROM semantic_session_behavior WHERE session_id = 'mixed-s'
            """
        ).fetchone()
        assert row == (6, 1, 1, 1, 1, 1)

    def test_shares_are_fractions_of_tool_uses(
        self, conn, mixed_tool_session, tmp_path
    ):
        run_v15_etl(conn, mixed_tool_session, project_name="test-project",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            """
            SELECT read_share, mutate_share, execute_share
            FROM semantic_session_behavior WHERE session_id = 'mixed-s'
            """
        ).fetchone()
        for share in row:
            assert share == pytest.approx(1 / 6)

    def test_percentile_ranks_present_and_bounded(
        self, conn, mixed_tool_session, tmp_path
    ):
        """Ranks are corpus-derived, which is what replaces a hardcoded cutoff."""
        run_v15_etl(conn, mixed_tool_session, project_name="test-project",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            """
            SELECT mutate_share_pctile, tokens_out_pctile, thinking_pctile
            FROM semantic_session_behavior WHERE session_id = 'mixed-s'
            """
        ).fetchone()
        for rank in row:
            assert rank is not None
            assert 0.0 <= rank <= 1.0

    def test_view_carries_no_archetype_label(self, conn):
        """The view must stay descriptive. An archetype column here would be
        a threshold frozen into the schema -- the thing this design avoids."""
        cols = [
            r[0] for r in conn.execute(
                "SELECT column_name FROM duckdb_columns() "
                "WHERE table_name = 'semantic_session_behavior'"
            ).fetchall()
        ]
        assert not any(
            "archetype" in c or "label" in c or "bucket" in c for c in cols
        ), f"classification leaked into the view: {cols}"
