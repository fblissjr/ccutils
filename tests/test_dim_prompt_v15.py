"""Tests for the v0.15 dim_prompt populator (Phase D).

Grain: one row per (display_text, timestamp) pair in history.jsonl.
Linked to dim_session via session_id when the history entry carries one.

Stays minimal -- matches the dim_tool / dim_model pattern (no lineage
block). Not part of run_v15_etl: history.jsonl is a global file, not
per-session, so the populator is called explicitly (e.g. by `ccutils all`
after the per-session loop).
"""

from __future__ import annotations

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.dim_prompt import import_history


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


@pytest.fixture
def history_jsonl(tmp_path):
    """A minimal history.jsonl with three prompts, two linked to a session."""
    path = tmp_path / "history.jsonl"
    lines = [
        {"display": "fix the bug",
         "project": "/work/proj",
         "sessionId": "sess-A",
         "timestamp": 1745052000000},
        {"display": "add a new feature",
         "project": "/work/proj",
         "sessionId": "sess-A",
         "timestamp": 1745052060000},
        {"display": "explore another thing",
         "project": "/work/other",
         "timestamp": 1745052120000},
    ]
    path.write_text("\n".join(json.dumps(d) for d in lines))
    return path


class TestDimPromptDates:
    def test_import_inserts_dim_date_rows_for_prompt_dates(
        self, conn, history_jsonl
    ):
        """history.jsonl carries dates no staged session covers; the
        import must add their dim_date rows or semantic_prompt_history
        returns NULL full_date."""
        import_history(conn, history_jsonl)
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM dim_prompt dp
            JOIN dim_date dd ON dp.date_key = dd.date_key
            """
        ).fetchone()
        assert row[0] == 3

    def test_semantic_prompt_history_full_date_not_null(
        self, conn, history_jsonl
    ):
        import_history(conn, history_jsonl)
        dates = [r[0] for r in conn.execute(
            "SELECT full_date FROM semantic_prompt_history"
        ).fetchall()]
        assert len(dates) == 3
        assert all(d is not None for d in dates)


class TestDimPrompt:
    def test_one_row_per_history_entry(self, conn, history_jsonl):
        import_history(conn, history_jsonl)
        n = conn.execute("SELECT COUNT(*) FROM dim_prompt").fetchone()[0]
        assert n == 3

    def test_display_text_captured(self, conn, history_jsonl):
        import_history(conn, history_jsonl)
        rows = conn.execute(
            "SELECT display_text FROM dim_prompt ORDER BY timestamp"
        ).fetchall()
        assert [r[0] for r in rows] == [
            "fix the bug",
            "add a new feature",
            "explore another thing",
        ]

    def test_project_name_derived(self, conn, history_jsonl):
        import_history(conn, history_jsonl)
        rows = conn.execute(
            "SELECT DISTINCT project_path, project_name FROM dim_prompt "
            "ORDER BY project_path"
        ).fetchall()
        assert ("/work/other", "other") in rows
        assert ("/work/proj", "proj") in rows

    def test_session_key_linked_when_session_id_present(
        self, conn, history_jsonl
    ):
        # Insert a stub dim_session row for sess-A so the link can resolve
        conn.execute(
            "INSERT INTO dim_session (session_key, session_id) "
            "VALUES (md5('sess-A'), 'sess-A')"
        )
        import_history(conn, history_jsonl)
        # The two sess-A prompts should have session_key set
        rows = conn.execute(
            "SELECT display_text, session_key FROM dim_prompt "
            "WHERE session_key IS NOT NULL"
        ).fetchall()
        assert len(rows) == 2
        for _, sk in rows:
            assert sk == conn.execute(
                "SELECT md5('sess-A')"
            ).fetchone()[0]

    def test_idempotent_reload(self, conn, history_jsonl):
        import_history(conn, history_jsonl)
        first = conn.execute(
            "SELECT prompt_key FROM dim_prompt ORDER BY prompt_key"
        ).fetchall()
        import_history(conn, history_jsonl)
        second = conn.execute(
            "SELECT prompt_key FROM dim_prompt ORDER BY prompt_key"
        ).fetchall()
        assert first == second
        n = conn.execute("SELECT COUNT(*) FROM dim_prompt").fetchone()[0]
        assert n == 3

    def test_missing_history_file_no_error(self, conn, tmp_path):
        """Importing a non-existent history.jsonl is a no-op."""
        import_history(conn, tmp_path / "does_not_exist.jsonl")
        n = conn.execute("SELECT COUNT(*) FROM dim_prompt").fetchone()[0]
        assert n == 0
