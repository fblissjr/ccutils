"""Tests for the v0.15 dim_prompt populator (Phase D).

Grain: one row per (display_text, timestamp) pair in history.jsonl.
Linked to dim_session via session_id when the history entry carries one.

Stays minimal -- matches the dim_tool / dim_model pattern (no lineage
block). Not part of run_v15_etl: history.jsonl is a global file, not
per-session, so the populator is called explicitly (e.g. by `ccutils --source`
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


class TestHistoryIsScopedToTheWarehousesProjects:
    """A warehouse must not contain prompts from projects it does not cover.

    `history.jsonl` is machine-wide. Importing it unscoped put every prompt
    the user had ever typed into every warehouse: measured 2026-08-28, a
    ONE-SESSION warehouse for one project held 11,606 prompts from 103
    projects.

    That contradicts the project's stated privacy boundary -- a shared
    artifact is SCOPED, not scrubbed, so a wiki generated for named projects
    must contain nothing else. `_import_auto_memory` already scopes this way
    off `dim_project`; history did not, although every history entry already
    carries the project path needed to do it.

    Delete these and a scoped export silently becomes a machine-wide one,
    which is the exact failure the scoping decision exists to prevent.
    """

    def _history(self, tmp_path, *projects):
        f = tmp_path / "history.jsonl"
        f.write_text("\n".join(json.dumps({
            "display": f"prompt for {p}",
            "timestamp": 1737000000000 + i,
            "project": p,
            "pastedContents": {},
        }) for i, p in enumerate(projects)))
        return f

    def _warehouse_covering(self, conn, *cwds):
        for i, cwd in enumerate(cwds):
            conn.execute(
                "INSERT INTO dim_project (project_key, project_name, project_path) "
                "VALUES (?, ?, ?)",
                [f"pk{i}", "-" + cwd.strip("/").replace("/", "-"), cwd],
            )
            conn.execute(
                "INSERT INTO dim_session (session_key, session_id, cwd) "
                "VALUES (?, ?, ?)", [f"sk{i}", f"s{i}", cwd],
            )

    def test_only_covered_projects_are_imported(self, conn, tmp_path):
        self._warehouse_covering(conn, "/home/user/projects/mine")
        hist = self._history(
            tmp_path, "/home/user/projects/mine", "/home/user/projects/other"
        )

        import_history(conn, hist, only_projects=True)

        rows = conn.execute(
            "SELECT DISTINCT project_path FROM dim_prompt"
        ).fetchall()
        assert rows == [("/home/user/projects/mine",)]

    def test_unscoped_import_still_takes_everything(self, conn, tmp_path):
        """A full-corpus build loses nothing -- scoping only bites on filters."""
        self._warehouse_covering(conn, "/home/user/projects/mine")
        hist = self._history(
            tmp_path, "/home/user/projects/mine", "/home/user/projects/other"
        )

        import_history(conn, hist, only_projects=False)

        assert conn.execute("SELECT COUNT(*) FROM dim_prompt").fetchone()[0] == 2

    def test_a_dashed_project_name_still_matches(self, conn, tmp_path):
        """The encoding is lossy in one direction only.

        `-home-user-projects-my-app-tools` decodes to
        `/home/user/projects/fb/claude/skills` -- a different, nonexistent
        path. A first cut decoded backward and silently dropped every prompt
        for any project whose name contains a dash. Encode FORWARD instead:
        that is exact.
        """
        conn.execute(
            "INSERT INTO dim_project (project_key, project_name, project_path) "
            "VALUES ('pk', '-home-user-projects-my-app-tools', '/x')"
        )
        hist = self._history(
            tmp_path,
            "/home/user/projects/my-app-tools",
            "/home/user/projects/unrelated",
        )

        import_history(conn, hist, only_projects=True)

        rows = conn.execute("SELECT project_path FROM dim_prompt").fetchall()
        assert rows == [("/home/user/projects/my-app-tools",)]

    def test_scoping_removes_prompts_a_previous_unscoped_run_left(
        self, conn, tmp_path
    ):
        """Scoping must be a STATE, not a filter on new inserts.

        `import_history` is insert-only, and both the picker and `--source`
        default to the SAME output directory. So the sequence a reader
        follows straight from the README -- build a full archive, then build
        a scoped one into the default location -- left every prompt on the
        machine sitting in a warehouse that reports itself as scoped.

        Deleting is safe here because dim_prompt is derived: the next
        unscoped run re-imports from history.jsonl.
        """
        self._warehouse_covering(conn, "/home/user/projects/mine")
        hist = self._history(
            tmp_path, "/home/user/projects/mine", "/home/user/projects/other"
        )

        import_history(conn, hist, only_projects=False)
        assert conn.execute("SELECT COUNT(*) FROM dim_prompt").fetchone()[0] == 2

        import_history(conn, hist, only_projects=True)

        rows = conn.execute("SELECT DISTINCT project_path FROM dim_prompt").fetchall()
        assert rows == [("/home/user/projects/mine",)], (
            "a scoped run must leave no out-of-scope prompt behind"
        )

    def test_encoding_collapses_dots_and_underscores_too(self):
        """Claude Code replaces `/`, `.` AND `_` when naming a project dir.

        A first cut replaced only `/`, so the encoded-match arm never fired
        for any path containing a dot or an underscore, silently dropping
        those projects' prompts from scoped builds. Verified on the real
        corpus: a cwd ending `/evalroot/_run_d90fe13e` lives in a directory
        ending `-evalroot--run-d90fe13e`.
        """
        from ccutils.etl.dim_prompt import _project_dir_name

        assert _project_dir_name("/a/evalroot/_run_x") == "-a-evalroot--run-x"
        assert _project_dir_name("/a/b/.claude/wt") == "-a-b--claude-wt"
        assert _project_dir_name("/a/plain/path") == "-a-plain-path"

    def test_a_warehouse_covering_nothing_imports_nothing(self, conn, tmp_path):
        """An empty scope means NOTHING, never everything.

        `_import_auto_memory` carries this same warning: treating an empty
        project set as "unfiltered" inverts the scoping and ingests the whole
        machine, which is the worst possible reading of an empty set.
        """
        hist = self._history(tmp_path, "/home/user/projects/mine")

        import_history(conn, hist, only_projects=True)

        assert conn.execute("SELECT COUNT(*) FROM dim_prompt").fetchone()[0] == 0
