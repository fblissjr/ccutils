"""Tests for dim_memory / bridge_memory_link (Claude Code auto memory).

Grain: one row per (memory file, content version). ``dim_memory`` is a Type 2
slowly-changing dimension -- ``memory_id`` is the stable identity of a memory
file, ``memory_key`` identifies one version of it.

Type 2 is not decoration here. Claude Code overwrites memory files in place
and keeps no history of its own: the ``modified:`` frontmatter field holds
only the last write time, and prior contents survive only incidentally in
``file-history`` rollback checkpoints, which are pruned. If the warehouse
stored one row per file, every re-ingest would silently destroy the previous
state and memory evolution would be unrecoverable.

Like ``dim_prompt``, this is not part of ``run_v15_etl``: memory directories
are per-repository, not per-session, so the importer is called once after the
per-session loop.
"""

from __future__ import annotations

import pytest

from ccutils import create_star_schema
from ccutils.etl.dim_memory import import_memories, run_memory_import

FEEDBACK = """\
---
name: guard-rail-ordering
description: "Ordering a guard rail after the mutation lets a bad write land first"
metadata:
  node_type: memory
  type: feedback
  originSessionId: sess-A
  modified: 2026-07-30T03:57:12.156Z
---

Run guard rails before the mutation, not after.

Related: [[timeout-defaults]] and [[nothing-here]].
"""

SIGNAL = """\
---
name: timeout-defaults
description: prefer explicit timeouts to library defaults
metadata:
  node_type: memory
  type: feedback
  originSessionId: sess-B
  modified: 2026-07-24T10:00:00.000Z
---

Body of the timeout-defaults note.
"""

INDEX = """\
# project memory

Index with no frontmatter.
"""

PROJECT_DIR = "-work-alpha"

# Mirrors the real layout: dim_project.project_path is the encoded directory
# under the Claude home, and project_name is its last segment.
_FAKE_CLAUDE_HOME = "/home/u/.claude/projects"  # path-privacy: ignore


@pytest.fixture
def conn(tmp_path):
    conn = create_star_schema(tmp_path / "test.duckdb")
    # dim_project rows are keyed by the encoded projects/ directory name,
    # which is exactly the memory directory's owner.
    conn.execute(
        "INSERT INTO dim_project (project_key, project_path, project_name) "
        "VALUES ('pk-alpha', ?, ?)",
        [f"{_FAKE_CLAUDE_HOME}/{PROJECT_DIR}", PROJECT_DIR],
    )
    conn.execute(
        """
        INSERT INTO dim_session (session_key, session_id, project_key)
        VALUES ('sk-A', 'sess-A', 'pk-alpha')
        """
    )
    return conn


@pytest.fixture
def projects_root(tmp_path):
    d = tmp_path / "projects" / PROJECT_DIR / "memory"
    d.mkdir(parents=True)
    (d / "guard-rail-ordering.md").write_text(FEEDBACK)
    (d / "timeout-defaults.md").write_text(SIGNAL)
    (d / "MEMORY.md").write_text(INDEX)
    return tmp_path / "projects"


def cols(conn, table):
    return {r[0] for r in conn.execute(f"DESCRIBE {table}").fetchall()}


class TestDdl:
    def test_dim_memory_has_scd_columns(self, conn):
        assert {
            "memory_key", "memory_id", "content_hash",
            "version_num", "valid_from", "valid_to", "is_current",
        } <= cols(conn, "dim_memory")

    def test_dim_memory_has_content_and_provenance_columns(self, conn):
        assert {
            "project_key", "session_key", "scope", "owner_key", "agent_scope",
            "source_path", "file_name", "memory_name", "description",
            "memory_type", "node_type", "origin_session_id", "is_index",
            "has_frontmatter", "body_text", "body_chars", "body_lines",
            "link_count", "modified_at", "file_mtime", "date_key", "time_key",
        } <= cols(conn, "dim_memory")

    def test_bridge_memory_link_has_edge_columns(self, conn):
        assert {
            "memory_link_key", "memory_key", "memory_id", "project_key",
            "scope", "owner_key", "target_name", "target_memory_id",
            "is_resolved", "ordinal",
        } <= cols(conn, "bridge_memory_link")

    def test_semantic_memory_view_exists(self, conn):
        conn.execute("SELECT * FROM semantic_memory LIMIT 0")

    def test_semantic_memory_links_view_exists(self, conn):
        conn.execute("SELECT * FROM semantic_memory_links LIMIT 0")


class TestFirstImport:
    def test_every_memory_file_lands_as_version_one(self, conn, projects_root):
        import_memories(conn, projects_root=projects_root)
        rows = conn.execute(
            "SELECT file_name, version_num, is_current, valid_to "
            "FROM dim_memory ORDER BY file_name"
        ).fetchall()
        assert [r[0] for r in rows] == [
            "MEMORY.md", "guard-rail-ordering.md", "timeout-defaults.md",
        ]
        assert all(r[1] == 1 and r[2] is True and r[3] is None for r in rows)

    def test_frontmatter_is_projected_onto_columns(self, conn, projects_root):
        import_memories(conn, projects_root=projects_root)
        row = conn.execute(
            "SELECT memory_name, memory_type, node_type, origin_session_id, "
            "is_index, has_frontmatter FROM dim_memory "
            "WHERE file_name = 'guard-rail-ordering.md'"
        ).fetchone()
        assert row == ("guard-rail-ordering", "feedback", "memory", "sess-A", False, True)

    def test_index_file_is_flagged(self, conn, projects_root):
        import_memories(conn, projects_root=projects_root)
        row = conn.execute(
            "SELECT is_index, has_frontmatter FROM dim_memory "
            "WHERE file_name = 'MEMORY.md'"
        ).fetchone()
        assert row == (True, False)

    def test_body_text_is_stored(self, conn, projects_root):
        import_memories(conn, projects_root=projects_root)
        body = conn.execute(
            "SELECT body_text FROM dim_memory WHERE file_name = 'guard-rail-ordering.md'"
        ).fetchone()[0]
        assert "Run guard rails before the mutation" in body
        assert "originSessionId" not in body


class TestKeyResolution:
    def test_project_key_resolves_via_dim_project(self, conn, projects_root):
        import_memories(conn, projects_root=projects_root)
        keys = {
            r[0] for r in conn.execute("SELECT DISTINCT project_key FROM dim_memory").fetchall()
        }
        assert keys == {"pk-alpha"}

    def test_session_key_resolves_from_origin_session_id(self, conn, projects_root):
        import_memories(conn, projects_root=projects_root)
        row = conn.execute(
            "SELECT session_key FROM dim_memory WHERE file_name = 'guard-rail-ordering.md'"
        ).fetchone()
        assert row[0] == "sk-A"

    def test_unresolvable_origin_session_is_kept_raw(self, conn, projects_root):
        """sess-B is not in dim_session. The link must degrade to a NULL
        session_key while the stated id survives -- dropping it would lose
        the only evidence of which session wrote the memory."""
        import_memories(conn, projects_root=projects_root)
        row = conn.execute(
            "SELECT session_key, origin_session_id FROM dim_memory "
            "WHERE file_name = 'timeout-defaults.md'"
        ).fetchone()
        assert row == (None, "sess-B")


class TestIdempotencyAndVersioning:
    def test_reimport_of_unchanged_files_adds_nothing(self, conn, projects_root):
        import_memories(conn, projects_root=projects_root)
        before = conn.execute("SELECT COUNT(*) FROM dim_memory").fetchone()[0]
        import_memories(conn, projects_root=projects_root)
        after = conn.execute("SELECT COUNT(*) FROM dim_memory").fetchone()[0]
        assert after == before

    def test_modified_stamp_alone_does_not_open_a_version(self, conn, projects_root):
        """Claude Code rewrites modified: on every write. Treating that as a
        content change would fill the history with versions that differ only
        by a timestamp."""
        import_memories(conn, projects_root=projects_root)
        path = projects_root / PROJECT_DIR / "memory" / "guard-rail-ordering.md"
        path.write_text(
            FEEDBACK.replace("2026-07-30T03:57:12.156Z", "2026-08-05T09:00:00.000Z")
        )
        import_memories(conn, projects_root=projects_root)
        assert conn.execute(
            "SELECT COUNT(*) FROM dim_memory WHERE file_name = 'guard-rail-ordering.md'"
        ).fetchone()[0] == 1

    def test_body_change_opens_a_new_version_and_closes_the_old(
        self, conn, projects_root
    ):
        import_memories(conn, projects_root=projects_root)
        path = projects_root / PROJECT_DIR / "memory" / "guard-rail-ordering.md"
        path.write_text(FEEDBACK.replace("before the mutation", "after the mutation"))
        import_memories(conn, projects_root=projects_root)

        rows = conn.execute(
            "SELECT version_num, is_current, valid_to IS NULL FROM dim_memory "
            "WHERE file_name = 'guard-rail-ordering.md' ORDER BY version_num"
        ).fetchall()
        assert rows == [(1, False, False), (2, True, True)]

    def test_exactly_one_current_row_per_memory(self, conn, projects_root):
        import_memories(conn, projects_root=projects_root)
        path = projects_root / PROJECT_DIR / "memory" / "guard-rail-ordering.md"
        path.write_text(FEEDBACK.replace("before the mutation", "after the mutation"))
        import_memories(conn, projects_root=projects_root)
        path.write_text(FEEDBACK.replace("before the mutation", "during the mutation"))
        import_memories(conn, projects_root=projects_root)

        dupes = conn.execute(
            "SELECT memory_id, COUNT(*) FROM dim_memory WHERE is_current "
            "GROUP BY memory_id HAVING COUNT(*) > 1"
        ).fetchall()
        assert dupes == []

    def test_reverting_to_earlier_content_is_a_new_version(self, conn, projects_root):
        """Content hashes repeat when a memory is reverted. Version identity
        must be (memory, version_num), not (memory, content) -- keying on
        content would silently drop the revert."""
        path = projects_root / PROJECT_DIR / "memory" / "guard-rail-ordering.md"
        import_memories(conn, projects_root=projects_root)
        path.write_text(FEEDBACK.replace("before the mutation", "after the mutation"))
        import_memories(conn, projects_root=projects_root)
        path.write_text(FEEDBACK)
        import_memories(conn, projects_root=projects_root)

        rows = conn.execute(
            "SELECT version_num, is_current FROM dim_memory "
            "WHERE file_name = 'guard-rail-ordering.md' ORDER BY version_num"
        ).fetchall()
        assert rows == [(1, False), (2, False), (3, True)]

    def test_deleting_the_whole_memory_directory_closes_its_rows(
        self, conn, projects_root
    ):
        """Closing must key off the scope that was SCANNED, not off the files
        that happened to come back. If it keyed off results, wiping a memory
        directory would leave every one of its rows open forever, and the
        warehouse would report retired memories as current."""
        import_memories(conn, projects_root=projects_root)
        for md in (projects_root / PROJECT_DIR / "memory").glob("*.md"):
            md.unlink()
        import_memories(conn, projects_root=projects_root)

        assert conn.execute(
            "SELECT COUNT(*) FROM dim_memory WHERE is_current"
        ).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM dim_memory").fetchone()[0] == 3

    def test_a_filtered_run_does_not_retire_another_projects_memory(
        self, conn, projects_root
    ):
        """-p alpha must not close beta's rows just because it did not look
        at them."""
        other = projects_root / "-work-beta" / "memory"
        other.mkdir(parents=True)
        (other / "MEMORY.md").write_text(INDEX)
        import_memories(conn, projects_root=projects_root)

        import_memories(conn, projects_root=projects_root, only={PROJECT_DIR})
        assert conn.execute(
            "SELECT COUNT(*) FROM dim_memory WHERE is_current AND owner_key = '-work-beta'"
        ).fetchone()[0] == 1

    def test_deleted_memory_file_is_closed_not_erased(self, conn, projects_root):
        """A retired memory is a fact about the project's history. Closing the
        row keeps it queryable; deleting it would make the warehouse forget
        the memory ever existed."""
        import_memories(conn, projects_root=projects_root)
        (projects_root / PROJECT_DIR / "memory" / "timeout-defaults.md").unlink()
        import_memories(conn, projects_root=projects_root)

        row = conn.execute(
            "SELECT is_current, valid_to IS NOT NULL FROM dim_memory "
            "WHERE file_name = 'timeout-defaults.md'"
        ).fetchone()
        assert row == (False, True)


class TestLinkGraph:
    def test_links_land_as_bridge_rows_in_order(self, conn, projects_root):
        import_memories(conn, projects_root=projects_root)
        rows = conn.execute(
            "SELECT target_name, ordinal FROM bridge_memory_link bl "
            "JOIN dim_memory m USING (memory_key) "
            "WHERE m.file_name = 'guard-rail-ordering.md' ORDER BY ordinal"
        ).fetchall()
        assert rows == [("timeout-defaults", 0), ("nothing-here", 1)]

    def test_link_resolves_to_a_sibling_memory(self, conn, projects_root):
        import_memories(conn, projects_root=projects_root)
        row = conn.execute(
            "SELECT bl.is_resolved, bl.target_memory_id = t.memory_id "
            "FROM bridge_memory_link bl "
            "JOIN dim_memory t ON t.file_name = 'timeout-defaults.md' "
            "WHERE bl.target_name = 'timeout-defaults'"
        ).fetchone()
        assert row == (True, True)

    def test_link_resolves_across_separator_conventions(self, conn, projects_root):
        """Real corpora write the file stem with underscores and the
        frontmatter name with hyphens, then link with either. `-` and `_` are
        the same identifier in different clothes, so matching modulo
        separator recovers real edges that an exact match drops."""
        d = projects_root / PROJECT_DIR / "memory"
        (d / "feedback_timeout_defaults.md").write_text(
            SIGNAL.replace("name: timeout-defaults", "name: timeout-defaults-not-retries")
        )
        (d / "linker.md").write_text(
            "---\nname: linker\n---\n\nsee [[feedback-timeout-defaults]]\n"
        )
        import_memories(conn, projects_root=projects_root)

        row = conn.execute(
            "SELECT is_resolved, target_file_name FROM semantic_memory_links "
            "WHERE target_name = 'feedback-timeout-defaults'"
        ).fetchone()
        assert row == (True, "feedback_timeout_defaults.md")

    def test_a_near_miss_link_is_not_fuzzy_matched(self, conn, projects_root):
        """[[signal]] is not [[timeout-defaults]]. Resolving on prefixes or
        substrings would invent edges the author never wrote -- an unresolved
        row is the honest answer."""
        d = projects_root / PROJECT_DIR / "memory"
        (d / "linker.md").write_text(
            "---\nname: linker\n---\n\nsee [[signal]] and [[timeout-defaults-extra]]\n"
        )
        import_memories(conn, projects_root=projects_root)

        rows = conn.execute(
            "SELECT target_name, is_resolved FROM bridge_memory_link "
            "WHERE target_name IN ('signal', 'timeout-defaults-extra') ORDER BY 1"
        ).fetchall()
        assert rows == [("signal", False), ("timeout-defaults-extra", False)]

    def test_dangling_link_is_kept_unresolved(self, conn, projects_root):
        """A [[link]] to a memory that was never written is real signal --
        it marks something Claude meant to record. Dropping the row would
        hide it."""
        import_memories(conn, projects_root=projects_root)
        row = conn.execute(
            "SELECT is_resolved, target_memory_id FROM bridge_memory_link "
            "WHERE target_name = 'nothing-here'"
        ).fetchone()
        assert row == (False, None)

    def test_links_are_rebuilt_for_the_new_version_only(self, conn, projects_root):
        """Each version owns its edges. The closed version keeps the edges it
        had, so the graph can be queried as of any point in time."""
        path = projects_root / PROJECT_DIR / "memory" / "guard-rail-ordering.md"
        import_memories(conn, projects_root=projects_root)
        path.write_text(FEEDBACK.replace("[[nothing-here]]", "[[still-nothing]]"))
        import_memories(conn, projects_root=projects_root)

        current = conn.execute(
            "SELECT bl.target_name FROM bridge_memory_link bl "
            "JOIN dim_memory m USING (memory_key) "
            "WHERE m.file_name = 'guard-rail-ordering.md' AND m.is_current "
            "ORDER BY bl.ordinal"
        ).fetchall()
        assert current == [("timeout-defaults",), ("still-nothing",)]

        historical = conn.execute(
            "SELECT COUNT(*) FROM bridge_memory_link bl "
            "JOIN dim_memory m USING (memory_key) "
            "WHERE m.file_name = 'guard-rail-ordering.md' AND NOT m.is_current"
        ).fetchone()[0]
        assert historical == 2


class TestIndexLinks:
    """MEMORY.md is the index, and it points at topic files with
    ``- [Title](file.md)``, not [[wiki]] syntax. Those index edges are the
    relationship that makes the corpus an index rather than a pile of files.
    Measured on a real corpus: 71 of 140 edges were markdown links, ALL of
    them originating in an index, and 70 duplicated no wiki edge."""

    @pytest.fixture
    def indexed(self, projects_root):
        (projects_root / PROJECT_DIR / "memory" / "MEMORY.md").write_text(
            "# project memory\n\n"
            "- [Exit codes](guard-rail-ordering.md) -- pipes hide failures\n"
            "- [Signal honesty](timeout-defaults.md) -- do not touch the check\n"
            "- [Missing](never-written.md) -- not on disk\n"
        )
        return projects_root

    def test_index_entries_become_resolved_edges(self, conn, indexed):
        import_memories(conn, projects_root=indexed)
        rows = conn.execute(
            "SELECT bl.target_name, bl.is_resolved, bl.link_syntax "
            "FROM bridge_memory_link bl JOIN dim_memory m USING (memory_key) "
            "WHERE m.file_name = 'MEMORY.md' ORDER BY bl.ordinal"
        ).fetchall()
        assert rows == [
            ("guard-rail-ordering.md", True, "markdown"),
            ("timeout-defaults.md", True, "markdown"),
            ("never-written.md", False, "markdown"),
        ]

    def test_index_label_is_captured(self, conn, indexed):
        """The index's own label for a memory is what a human reads in the
        table of contents; it is not recoverable from the target file."""
        import_memories(conn, projects_root=indexed)
        row = conn.execute(
            "SELECT link_text FROM bridge_memory_link "
            "WHERE target_name = 'guard-rail-ordering.md'"
        ).fetchone()
        assert row == ("Exit codes",)

    def test_markdown_target_resolves_on_file_name_not_memory_name(
        self, conn, indexed
    ):
        """guard-rail-ordering.md carries `name: guard-rail-ordering`, but a
        markdown target is a path the author wrote -- it must match the file
        name, so an index still resolves when the two disagree."""
        d = indexed / PROJECT_DIR / "memory"
        (d / "renamed_file.md").write_text(
            FEEDBACK.replace("name: guard-rail-ordering", "name: totally-different")
        )
        (d / "MEMORY.md").write_text("# index\n\n- [R](renamed_file.md) -- x\n")
        import_memories(conn, projects_root=indexed)

        row = conn.execute(
            "SELECT bl.is_resolved, t.memory_name FROM bridge_memory_link bl "
            "LEFT JOIN dim_memory t ON bl.target_memory_id = t.memory_id "
            "WHERE bl.target_name = 'renamed_file.md'"
        ).fetchone()
        assert row == (True, "totally-different")

    def test_both_syntaxes_coexist_in_one_graph(self, conn, indexed):
        import_memories(conn, projects_root=indexed)
        counts = dict(
            conn.execute(
                "SELECT link_syntax, COUNT(*) FROM bridge_memory_link "
                "GROUP BY link_syntax"
            ).fetchall()
        )
        # 3 index entries + the 2 [[wiki]] links in guard-rail-ordering.md
        assert counts == {"markdown": 3, "wiki": 2}

    def test_semantic_view_distinguishes_index_edges(self, conn, indexed):
        import_memories(conn, projects_root=indexed)
        row = conn.execute(
            "SELECT source_is_index, link_syntax FROM semantic_memory_links "
            "WHERE target_name = 'timeout-defaults.md'"
        ).fetchone()
        assert row == (True, "markdown")


class TestAgentScope:
    @pytest.fixture
    def agent_root(self, tmp_path):
        d = tmp_path / "agent-memory" / "prompt-engineer"
        d.mkdir(parents=True)
        (d / "MEMORY.md").write_text(INDEX)
        return tmp_path / "agent-memory"

    def test_agent_memory_is_ingested_with_its_scope(self, conn, agent_root):
        import_memories(conn, agent_user_root=agent_root)
        row = conn.execute(
            "SELECT scope, owner_key, agent_scope, project_key FROM dim_memory"
        ).fetchone()
        assert row == ("agent", "prompt-engineer", "user", None)

    def test_agent_and_project_memory_coexist(self, conn, projects_root, agent_root):
        import_memories(
            conn, projects_root=projects_root, agent_user_root=agent_root
        )
        counts = dict(
            conn.execute("SELECT scope, COUNT(*) FROM dim_memory GROUP BY scope").fetchall()
        )
        assert counts == {"project": 3, "agent": 1}

    def test_same_agent_name_in_different_agent_scopes_stays_distinct(
        self, conn, tmp_path
    ):
        """A subagent can declare `memory: user` in one place and
        `memory: project` in another under the same name. If agent_scope is
        not part of the identity they collapse into one memory that appears
        to flip-flop between two bodies on every import."""
        user = tmp_path / "agent-memory" / "reviewer"
        user.mkdir(parents=True)
        (user / "MEMORY.md").write_text("# user scope body\n")

        repo = tmp_path / "repo"
        committed = repo / ".claude" / "agent-memory" / "reviewer"
        committed.mkdir(parents=True)
        (committed / "MEMORY.md").write_text("# project scope body\n")

        import_memories(
            conn,
            agent_user_root=tmp_path / "agent-memory",
            agent_repo_paths=[repo],
        )
        rows = conn.execute(
            "SELECT agent_scope, version_num, is_current FROM dim_memory "
            "ORDER BY agent_scope"
        ).fetchall()
        assert rows == [("project", 1, True), ("user", 1, True)]

    def test_same_agent_name_in_two_repos_stays_distinct(self, conn, tmp_path):
        """Committed subagent memory is shared with a team, so a `reviewer`
        agent existing in two repositories is ordinary. Keying identity on
        the agent name alone would merge them."""
        repos = []
        for name, body in (("alpha", "# alpha body\n"), ("beta", "# beta body\n")):
            repo = tmp_path / name
            d = repo / ".claude" / "agent-memory" / "reviewer"
            d.mkdir(parents=True)
            (d / "MEMORY.md").write_text(body)
            repos.append(repo)

        import_memories(conn, agent_repo_paths=repos)
        assert conn.execute(
            "SELECT COUNT(DISTINCT memory_id) FROM dim_memory"
        ).fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM dim_memory WHERE is_current"
        ).fetchone()[0] == 2

    def test_same_file_name_in_both_scopes_stays_distinct(
        self, conn, projects_root, agent_root
    ):
        """Both scopes have a MEMORY.md. Keying identity on file name alone
        would collapse them into one memory."""
        import_memories(
            conn, projects_root=projects_root, agent_user_root=agent_root
        )
        ids = conn.execute(
            "SELECT COUNT(DISTINCT memory_id) FROM dim_memory WHERE file_name = 'MEMORY.md'"
        ).fetchone()[0]
        assert ids == 2


class TestEtlIntegration:
    """Memory ingestion has to be visible to the run-metadata system.

    The warehouse records three grains (batch / run / step) and every other
    populator reports into them. A global source that writes rows outside
    that system is invisible: nothing says how many memory versions a run
    wrote, nothing says which run wrote a given version, and a failure
    leaves no trace at all. For a Type 2 SCD the provenance question --
    "which run observed this version" -- is exactly the one you want
    answered.
    """

    def test_dim_memory_carries_run_provenance_columns(self, conn):
        assert {
            "etl_run_id", "record_source", "created_at", "created_by_version_key",
        } <= cols(conn, "dim_memory")

    def test_bridge_memory_link_carries_run_provenance_columns(self, conn):
        assert {
            "etl_run_id", "record_source", "created_at", "created_by_version_key",
        } <= cols(conn, "bridge_memory_link")

    def test_dim_memory_has_no_is_deleted_column(self, conn):
        """Deliberate. The Type 2 columns (is_current / valid_to) ARE the
        deletion mechanism here. A second one would let a row be closed by
        one convention and deleted by the other, and the two would disagree
        the first time a memory was retired."""
        assert "is_deleted" not in cols(conn, "dim_memory")

    def test_import_records_a_run_of_its_own_kind(self, conn, projects_root):
        run_memory_import(conn, projects_root=projects_root)
        rows = conn.execute(
            "SELECT run_kind, status FROM fact_etl_runs "
            "WHERE source_path = '<auto-memory>'"
        ).fetchall()
        assert rows == [("global_source", "success")]

    def test_semantic_etl_runs_exposes_the_run_kind(self, conn, projects_root):
        """The documented observability view must be able to tell a memory
        import from a session run. Without run_kind exposed, an unqualified
        count over this view silently mixes three different kinds of run."""
        run_memory_import(conn, projects_root=projects_root)
        row = conn.execute(
            "SELECT run_kind, status FROM semantic_etl_runs "
            "WHERE source_path = '<auto-memory>'"
        ).fetchone()
        assert row == ("global_source", "success")

    def test_import_records_a_step_with_real_counts(self, conn, projects_root):
        """Counts must be derived from the work actually done, not asserted
        by the caller -- a hand-tallied count is how a step row starts
        lying."""
        run_memory_import(conn, projects_root=projects_root)
        row = conn.execute(
            "SELECT step_name, status, rows_inserted FROM fact_etl_steps "
            "WHERE step_name = 'dim_memory'"
        ).fetchone()
        assert row == ("dim_memory", "success", 3)

    def test_rows_are_stamped_with_the_run_that_wrote_them(
        self, conn, projects_root
    ):
        run_memory_import(conn, projects_root=projects_root)
        run_id = conn.execute(
            "SELECT etl_run_id FROM fact_etl_runs WHERE source_path = '<auto-memory>'"
        ).fetchone()[0]
        assert conn.execute(
            "SELECT COUNT(*) FROM dim_memory WHERE etl_run_id = ?", [run_id]
        ).fetchone()[0] == 3
        assert conn.execute(
            "SELECT COUNT(*) FROM bridge_memory_link WHERE etl_run_id IS NULL"
        ).fetchone()[0] == 0

    def test_a_new_version_is_stamped_with_the_later_run(
        self, conn, projects_root
    ):
        """The whole point of provenance on a Type 2 row: two versions of one
        memory must name the two different runs that observed them."""
        run_memory_import(conn, projects_root=projects_root)
        path = projects_root / PROJECT_DIR / "memory" / "guard-rail-ordering.md"
        path.write_text(FEEDBACK.replace("before the mutation", "after the mutation"))
        run_memory_import(conn, projects_root=projects_root)

        runs = conn.execute(
            "SELECT version_num, etl_run_id FROM dim_memory "
            "WHERE file_name = 'guard-rail-ordering.md' ORDER BY version_num"
        ).fetchall()
        assert len(runs) == 2
        assert runs[0][1] != runs[1][1]

    def test_an_unchanged_reimport_records_a_run_that_wrote_nothing(
        self, conn, projects_root
    ):
        """Idempotency has to be observable. A second run that inserts zero
        rows is the evidence that nothing changed -- silence would be
        indistinguishable from the import never happening."""
        run_memory_import(conn, projects_root=projects_root)
        run_memory_import(conn, projects_root=projects_root)
        counts = [
            r[0] for r in conn.execute(
                "SELECT rows_inserted FROM fact_etl_steps "
                "WHERE step_name = 'dim_memory' ORDER BY started_at"
            ).fetchall()
        ]
        assert counts == [3, 0]

    def test_a_failure_is_recorded_rather_than_swallowed(
        self, conn, projects_root, monkeypatch
    ):
        """A best-effort import must still leave a trace when it fails.
        `except Exception: pass` loses the fact that memory was supposed to
        be there -- the warehouse then looks identical to one built on a
        machine with auto memory disabled."""
        import ccutils.etl.dim_memory as mod

        def boom(*a, **k):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(mod, "_collect", boom)
        run_memory_import(conn, projects_root=projects_root)

        row = conn.execute(
            "SELECT status, error_message FROM fact_etl_runs "
            "WHERE source_path = '<auto-memory>'"
        ).fetchone()
        assert row[0] == "failed"
        assert "disk on fire" in row[1]

    def test_a_failure_does_not_abort_the_archive(
        self, conn, projects_root, monkeypatch
    ):
        """Memory is additive -- losing it corrupts nothing else, so unlike
        the cross-session reconciliation pass this one records and returns
        instead of re-raising."""
        import ccutils.etl.dim_memory as mod

        monkeypatch.setattr(
            mod, "_collect", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        )
        assert run_memory_import(conn, projects_root=projects_root) == 0


class TestUpgradePath:
    """A warehouse built before the provenance/link_syntax columns existed
    must heal on open, not fail.

    ``CREATE TABLE IF NOT EXISTS`` never widens an existing table, so the
    narrow shape survives every subsequent create_star_schema() call unless
    _COLUMN_MIGRATIONS re-adds the columns. This is not a cosmetic gap here:
    the import writes those columns directly, so an un-migrated warehouse
    fails outright rather than degrading.
    """

    def _narrow_warehouse(self, tmp_path):
        """A database holding the memory tables as originally shipped."""
        import duckdb

        path = tmp_path / "old.duckdb"
        conn = duckdb.connect(str(path))
        conn.execute(
            """
            CREATE TABLE dim_memory (
                memory_key VARCHAR, memory_id VARCHAR, project_key VARCHAR,
                session_key VARCHAR, scope VARCHAR, owner_key VARCHAR,
                owner_root VARCHAR, agent_scope VARCHAR, source_path VARCHAR,
                file_name VARCHAR, memory_name VARCHAR, description TEXT,
                memory_type VARCHAR, node_type VARCHAR,
                origin_session_id VARCHAR, is_index BOOLEAN,
                has_frontmatter BOOLEAN, body_text TEXT, content_hash VARCHAR,
                body_chars INTEGER, body_lines INTEGER, link_count INTEGER,
                modified_at TIMESTAMP, file_mtime TIMESTAMP,
                version_num INTEGER, valid_from TIMESTAMP, valid_to TIMESTAMP,
                is_current BOOLEAN, date_key INTEGER, time_key INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE bridge_memory_link (
                memory_link_key VARCHAR, memory_key VARCHAR,
                memory_id VARCHAR, project_key VARCHAR, scope VARCHAR,
                owner_key VARCHAR, target_name VARCHAR,
                target_memory_id VARCHAR, is_resolved BOOLEAN, ordinal INTEGER
            )
            """
        )
        conn.close()
        return path

    def test_narrow_warehouse_gains_the_columns_on_open(self, tmp_path):
        path = self._narrow_warehouse(tmp_path)
        conn = create_star_schema(path)
        assert {
            "created_at", "created_by_version_key", "etl_run_id", "record_source",
        } <= cols(conn, "dim_memory")
        assert {
            "created_at", "created_by_version_key", "etl_run_id",
            "record_source", "link_syntax", "link_text",
        } <= cols(conn, "bridge_memory_link")

    def test_import_succeeds_against_an_upgraded_warehouse(
        self, tmp_path, projects_root
    ):
        """The end the migration exists for. Without it this raises on a
        missing column instead of writing rows."""
        conn = create_star_schema(self._narrow_warehouse(tmp_path))
        assert run_memory_import(conn, projects_root=projects_root) == 3
        assert conn.execute(
            "SELECT COUNT(*) FROM dim_memory WHERE etl_run_id IS NOT NULL"
        ).fetchone()[0] == 3

    def test_markdown_links_resolve_on_an_upgraded_warehouse(
        self, tmp_path, projects_root
    ):
        """link_syntax is branched on during resolution, so an un-migrated
        warehouse would send every index edge down the identifier path and
        silently resolve none of them."""
        (projects_root / PROJECT_DIR / "memory" / "MEMORY.md").write_text(
            "# index\n\n- [Exit codes](guard-rail-ordering.md) -- hook\n"
        )
        conn = create_star_schema(self._narrow_warehouse(tmp_path))
        run_memory_import(conn, projects_root=projects_root)

        row = conn.execute(
            "SELECT link_syntax, is_resolved FROM bridge_memory_link "
            "WHERE target_name = 'guard-rail-ordering.md'"
        ).fetchone()
        assert row == ("markdown", True)


class TestScoping:
    def test_only_named_projects_are_ingested(self, conn, projects_root):
        """A filtered archive run (-p alpha) must not pull in every other
        project's memory from the same machine."""
        other = projects_root / "-work-beta" / "memory"
        other.mkdir(parents=True)
        (other / "MEMORY.md").write_text(INDEX)

        import_memories(conn, projects_root=projects_root, only={PROJECT_DIR})
        owners = {r[0] for r in conn.execute("SELECT DISTINCT owner_key FROM dim_memory").fetchall()}
        assert owners == {PROJECT_DIR}

    def test_missing_roots_are_a_no_op(self, conn, tmp_path):
        assert import_memories(conn, projects_root=tmp_path / "nope") == 0
        assert conn.execute("SELECT COUNT(*) FROM dim_memory").fetchone()[0] == 0


class TestSemanticViews:
    def test_semantic_memory_exposes_only_current_versions(self, conn, projects_root):
        path = projects_root / PROJECT_DIR / "memory" / "guard-rail-ordering.md"
        import_memories(conn, projects_root=projects_root)
        path.write_text(FEEDBACK.replace("before the mutation", "after the mutation"))
        import_memories(conn, projects_root=projects_root)

        assert conn.execute(
            "SELECT COUNT(*) FROM semantic_memory WHERE file_name = 'guard-rail-ordering.md'"
        ).fetchone()[0] == 1

    def test_semantic_memory_carries_project_name(self, conn, projects_root):
        import_memories(conn, projects_root=projects_root)
        names = {
            r[0] for r in conn.execute("SELECT DISTINCT project_name FROM semantic_memory").fetchall()
        }
        assert names == {PROJECT_DIR}

    def test_semantic_memory_links_names_both_ends(self, conn, projects_root):
        import_memories(conn, projects_root=projects_root)
        row = conn.execute(
            "SELECT source_name, target_name, is_resolved FROM semantic_memory_links "
            "WHERE target_name = 'timeout-defaults'"
        ).fetchone()
        assert row == ("guard-rail-ordering", "timeout-defaults", True)
