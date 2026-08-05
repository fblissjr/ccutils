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
name: first-topic
description: "First placeholder description for the example memory"
metadata:
  node_type: memory
  type: feedback
  originSessionId: sess-A
  modified: 2026-07-30T03:57:12.156Z
---

Body text for the first example topic.

Related: [[second-topic]] and [[missing-topic]].
"""

SIGNAL = """\
---
name: second-topic
description: Second placeholder description for the example memory
metadata:
  node_type: memory
  type: feedback
  originSessionId: sess-B
  modified: 2026-07-24T10:00:00.000Z
---

Body text for the second example topic.
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
    (d / "topic_one.md").write_text(FEEDBACK)
    (d / "topic_two.md").write_text(SIGNAL)
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
            "MEMORY.md", "topic_one.md", "topic_two.md",
        ]
        assert all(r[1] == 1 and r[2] is True and r[3] is None for r in rows)

    def test_frontmatter_is_projected_onto_columns(self, conn, projects_root):
        import_memories(conn, projects_root=projects_root)
        row = conn.execute(
            "SELECT memory_name, memory_type, node_type, origin_session_id, "
            "is_index, has_frontmatter FROM dim_memory "
            "WHERE file_name = 'topic_one.md'"
        ).fetchone()
        assert row == ("first-topic", "feedback", "memory", "sess-A", False, True)

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
            "SELECT body_text FROM dim_memory WHERE file_name = 'topic_one.md'"
        ).fetchone()[0]
        assert "Body text for the first example topic" in body
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
            "SELECT session_key FROM dim_memory WHERE file_name = 'topic_one.md'"
        ).fetchone()
        assert row[0] == "sk-A"

    def test_unresolvable_origin_session_is_kept_raw(self, conn, projects_root):
        """sess-B is not in dim_session. The link must degrade to a NULL
        session_key while the stated id survives -- dropping it would lose
        the only evidence of which session wrote the memory."""
        import_memories(conn, projects_root=projects_root)
        row = conn.execute(
            "SELECT session_key, origin_session_id FROM dim_memory "
            "WHERE file_name = 'topic_two.md'"
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
        path = projects_root / PROJECT_DIR / "memory" / "topic_one.md"
        path.write_text(
            FEEDBACK.replace("2026-07-30T03:57:12.156Z", "2026-08-05T09:00:00.000Z")
        )
        import_memories(conn, projects_root=projects_root)
        assert conn.execute(
            "SELECT COUNT(*) FROM dim_memory WHERE file_name = 'topic_one.md'"
        ).fetchone()[0] == 1

    def test_body_change_opens_a_new_version_and_closes_the_old(
        self, conn, projects_root
    ):
        import_memories(conn, projects_root=projects_root)
        path = projects_root / PROJECT_DIR / "memory" / "topic_one.md"
        path.write_text(FEEDBACK.replace("for the first example", "after the mutation"))
        import_memories(conn, projects_root=projects_root)

        rows = conn.execute(
            "SELECT version_num, is_current, valid_to IS NULL FROM dim_memory "
            "WHERE file_name = 'topic_one.md' ORDER BY version_num"
        ).fetchall()
        assert rows == [(1, False, False), (2, True, True)]

    def test_exactly_one_current_row_per_memory(self, conn, projects_root):
        import_memories(conn, projects_root=projects_root)
        path = projects_root / PROJECT_DIR / "memory" / "topic_one.md"
        path.write_text(FEEDBACK.replace("for the first example", "after the mutation"))
        import_memories(conn, projects_root=projects_root)
        path.write_text(FEEDBACK.replace("for the first example", "during the mutation"))
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
        path = projects_root / PROJECT_DIR / "memory" / "topic_one.md"
        import_memories(conn, projects_root=projects_root)
        path.write_text(FEEDBACK.replace("for the first example", "after the mutation"))
        import_memories(conn, projects_root=projects_root)
        path.write_text(FEEDBACK)
        import_memories(conn, projects_root=projects_root)

        rows = conn.execute(
            "SELECT version_num, is_current FROM dim_memory "
            "WHERE file_name = 'topic_one.md' ORDER BY version_num"
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
        (projects_root / PROJECT_DIR / "memory" / "topic_two.md").unlink()
        import_memories(conn, projects_root=projects_root)

        row = conn.execute(
            "SELECT is_current, valid_to IS NOT NULL FROM dim_memory "
            "WHERE file_name = 'topic_two.md'"
        ).fetchone()
        assert row == (False, True)


class TestLinkGraph:
    def test_links_land_as_bridge_rows_in_order(self, conn, projects_root):
        import_memories(conn, projects_root=projects_root)
        rows = conn.execute(
            "SELECT target_name, ordinal FROM bridge_memory_link bl "
            "JOIN dim_memory m USING (memory_key) "
            "WHERE m.file_name = 'topic_one.md' ORDER BY ordinal"
        ).fetchall()
        assert rows == [("second-topic", 0), ("missing-topic", 1)]

    def test_link_resolves_to_a_sibling_memory(self, conn, projects_root):
        import_memories(conn, projects_root=projects_root)
        row = conn.execute(
            "SELECT bl.is_resolved, bl.target_memory_id = t.memory_id "
            "FROM bridge_memory_link bl "
            "JOIN dim_memory t ON t.file_name = 'topic_two.md' "
            "WHERE bl.target_name = 'second-topic'"
        ).fetchone()
        assert row == (True, True)

    def test_link_resolves_across_separator_conventions(self, conn, projects_root):
        """Real corpora write the file stem with underscores and the
        frontmatter name with hyphens, then link with either. `-` and `_` are
        the same identifier in different clothes, so matching modulo
        separator recovers real edges that an exact match drops."""
        d = projects_root / PROJECT_DIR / "memory"
        (d / "notes_archive.md").write_text(
            SIGNAL.replace("name: second-topic", "name: archive-notes")
        )
        (d / "linker.md").write_text(
            "---\nname: linker\n---\n\nsee [[notes-archive]]\n"
        )
        import_memories(conn, projects_root=projects_root)

        row = conn.execute(
            "SELECT is_resolved, target_file_name FROM semantic_memory_links "
            "WHERE target_name = 'notes-archive'"
        ).fetchone()
        assert row == (True, "notes_archive.md")

    def test_a_near_miss_link_is_not_fuzzy_matched(self, conn, projects_root):
        """[[signal]] is not [[second-topic]]. Resolving on prefixes or
        substrings would invent edges the author never wrote -- an unresolved
        row is the honest answer."""
        d = projects_root / PROJECT_DIR / "memory"
        (d / "linker.md").write_text(
            "---\nname: linker\n---\n\nsee [[second]] and [[second-topic-extra]]\n"
        )
        import_memories(conn, projects_root=projects_root)

        rows = conn.execute(
            "SELECT target_name, is_resolved FROM bridge_memory_link "
            "WHERE target_name IN ('second', 'second-topic-extra') ORDER BY 1"
        ).fetchall()
        assert rows == [("second", False), ("second-topic-extra", False)]

    def test_dangling_link_is_kept_unresolved(self, conn, projects_root):
        """A [[link]] to a memory that was never written is real signal --
        it marks something Claude meant to record. Dropping the row would
        hide it."""
        import_memories(conn, projects_root=projects_root)
        row = conn.execute(
            "SELECT is_resolved, target_memory_id FROM bridge_memory_link "
            "WHERE target_name = 'missing-topic'"
        ).fetchone()
        assert row == (False, None)

    def test_links_are_rebuilt_for_the_new_version_only(self, conn, projects_root):
        """Each version owns its edges. The closed version keeps the edges it
        had, so the graph can be queried as of any point in time."""
        path = projects_root / PROJECT_DIR / "memory" / "topic_one.md"
        import_memories(conn, projects_root=projects_root)
        path.write_text(FEEDBACK.replace("[[missing-topic]]", "[[other-missing-topic]]"))
        import_memories(conn, projects_root=projects_root)

        current = conn.execute(
            "SELECT bl.target_name FROM bridge_memory_link bl "
            "JOIN dim_memory m USING (memory_key) "
            "WHERE m.file_name = 'topic_one.md' AND m.is_current "
            "ORDER BY bl.ordinal"
        ).fetchall()
        assert current == [("second-topic",), ("other-missing-topic",)]

        historical = conn.execute(
            "SELECT COUNT(*) FROM bridge_memory_link bl "
            "JOIN dim_memory m USING (memory_key) "
            "WHERE m.file_name = 'topic_one.md' AND NOT m.is_current"
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
            "- [Exit codes](topic_one.md) -- pipes hide failures\n"
            "- [Signal honesty](topic_two.md) -- do not touch the check\n"
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
            ("topic_one.md", True, "markdown"),
            ("topic_two.md", True, "markdown"),
            ("never-written.md", False, "markdown"),
        ]

    def test_index_label_is_captured(self, conn, indexed):
        """The index's own label for a memory is what a human reads in the
        table of contents; it is not recoverable from the target file."""
        import_memories(conn, projects_root=indexed)
        row = conn.execute(
            "SELECT link_text FROM bridge_memory_link "
            "WHERE target_name = 'topic_one.md'"
        ).fetchone()
        assert row == ("Exit codes",)

    def test_markdown_target_resolves_on_file_name_not_memory_name(
        self, conn, indexed
    ):
        """topic_one.md carries `name: first-topic`, but a
        markdown target is a path the author wrote -- it must match the file
        name, so an index still resolves when the two disagree."""
        d = indexed / PROJECT_DIR / "memory"
        (d / "renamed_file.md").write_text(
            FEEDBACK.replace("name: first-topic", "name: totally-different")
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
        # 3 index entries + the 2 [[wiki]] links in topic_one.md
        assert counts == {"markdown": 3, "wiki": 2}

    def test_semantic_view_distinguishes_index_edges(self, conn, indexed):
        import_memories(conn, projects_root=indexed)
        row = conn.execute(
            "SELECT source_is_index, link_syntax FROM semantic_memory_links "
            "WHERE target_name = 'topic_two.md'"
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
        path = projects_root / PROJECT_DIR / "memory" / "topic_one.md"
        path.write_text(FEEDBACK.replace("for the first example", "after the mutation"))
        run_memory_import(conn, projects_root=projects_root)

        runs = conn.execute(
            "SELECT version_num, etl_run_id FROM dim_memory "
            "WHERE file_name = 'topic_one.md' ORDER BY version_num"
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
            "# index\n\n- [Exit codes](topic_one.md) -- hook\n"
        )
        conn = create_star_schema(self._narrow_warehouse(tmp_path))
        run_memory_import(conn, projects_root=projects_root)

        row = conn.execute(
            "SELECT link_syntax, is_resolved FROM bridge_memory_link "
            "WHERE target_name = 'topic_one.md'"
        ).fetchone()
        assert row == ("markdown", True)


class TestParserContractUpgrade:
    """Findings from the v0.19.0 review. Each of these passes on a freshly
    built warehouse and fails on an upgraded one, which is why none was
    caught before release -- every verification rebuilt from scratch."""

    def _import_with_wiki_links_only(self, conn, projects_root):
        """Import as the pre-0.19 parser did: wiki links only."""
        import ccutils.parsers.memory as pm

        original = pm._extract_links
        pm._extract_links = lambda body: [
            link for link in original(body) if link.syntax == "wiki"
        ]
        try:
            run_memory_import(conn, projects_root=projects_root)
        finally:
            pm._extract_links = original

    @pytest.fixture
    def indexed_root(self, projects_root):
        (projects_root / PROJECT_DIR / "memory" / "MEMORY.md").write_text(
            "# index\n\n- [Guard rails](topic_one.md) -- hook\n"
        )
        return projects_root

    def test_index_edges_appear_when_only_the_parser_changed(
        self, conn, indexed_root
    ):
        """content_hash covers the memory's CONTENT, and links are derived
        from it -- so an unchanged MEMORY.md takes the no-op path and its
        markdown edges never materialize. Every markdown edge lives in an
        index, so upgrading delivered zero of them until the index text
        happened to change."""
        self._import_with_wiki_links_only(conn, indexed_root)
        assert conn.execute(
            "SELECT COUNT(*) FROM bridge_memory_link WHERE link_syntax = 'markdown'"
        ).fetchone()[0] == 0

        run_memory_import(conn, projects_root=indexed_root)

        # Assert the edge RESOLVES, not merely that a row exists. A row with
        # is_resolved = FALSE and a NULL target is an unusable edge -- the
        # index graph is still dead -- and asserting COUNT(*) alone passes
        # straight over that.
        row = conn.execute(
            "SELECT is_resolved, target_memory_id IS NOT NULL, project_key IS NOT NULL "
            "FROM bridge_memory_link WHERE link_syntax = 'markdown'"
        ).fetchone()
        assert row == (True, True, True)

    def test_relinked_edge_is_visible_through_the_semantic_view(
        self, conn, indexed_root
    ):
        """The view is what a consumer actually reads. An edge that resolves
        in the bridge but shows NULL names through the view is still dead."""
        self._import_with_wiki_links_only(conn, indexed_root)
        run_memory_import(conn, projects_root=indexed_root)

        row = conn.execute(
            "SELECT is_resolved, target_file_name, project_name "
            "FROM semantic_memory_links WHERE link_syntax = 'markdown'"
        ).fetchone()
        assert row == (True, "topic_one.md", PROJECT_DIR)

    def test_relink_is_not_self_sealing(self, conn, indexed_root):
        """Second and third runs must not leave the edge stuck unresolved.
        _sync_links short-circuits once stored == wanted, so a relink that
        failed to resolve would never get another chance."""
        self._import_with_wiki_links_only(conn, indexed_root)
        run_memory_import(conn, projects_root=indexed_root)
        run_memory_import(conn, projects_root=indexed_root)

        assert conn.execute(
            "SELECT COUNT(*) FROM bridge_memory_link "
            "WHERE link_syntax = 'markdown' AND NOT is_resolved"
        ).fetchone()[0] == 0

    def test_a_relink_only_run_reports_the_work_it_did(self, conn, indexed_root):
        """A DELETE + INSERT + link_count UPDATE across the corpus recorded as
        zero work is how an operator upgrading concludes memory did nothing --
        especially now that the gotchas doc points them at this exact step."""
        self._import_with_wiki_links_only(conn, indexed_root)
        run_memory_import(conn, projects_root=indexed_root)

        row = conn.execute(
            "SELECT rows_inserted, rows_updated FROM fact_etl_steps "
            "WHERE step_name = 'dim_memory' ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        assert row[1] > 0, "relink work must be recorded somewhere non-zero"

    def test_relinking_does_not_open_a_spurious_version(self, conn, indexed_root):
        """The memory did not change -- our reading of it did. Opening a
        version would assert an edit that never happened, and would do it for
        every memory in the corpus at once."""
        self._import_with_wiki_links_only(conn, indexed_root)
        run_memory_import(conn, projects_root=indexed_root)

        assert conn.execute(
            "SELECT COUNT(*) FROM dim_memory WHERE file_name = 'MEMORY.md'"
        ).fetchone()[0] == 1

    def test_link_count_is_corrected_when_links_are_rebuilt(
        self, conn, indexed_root
    ):
        """link_count is a denormalized copy of the edge count; leaving it
        stale makes the column disagree with the bridge table."""
        self._import_with_wiki_links_only(conn, indexed_root)
        run_memory_import(conn, projects_root=indexed_root)

        row = conn.execute(
            "SELECT m.link_count, (SELECT COUNT(*) FROM bridge_memory_link bl "
            "WHERE bl.memory_key = m.memory_key) FROM dim_memory m "
            "WHERE m.file_name = 'MEMORY.md' AND m.is_current"
        ).fetchone()
        assert row[0] == row[1] == 1

    def test_relink_is_idempotent(self, conn, indexed_root):
        """A third import with nothing changed must not duplicate edges."""
        self._import_with_wiki_links_only(conn, indexed_root)
        run_memory_import(conn, projects_root=indexed_root)
        run_memory_import(conn, projects_root=indexed_root)

        assert conn.execute(
            "SELECT COUNT(*) FROM bridge_memory_link bl JOIN dim_memory m "
            "USING (memory_key) WHERE m.file_name = 'MEMORY.md'"
        ).fetchone()[0] == 1

    def test_legacy_rows_with_null_link_syntax_still_resolve(
        self, conn, projects_root
    ):
        """Pre-0.19 bridge rows carry link_syntax NULL. `link_syntax <>
        'markdown'` is NULL for them, so the wiki branch skipped them
        forever: a dangling link whose target was written later could never
        resolve."""
        import_memories(conn, projects_root=projects_root)
        conn.execute("UPDATE bridge_memory_link SET link_syntax = NULL")
        conn.execute(
            "UPDATE bridge_memory_link SET target_memory_id = NULL, is_resolved = FALSE"
        )

        from ccutils.etl.dim_memory import _resolve_link_targets

        _resolve_link_targets(conn)
        assert conn.execute(
            "SELECT COUNT(*) FROM bridge_memory_link WHERE is_resolved"
        ).fetchone()[0] >= 1


class TestMigratedWarehouseWrites:
    """created_at comes from a table DEFAULT, and ALTER TABLE ADD COLUMN
    carries no default -- so on a migrated warehouse the column exists but
    every row written into it is NULL, silently diverging from a freshly
    built one."""

    def _narrow(self, tmp_path):
        import duckdb

        path = tmp_path / "narrow.duckdb"
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
                memory_link_key VARCHAR, memory_key VARCHAR, memory_id VARCHAR,
                project_key VARCHAR, scope VARCHAR, owner_key VARCHAR,
                target_name VARCHAR, target_memory_id VARCHAR,
                is_resolved BOOLEAN, ordinal INTEGER
            )
            """
        )
        conn.close()
        return path

    def test_created_at_is_populated_on_a_migrated_warehouse(
        self, tmp_path, projects_root
    ):
        conn = create_star_schema(self._narrow(tmp_path))
        run_memory_import(conn, projects_root=projects_root)

        assert conn.execute(
            "SELECT COUNT(*) FROM dim_memory WHERE created_at IS NULL"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM bridge_memory_link WHERE created_at IS NULL"
        ).fetchone()[0] == 0


class TestInterruptAndGuarding:
    def test_keyboard_interrupt_is_not_swallowed(self, conn, projects_root):
        """Ctrl-C during a long archive build must reach the caller. Catching
        BaseException and returning 0 records a failed run and lets the batch
        sail on to complete(), discarding the user's interrupt."""
        import ccutils.etl.dim_memory as mod

        def interrupt(*a, **k):
            raise KeyboardInterrupt

        original = mod._collect
        mod._collect = interrupt
        try:
            with pytest.raises(KeyboardInterrupt):
                run_memory_import(conn, projects_root=projects_root)
        finally:
            mod._collect = original

        # The run must still be marked failed on the way out.
        assert conn.execute(
            "SELECT status FROM fact_etl_runs WHERE source_path = '<auto-memory>'"
        ).fetchone()[0] == "failed"

    def test_ordinary_errors_are_still_recorded_and_swallowed(
        self, conn, projects_root
    ):
        import ccutils.etl.dim_memory as mod

        original = mod._collect
        mod._collect = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        try:
            assert run_memory_import(conn, projects_root=projects_root) == 0
        finally:
            mod._collect = original


class TestRecordSource:
    def test_record_source_is_a_known_provenance_label(self, conn, projects_root):
        """lineage.py keeps an allow-list so a typo cannot land in 100k rows.
        Writing a value straight past record_source_label bypasses it."""
        from ccutils.etl.lineage import record_source_label

        run_memory_import(conn, projects_root=projects_root)
        # Every distinct value, both tables -- fetchone() validates one
        # arbitrary row and would pass over a table holding a mix.
        for table in ("dim_memory", "bridge_memory_link"):
            for (value,) in conn.execute(
                f"SELECT DISTINCT record_source FROM {table}"
            ).fetchall():
                assert record_source_label(value) == value

    def test_legacy_record_source_is_backfilled_on_open(self, tmp_path, projects_root):
        """Rows written before the label changed carry the run sentinel, which
        is not in the allow-list -- so record_source_label() raises on them and
        any filter on the new value silently misses them."""
        from ccutils.etl.lineage import record_source_label

        conn = create_star_schema(tmp_path / "legacy.duckdb")
        run_memory_import(conn, projects_root=projects_root)
        conn.execute("UPDATE dim_memory SET record_source = '<auto-memory>'")
        conn.execute("UPDATE bridge_memory_link SET record_source = '<auto-memory>'")
        conn.close()

        conn = create_star_schema(tmp_path / "legacy.duckdb")
        for table in ("dim_memory", "bridge_memory_link"):
            for (value,) in conn.execute(
                f"SELECT DISTINCT record_source FROM {table}"
            ).fetchall():
                assert record_source_label(value) == value


class TestAgentLinkIsolation:
    def test_markdown_link_does_not_cross_repositories(self, conn, tmp_path):
        """owner_key alone is not unique for agent memory -- that is why
        memory_id includes owner_root. Link resolution matching only on
        scope+owner_key+file_name can point one repo's index at another
        repo's memory."""
        repos = []
        for name in ("alpha", "beta"):
            repo = tmp_path / name
            d = repo / ".claude" / "agent-memory" / "reviewer"
            d.mkdir(parents=True)
            (d / "MEMORY.md").write_text(f"# {name}\n\n- [T](topic.md) -- hook\n")
            (d / "topic.md").write_text(f"---\nname: topic\n---\n\n{name} body\n")
            repos.append(repo)

        run_memory_import(conn, agent_repo_paths=repos)

        rows = conn.execute(
            """
            SELECT src.owner_root, tgt.owner_root
            FROM bridge_memory_link bl
            JOIN dim_memory src ON bl.memory_key = src.memory_key
            JOIN dim_memory tgt ON bl.target_memory_id = tgt.memory_id
            WHERE bl.link_syntax = 'markdown'
            """
        ).fetchall()
        assert len(rows) == 2
        for source_root, target_root in rows:
            assert source_root == target_root


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
        path = projects_root / PROJECT_DIR / "memory" / "topic_one.md"
        import_memories(conn, projects_root=projects_root)
        path.write_text(FEEDBACK.replace("for the first example", "after the mutation"))
        import_memories(conn, projects_root=projects_root)

        assert conn.execute(
            "SELECT COUNT(*) FROM semantic_memory WHERE file_name = 'topic_one.md'"
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
            "WHERE target_name = 'second-topic'"
        ).fetchone()
        assert row == ("first-topic", "second-topic", True)
