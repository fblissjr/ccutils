"""Tests for the auto-memory parser (parsers/memory.py).

Claude Code writes auto memory as plain markdown under
``<HOME>/.claude/projects/<project>/memory/`` (project scope) and under the
``agent-memory`` directories (subagent scope). Files carry optional YAML
frontmatter; ``MEMORY.md`` is the index and often has none.

The parser is deliberately hand-rolled rather than pulling in PyYAML: every
frontmatter line in the real corpus is a plain ``key: value`` with at most one
level of nesting under ``metadata:``, and the project has no yaml dependency.

The load-bearing behaviour here is ``content_hash``. Claude Code rewrites the
``modified:`` frontmatter stamp on every write, so hashing the raw file would
manufacture a new SCD version every time a memory was touched without its
content changing. The hash must cover the meaning of the memory and exclude
the write stamp.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ccutils.parsers.memory import (
    MemoryFile,
    MemoryLink,
    iter_agent_memories,
    iter_project_memories,
    parse_memory_file,
)

FULL = """\
---
name: check-exit-codes
description: "Piping a validator into tail reports the filter's status, not the check's"
metadata:
  node_type: memory
  type: feedback
  originSessionId: sess-A
  modified: 2026-07-30T03:57:12.156Z
---

Never verify a gate by piping it into `tail`.

Related: [[signal-honesty]] and [[derive-at-write-time]].
"""

FLAT = """\
---
name: older-format
description: written before metadata nesting existed
type: project
---

Body of the older-format memory.
"""

NO_FRONTMATTER = """\
# project memory

An index with no frontmatter at all. Links to [[check-exit-codes]].
"""


def write(tmp_path, name: str, text: str):
    p = tmp_path / name
    p.write_text(text)
    return p


class TestFrontmatterParsing:
    def test_nested_metadata_fields_are_extracted(self, tmp_path):
        path = write(tmp_path, "check-exit-codes.md", FULL)
        m = parse_memory_file(path, scope="project", owner_key="-work-proj")

        assert isinstance(m, MemoryFile)
        assert m.memory_name == "check-exit-codes"
        assert m.description.startswith("Piping a validator")
        assert m.memory_type == "feedback"
        assert m.node_type == "memory"
        assert m.origin_session_id == "sess-A"
        assert m.modified == datetime(
            2026, 7, 30, 3, 57, 12, 156000, tzinfo=timezone.utc
        )
        assert m.has_frontmatter is True
        assert m.is_index is False
        assert m.scope == "project"
        assert m.owner_key == "-work-proj"

    def test_quoted_description_keeps_inner_punctuation(self, tmp_path):
        """Surrounding quotes are stripped; a colon inside the value is not a
        key separator. Four descriptions in the real corpus contain colons."""
        path = write(
            tmp_path,
            "quoted.md",
            '---\nname: q\ndescription: "a: b, and c"\n---\n\nbody\n',
        )
        m = parse_memory_file(path, scope="project", owner_key="p")
        assert m.description == "a: b, and c"

    def test_flat_type_is_read_when_metadata_block_absent(self, tmp_path):
        """15 files in the real corpus predate the metadata: block and carry
        a top-level type:. Losing them would blank memory_type for a sixth
        of the corpus."""
        path = write(tmp_path, "older-format.md", FLAT)
        m = parse_memory_file(path, scope="project", owner_key="p")
        assert m.memory_type == "project"
        assert m.node_type is None
        assert m.origin_session_id is None

    def test_file_without_frontmatter_falls_back_to_stem(self, tmp_path):
        path = write(tmp_path, "MEMORY.md", NO_FRONTMATTER)
        m = parse_memory_file(path, scope="project", owner_key="p")
        assert m.has_frontmatter is False
        assert m.memory_name == "MEMORY"
        assert m.memory_type is None
        assert m.is_index is True
        assert m.body_text.strip().startswith("# project memory")

    def test_body_excludes_the_frontmatter_block(self, tmp_path):
        path = write(tmp_path, "check-exit-codes.md", FULL)
        m = parse_memory_file(path, scope="project", owner_key="p")
        assert "originSessionId" not in m.body_text
        assert m.body_text.strip().startswith("Never verify a gate")


class TestWikiLinks:
    def test_links_are_extracted_in_order(self, tmp_path):
        path = write(tmp_path, "a.md", FULL)
        m = parse_memory_file(path, scope="project", owner_key="p")
        assert [(x.target, x.syntax) for x in m.links] == [
            ("signal-honesty", "wiki"),
            ("derive-at-write-time", "wiki"),
        ]

    def test_links_inside_fenced_code_are_ignored(self, tmp_path):
        """A fenced block showing the link syntax is documentation, not an
        edge in the memory graph."""
        text = (
            "---\nname: a\n---\n\nreal [[one]]\n\n"
            "```\nliteral [[not-an-edge]]\n```\n\ntrailing [[two]]\n"
        )
        path = write(tmp_path, "a.md", text)
        m = parse_memory_file(path, scope="project", owner_key="p")
        assert [x.target for x in m.links] == ["one", "two"]

    def test_repeated_link_is_kept_once_per_occurrence(self, tmp_path):
        text = "---\nname: a\n---\n\n[[x]] then [[x]] again\n"
        path = write(tmp_path, "a.md", text)
        m = parse_memory_file(path, scope="project", owner_key="p")
        assert [x.target for x in m.links] == ["x", "x"]


class TestMarkdownIndexLinks:
    """MEMORY.md is an index and points at topic files with
    ``- [Title](file.md) -- hook``, not with [[wiki]] syntax. Reading only
    the wiki form drops every index edge -- on a real corpus that was 71 of
    140 edges, all of them originating in an index."""

    def test_markdown_link_to_a_sibling_is_an_edge(self, tmp_path):
        path = write(
            tmp_path,
            "MEMORY.md",
            "# index\n\n- [Cowork transcripts](cowork-local.md) -- hook text\n",
        )
        m = parse_memory_file(path, scope="project", owner_key="p")
        assert m.links == [
            MemoryLink(
                target="cowork-local.md",
                syntax="markdown",
                text="Cowork transcripts",
            )
        ]

    def test_both_syntaxes_share_one_document_order(self, tmp_path):
        path = write(
            tmp_path,
            "a.md",
            "---\nname: a\n---\n\n[[first]] then [second](two.md) then [[third]]\n",
        )
        m = parse_memory_file(path, scope="project", owner_key="p")
        assert [(x.target, x.syntax) for x in m.links] == [
            ("first", "wiki"),
            ("two.md", "markdown"),
            ("third", "wiki"),
        ]

    def test_non_memory_targets_are_not_edges(self, tmp_path):
        """An external URL, an in-document anchor, an image, and a path
        reaching outside the memory directory are all links, but none of
        them is a memory-to-memory edge."""
        path = write(
            tmp_path,
            "a.md",
            "---\nname: a\n---\n\n"
            "[ext](https://example.test/page.md) [top](#section) "
            "[nested](sub/dir/file.md) ![img](pic.png) [ok](real.md)\n",
        )
        m = parse_memory_file(path, scope="project", owner_key="p")
        assert [x.target for x in m.links] == ["real.md"]

    def test_fragment_is_stripped_from_the_target(self, tmp_path):
        path = write(
            tmp_path,
            "a.md",
            "---\nname: a\n---\n\nsee [part](topic.md#a-section)\n",
        )
        m = parse_memory_file(path, scope="project", owner_key="p")
        assert [x.target for x in m.links] == ["topic.md"]

    def test_markdown_links_in_fenced_code_are_ignored(self, tmp_path):
        path = write(
            tmp_path,
            "a.md",
            "---\nname: a\n---\n\nreal [a](one.md)\n\n"
            "```\n[doc](not-an-edge.md)\n```\n",
        )
        m = parse_memory_file(path, scope="project", owner_key="p")
        assert [x.target for x in m.links] == ["one.md"]


class TestContentHash:
    def test_hash_ignores_the_modified_stamp(self, tmp_path):
        """Claude Code rewrites modified: on every write. If that moved the
        hash, every touch would open a spurious SCD version and the memory
        history would be noise."""
        a = write(tmp_path, "a.md", FULL)
        b = write(
            tmp_path,
            "b.md",
            FULL.replace("2026-07-30T03:57:12.156Z", "2026-08-05T09:00:00.000Z"),
        )
        ha = parse_memory_file(a, scope="project", owner_key="p").content_hash
        hb = parse_memory_file(b, scope="project", owner_key="p").content_hash
        assert ha == hb

    def test_hash_changes_when_body_changes(self, tmp_path):
        a = write(tmp_path, "a.md", FULL)
        b = write(tmp_path, "b.md", FULL.replace("Never verify", "Always verify"))
        ha = parse_memory_file(a, scope="project", owner_key="p").content_hash
        hb = parse_memory_file(b, scope="project", owner_key="p").content_hash
        assert ha != hb

    def test_hash_changes_when_description_changes(self, tmp_path):
        """description is meaning, not a write stamp -- it is what recall
        matches on, so a change to it is a new version."""
        a = write(tmp_path, "a.md", FULL)
        b = write(tmp_path, "b.md", FULL.replace("Piping a validator", "Piping a check"))
        ha = parse_memory_file(a, scope="project", owner_key="p").content_hash
        hb = parse_memory_file(b, scope="project", owner_key="p").content_hash
        assert ha != hb


class TestProjectDiscovery:
    @pytest.fixture
    def projects_root(self, tmp_path):
        root = tmp_path / "projects"
        for proj, files in (
            ("-work-alpha", {"MEMORY.md": NO_FRONTMATTER, "check.md": FULL}),
            ("-work-beta", {"older.md": FLAT}),
            ("-work-empty", {}),
        ):
            d = root / proj / "memory"
            d.mkdir(parents=True)
            for name, text in files.items():
                (d / name).write_text(text)
        # A session transcript alongside the memory dir must not be picked up.
        (root / "-work-alpha" / "sess.jsonl").write_text('{"type":"user"}\n')
        return root

    def test_discovers_every_memory_file_with_its_project_key(self, projects_root):
        found = list(iter_project_memories(projects_root))
        assert {(m.owner_key, m.file_name) for m in found} == {
            ("-work-alpha", "MEMORY.md"),
            ("-work-alpha", "check.md"),
            ("-work-beta", "older.md"),
        }
        assert all(m.scope == "project" for m in found)

    def test_only_markdown_under_a_memory_dir_is_read(self, projects_root):
        (projects_root / "-work-alpha" / "memory" / "notes.txt").write_text("x")
        names = {m.file_name for m in iter_project_memories(projects_root)}
        assert "notes.txt" not in names
        assert "sess.jsonl" not in names

    def test_only_named_projects_are_scanned_when_filtered(self, projects_root):
        found = list(iter_project_memories(projects_root, only={"-work-beta"}))
        assert {m.owner_key for m in found} == {"-work-beta"}

    def test_missing_root_is_not_an_error(self, tmp_path):
        assert list(iter_project_memories(tmp_path / "nope")) == []


class TestAgentDiscovery:
    def test_user_scope_agent_memory_is_keyed_by_agent_name(self, tmp_path):
        d = tmp_path / "agent-memory" / "prompt-engineer"
        d.mkdir(parents=True)
        (d / "MEMORY.md").write_text(NO_FRONTMATTER)

        found = list(iter_agent_memories(user_root=tmp_path / "agent-memory"))
        assert len(found) == 1
        m = found[0]
        assert m.scope == "agent"
        assert m.owner_key == "prompt-engineer"
        assert m.agent_scope == "user"

    def test_repo_scope_covers_committed_and_local_directories(self, tmp_path):
        repo = tmp_path / "repo"
        for sub, agent in (("agent-memory", "reviewer"), ("agent-memory-local", "scout")):
            d = repo / ".claude" / sub / agent
            d.mkdir(parents=True)
            (d / "MEMORY.md").write_text(NO_FRONTMATTER)

        found = list(iter_agent_memories(repo_paths=[repo]))
        assert {(m.owner_key, m.agent_scope) for m in found} == {
            ("reviewer", "project"),
            ("scout", "local"),
        }

    def test_missing_roots_are_not_an_error(self, tmp_path):
        assert list(
            iter_agent_memories(user_root=tmp_path / "nope", repo_paths=[tmp_path / "gone"])
        ) == []
