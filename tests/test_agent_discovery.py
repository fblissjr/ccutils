"""Tests for agent discovery and multi-select functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from ccutils import (
    extract_session_metadata,
    find_agent_sessions,
    generate_multi_session_index,
)
from ccutils.parsers.discovery import find_all_sessions, is_curated_out
from ccutils.parsers.session import get_session_summary


def _entry(**over):
    """One JSONL line. Agent transcripts carry the PARENT's sessionId --
    identity comes from the file stem (CLAUDE.md subagent contract)."""
    base = {
        "type": "user",
        "uuid": "u-001",
        "parentUuid": None,
        "timestamp": "2025-01-15T10:00:00.000Z",
        "message": {"role": "user", "content": "Hello"},
    }
    base.update(over)
    return json.dumps(base) + "\n"


def _write(path, **over):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_entry(**over))
    return path


@pytest.fixture
def session_dir(tmp_path):
    """Parent and agents in the REAL on-disk layout.

    ``<project>/<parent-uuid>.jsonl`` with the agents one directory down at
    ``<project>/<parent-uuid>/subagents/agent-<id>.jsonl``. The previous
    version of this fixture wrote the agents flat beside the parent, which
    is the pre-2026 layout -- no such file exists on disk any more, and
    every assertion here passed against a shape that had stopped existing.
    """
    project = tmp_path / "-home-user-projects-demo"

    parent = _write(project / "abc123.jsonl", sessionId="abc123",
                    cwd="/home/user/project")

    agents = project / "abc123" / "subagents"
    agent1 = _write(agents / "agent-xyz789.jsonl", sessionId="abc123",
                    agentId="xyz789", isSidechain=True,
                    message={"role": "user", "content": "Agent task"})
    agent2 = _write(agents / "agent-def456.jsonl", sessionId="abc123",
                    agentId="def456", isSidechain=True,
                    message={"role": "user", "content": "Another agent task"})

    other = _write(project / "other999.jsonl", sessionId="other999",
                   message={"role": "user", "content": "Different session"})

    return project, parent, agent1, agent2, other


@pytest.fixture
def nested_agent_dir(tmp_path):
    """An agent that spawned an agent, as the layout actually records it.

    Measured against 300 real agent transcripts: every one carries its ROOT
    parent's sessionId, and nested agents (sidecar ``spawnDepth`` up to 5)
    sit FLAT in the same ``subagents/`` directory as their level-1 siblings.
    The old fixture gave the level-2 agent ``sessionId: "agent-level1"`` and
    a chain to walk; no such file exists.
    """
    project = tmp_path / "-home-user-projects-demo"
    parent = _write(project / "parent123.jsonl", sessionId="parent123")

    agents = project / "parent123" / "subagents"
    agent_l1 = _write(agents / "agent-level1.jsonl", sessionId="parent123",
                      agentId="level1", isSidechain=True,
                      message={"role": "user", "content": "Level 1 agent"})
    (agents / "agent-level1.meta.json").write_text(
        json.dumps({"agentType": "Explore", "spawnDepth": 1})
    )

    agent_l2 = _write(agents / "agent-level2.jsonl", sessionId="parent123",
                      agentId="level2", isSidechain=True,
                      message={"role": "user", "content": "Level 2 agent"})
    (agents / "agent-level2.meta.json").write_text(
        json.dumps({"agentType": "Explore", "spawnDepth": 2})
    )

    return project, parent, agent_l1, agent_l2


@pytest.fixture
def workflow_agent_dir(tmp_path):
    """Workflow-tool agents nest one directory deeper.

    ``<project>/<uuid>/subagents/workflows/<wf_id>/agent-<id>.jsonl``.
    34 such files exist on disk; nothing in the pipeline knew the shape.
    A ``journal.jsonl`` sits beside them and is not an agent transcript.
    """
    project = tmp_path / "-home-user-projects-demo"
    parent = _write(project / "wfparent.jsonl", sessionId="wfparent")

    wf = project / "wfparent" / "subagents" / "workflows" / "wf_c4e3dd50"
    agent = _write(wf / "agent-w1.jsonl", sessionId="wfparent", agentId="w1",
                   isSidechain=True,
                   message={"role": "user", "content": "Workflow agent"})
    journal = wf / "journal.jsonl"
    journal.write_text(json.dumps({"type": "journal", "note": "step 1"}) + "\n")

    return project, parent, agent, journal


class TestExtractSessionMetadata:
    """Tests for extract_session_metadata function."""

    def test_regular_session(self, session_dir):
        """Regular session has sessionId but no agentId."""
        _, parent, _, _, _ = session_dir
        meta = extract_session_metadata(parent)

        assert meta["sessionId"] == "abc123"
        assert meta["agentId"] is None
        assert meta["isSidechain"] is False

    def test_agent_session(self, session_dir):
        """Agent session has agentId and isSidechain=True."""
        _, _, agent1, _, _ = session_dir
        meta = extract_session_metadata(agent1)

        assert meta["sessionId"] == "abc123"
        assert meta["agentId"] == "xyz789"
        assert meta["isSidechain"] is True

    def test_missing_fields_defaults(self, tmp_path):
        """Missing fields should have sensible defaults."""
        minimal = tmp_path / "minimal.jsonl"
        minimal.write_text(
            json.dumps(
                {
                    "type": "user",
                    "message": {"content": "Hello"},
                }
            )
            + "\n"
        )

        meta = extract_session_metadata(minimal)
        assert meta["sessionId"] is None
        assert meta["agentId"] is None
        assert meta["isSidechain"] is False

    def test_bare_scalar_lines_do_not_crash(self, tmp_path):
        """A line that parses to a non-dict (bare scalar) must be skipped,
        not crash with AttributeError on .get()."""
        f = tmp_path / "scalar.jsonl"
        f.write_text('"just a string"\n42\nnull\n'
                     + json.dumps({"type": "user", "sessionId": "s1",
                                   "message": {"content": "hi"}}) + "\n")
        meta = extract_session_metadata(f)
        assert meta["sessionId"] == "s1"

    def test_session_id_found_past_long_headerless_prefix(self, tmp_path):
        """No arbitrary line cap: sessionId after 25+ headerless summary
        lines is still found (matches extract_header_fields, which is
        unbounded -- the two must agree on the same file)."""
        lines = [json.dumps({"type": "summary", "summary": f"recap {i}"})
                 for i in range(40)]
        lines.append(json.dumps({"type": "user", "sessionId": "deep-s",
                                 "agentId": "a1", "isSidechain": True,
                                 "message": {"content": "hi"}}))
        f = tmp_path / "deep.jsonl"
        f.write_text("\n".join(lines) + "\n")
        meta = extract_session_metadata(f)
        assert meta["sessionId"] == "deep-s"
        assert meta["agentId"] == "a1"
        assert meta["isSidechain"] is True

    def test_empty_file_returns_empty_dict(self, tmp_path):
        """Empty file returns empty metadata."""
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")

        meta = extract_session_metadata(empty)
        assert meta == {}

    def test_cwd_trailing_a_later_line_is_still_found(self, tmp_path):
        """Real data: queue-operation/last-prompt entries carry sessionId
        but no cwd, and can precede the user/attachment entries that do --
        cwd must not be pinned to whichever line supplies sessionId first."""
        lines = [
            json.dumps({"type": "queue-operation", "sessionId": "s1",
                        "operation": "enqueue"}),
            json.dumps({"type": "user", "sessionId": "s1",
                        "cwd": "/private/tmp/sandbox/run-1",
                        "message": {"content": "hi"}}),
        ]
        f = tmp_path / "queued.jsonl"
        f.write_text("\n".join(lines) + "\n")

        meta = extract_session_metadata(f)
        assert meta["sessionId"] == "s1"
        assert meta["cwd"] == "/private/tmp/sandbox/run-1"


class TestFindAgentSessions:
    """Attaching a session's subagent transcripts, against the real layout.

    Every assertion here was previously written against agents sitting flat
    beside their parent. They all passed and the function returned nothing
    on real data -- the picker path shipped with subagents silently absent.
    """

    def test_finds_agents_in_the_subagents_directory(self, session_dir):
        _, parent, agent1, agent2, _ = session_dir

        result = find_agent_sessions([parent])

        assert parent in result
        assert set(result[parent]) == {agent1, agent2}

    def test_ignores_unrelated_sessions(self, session_dir):
        _, parent, _, _, other = session_dir

        result = find_agent_sessions([parent])

        for agents in result.values():
            assert other not in agents

    def test_multiple_parents(self, session_dir):
        project, parent, agent1, agent2, other = session_dir
        other_agent = _write(
            project / "other999" / "subagents" / "agent-other.jsonl",
            sessionId="other999", agentId="other", isSidechain=True,
            message={"role": "user", "content": "Other agent"},
        )

        result = find_agent_sessions([parent, other])

        assert set(result[parent]) == {agent1, agent2}
        assert result[other] == [other_agent]

    def test_nested_agents_are_found(self, nested_agent_dir):
        """A nested agent is a sibling file, not a deeper directory."""
        _, parent, agent_l1, agent_l2 = nested_agent_dir

        result = find_agent_sessions([parent], recursive=True)

        assert set(result[parent]) == {agent_l1, agent_l2}

    def test_recursive_false_returns_the_same_set(self, nested_agent_dir):
        """`recursive` no longer selects anything, and says so.

        The old test asserted recursive=False returned level 1 only. The
        layout cannot express that: every descendant sits flat in one
        directory carrying the same sessionId, so depth is not derivable
        from the files this function looks at. Retained as a no-op kwarg
        rather than silently returning a subset that depth never defined.
        Stated depth lives in the `.meta.json` `spawnDepth` sidecar; if a
        depth filter is ever wanted, it belongs there.
        """
        _, parent, agent_l1, agent_l2 = nested_agent_dir

        assert set(find_agent_sessions([parent], recursive=False)[parent]) == {
            agent_l1,
            agent_l2,
        }

    def test_workflow_agents_are_found(self, workflow_agent_dir):
        """Workflow agents nest a directory deeper and must still attach."""
        _, parent, agent, _journal = workflow_agent_dir

        result = find_agent_sessions([parent])

        assert result[parent] == [agent]

    def test_workflow_journal_is_not_an_agent(self, workflow_agent_dir):
        """journal.jsonl shares the directory and is not a transcript."""
        _, parent, _agent, journal = workflow_agent_dir

        assert journal not in find_agent_sessions([parent])[parent]

    def test_empty_list_returns_empty(self, session_dir):
        assert find_agent_sessions([]) == {}

    def test_session_with_no_agents(self, session_dir):
        _, _, _, _, other = session_dir

        result = find_agent_sessions([other])

        assert result[other] == []

    def test_an_agent_path_has_no_agents_of_its_own(self, nested_agent_dir):
        """Passing an agent file must not walk into a sibling's directory."""
        _, _parent, agent_l1, _ = nested_agent_dir

        assert find_agent_sessions([agent_l1])[agent_l1] == []


class TestAgentTranscriptSummary:
    """An agent's task prompt arrives as an `isMeta` user entry.

    `_get_jsonl_summary` skips isMeta entries, which is right for a parent
    session (they are harness injections) and wrong for an agent transcript,
    where that entry IS the delegated task -- the most descriptive line in
    the file. The consequence was not cosmetic: html and markdown exports
    drop anything summarising to "(no summary)", so those agents vanished
    from browsable output. Measured over a 400-file sample of real agent
    transcripts, 23 summarised to "(no summary)"; 17 of them recover here.
    """

    def _agent_with_meta_prompt(self, tmp_path):
        f = tmp_path / "-proj" / "s1" / "subagents" / "agent-a1.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(
            json.dumps({
                "type": "user", "isMeta": True, "isSidechain": True,
                "sessionId": "s1", "agentId": "a1",
                "timestamp": "2025-01-15T10:00:00.000Z",
                "message": {"role": "user", "content": "Review the diff for bugs"},
            }) + "\n"
            + json.dumps({
                "type": "attachment", "isSidechain": True, "sessionId": "s1",
                "timestamp": "2025-01-15T10:00:01.000Z",
            }) + "\n"
        )
        return f

    def test_meta_task_prompt_becomes_the_summary(self, tmp_path):
        agent = self._agent_with_meta_prompt(tmp_path)

        assert get_session_summary(agent) == "Review the diff for bugs"

    def test_such_an_agent_is_not_curated_out(self, tmp_path):
        agent = self._agent_with_meta_prompt(tmp_path)

        assert not is_curated_out(get_session_summary(agent))

    def test_render_exports_keep_it(self, tmp_path):
        """The end the user sees: default (curated) discovery keeps it."""
        self._agent_with_meta_prompt(tmp_path)

        found = find_all_sessions(tmp_path, include_agents=True)

        stems = [s["path"].stem for p in found for s in p["sessions"]]
        assert "agent-a1" in stems

    def test_parent_sessions_still_skip_meta_entries(self, tmp_path):
        """No regression: isMeta on a normal session is still not a summary."""
        f = tmp_path / "-proj" / "s2.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(
            json.dumps({
                "type": "user", "isMeta": True, "sessionId": "s2",
                "timestamp": "2025-01-15T10:00:00.000Z",
                "message": {"role": "user", "content": "injected harness text"},
            }) + "\n"
            + json.dumps({
                "type": "user", "sessionId": "s2",
                "timestamp": "2025-01-15T10:00:01.000Z",
                "message": {"role": "user", "content": "the real first prompt"},
            }) + "\n"
        )

        assert get_session_summary(f) == "the real first prompt"

    def test_a_warmup_agent_is_still_curated_out(self, tmp_path):
        """Recovering real agents must not also un-hide warmup ones."""
        f = tmp_path / "-proj" / "s3" / "subagents" / "agent-w.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(
            json.dumps({
                "type": "user", "isMeta": True, "isSidechain": True,
                "sessionId": "s3", "agentId": "w",
                "timestamp": "2025-01-15T10:00:00.000Z",
                "message": {"role": "user", "content": "Warmup"},
            }) + "\n"
        )

        assert is_curated_out(get_session_summary(f))


class TestGenerateMultiSessionIndex:
    """The flat index for a multi-session `local` run.

    Claim: these assertions previously encoded the nested layout --
    `href="<stem>/index.html"`, a sibling transcript.css, an `index-item`
    class. C2/C3 made the archive flat and self-contained, so those are the
    wrong shape now. The CLAIMS survive: every session is reachable, agents
    are distinguishable, a custom title is honoured. Only the format moved.
    """

    def test_generates_index_html(self, session_dir, tmp_path):
        _, parent, agent1, agent2, _ = session_dir
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        agent_map = {parent: [agent1, agent2]}
        result = generate_multi_session_index(output_dir, [parent], agent_map=agent_map)
        assert result == output_dir / "index.html"
        assert result.exists()

    def test_links_straight_to_the_transcript(self, session_dir, tmp_path):
        """Flat layout: <stem>.html, not <stem>/index.html."""
        _, parent, _, _, _ = session_dir
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generate_multi_session_index(output_dir, [parent])
        html_out = (output_dir / "index.html").read_text()
        assert parent.stem in html_out
        assert f'href="{parent.stem}.html"' in html_out
        assert f'href="{parent.stem}/index.html"' not in html_out

    def test_agent_sessions_are_labelled(self, session_dir, tmp_path):
        _, parent, agent1, _, _ = session_dir
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        agent_map = {parent: [agent1]}
        generate_multi_session_index(output_dir, [parent, agent1], agent_map=agent_map)
        html_out = (output_dir / "index.html").read_text()
        assert ">agent<" in html_out

    def test_parents_are_not_labelled_agent(self, session_dir, tmp_path):
        """agent_map keys are PARENTS, so a lookup against it must never
        drive the agent label. The removed branch tested a stem against a
        Path-keyed dict: dead as written, and wrong if repaired literally.
        The `agent-` filename prefix is the only signal."""
        _, parent, agent1, _, _ = session_dir
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        agent_map = {parent: [agent1]}
        generate_multi_session_index(
            output_dir, [parent, agent1], agent_map=agent_map
        )
        html_out = (output_dir / "index.html").read_text()
        assert html_out.count(">agent<") == 1
        assert html_out.count(">session<") == 1

    def test_is_self_contained(self, session_dir, tmp_path):
        """Styling and script are inlined and hash-pinned -- no siblings."""
        _, parent, _, _, _ = session_dir
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generate_multi_session_index(output_dir, [parent])
        html_out = (output_dir / "index.html").read_text()
        assert 'href="transcript.css"' not in html_out
        assert 'src="transcript.js"' not in html_out
        assert "<style>" in html_out and "sha256-" in html_out

    def test_every_card_is_filterable(self, session_dir, tmp_path):
        _, parent, agent1, _, _ = session_dir
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generate_multi_session_index(output_dir, [parent, agent1])
        html_out = (output_dir / "index.html").read_text()
        assert html_out.count("data-search=") == 2


