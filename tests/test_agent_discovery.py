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


@pytest.fixture
def session_dir(tmp_path):
    """Create a directory structure with parent and agent sessions."""
    # Parent session
    parent = tmp_path / "abc123.jsonl"
    parent.write_text(
        json.dumps(
            {
                "type": "user",
                "uuid": "user-001",
                "parentUuid": None,
                "sessionId": "abc123",
                "timestamp": "2025-01-15T10:00:00.000Z",
                "cwd": "/home/user/project",
                "message": {"role": "user", "content": "Hello"},
            }
        )
        + "\n"
    )

    # Agent session linked to parent
    agent1 = tmp_path / "agent-xyz789.jsonl"
    agent1.write_text(
        json.dumps(
            {
                "type": "user",
                "uuid": "agent-user-001",
                "parentUuid": None,
                "sessionId": "abc123",
                "agentId": "xyz789",
                "isSidechain": True,
                "timestamp": "2025-01-15T10:05:00.000Z",
                "message": {"role": "user", "content": "Agent task"},
            }
        )
        + "\n"
    )

    # Another agent session linked to same parent
    agent2 = tmp_path / "agent-def456.jsonl"
    agent2.write_text(
        json.dumps(
            {
                "type": "user",
                "uuid": "agent-user-002",
                "parentUuid": None,
                "sessionId": "abc123",
                "agentId": "def456",
                "isSidechain": True,
                "timestamp": "2025-01-15T10:10:00.000Z",
                "message": {"role": "user", "content": "Another agent task"},
            }
        )
        + "\n"
    )

    # Unrelated session
    other = tmp_path / "other999.jsonl"
    other.write_text(
        json.dumps(
            {
                "type": "user",
                "uuid": "other-001",
                "parentUuid": None,
                "sessionId": "other999",
                "timestamp": "2025-01-15T11:00:00.000Z",
                "message": {"role": "user", "content": "Different session"},
            }
        )
        + "\n"
    )

    return tmp_path, parent, agent1, agent2, other


@pytest.fixture
def nested_agent_dir(tmp_path):
    """Create a directory with nested agents (agent spawning agent)."""
    # Parent session
    parent = tmp_path / "parent123.jsonl"
    parent.write_text(
        json.dumps(
            {
                "type": "user",
                "uuid": "user-001",
                "sessionId": "parent123",
                "timestamp": "2025-01-15T10:00:00.000Z",
                "message": {"role": "user", "content": "Start"},
            }
        )
        + "\n"
    )

    # Level 1 agent (child of parent)
    agent_l1 = tmp_path / "agent-level1.jsonl"
    agent_l1.write_text(
        json.dumps(
            {
                "type": "user",
                "uuid": "l1-001",
                "sessionId": "parent123",
                "agentId": "level1",
                "isSidechain": True,
                "timestamp": "2025-01-15T10:05:00.000Z",
                "message": {"role": "user", "content": "Level 1 agent"},
            }
        )
        + "\n"
    )

    # Level 2 agent (child of level 1 agent)
    # Note: sessionId matches the agent-level1 file stem
    agent_l2 = tmp_path / "agent-level2.jsonl"
    agent_l2.write_text(
        json.dumps(
            {
                "type": "user",
                "uuid": "l2-001",
                "sessionId": "agent-level1",
                "agentId": "level2",
                "isSidechain": True,
                "timestamp": "2025-01-15T10:10:00.000Z",
                "message": {"role": "user", "content": "Level 2 agent"},
            }
        )
        + "\n"
    )

    return tmp_path, parent, agent_l1, agent_l2


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
    """Tests for find_agent_sessions function."""

    def test_finds_agents_for_parent(self, session_dir):
        """Should find all agents linked to a parent session."""
        _, parent, agent1, agent2, _ = session_dir

        result = find_agent_sessions([parent])

        assert parent in result
        assert len(result[parent]) == 2
        assert agent1 in result[parent]
        assert agent2 in result[parent]

    def test_ignores_unrelated_sessions(self, session_dir):
        """Should not include agents from unrelated sessions."""
        _, parent, _, _, other = session_dir

        result = find_agent_sessions([parent])

        # other session's agents should not be included
        for agents in result.values():
            assert other not in agents

    def test_multiple_parents(self, session_dir, tmp_path):
        """Should handle multiple parent sessions."""
        _, parent, agent1, agent2, other = session_dir

        # Create an agent for the 'other' session
        other_agent = tmp_path / "agent-other.jsonl"
        other_agent.write_text(
            json.dumps(
                {
                    "type": "user",
                    "sessionId": "other999",
                    "agentId": "other",
                    "isSidechain": True,
                    "message": {"content": "Other agent"},
                }
            )
            + "\n"
        )

        result = find_agent_sessions([parent, other])

        assert parent in result
        assert other in result
        assert len(result[parent]) == 2
        assert len(result[other]) == 1
        assert other_agent in result[other]

    def test_recursive_finds_nested_agents(self, nested_agent_dir):
        """Should recursively find agents spawned by agents."""
        _, parent, agent_l1, agent_l2 = nested_agent_dir

        result = find_agent_sessions([parent], recursive=True)

        # Should find both levels
        all_agents = []
        for agents in result.values():
            all_agents.extend(agents)

        assert agent_l1 in all_agents
        assert agent_l2 in all_agents

    def test_non_recursive_skips_nested(self, nested_agent_dir):
        """Non-recursive mode should only find direct children."""
        _, parent, agent_l1, agent_l2 = nested_agent_dir

        result = find_agent_sessions([parent], recursive=False)

        # Should only find level 1
        assert parent in result
        assert agent_l1 in result[parent]
        # Level 2 should not be directly under parent
        assert agent_l2 not in result[parent]

    def test_empty_list_returns_empty(self, session_dir):
        """Empty input returns empty result."""
        result = find_agent_sessions([])
        assert result == {}

    def test_session_with_no_agents(self, session_dir):
        """Session with no agents returns empty list for that session."""
        _, _, _, _, other = session_dir

        result = find_agent_sessions([other])

        assert other in result
        assert result[other] == []


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


